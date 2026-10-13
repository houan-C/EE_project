#!/usr/bin/env python3
"""
seq_rx_logger.py — 實驗 1：逐包紀錄與統計（接 RX 板的筆電執行）

搭配現有的 CC1352 v8 RX 韌體（rfUARTBridge_CC1352R1_RX_WB_v8_DIO_5_enabled），不需要改韌體。
韌體每收到一個 RF 封包，就從 UART 送出：
    [長度 L (1 B)] [封包內容 (L B)] [RSSI (1 B, 有號數 dBm)] [狀態 (1 B)]
CRC 錯誤的 RF 封包會被韌體直接丟掉，所以「掉包」就是序號沒出現。

用法範例：
    python seq_rx_logger.py --list                 列出所有 COM port
    python seq_rx_logger.py --label 2km_LNA        開始紀錄（先開 RX，再開 TX）
    python seq_rx_logger.py --replay results/2km_LNA_20261020_101500/raw.bin
                                                   用之前存下的原始資料重新分析

輸出（每次執行一個資料夾 results/<label>_<日期時間>/）：
    raw.bin          序列埠收到的原始位元組（含時間戳），可用 --replay 重新分析
    packets.csv      每收到一包寫一列
    per_second.csv   每個段落、每一秒的成功率與 RSSI
    bursts.csv       連續掉包長度的分布
    summary.txt      人看的總結（和結束時畫面上印的一樣）
    summary.json     程式讀的總結
"""

import argparse
import csv
import json
import os
import statistics
import struct
import sys
import time
import zlib
from datetime import datetime

# ---------------------------------------------------------------------------
# 封包格式（必須與 seq_tx.py 一致）
# ---------------------------------------------------------------------------
PKT_LEN = 230
MAGIC = b'\xA7SEQ'
HDR_FMT = '<4sIHHIId'          # magic, seq, seg_id, rate, seg_index, seg_total, tx_time
HDR_LEN = struct.calcsize(HDR_FMT)   # 28
MAX_FW_LEN = 240                     # 韌體 MAX_LENGTH
RAW_CHUNK_HDR = '<dI'                # raw.bin：每一段 = (收到時間, 長度) + 資料
RAW_CHUNK_HDR_LEN = struct.calcsize(RAW_CHUNK_HDR)

HEALTH_PSR = 99.5   # 健康檢查：每段成功率至少要這麼高


def int8(b):
    return b - 256 if b > 127 else b


# ---------------------------------------------------------------------------
# 解析韌體輸出的資料流
# ---------------------------------------------------------------------------
class FirmwareStreamParser:
    """把 [L][payload][rssi][status] 一筆一筆切出來，並在失去同步時自動重新對齊。"""

    def __init__(self):
        self.buf = bytearray()
        self.synced = False
        self.skipped_bytes = 0
        self.resyncs = 0

    def feed(self, data):
        self.buf.extend(data)
        out = []
        while True:
            if not self.synced:
                idx = self.buf.find(MAGIC)
                if idx == -1:
                    keep = len(MAGIC) - 1
                    if len(self.buf) > keep:
                        self.skipped_bytes += len(self.buf) - keep
                        del self.buf[:len(self.buf) - keep]
                    break
                if idx == 0:            # 前面沒有長度位元組，不可能是一筆完整紀錄
                    self.skipped_bytes += 1
                    del self.buf[:1]
                    continue
                self.skipped_bytes += idx - 1
                del self.buf[:idx - 1]
                self.synced = True
            if len(self.buf) < 1:
                break
            L = self.buf[0]
            if L == 0 or L > MAX_FW_LEN:
                self._lose_sync()
                continue
            if len(self.buf) < L + 3:
                break
            if self.buf[1:5] != MAGIC:
                # 內容開頭不是我們的標記：可能是 UART 上的雜訊，也可能是 TX 端位元組錯位。
                # 用下一筆紀錄的長度位元組來判斷：錯位時每一筆長度都一樣，雜訊則不會。
                if len(self.buf) < L + 4:
                    break               # 等下一筆資料進來再判斷
                if self.buf[L + 3] != L:
                    self._lose_sync()
                    continue
            payload = bytes(self.buf[1:1 + L])
            rssi = int8(self.buf[1 + L])
            status = self.buf[2 + L]
            del self.buf[:L + 3]
            out.append((payload, rssi, status))
        return out

    def _lose_sync(self):
        self.synced = False
        self.resyncs += 1
        self.skipped_bytes += 1
        del self.buf[:1]


# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
class Segment:
    def __init__(self, seg_id, rate, total):
        self.seg_id = seg_id
        self.rate = rate
        self.total = total
        self.rssi = {}            # seg_index -> rssi（第一次收到的）
        self.duplicates = 0
        self.latency_ms = []
        self.first_rx = None
        self.last_rx = None


class Stats:
    def __init__(self):
        self.segments = {}
        self.crc_errors = 0
        self.misaligned = 0
        self.foreign = 0
        self.short = 0
        self.records = 0

    def add(self, payload, rssi, status, rx_time):
        """回傳要寫進 packets.csv 的一列（或 None）"""
        self.records += 1
        L = len(payload)
        if payload[:4] != MAGIC:
            if MAGIC in payload:
                self.misaligned += 1
                kind = 'misaligned'
            else:
                self.foreign += 1
                kind = 'foreign'
            return [f"{rx_time:.6f}", '', '', '', '', '', rssi, f"{status:02X}", L, 0, kind, '']
        if L < PKT_LEN:
            self.short += 1
            return [f"{rx_time:.6f}", '', '', '', '', '', rssi, f"{status:02X}", L, 0, 'short', '']
        rec = payload[:PKT_LEN]
        crc_ok = (zlib.crc32(rec[:-4]) & 0xFFFFFFFF) == struct.unpack('<I', rec[-4:])[0]
        if not crc_ok:
            self.crc_errors += 1
            return [f"{rx_time:.6f}", '', '', '', '', '', rssi, f"{status:02X}", L, 0, 'crc_error', '']
        _, seq, seg_id, rate, idx, total, tx_time = struct.unpack(HDR_FMT, rec[:HDR_LEN])
        seg = self.segments.get(seg_id)
        if seg is None:
            seg = self.segments[seg_id] = Segment(seg_id, rate, total)
        if seg.first_rx is None:
            seg.first_rx = rx_time
        seg.last_rx = rx_time
        if idx in seg.rssi:
            seg.duplicates += 1
            kind = 'duplicate'
        else:
            seg.rssi[idx] = rssi
            kind = 'ok'
        latency = (rx_time - tx_time) * 1000.0
        seg.latency_ms.append(latency)
        return [f"{rx_time:.6f}", seq, seg_id, idx, total, rate, rssi, f"{status:02X}", L, 1, kind,
                f"{latency:.1f}"]


def burst_lengths(received_idx, total):
    """連續掉包長度清單（含段落開頭、結尾的掉包）"""
    bursts = []
    prev = -1
    for i in sorted(received_idx):
        if i - prev > 1:
            bursts.append(i - prev - 1)
        prev = i
    if total - 1 - prev > 0:
        bursts.append(total - 1 - prev)
    return bursts


def summarize(stats, parser, label, out_dir, bytes_total, t_begin, t_end):
    seg_rows, per_sec_rows, burst_rows = [], [], []
    lines = []
    lines.append(f"=== 實驗 1 總結：{label} ===")
    lines.append(f"紀錄時間：{datetime.fromtimestamp(t_begin):%Y-%m-%d %H:%M:%S} 起，共 {t_end - t_begin:.0f} 秒")
    lines.append(f"收到位元組 {bytes_total}，韌體紀錄 {stats.records} 筆")
    lines.append("")

    seg_ids = sorted(stats.segments)
    missing_segs = []
    if seg_ids:
        missing_segs = [s for s in range(seg_ids[0], seg_ids[-1] + 1) if s not in stats.segments]

    for sid in seg_ids:
        seg = stats.segments[sid]
        got = len(seg.rssi)
        psr = 100.0 * got / seg.total if seg.total else 0.0
        rssis = list(seg.rssi.values())
        bursts = burst_lengths(seg.rssi.keys(), seg.total)
        hist = {}
        for b in bursts:
            hist[b] = hist.get(b, 0) + 1
        for b in sorted(hist):
            burst_rows.append([sid, seg.rate, b, hist[b]])

        # 每秒：以「送出順序」切塊，每 rate 包 = TX 的 1 秒
        n_blocks = (seg.total + seg.rate - 1) // seg.rate
        for blk in range(n_blocks):
            lo, hi = blk * seg.rate, min((blk + 1) * seg.rate, seg.total)
            r = [seg.rssi[i] for i in range(lo, hi) if i in seg.rssi]
            exp = hi - lo
            per_sec_rows.append([sid, seg.rate, blk, exp, len(r),
                                 f"{100.0 * len(r) / exp:.1f}",
                                 f"{statistics.mean(r):.1f}" if r else '',
                                 min(r) if r else '', max(r) if r else ''])

        row = {
            'seg_id': sid, 'rate_pps': seg.rate, 'planned': seg.total, 'received': got,
            'psr_percent': round(psr, 2), 'duplicates': seg.duplicates,
            'rssi_mean': round(statistics.mean(rssis), 1) if rssis else None,
            'rssi_median': statistics.median(rssis) if rssis else None,
            'rssi_min': min(rssis) if rssis else None,
            'rssi_max': max(rssis) if rssis else None,
            'rssi_std': round(statistics.pstdev(rssis), 1) if len(rssis) > 1 else None,
            'loss_events': len(bursts),
            'lost_packets': seg.total - got,
            'burst_max': max(bursts) if bursts else 0,
            'burst_mean': round(statistics.mean(bursts), 2) if bursts else 0,
            'latency_ms_median': round(statistics.median(seg.latency_ms), 1) if seg.latency_ms else None,
        }
        seg_rows.append(row)
        lines.append(f"[段落 {sid}] {seg.rate} 包/秒：應收 {seg.total}，實收 {got}，成功率 {psr:.2f}%")
        if rssis:
            lines.append(f"    RSSI 平均 {row['rssi_mean']} dBm（中位數 {row['rssi_median']}，"
                         f"最小 {row['rssi_min']}，最大 {row['rssi_max']}，標準差 {row['rssi_std']}）")
        lines.append(f"    掉包 {row['lost_packets']} 包，分成 {row['loss_events']} 次；"
                     f"最長連續掉 {row['burst_max']} 包，平均每次 {row['burst_mean']} 包")
        if seg.duplicates:
            lines.append(f"    重複收到 {seg.duplicates} 包")
        if row['latency_ms_median'] is not None:
            lines.append(f"    延遲中位數約 {row['latency_ms_median']} ms（需兩台電腦已對時才有意義）")

    for s in missing_segs:
        lines.append(f"[段落 {s}] 一包都沒收到（請對照 TX 紀錄檔看這一段送了幾包）")

    lines.append("")
    lines.append("資料品質：")
    lines.append(f"    CRC 錯誤 {stats.crc_errors}、未對齊 {stats.misaligned}、過短 {stats.short}、"
                 f"非本實驗封包 {stats.foreign}、重新對齊 {parser.resyncs} 次、略過位元組 {parser.skipped_bytes}")

    problems = []
    if not seg_rows:
        problems.append("沒有收到任何本實驗的封包")
    if stats.misaligned:
        problems.append("有未對齊的封包：TX 端的位元組和 RF 封包邊界錯開了，"
                        "數據不可用。請重開 TX 板與 seq_tx.py 後重測，並降低速率")
    if stats.crc_errors or stats.short:
        problems.append("有 CRC 錯誤或過短的封包：UART 傳輸或對齊有問題，"
                        "請重開 TX 板與 seq_tx.py 後重測")
    lines.append("")
    if problems:
        lines.append("[!] 注意：")
        for p in problems:
            lines.append(f"    - {p}")
    health_ok = (not problems and not missing_segs and
                 all(r['psr_percent'] >= HEALTH_PSR for r in seg_rows))
    lines.append(f"桌面健康檢查判定（每段成功率 >= {HEALTH_PSR}%、無 CRC 錯誤、無未對齊）："
                 f"{'通過' if health_ok else '未通過'}")
    lines.append("（這個判定只在兩塊板子放在同一張桌上時有意義；場測時請看各段成功率。）")
    text = "\n".join(lines)

    with open(os.path.join(out_dir, 'per_second.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['seg_id', 'rate_pps', 'second', 'expected', 'received', 'psr_percent',
                    'rssi_mean', 'rssi_min', 'rssi_max'])
        w.writerows(per_sec_rows)
    with open(os.path.join(out_dir, 'bursts.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['seg_id', 'rate_pps', 'burst_length', 'count'])
        w.writerows(burst_rows)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8-sig') as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'label': label, 'start_time': t_begin, 'end_time': t_end, 'bytes': bytes_total,
            'segments': seg_rows, 'missing_segments': missing_segs,
            'crc_errors': stats.crc_errors, 'misaligned': stats.misaligned, 'short': stats.short,
            'foreign': stats.foreign, 'resyncs': parser.resyncs,
            'skipped_bytes': parser.skipped_bytes, 'health_check_pass': health_ok,
        }, f, ensure_ascii=False, indent=2)
    return text


# ---------------------------------------------------------------------------
# 執行
# ---------------------------------------------------------------------------
PACKET_HEADER = ['rx_time', 'seq', 'seg_id', 'seg_index', 'seg_total', 'rate_pps', 'rssi_dbm',
                 'status_hex', 'fw_len', 'crc_ok', 'kind', 'latency_ms']


def find_port(explicit):
    if explicit:
        return explicit
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    app = [p for p in ports if 'Application/User UART' in (p.description or '')]
    if len(app) == 1:
        return app[0].device
    if len(app) > 1:
        print("[錯誤] 偵測到多塊 LaunchPad，請用 --port 指定 RX 板的 COM port：")
        for p in app:
            print(f"  {p.device:8s}  {p.description}")
        sys.exit(1)
    xds = [p for p in ports if 'XDS110' in (p.description or '')]
    if xds:
        return xds[0].device
    print("[錯誤] 找不到 LaunchPad，請用 --port 指定。目前的 COM port：")
    for p in ports:
        print(f"  {p.device:8s}  {p.description}")
    sys.exit(1)


def make_out_dir(base, label):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(base, f"{label}_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def live_line(stats, recent):
    if not stats.segments:
        return "等待封包中…"
    sid = max(stats.segments)
    seg = stats.segments[sid]
    got = len(seg.rssi)
    expected_so_far = (max(seg.rssi) + 1) if seg.rssi else 0
    psr = 100.0 * got / expected_so_far if expected_so_far else 0.0
    r = [x for x in recent]
    rs = f"RSSI {statistics.mean(r):.0f} dBm（最低 {min(r)}）" if r else "RSSI --"
    warn = ""
    if stats.misaligned or stats.crc_errors:
        warn = f"  [!] 未對齊 {stats.misaligned}、CRC 錯 {stats.crc_errors}"
    return (f"[段落 {sid} | {seg.rate} 包/秒] 已收 {got}/{expected_so_far}（{psr:.1f}%）"
            f"  本秒 {len(r)} 包  {rs}{warn}")


def run_live(args):
    import serial
    port = find_port(args.port)
    try:
        ser = serial.Serial(port, args.baud, timeout=0.05)
    except Exception as e:
        sys.exit(f"[錯誤] 無法開啟 {port}：{e}")
    out_dir = make_out_dir(args.outdir, args.label)
    print(f"RX 板：{port} @ {args.baud}")
    print(f"輸出資料夾：{out_dir}")
    print(f"現在可以啟動 TX 端。按 Ctrl+C 結束；收到封包後若 {args.idle_stop:g} 秒沒有新封包也會自動結束。\n")

    parser, stats = FirmwareStreamParser(), Stats()
    raw_f = open(os.path.join(out_dir, 'raw.bin'), 'wb')
    pkt_f = open(os.path.join(out_dir, 'packets.csv'), 'w', newline='', encoding='utf-8-sig')
    pkt_w = csv.writer(pkt_f)
    pkt_w.writerow(PACKET_HEADER)

    t_begin = time.time()
    last_pkt_time = None
    last_print = time.time()
    last_flush = time.time()
    recent = []
    bytes_total = 0
    try:
        while True:
            data = ser.read(4096)
            now = time.time()
            if data:
                bytes_total += len(data)
                raw_f.write(struct.pack(RAW_CHUNK_HDR, now, len(data)) + data)
                for payload, rssi, status in parser.feed(data):
                    row = stats.add(payload, rssi, status, now)
                    pkt_w.writerow(row)
                    if row[10] == 'ok':
                        recent.append(rssi)
                        last_pkt_time = now
            if now - last_print >= 1.0:
                print("\r" + live_line(stats, recent) + "        ", end='', flush=True)
                recent = []
                last_print = now
            if now - last_flush >= 5.0:
                raw_f.flush()
                pkt_f.flush()
                last_flush = now
            if last_pkt_time is not None and now - last_pkt_time > args.idle_stop:
                print(f"\n{args.idle_stop:g} 秒沒有新封包，自動結束。")
                break
            if args.max_time and now - t_begin > args.max_time:
                print("\n到達 --max-time，結束。")
                break
    except KeyboardInterrupt:
        print("\n使用者按下 Ctrl+C，結束。")
    finally:
        ser.close()
        raw_f.close()
        pkt_f.close()

    text = summarize(stats, parser, args.label, out_dir, bytes_total, t_begin, time.time())
    print("\n" + text)
    print(f"\n所有檔案在：{out_dir}")


def run_replay(args):
    src = args.replay
    label = args.label if args.label != 'test' else os.path.basename(os.path.dirname(os.path.abspath(src)))
    out_dir = make_out_dir(args.outdir, label + "_replay")
    parser, stats = FirmwareStreamParser(), Stats()
    bytes_total = 0
    t_begin = t_end = None
    with open(src, 'rb') as f, \
            open(os.path.join(out_dir, 'packets.csv'), 'w', newline='', encoding='utf-8-sig') as pkt_f:
        pkt_w = csv.writer(pkt_f)
        pkt_w.writerow(PACKET_HEADER)
        while True:
            hdr = f.read(RAW_CHUNK_HDR_LEN)
            if len(hdr) < RAW_CHUNK_HDR_LEN:
                break
            t, n = struct.unpack(RAW_CHUNK_HDR, hdr)
            data = f.read(n)
            if t_begin is None:
                t_begin = t
            t_end = t
            bytes_total += len(data)
            for payload, rssi, status in parser.feed(data):
                pkt_w.writerow(stats.add(payload, rssi, status, t))
    if t_begin is None:
        sys.exit("[錯誤] raw.bin 是空的")
    text = summarize(stats, parser, label, out_dir, bytes_total, t_begin, t_end)
    print(text)
    print(f"\n所有檔案在：{out_dir}")


def main():
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="實驗 1：逐包紀錄與統計")
    ap.add_argument('--port', help='RX 板的 COM port（預設自動偵測）')
    ap.add_argument('--baud', type=int, default=921600)
    ap.add_argument('--label', default='test', help='測試標籤，會用在資料夾名稱，例如 2km_LNA')
    ap.add_argument('--outdir', default='results', help='輸出資料夾（預設 results）')
    ap.add_argument('--idle-stop', type=float, default=15.0,
                    help='收到封包後，幾秒沒有新封包就自動結束（預設 15）')
    ap.add_argument('--max-time', type=float, default=0, help='最長紀錄秒數，0 = 不限（預設 0）')
    ap.add_argument('--replay', help='重新分析之前存下的 raw.bin，不開序列埠')
    ap.add_argument('--list', action='store_true', help='列出所有 COM port 後結束')
    args = ap.parse_args()

    if args.list:
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device:8s}  {p.description}")
        return
    if args.replay:
        run_replay(args)
    else:
        run_live(args)


if __name__ == '__main__':
    main()
