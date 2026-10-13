#!/usr/bin/env python3
"""
seq_tx.py — 實驗 1：帶序號的封包發送端（接 TX 板的筆電執行）

搭配現有的 CC1352 v8 TX 韌體（rfUARTBridge_CC1352R1_TX_WB_v8），不需要改韌體。
韌體每收滿 230 bytes 就發一個 RF 封包，所以本程式每次剛好寫入 230 bytes，
讓「一筆測試封包 = 一個 RF 封包」。

封包格式（230 bytes，little-endian，必須與 seq_rx_logger.py 完全一致）：
    magic      4 B   b'\\xA7SEQ'，RX 用來確認封包有對齊
    seq        4 B   全域序號，從 0 開始
    seg_id     2 B   段落編號（每一段 = 一種速率跑一段時間）
    rate       2 B   這一段的發送速率（包/秒）
    seg_index  4 B   這一包在段落內的編號，從 0 開始
    seg_total  4 B   這一段總共要送幾包（RX 靠它算出「應收」數量）
    tx_time    8 B   TX 電腦送出的時間（Unix 秒，double）
    padding  198 B   固定內容，讓封包長度和影像封包一樣
    crc32      4 B   前 226 bytes 的 CRC32

用法範例：
    python seq_tx.py --list                          列出所有 COM port
    python seq_tx.py --plan 10x60,100x60             健康檢查：低負載、影像負載各 1 分鐘
    python seq_tx.py --plan 10x120,100x120 --repeat 3 --label 2km_LNA
                                                     場測：低負載 2 分、影像負載 2 分，重複 3 次
    python seq_tx.py --plan 50x20,100x20,150x20,200x20 --label rate_sweep
                                                     桌面找出最高能穩定送出的速率
"""

import argparse
import csv
import os
import struct
import sys
import time
import zlib
from datetime import datetime

# ---------------------------------------------------------------------------
# 封包格式（必須與 seq_rx_logger.py 一致）
# ---------------------------------------------------------------------------
PKT_LEN = 230
MAGIC = b'\xA7SEQ'
HDR_FMT = '<4sIHHIId'          # magic, seq, seg_id, rate, seg_index, seg_total, tx_time
HDR_LEN = struct.calcsize(HDR_FMT)   # 28
CRC_LEN = 4
PAD_LEN = PKT_LEN - HDR_LEN - CRC_LEN  # 198
PADDING = bytes(((i * 73 + 41) & 0xFF) for i in range(PAD_LEN))
assert MAGIC not in PADDING

SEGMENT_GAP_S = 2.0   # 段落之間停多久（要大於韌體 100 ms 的 flush timeout）


def build_packet(seq, seg_id, rate, seg_index, seg_total, tx_time=None):
    if tx_time is None:
        tx_time = time.time()
    body = struct.pack(HDR_FMT, MAGIC, seq, seg_id, rate, seg_index, seg_total, tx_time) + PADDING
    crc = zlib.crc32(body) & 0xFFFFFFFF
    pkt = body + struct.pack('<I', crc)
    assert len(pkt) == PKT_LEN
    return pkt


def parse_plan(plan_str, repeat):
    """'10x120,100x120' + repeat 3  →  [(rate, duration_s), ...] 共 6 段"""
    base = []
    for item in plan_str.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            rate_s, dur_s = item.lower().split('x')
            rate, dur = int(rate_s), float(dur_s)
        except ValueError:
            sys.exit(f"[錯誤] --plan 格式不對：'{item}'，應為 速率x秒數，例如 100x120")
        if not (1 <= rate <= 2000) or dur <= 0:
            sys.exit(f"[錯誤] --plan 數值不合理：'{item}'")
        base.append((rate, dur))
    if not base:
        sys.exit("[錯誤] --plan 是空的")
    return base * repeat


# ---------------------------------------------------------------------------
# COM port
# ---------------------------------------------------------------------------
def list_ports():
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("找不到任何 COM port。")
    for p in ports:
        print(f"  {p.device:8s}  {p.description}")


def find_port(explicit):
    if explicit:
        return explicit
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    app = [p for p in ports if 'Application/User UART' in (p.description or '')]
    if len(app) == 1:
        return app[0].device
    if len(app) > 1:
        print("[錯誤] 偵測到多塊 LaunchPad，請用 --port 指定 TX 板的 COM port：")
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


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="實驗 1：帶序號的封包發送端")
    ap.add_argument('--port', help='TX 板的 COM port（預設自動偵測）')
    ap.add_argument('--baud', type=int, default=921600)
    ap.add_argument('--plan', default='10x120,100x120',
                    help='段落計畫，速率x秒數，逗號分隔（預設 10x120,100x120）')
    ap.add_argument('--repeat', type=int, default=1, help='整個計畫重複幾次（預設 1）')
    ap.add_argument('--label', default='test', help='測試標籤，會用在紀錄檔名，例如 2km_LNA')
    ap.add_argument('--start-delay', type=float, default=3.0, help='開始前倒數幾秒（預設 3）')
    ap.add_argument('--outdir', default='results', help='紀錄檔資料夾（預設 results）')
    ap.add_argument('--list', action='store_true', help='列出所有 COM port 後結束')
    args = ap.parse_args()

    if args.list:
        list_ports()
        return

    segments = parse_plan(args.plan, args.repeat)
    total_pkts = sum(int(round(r * d)) for r, d in segments)
    total_time = sum(d for _, d in segments) + SEGMENT_GAP_S * (len(segments) - 1)

    import serial
    port = find_port(args.port)
    try:
        ser = serial.Serial(port, args.baud, timeout=1, write_timeout=5)
    except Exception as e:
        sys.exit(f"[錯誤] 無法開啟 {port}：{e}")

    print(f"TX 板：{port} @ {args.baud}")
    print(f"計畫：{len(segments)} 段，共 {total_pkts} 包，約 {total_time / 60:.1f} 分鐘")
    for i, (r, d) in enumerate(segments):
        print(f"  段落 {i}: {r} 包/秒 x {d:g} 秒 = {int(round(r * d))} 包")
    print("請確認 RX 端的 seq_rx_logger.py 已經在執行。")

    # 讓韌體把上一次殘留的半包 flush 掉（韌體 flush timeout 是 100 ms）
    time.sleep(0.5)
    for s in range(int(args.start_delay), 0, -1):
        print(f"  {s} 秒後開始…")
        time.sleep(1)

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.outdir, f"tx_{args.label}_{stamp}.csv")
    log_rows = []

    seq = 0
    aborted = False
    try:
        for seg_id, (rate, dur) in enumerate(segments):
            seg_total = int(round(rate * dur))
            interval = 1.0 / rate
            print(f"\n[段落 {seg_id}] {rate} 包/秒，共 {seg_total} 包")
            t_start_wall = time.time()
            t0 = time.perf_counter()
            sent = 0
            max_lag = 0.0
            last_print = t0
            for idx in range(seg_total):
                target = t0 + idx * interval
                now = time.perf_counter()
                if target > now:
                    time.sleep(target - now)
                else:
                    max_lag = max(max_lag, now - target)
                ser.write(build_packet(seq, seg_id, rate, idx, seg_total))
                seq += 1
                sent += 1
                now = time.perf_counter()
                if now - last_print >= 1.0:
                    last_print = now
                    lag_warn = "  [!] 電腦送不及，實際速率低於設定" if now - target > 0.5 else ""
                    print(f"\r  已送 {sent}/{seg_total}{lag_warn}      ", end='', flush=True)
            elapsed = time.perf_counter() - t0
            actual_rate = sent / elapsed if elapsed > 0 else 0.0
            print(f"\r  已送 {sent}/{seg_total}，實際 {actual_rate:.1f} 包/秒，最大延遲 {max_lag * 1000:.0f} ms")
            log_rows.append([seg_id, rate, dur, seg_total, sent, f"{t_start_wall:.3f}",
                             f"{time.time():.3f}", f"{actual_rate:.2f}", f"{max_lag * 1000:.1f}"])
            if seg_id != len(segments) - 1:
                time.sleep(SEGMENT_GAP_S)
    except KeyboardInterrupt:
        aborted = True
        print("\n[中斷] 使用者按下 Ctrl+C")
        if 'seg_total' in locals():
            log_rows.append([seg_id, rate, dur, seg_total, sent, f"{t_start_wall:.3f}",
                             f"{time.time():.3f}", "", "aborted"])
    except Exception as e:
        aborted = True
        print(f"\n[錯誤] 寫入序列埠失敗：{e}")
    finally:
        ser.close()

    with open(log_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['seg_id', 'rate_pps', 'duration_s', 'planned', 'sent',
                    'start_time', 'end_time', 'actual_rate_pps', 'max_lag_ms'])
        w.writerows(log_rows)

    print(f"\n{'已中斷' if aborted else '完成'}：共送出 {seq} 包。TX 紀錄：{log_path}")
    print("請等 RX 端出現總結（或約 15 秒沒收到封包會自動結束）。")


if __name__ == '__main__':
    main()
