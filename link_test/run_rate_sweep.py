#!/usr/bin/env python3
"""
run_rate_sweep.py — 一台筆電同時接 TX、RX 兩塊板子時，自動跑速率掃描。

流程：
  1. 鏈路檢查與對齊校正（約 2 秒）：TX 端只會以 32 bytes 為單位把資料交給韌體，
     不滿 32 的尾巴會卡住。先量出卡住多少、補齊歸零，否則之後每個封包都會錯位。
  2. 在另一個視窗啟動 seq_rx_logger.py（RX），等它開好序列埠。
  3. 在這個視窗執行 seq_tx.py（TX）照計畫送包。
  4. TX 送完後，等 RX 15 秒沒收到新封包自動結束，再把總結印在這個視窗。

用法：
  python run_rate_sweep.py --tx COM6 --rx COM3
  python run_rate_sweep.py --tx COM6 --rx COM3 --plan 150x20,160x20,170x20 --label rate_fine
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKT_LEN = 230
ALIGN = 32


def parse_records(buf):
    """把 RX 韌體輸出切成 [長度][內容][RSSI][狀態] 紀錄，回傳內容清單。"""
    out, i = [], 0
    while i < len(buf):
        L = buf[i]
        if L == 0 or i + L + 3 > len(buf):
            break
        out.append(bytes(buf[i + 1:i + 1 + L]))
        i += L + 3
    return out


def send_and_listen(tx, rx, data, wait=0.4):
    rx.reset_input_buffer()
    if data:
        tx.write(data)
        tx.flush()
    t_end = time.time() + wait
    buf = bytearray()
    while time.time() < t_end:
        buf.extend(rx.read(4096))
    return buf


def calibrate(tx_port, rx_port, baud):
    """
    TX 端只會以 32 bytes 為單位把 UART 資料交給韌體，不滿 32 的尾巴會卡住。
    這裡先量出目前卡住幾個 byte，再補齊歸零，讓之後的封包邊界對齊。
    """
    import serial
    print("=== 步驟 1：鏈路檢查與 32-byte 對齊校正 ===")
    try:
        rx = serial.Serial(rx_port, baud, timeout=0.05)
    except Exception as e:
        print(f"[錯誤] 打不開 RX 的 {rx_port}：{e}")
        return False
    try:
        tx = serial.Serial(tx_port, baud, timeout=1, write_timeout=5)
    except Exception as e:
        rx.close()
        print(f"[錯誤] 打不開 TX 的 {tx_port}：{e}")
        return False
    time.sleep(0.3)
    ok = False
    try:
        seq_a = bytes(range(0x80, 0x80 + ALIGN))   # 量卡住量用
        seq_b = bytes(range(0xC0, 0xC0 + ALIGN))   # 驗證用
        for attempt in range(1, 4):
            # (1) 送 32 bytes 的序列 A：卡住的 s 個舊位元組 + 序列 A 的前 (32-s) 個會一起被送出
            buf = send_and_listen(tx, rx, seq_a)
            recs = parse_records(buf)
            if not recs:
                print(f"  第 {attempt} 次：RX 沒收到任何東西。請確認兩塊板子的天線都有接、電源有開。")
                continue
            payload = b''.join(recs)
            k = max((k for k in range(0, ALIGN + 1)
                     if len(payload) >= k and payload[len(payload) - k:] == seq_a[:k]), default=0)
            s = ALIGN - k
            print(f"  第 {attempt} 次：收到 {len(recs)} 筆共 {len(payload)} bytes，"
                  f"推算原本卡住 {len(payload) - k} bytes，現在卡住 {s} bytes")
            # (2) 補 (32 - s) 個位元組，把卡住的量湊成 32 → 全部送出、歸零
            if s % ALIGN:
                send_and_listen(tx, rx, bytes(ALIGN - s), wait=0.3)
            # (3) 驗證：送序列 B，應該剛好收到完整的序列 B、前後沒有其他位元組
            buf = send_and_listen(tx, rx, seq_b)
            payload = b''.join(parse_records(buf))
            if payload == seq_b:
                print("  校正完成：TX 端沒有卡住的位元組，封包邊界已對齊。")
                ok = True
                break
            print(f"  驗證沒通過（收到 {len(payload)} bytes：{payload[:8].hex(' ')}…），重試")
    finally:
        tx.close()
        rx.close()
    if not ok:
        print("[錯誤] 校正失敗。")
        list_ports()
    return ok


def list_ports():
    import serial.tools.list_ports
    print("\n目前電腦上的 COM port（資料要走「Application/User UART」那一個）：")
    for p in serial.tools.list_ports.comports():
        print(f"  {p.device:8s}  {p.description}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tx', required=True, help='TX 板的 COM port，例如 COM6')
    ap.add_argument('--rx', required=True, help='RX 板的 COM port，例如 COM3')
    ap.add_argument('--baud', type=int, default=921600)
    ap.add_argument('--plan', default='50x20,100x20,150x20,200x20,250x20')
    ap.add_argument('--label', default='rate_sweep')
    ap.add_argument('--skip-probe', action='store_true')
    args = ap.parse_args()
    py = sys.executable

    if not args.skip_probe:
        if not calibrate(args.tx, args.rx, args.baud):
            return 1
        print()
        time.sleep(0.5)

    print("=== 步驟 2：在另一個視窗啟動 RX 紀錄 ===")
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    before = set(glob.glob(os.path.join(HERE, 'results', f'{args.label}_*')))
    rx_proc = subprocess.Popen([py, 'seq_rx_logger.py', '--port', args.rx, '--baud', str(args.baud),
                                '--label', args.label, '--idle-stop', '15'],
                               cwd=HERE, creationflags=flags)
    time.sleep(4.0)
    if rx_proc.poll() is not None:
        print("[錯誤] RX 程式一啟動就結束了，請看 RX 視窗的錯誤訊息。")
        return 3

    print("=== 步驟 3：TX 送包 ===")
    try:
        tx_rc = subprocess.call([py, 'seq_tx.py', '--port', args.tx, '--baud', str(args.baud),
                                 '--plan', args.plan, '--label', args.label, '--start-delay', '2'],
                                cwd=HERE)
    except KeyboardInterrupt:
        tx_rc = -1
        print("\n[中斷] TX 已停止，等 RX 收尾…")

    print("\n=== 步驟 4：等 RX 收尾（約 15 秒）===")
    try:
        rx_proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        print("[警告] RX 90 秒內沒有自己結束，強制停止。")
        rx_proc.terminate()

    after = set(glob.glob(os.path.join(HERE, 'results', f'{args.label}_*')))
    new_dirs = sorted(d for d in after - before if os.path.isdir(d))
    if not new_dirs:
        print("[錯誤] 找不到 RX 的結果資料夾。")
        return 4
    out = new_dirs[-1]
    summ = os.path.join(out, 'summary.txt')
    print(f"\n結果資料夾：{out}\n")
    if os.path.exists(summ):
        with open(summ, encoding='utf-8-sig') as f:
            print(f.read())
    print("\n完成。可以告訴 Claude 跑完了。")
    return 0 if tx_rc == 0 else 5


if __name__ == '__main__':
    sys.exit(main())
