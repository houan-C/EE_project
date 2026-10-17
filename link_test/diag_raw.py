#!/usr/bin/env python3
"""diag_raw.py — 擷取 RX 原始位元組，分析韌體怎麼把 UART 資料切成 RF 封包。"""
import json
import os
import threading
import time
from datetime import datetime
import serial

BAUD = 921600
TX_PORT = 'COM6'
RX_PORT = 'COM3'
PKT_LEN = 230
HERE = os.path.dirname(os.path.abspath(__file__))


def make_pkt(test_id, idx):
    head = b'PK' + bytes([test_id, idx])
    tail = b'EN' + bytes([test_id, idx])
    return head + bytes([0x5A]) * (PKT_LEN - 8) + tail


def main():
    rx = serial.Serial(RX_PORT, BAUD, timeout=0.02)
    tx = serial.Serial(TX_PORT, BAUD, timeout=1, write_timeout=5)
    time.sleep(0.5)
    rx.reset_input_buffer()

    chunks = []          # (t, bytes)
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            d = rx.read(4096)
            if d:
                chunks.append((time.perf_counter(), d))

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    t0 = time.perf_counter()
    tx_log = []

    print("測試 A：5 包，每包間隔 300 ms")
    for k in range(5):
        tx.write(make_pkt(1, k)); tx.flush()
        tx_log.append((time.perf_counter() - t0, 1, k))
        time.sleep(0.3)
    time.sleep(1.0)

    print("測試 B：20 包，每秒 100 包")
    nxt = time.perf_counter()
    for k in range(20):
        now = time.perf_counter()
        if nxt > now:
            time.sleep(nxt - now)
        tx.write(make_pkt(2, k))
        tx_log.append((time.perf_counter() - t0, 2, k))
        nxt += 0.01
    tx.flush()
    time.sleep(1.0)

    print("測試 C：10 包一次寫入（連續資料流）")
    tx.write(b''.join(make_pkt(3, k) for k in range(10))); tx.flush()
    tx_log.append((time.perf_counter() - t0, 3, -1))
    time.sleep(2.0)

    stop.set()
    th.join(timeout=1)
    tx.close(); rx.close()

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(HERE, 'results', f'diag_raw_{stamp}')
    os.makedirs(out, exist_ok=True)
    raw = b''.join(d for _, d in chunks)
    with open(os.path.join(out, 'rx_raw.bin'), 'wb') as f:
        f.write(raw)
    with open(os.path.join(out, 'rx_chunks.json'), 'w') as f:
        json.dump([[t - t0, len(d)] for t, d in chunks], f)
    with open(os.path.join(out, 'tx_log.json'), 'w') as f:
        json.dump(tx_log, f)
    print(f"\nRX 共收到 {len(raw)} bytes（送出 {35 * PKT_LEN} bytes 資料）。")
    print(f"已存到 {out}\n請告訴 Claude 跑完了。")


if __name__ == '__main__':
    main()
