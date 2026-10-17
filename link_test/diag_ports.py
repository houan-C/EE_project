#!/usr/bin/env python3
"""diag_ports.py — 找出哪個 COM port 是 TX、哪個是 RX，以及無線鏈路有沒有通。"""
import sys
import time
import serial
import serial.tools.list_ports

BAUD = 921600
RX_PORT = 'COM3'
TX_CANDIDATES = ['COM6']
PKT = bytes([0x5A]) * 230


def listen(rx, seconds):
    buf = bytearray()
    t_end = time.time() + seconds
    while time.time() < t_end:
        buf.extend(rx.read(4096))
    return buf


def main():
    print("=== A. 每個 COM port 的硬體資訊 ===")
    for p in serial.tools.list_ports.comports():
        vidpid = f"{p.vid:04X}:{p.pid:04X}" if p.vid is not None else "?"
        print(f"  {p.device:6s} VID:PID={vidpid}  序號={p.serial_number}  位置={p.location}  ({p.description})")
    print("  （TI LaunchPad 的 VID:PID 通常是 0451:BEF3，Silicon Labs CP210x 是 10C4:EA60。"
          "同一塊板子的兩個 port 序號會一樣。）\n")

    try:
        rx = serial.Serial(RX_PORT, BAUD, timeout=0.1)
    except Exception as e:
        print(f"[錯誤] 打不開 {RX_PORT}：{e}")
        return
    print(f"=== B. 只聽 {RX_PORT}，什麼都不送（3 秒）===")
    rx.reset_input_buffer()
    b = listen(rx, 3)
    print(f"  收到 {len(b)} bytes  開頭：{bytes(b[:16]).hex(' ')}\n")

    for txp in TX_CANDIDATES:
        print(f"=== C. 從 {txp} 送 10 包（每包 230 B，約 1 秒），同時聽 {RX_PORT} ===")
        print("  >>> 請看 TX 板的綠燈、RX 板上有沒有任何燈在閃 <<<")
        for s in (3, 2, 1):
            print(f"  {s}…", end=' ', flush=True)
            time.sleep(1)
        print()
        try:
            tx = serial.Serial(txp, BAUD, timeout=1, write_timeout=3)
        except Exception as e:
            print(f"  [錯誤] 打不開 {txp}：{e}\n")
            continue
        rx.reset_input_buffer()
        ok = True
        for _ in range(10):
            try:
                tx.write(PKT)
            except Exception as e:
                print(f"  [錯誤] 寫入 {txp} 失敗：{e}")
                ok = False
                break
            time.sleep(0.1)
        b = listen(rx, 1.5)
        tx.close()
        print(f"  寫入{'成功' if ok else '失敗'}。{RX_PORT} 收到 {len(b)} bytes  開頭：{bytes(b[:16]).hex(' ')}\n")
        time.sleep(0.5)
    rx.close()
    print("診斷結束。請把整個畫面截圖給 Claude，並說明兩塊板子的燈有沒有閃。")


if __name__ == '__main__':
    main()
