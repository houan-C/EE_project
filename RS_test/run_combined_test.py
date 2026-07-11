import time
import serial
import serial.tools.list_ports
import struct
import sys
import argparse
import threading
from reedsolo import RSCodec, ReedSolomonError
import numpy as np

# Find default serial port as fallback
def find_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    for p in ports:
        if "XDS110" in p.description:
            return p.device
    for p in ports:
        if "USB" in p.description or "UART" in p.description:
            return p.device
    return ports[0].device

# TX background worker
def tx_worker(tx_port, baud, interval, count, stop_event, init_event, error_holder):
    print(f"[TX Thread] Opening TX port: {tx_port} at {baud} baud...")
    try:
        ser_tx = serial.Serial(tx_port, baud, timeout=1)
        init_event.set()
    except Exception as e:
        error_holder[0] = str(e)
        return

    rs = RSCodec(32)
    header_format = '<4sBhhHHI'
    seq_no = 0

    try:
        while not stop_event.is_set():
            if count > 0 and seq_no >= count:
                break

            # Message: [SeqNo (4 bytes) | Payload (219 bytes)]
            payload = bytes(i % 256 for i in range(219))
            message = struct.pack(">I", seq_no) + payload
            rs_block = rs.encode(message)
            
            # Wrap in 17-byte AVIF header so CC1310 firmware accepts and transmits it
            header = struct.pack(header_format, b'AVIF', 0, 0, 0, 0, 0, len(rs_block))
            packet = header + rs_block

            ser_tx.write(packet)
            seq_no += 1
            time.sleep(interval)
            
    except Exception as e:
        print(f"[TX Thread] Exception: {e}")
    finally:
        if 'ser_tx' in locals() and ser_tx.is_open:
            ser_tx.close()
        print(f"[TX Thread] Stopped. Sent {seq_no} packets.")

def main():
    parser = argparse.ArgumentParser(description="Reed-Solomon Combined Hardware Test Runner")
    parser.add_argument("--rx", type=str, required=True, help="RX COM port (e.g. COM3)")
    parser.add_argument("--tx", type=str, required=True, help="TX COM port (e.g. COM5)")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (default: 921600)")
    parser.add_argument("--interval", type=float, default=0.02, help="Transmit interval in seconds (default: 0.02)")
    parser.add_argument("--count", type=int, default=2000, help="Total packets to send, 0 for infinite (default: 2000)")
    parser.add_argument("--raw", action="store_true", help="Use raw UART mode (disable CC1310 DSSS MAC parsing)")
    args = parser.parse_args()

    # Initialize Reed-Solomon
    rs = RSCodec(32)
    header_magic = b'AVIF'
    header_format = '<4sBhhHHI'
    header_size = struct.calcsize(header_format)
    rs_block_size = 255

    # RX setup
    print(f"[RX] Opening RX port: {args.rx} at {args.baud} baud...")
    try:
        ser_rx = serial.Serial(args.rx, args.baud, timeout=0.01)
    except Exception as e:
        print(f"[RX] Error: Could not open RX port {args.rx}: {e}")
        sys.exit(1)

    # Start TX thread
    stop_event = threading.Event()
    tx_init_event = threading.Event()
    tx_error = [None]
    tx_thread = threading.Thread(
        target=tx_worker, 
        args=(args.tx, args.baud, args.interval, args.count, stop_event, tx_init_event, tx_error),
        daemon=True
    )
    tx_thread.start()

    # Wait for TX thread to initialize (up to 2 seconds)
    print("Waiting for TX thread to initialize...")
    tx_init_event.wait(timeout=2.0)
    if not tx_init_event.is_set():
        err_msg = tx_error[0] if tx_error[0] else "Unknown initialization timeout"
        print(f"\nError: TX thread failed to initialize: {err_msg}")
        ser_rx.close()
        sys.exit(1)

    # Statistics counters
    start_seq_no = None
    last_seq_no = None
    
    stat_received = 0
    stat_lost = 0
    stat_clean = 0
    stat_corrected = 0
    stat_uncorrectable = 0
    stat_raw_bytes = 0
    
    rssi_list = []
    error_distribution = {i: 0 for i in range(17)}
    error_distribution['>16'] = 0

    chunk_buffer = bytearray()
    stream_buffer = bytearray()

    print(f"\n--- Starting Combined RS Test Run ---")
    print(f"Baud: {args.baud} | Interval: {args.interval}s | Target Packets: {args.count}")
    print("Press Ctrl+C to terminate early.\n")
    time.sleep(1.0) # Let user see ports opening

    last_display_time = time.time()
    run_finished = False

    try:
        while True:
            # Read from serial port
            data = ser_rx.read(1024)
            if not data:
                time.sleep(0.002)
            else:
                stat_raw_bytes += len(data)
                chunk_buffer.extend(data)

            if args.raw:
                stream_buffer.extend(chunk_buffer)
                chunk_buffer.clear()
            else:
                # CC1310 DSSS MAC Frame parser
                while len(chunk_buffer) >= 3:
                    payload_len = chunk_buffer[0]
                    if len(chunk_buffer) < payload_len + 3:
                        break
                    rssi_val = chunk_buffer[payload_len + 1] - 256
                    rssi_list.append(rssi_val)
                    stream_buffer.extend(chunk_buffer[1 : payload_len + 1])
                    chunk_buffer = chunk_buffer[payload_len + 3 :]

            # Process stream to align and extract packets
            while True:
                idx = stream_buffer.find(header_magic)
                if idx == -1:
                    if len(stream_buffer) > len(header_magic):
                        stream_buffer = stream_buffer[-len(header_magic):]
                    break
                
                if idx > 0:
                    stream_buffer = stream_buffer[idx:]
                    idx = 0

                if len(stream_buffer) < header_size:
                    break

                unpacked = struct.unpack(header_format, stream_buffer[:header_size])
                magic, pkt_type, dx, dy, crop_x, crop_y, payload_len = unpacked

                if len(stream_buffer) < header_size + payload_len:
                    break

                rs_block = stream_buffer[header_size : header_size + payload_len]
                stream_buffer = stream_buffer[header_size + payload_len:]

                stat_received += 1

                try:
                    decoded_tuple = rs.decode(rs_block)
                    decoded_msg = decoded_tuple[0]
                    seq_no = struct.unpack(">I", decoded_msg[:4])[0]

                    # Diff to count corrected errors
                    corrected_block = rs.encode(decoded_msg)
                    diff_positions = [i for i in range(rs_block_size) if rs_block[i] != corrected_block[i]]
                    num_errors = len(diff_positions)

                    if num_errors == 0:
                        stat_clean += 1
                        error_distribution[0] += 1
                    else:
                        stat_corrected += 1
                        error_distribution[num_errors] += 1

                    if start_seq_no is None:
                        start_seq_no = seq_no
                        last_seq_no = seq_no
                    else:
                        if seq_no > last_seq_no + 1:
                            gaps = seq_no - last_seq_no - 1
                            stat_lost += gaps
                            last_seq_no = seq_no
                        elif seq_no == last_seq_no + 1:
                            last_seq_no = seq_no

                except ReedSolomonError:
                    stat_uncorrectable += 1
                    error_distribution['>16'] += 1
                    if last_seq_no is not None:
                        last_seq_no += 1

            # Check if TX has finished and all sent packets have had time to arrive
            if not tx_thread.is_alive() and args.count > 0:
                # Wait 1.0s after TX ends to gather any remaining packets in transit
                time.sleep(1.0)
                run_finished = True
                break

            # Update display dashboard every 1.0s
            now = time.time()
            if now - last_display_time >= 1.0:
                last_display_time = now
                total_sent = (last_seq_no - start_seq_no + 1) if start_seq_no is not None else 0
                total_sent = max(total_sent, stat_received + stat_lost)
                loss_rate = (stat_lost / total_sent * 100) if total_sent > 0 else 0
                corr_rate = (stat_corrected / stat_received * 100) if stat_received > 0 else 0
                uncorr_rate = (stat_uncorrectable / stat_received * 100) if stat_received > 0 else 0
                avg_rssi = np.mean(rssi_list[-50:]) if rssi_list else 0.0

                # Print Dashboard
                sys.stdout.write("\033[H\033[J")
                sys.stdout.write("==================================================\n")
                sys.stdout.write("      Reed-Solomon Combined Test Dashboard        \n")
                sys.stdout.write("==================================================\n")
                sys.stdout.write(f"RX Port: {args.rx} | TX Port: {args.tx} | Baud: {args.baud}\n")
                sys.stdout.write(f"TX Progress: {'Sending' if tx_thread.is_alive() else 'Finished'}\n")
                if last_seq_no is not None:
                    sys.stdout.write(f"Seq Range: #{start_seq_no} to #{last_seq_no}\n")
                sys.stdout.write(f"Avg RSSI (Last 50): {avg_rssi:.1f} dBm\n")
                sys.stdout.write(f"Raw Bytes Recv'd:    {stat_raw_bytes} bytes\n")
                sys.stdout.write("--------------------------------------------------\n")
                sys.stdout.write(f"Estimated Sent:      {total_sent:<6d}\n")
                sys.stdout.write(f"Packets Lost:        {stat_lost:<6d} (Loss Rate: {loss_rate:.2f}%)\n")
                sys.stdout.write(f"Packets Received:    {stat_received:<6d}\n")
                sys.stdout.write(f"  - Clean (0 errors):{stat_clean:<6d}\n")
                sys.stdout.write(f"  - Corrected (1-16):{stat_corrected:<6d} (Corr Rate: {corr_rate:.2f}%)\n")
                sys.stdout.write(f"  - Uncorrectable:   {stat_uncorrectable:<6d} (Uncorr Rate: {uncorr_rate:.2f}%)\n")
                sys.stdout.write("--------------------------------------------------\n")
                sys.stdout.write("Error distribution in received blocks (0-16 bytes):\n")
                for err_cnt in [0, 1, 2, 3, 4, 8, 12, 16, '>16']:
                    count = error_distribution[err_cnt]
                    bar = "#" * min(20, int(count / max(1, stat_received) * 40))
                    sys.stdout.write(f"  {str(err_cnt):>3s} bytes error: {count:<5d} {bar}\n")
                sys.stdout.write("==================================================\n")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nExperiment stopped by user. Generating final report...")
    finally:
        stop_event.set()
        tx_thread.join(timeout=1.0)
        ser_rx.close()
        
        # Print final report summary
        total_sent = (last_seq_no - start_seq_no + 1) if start_seq_no is not None else 0
        total_sent = max(total_sent, stat_received + stat_lost)
        loss_rate = (stat_lost / total_sent * 100) if total_sent > 0 else 0
        corr_rate = (stat_corrected / stat_received * 100) if stat_received > 0 else 0
        uncorr_rate = (stat_uncorrectable / stat_received * 100) if stat_received > 0 else 0
        avg_rssi = np.mean(rssi_list) if rssi_list else 0.0

        sys.stdout.write("\033[H\033[J")
        print("=" * 50)
        print("                 FINAL EXPERIMENT REPORT          ")
        print("=" * 50)
        if run_finished:
            print("Run completed automatically.")
        print(f"RX Port:             {args.rx}")
        print(f"TX Port:             {args.tx}")
        print(f"Baud Rate:           {args.baud}")
        print(f"Raw Bytes Recv'd:    {stat_raw_bytes} bytes")
        print(f"Estimated Sent:      {total_sent} packets")
        print(f"Packets Lost:        {stat_lost} packets ({loss_rate:.2f}% packet loss)")
        print(f"Packets Received:    {stat_received} packets")
        print(f"  - Clean:           {stat_clean} packets")
        print(f"  - Corrected:       {stat_corrected} packets ({corr_rate:.2f}%)")
        print(f"  - Uncorrectable:   {stat_uncorrectable} packets ({uncorr_rate:.2f}%)")
        print(f"Average Link RSSI:   {avg_rssi:.1f} dBm")
        print("-" * 50)
        print("Error Distribution Table:")
        print("  Errors (bytes) | Packet Count | Percentage")
        for err_cnt in sorted([k for k in error_distribution.keys() if isinstance(k, int)]) + ['>16']:
            cnt = error_distribution[err_cnt]
            pct = (cnt / stat_received * 100) if stat_received > 0 else 0
            print(f"    {str(err_cnt):<12s} | {cnt:<12d} | {pct:.2f}%")
        print("=" * 50)
        print("Scientific Analysis Conclusions:")
        if loss_rate > 5.0 and stat_corrected == 0:
            print("  [Erase-Only Channel]: Bit/byte corruption is extremely rare.")
            print("  The hardware CRC layer automatically drops packets with errors,")
            print("  causing chunk-based erasures. Standard byte-level RS(255, 223) within")
            print("  packets is INEFFECTIVE here. Recommend switching to Packet-Level")
            print("  Erasure Coding across multiple packets.")
        elif stat_corrected > 0:
            print("  [Hybrid Channel]: Both random byte corruption and packet drops are present.")
            print(f"  Byte-level RS successfully repaired {stat_corrected} packets ({corr_rate:.2f}%).")
            print("  Byte-level RS is effective at improving link reliability.")
        else:
            print("  Link was clean during this run. No errors or drops detected.")
        print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
