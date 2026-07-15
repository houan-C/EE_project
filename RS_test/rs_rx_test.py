import time
import serial
import serial.tools.list_ports
import struct
import sys
import argparse
from reedsolo import RSCodec, ReedSolomonError
import numpy as np
import os

# Enable ANSI escape sequences on Windows CMD/PowerShell
if os.name == 'nt':
    os.system('')

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

def main():
    parser = argparse.ArgumentParser(description="Reed-Solomon Hardware Test Receiver")
    parser.add_argument("--port", type=str, default=None, help="COM port (default: auto-discover)")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (default: 921600)")
    parser.add_argument("--raw", action="store_true", help="Use raw UART mode (disable CC1310 DSSS MAC parsing)")
    parser.add_argument("--interval", type=float, default=0.02, help="Expected packet interval in seconds (default: 0.02)")
    parser.add_argument("--test-len", type=int, default=2000, help="Number of packets for test run (default: 2000)")
    parser.add_argument("--sync-len", type=int, default=100, help="Consecutive packets to establish stable connection (default: 100)")
    args = parser.parse_args()

    com_port = args.port if args.port else find_serial_port()
    if not com_port:
        print("Error: No serial port found.")
        sys.exit(1)

    print(f"Opening serial port: {com_port} at {args.baud} baud...")
    try:
        ser = serial.Serial(com_port, args.baud, timeout=0.01)
    except Exception as e:
        print(f"Error: Could not open serial port {com_port}: {e}")
        sys.exit(1)

    # Perform hardware reset on CC1310 LaunchPad
    print("Performing CC1310 RX hardware reset...")
    ser.dtr = True
    ser.rts = True
    time.sleep(0.2)
    ser.dtr = False
    ser.rts = False
    print("Waiting 3 seconds for CC1310 RX board to boot...")
    time.sleep(3.0)
    ser.reset_input_buffer()

    print("Initializing Reed-Solomon Decoder (nsym=32)...")
    rs = RSCodec(32)

    rs_block_size = 255

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

    # Connection state machine variables
    connection_stable = False
    consecutive_count = 0
    timeout_lost_seqs = set()
    last_packet_time = None

    def print_and_reset_report(title):
        nonlocal start_seq_no, last_seq_no, stat_received, stat_lost, stat_clean
        nonlocal stat_corrected, stat_uncorrectable, stat_raw_bytes, rssi_list, error_distribution
        nonlocal connection_stable, consecutive_count, timeout_lost_seqs, last_packet_time

        if stat_received == 0 and stat_lost == 0:
            return

        total_sent = (last_seq_no - start_seq_no + 1) if start_seq_no is not None else 0
        total_sent = max(total_sent, stat_received + stat_lost)
        loss_rate = (stat_lost / total_sent * 100) if total_sent > 0 else 0
        corr_rate = (stat_corrected / stat_received * 100) if stat_received > 0 else 0
        uncorr_rate = (stat_uncorrectable / stat_received * 100) if stat_received > 0 else 0
        avg_rssi = np.mean(rssi_list) if rssi_list else 0.0

        print("\n" + "=" * 50)
        print(f"{title:^50}")
        print("=" * 50)
        print(f"COM Port:            {com_port}")
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

        # Reset for next window
        start_seq_no = None
        last_seq_no = None
        stat_received = 0
        stat_lost = 0
        stat_clean = 0
        stat_corrected = 0
        stat_uncorrectable = 0
        stat_raw_bytes = 0
        rssi_list.clear()
        for k in error_distribution:
            error_distribution[k] = 0
        timeout_lost_seqs.clear()
        connection_stable = False
        consecutive_count = 0
        last_packet_time = None

    chunk_buffer = bytearray()
    stream_buffer = bytearray()

    print(f"\n--- Starting RS Reception & Analysis ---")
    print(f"Mode: {'Raw Serial' if args.raw else 'CC1310 DSSS MAC Frame'}")
    print("Press Ctrl+C to stop and print final report.\n")

    last_display_time = time.time()

    try:
        while True:
            # Read from serial port
            data = ser.read(1024)
            if not data:
                time.sleep(0.002)
            else:
                stat_raw_bytes += len(data)
                chunk_buffer.extend(data)

            if args.raw:
                # Direct raw streaming
                stream_buffer.extend(chunk_buffer)
                chunk_buffer.clear()
            else:
                # Parse DSSS MAC Frame format: [Len(1B) | Payload(Len) | RSSI(1B) | Status(1B)]
                while len(chunk_buffer) >= 3:
                    payload_len = chunk_buffer[0]
                    if payload_len > 255 or payload_len == 0:
                        chunk_buffer.pop(0)
                        continue

                    if len(chunk_buffer) < payload_len + 3:
                        break
                    
                    # Verify status/dummy byte is 0x00
                    if chunk_buffer[payload_len + 2] != 0x00:
                        chunk_buffer.pop(0)
                        continue

                    rssi_val = chunk_buffer[payload_len + 1] - 256
                    rssi_list.append(rssi_val)
                    
                    payload = chunk_buffer[1 : payload_len + 1]
                    stream_buffer.extend(payload)
                    chunk_buffer = chunk_buffer[payload_len + 3 :]

            # Parse stream_buffer using magic header
            RS_MAGIC = b'RSTST'
            PACKET_SIZE = 260  # 5B magic + 255B RS block

            # Helper to record statistics when stable
            def record_packet_stats(is_dec, raw_blk, dec_msg):
                nonlocal stat_clean, stat_corrected, stat_uncorrectable, error_distribution
                if is_dec:
                    corrected_block = rs.encode(dec_msg)
                    diff_positions = [i for i in range(rs_block_size) if raw_blk[i] != corrected_block[i]]
                    num_errors = len(diff_positions)
                    if num_errors == 0:
                        stat_clean += 1
                        error_distribution[0] += 1
                    else:
                        stat_corrected += 1
                        error_distribution[num_errors] += 1
                else:
                    stat_uncorrectable += 1
                    error_distribution['>16'] += 1

            while True:
                magic_pos = stream_buffer.find(RS_MAGIC)
                if magic_pos == -1:
                    if len(stream_buffer) > PACKET_SIZE:
                        stream_buffer = stream_buffer[-(PACKET_SIZE - 1):]
                    break

                if magic_pos > 0:
                    stream_buffer = stream_buffer[magic_pos:]

                if len(stream_buffer) < PACKET_SIZE:
                    break

                rs_block = bytes(stream_buffer[5:260])
                stream_buffer = stream_buffer[PACKET_SIZE:]

                # Try decoding to extract seq_no
                try:
                    decoded_tuple = rs.decode(rs_block)
                    decoded_msg = decoded_tuple[0]
                    seq_no = struct.unpack(">I", decoded_msg[:4])[0]
                    is_decoded = True
                except ReedSolomonError:
                    seq_no = struct.unpack(">I", rs_block[:4])[0]
                    decoded_msg = None
                    is_decoded = False

                if not connection_stable:
                    # Sync phase
                    if last_seq_no is None:
                        consecutive_count = 1
                        last_seq_no = seq_no
                    else:
                        if seq_no == last_seq_no + 1:
                            consecutive_count += 1
                        else:
                            consecutive_count = 1
                        last_seq_no = seq_no
                    
                    last_packet_time = time.time()
                    
                    if consecutive_count >= args.sync_len:
                        connection_stable = True
                        start_seq_no = seq_no
                        last_seq_no = seq_no
                        stat_received = 1
                        stat_lost = 0
                        stat_clean = 0
                        stat_corrected = 0
                        stat_uncorrectable = 0
                        rssi_list = rssi_list[-1:] if rssi_list else []
                        for k in error_distribution:
                            error_distribution[k] = 0
                        timeout_lost_seqs.clear()
                        record_packet_stats(is_decoded, rs_block, decoded_msg)
                else:
                    # Stable phase
                    now = time.time()
                    if seq_no <= last_seq_no:
                        if seq_no in timeout_lost_seqs:
                            timeout_lost_seqs.remove(seq_no)
                            stat_lost -= 1
                            stat_uncorrectable -= 1
                            error_distribution['>16'] -= 1
                            stat_received += 1
                            record_packet_stats(is_decoded, rs_block, decoded_msg)
                    else:
                        stat_received += 1
                        # Check for sequence gaps
                        if seq_no > last_seq_no + 1:
                            for g_seq in range(last_seq_no + 1, seq_no):
                                if g_seq not in timeout_lost_seqs:
                                    stat_lost += 1
                                    stat_uncorrectable += 1
                                    error_distribution['>16'] += 1
                        record_packet_stats(is_decoded, rs_block, decoded_msg)
                        last_seq_no = seq_no
                        last_packet_time = now

            # Timeout check at the end of loop iteration
            now = time.time()
            if last_packet_time is not None:
                if connection_stable:
                    elapsed = now - last_packet_time
                    while elapsed >= 1.8 * args.interval:
                        lost_seq = last_seq_no + 1
                        stat_lost += 1
                        stat_uncorrectable += 1
                        error_distribution['>16'] += 1
                        timeout_lost_seqs.add(lost_seq)
                        
                        last_seq_no = lost_seq
                        last_packet_time += args.interval
                        elapsed = now - last_packet_time
                else:
                    # Reset consecutive sync packets if no packets received for 1.8 * interval
                    if now - last_packet_time >= 1.8 * args.interval:
                        consecutive_count = 0
                        last_packet_time = now

            # Check if test run is completed
            if connection_stable and start_seq_no is not None:
                total_sent = (last_seq_no - start_seq_no + 1)
                total_sent = max(total_sent, stat_received + stat_lost)
                if total_sent >= args.test_len:
                    print("\n\n" + "*" * 50)
                    print_and_reset_report(f"{args.test_len} PACKETS COMPLETED REPORT")
                    time.sleep(2.0)

            # Update display dashboard every 1.0 second
            now = time.time()
            if now - last_display_time >= 1.0:
                last_display_time = now
                
                # Print Dashboard
                sys.stdout.write("\033[H\033[J")  # Clear screen
                sys.stdout.write("==================================================\n")
                sys.stdout.write("        Reed-Solomon Hardware Test Dashboard      \n")
                sys.stdout.write("==================================================\n")
                sys.stdout.write(f"Port: {com_port} | Baud: {args.baud}\n")
                
                if not connection_stable:
                    # Show RED indicator and synchronization progress
                    sys.stdout.write(f"Connection Status: \033[91m[UNSTABLE / DISCONNECTED]\033[0m\n")
                    sys.stdout.write(f"Sync Progress:     {consecutive_count} / {args.sync_len} consecutive packets\n")
                    sys.stdout.write("==================================================\n")
                else:
                    # Calculations for Stable state
                    total_sent = (last_seq_no - start_seq_no + 1) if start_seq_no is not None else 0
                    total_sent = max(total_sent, stat_received + stat_lost)
                    loss_rate = (stat_lost / total_sent * 100) if total_sent > 0 else 0
                    corr_rate = (stat_corrected / stat_received * 100) if stat_received > 0 else 0
                    uncorr_rate = (stat_uncorrectable / stat_received * 100) if stat_received > 0 else 0
                    avg_rssi = np.mean(rssi_list[-50:]) if rssi_list else 0.0
                    progress_pct = (total_sent / args.test_len * 100) if args.test_len > 0 else 0.0

                    # Show GREEN indicator and test metrics
                    sys.stdout.write(f"Connection Status: \033[92m[STABLE CONNECTED]\033[0m\n")
                    sys.stdout.write(f"Test Progress:     {total_sent} / {args.test_len} packets ({progress_pct:.1f}%)\n")
                    sys.stdout.write(f"Telemetry Status:  Active\n")
                    sys.stdout.write(f"Seq Range:         #{start_seq_no} to #{last_seq_no}\n")
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
                    
                    # Print histogram inline
                    for err_cnt in [0, 1, 2, 3, 4, 8, 12, 16, '>16']:
                        count = error_distribution[err_cnt]
                        bar = "#" * min(20, int(count / max(1, stat_received) * 40))
                        sys.stdout.write(f"  {str(err_cnt):>3s} bytes error: {count:<5d} {bar}\n")
                    sys.stdout.write("==================================================\n")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nExperiment stopped by user. Generating final report...")
    finally:
        ser.close()
        
        print_and_reset_report("FINAL EXPERIMENT REPORT")

if __name__ == "__main__":
    main()
