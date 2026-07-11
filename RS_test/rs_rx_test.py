import time
import serial
import serial.tools.list_ports
import struct
import sys
import argparse
from reedsolo import RSCodec, ReedSolomonError
import numpy as np

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

    rs_block_size = 200

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
                # Direct raw streaming (200-byte blocks)
                stream_buffer.extend(chunk_buffer)
                chunk_buffer.clear()
                while len(stream_buffer) >= rs_block_size:
                    rs_block = stream_buffer[:rs_block_size]
                    stream_buffer = stream_buffer[rs_block_size:]
                    stat_received += 1
                    
                    # Process RS Block
                    try:
                        decoded_tuple = rs.decode(rs_block)
                        decoded_msg = decoded_tuple[0]
                        seq_no = struct.unpack(">I", decoded_msg[:4])[0]
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
                                stat_lost += (seq_no - last_seq_no - 1)
                            last_seq_no = seq_no
                    except ReedSolomonError:
                        stat_uncorrectable += 1
                        error_distribution['>16'] += 1
                        if last_seq_no is not None:
                            last_seq_no += 1
            else:
                # Parse DSSS MAC Frame format: [Len(1, always 200) | Payload(200) | RSSI(1) | Status(1)] = 203 bytes
                while len(chunk_buffer) >= 3:
                    payload_len = chunk_buffer[0]
                    if payload_len != 200:
                        # Corrupted or shifted framing byte, discard and scan
                        chunk_buffer.pop(0)
                        continue

                    if len(chunk_buffer) < payload_len + 3:
                        break
                    
                    # Verify dummy/status byte at the end of the frame is 0x00
                    if chunk_buffer[payload_len + 2] != 0x00:
                        chunk_buffer.pop(0)
                        continue

                    rssi_val = chunk_buffer[payload_len + 1] - 256
                    rssi_list.append(rssi_val)
                    
                    rs_block = chunk_buffer[1 : payload_len + 1]
                    chunk_buffer = chunk_buffer[payload_len + 3 :]

                    stat_received += 1

                    # Process RS Block
                    try:
                        decoded_tuple = rs.decode(rs_block)
                        decoded_msg = decoded_tuple[0]
                        seq_no = struct.unpack(">I", decoded_msg[:4])[0]
                        
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
                            
                    except ReedSolomonError:
                        stat_uncorrectable += 1
                        error_distribution['>16'] += 1
                        if last_seq_no is not None:
                            last_seq_no += 1

            # Update display dashboard every 1.0 second
            now = time.time()
            if now - last_display_time >= 1.0:
                last_display_time = now
                
                # Calculations
                total_sent = (last_seq_no - start_seq_no + 1) if start_seq_no is not None else 0
                total_sent = max(total_sent, stat_received + stat_lost)
                
                loss_rate = (stat_lost / total_sent * 100) if total_sent > 0 else 0
                corr_rate = (stat_corrected / stat_received * 100) if stat_received > 0 else 0
                uncorr_rate = (stat_uncorrectable / stat_received * 100) if stat_received > 0 else 0
                avg_rssi = np.mean(rssi_list[-50:]) if rssi_list else 0.0

                # Print Dashboard
                sys.stdout.write("\033[H\033[J")  # Clear screen
                sys.stdout.write("==================================================\n")
                sys.stdout.write("        Reed-Solomon Hardware Test Dashboard      \n")
                sys.stdout.write("==================================================\n")
                sys.stdout.write(f"Port: {com_port} | Baud: {args.baud}\n")
                sys.stdout.write(f"Telemetry Status: {'Active' if last_seq_no is not None else 'Waiting for data...'}\n")
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
        
        # Print final report summary
        total_sent = (last_seq_no - start_seq_no + 1) if start_seq_no is not None else 0
        total_sent = max(total_sent, stat_received + stat_lost)
        loss_rate = (stat_lost / total_sent * 100) if total_sent > 0 else 0
        corr_rate = (stat_corrected / stat_received * 100) if stat_received > 0 else 0
        uncorr_rate = (stat_uncorrectable / stat_received * 100) if stat_received > 0 else 0
        avg_rssi = np.mean(rssi_list) if rssi_list else 0.0

        print("\n" + "=" * 50)
        print("                 FINAL EXPERIMENT REPORT          ")
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

if __name__ == "__main__":
    main()
