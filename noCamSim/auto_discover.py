import serial
import serial.tools.list_ports
import time
import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "board_config.json")

def get_xds110_ports():
    ports = list(serial.tools.list_ports.comports())
    xds_ports = []
    
    for p in ports:
        # VID 0x0451 is Texas Instruments (XDS110)
        if p.vid == 0x0451:
            loc = p.location or ""
            hwid = p.hwid or ""
            # Only keep the Application/User UART interface (interface 0 / MI_00 / ends with 0)
            if "MI_00" in hwid or loc.endswith("0") or "x.0" in loc:
                xds_ports.append(p)
                
    return xds_ports

def test_tx_rx(portA, portB):
    try:
        sa = serial.Serial(portA.device, 921600, timeout=0.1)
        sb = serial.Serial(portB.device, 921600, timeout=0.1)
    except Exception as e:
        print(f"Error opening ports for testing: {e}")
        return None, None

    time.sleep(0.1)
    sa.reset_input_buffer()
    sb.reset_input_buffer()

    test_msg = b'PING_TEST_123'
    sa.write(test_msg)
    time.sleep(0.1)
    ans = sb.read(100)
    
    if test_msg in ans:
        sa.close()
        sb.close()
        return portA, portB

    sa.reset_input_buffer()
    sb.reset_input_buffer()
    sb.write(test_msg)
    time.sleep(0.1)
    ans = sa.read(100)

    if test_msg in ans:
        sa.close()
        sb.close()
        return portB, portA

    sa.close()
    sb.close()
    return None, None

def get_role_port(role):
    xds_ports = get_xds110_ports()
    
    if len(xds_ports) == 0:
        print(f"[{role}] No XDS110 ports found! Connect your board.")
        return None
        
    if len(xds_ports) == 1:
        print(f"[{role}] Only one board attached. Assuming it matches role {role}.")
        return xds_ports[0].device
        
    if len(xds_ports) >= 2:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                for p in xds_ports:
                    if p.serial_number == config.get(role):
                        print(f"[{role}] Loaded from config -> {p.device}")
                        return p.device
            except:
                pass
                
        print(f"[{role}] Multiple boards. Running RF Ping Test...")
        tx_port, rx_port = test_tx_rx(xds_ports[0], xds_ports[1])
        
        if tx_port and rx_port:
            config = {
                "TX": tx_port.serial_number or tx_port.device,
                "RX": rx_port.serial_number or rx_port.device
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f)
            print(f"[{role}] Ping test success! Config saved.")
            return tx_port.device if role == "TX" else rx_port.device
        else:
            print(f"[{role}] RF Ping auto-discover failed (Port in use or no signal).")
            print("Since we don't have a configuration yet, please identify it once manually:")
            print("Available Boards:")
            for i, p in enumerate(xds_ports):
                print(f"  {i+1}: {p.device} (Serial: {p.serial_number})")
                
            try:
                sel = int(input(f"Enter the number for the {role} board (1-{len(xds_ports)}): ").strip())
                chosen = xds_ports[sel-1]
                
                config = {}
                if os.path.exists(CONFIG_FILE):
                    try:
                        with open(CONFIG_FILE, 'r') as f:
                            config = json.load(f)
                    except: pass
                
                config[role] = chosen.serial_number or chosen.device
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(config, f)
                print(f"[{role}] Saved Configuration securely! It will auto-detect next time.")
                return chosen.device
            except Exception as e:
                print("Invalid input or error, falling back to basic guess...")
                return xds_ports[0].device if role == "TX" else xds_ports[1].device
