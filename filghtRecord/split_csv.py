import csv
import os

input_csv = r"g:\code\EE_project\filghtRecord\FlightRecord_2026-04-17_[17-07-19].csv"
out_dir = r"g:\code\EE_project\filghtRecord"

with open(input_csv, 'r', encoding='utf-8') as f:
    lines = f.readlines()

row0 = lines[0]
if "sep=" not in row0:
    row1 = row0
    row0 = None
    start_idx = 1
else:
    row1 = lines[1]
    start_idx = 2

headers = next(csv.reader([row1]))
try:
    col_idx = headers.index("CAMERA.isVideo")
except ValueError:
    print("Could not find 'CAMERA.isVideo' column.")
    exit(1)

blocks = []
current_block = []
is_recording = False

reader = csv.reader(lines[start_idx:])
for row in reader:
    if len(row) <= col_idx:
        # Malformed row? Just append if recording
        if is_recording:
            current_block.append(row)
        continue
    
    val = row[col_idx].strip()
    if val == 'True':
        if not is_recording:
            is_recording = True
            current_block = [row]
        else:
            current_block.append(row)
    else:
        if is_recording:
            is_recording = False
            blocks.append(current_block)

if is_recording:
    blocks.append(current_block)

print(f"Found {len(blocks)} video recording blocks.")

for i, block in enumerate(blocks):
    print(f"Block {i+1}: {len(block)} rows")
    if i < 6:
        out_name = os.path.join(out_dir, f"{i+1}.csv")
        with open(out_name, 'w', encoding='utf-8', newline='') as out_f:
            if row0:
                out_f.write(row0)
            writer = csv.writer(out_f)
            writer.writerow(headers)
            writer.writerows(block)
        print(f"  -> Saved to {out_name}")
