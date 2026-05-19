from pathlib import Path

# Your label directory
labels_root = Path("rdd2022_china_yolo/labels")

fixed_files = 0
removed_labels = 0

for txt_file in labels_root.rglob("*.txt"):

    valid_lines = []

    for line in txt_file.read_text().splitlines():

        parts = line.strip().split()

        if not parts:
            continue

        cls_id = int(parts[0])

        # Keep only valid classes 0-3
        if cls_id <= 3:
            valid_lines.append(line)
        else:
            removed_labels += 1

    txt_file.write_text("\n".join(valid_lines))
    fixed_files += 1

print(f"Fixed files      : {fixed_files}")
print(f"Removed labels   : {removed_labels}")
print("Done.")