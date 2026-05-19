"""
prepare_dataset.py
==================
RDD2022 – China only  |  Category (stratified) sampling  |  70 / 20 / 10

Folder structure (your actual layout):

    dml\
    └── archive\
        └── RDD_SPLIT\
            ├── train\
            │   ├── images\   China_Drone_*.jpg  (+ possibly other countries)
            │   └── labels\   China_Drone_*.txt
            ├── val\
            │   ├── images\
            │   └── labels\
            └── test\
                ├── images\
                └── labels\

Only files whose name starts with  "China_"  are kept.

Steps
-----
1. Convert Pascal VOC XMLs → YOLO .txt  (skipped if labels/ already exists)
2. Pool China-only images from train / val / test
3. Build per-category index (dominant class per image)
4. Stratified re-split  →  70 / 20 / 10
5. Copy into  rdd2022_china_yolo\
6. Write dataset.yaml
7. Save category_distribution.png
"""

import shutil
import random
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent          # …\dml
RDD_SPLIT   = BASE_DIR / "archive" / "RDD_SPLIT"      # …\dml\archive\RDD_SPLIT
OUTPUT_ROOT = BASE_DIR / "rdd2022_china_yolo"

SPLITS_IN   = ("train", "val", "test")   # existing sub-folders to pool from

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SEED             = 42
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.20
TEST_RATIO       = 0.10
MAX_PER_CATEGORY = None     # int to cap per class, None = keep all

CATEGORY_NAMES = {
    0: "D00_Longitudinal_Crack",
    1: "D10_Transverse_Crack",
    2: "D20_Alligator_Crack",
    3: "D40_Pothole",
}
LABEL_MAP = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}

random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────
# STEP 1 – VOC XML  →  YOLO .txt
# ─────────────────────────────────────────────────────────────
def parse_voc_xml(xml_path, img_w, img_h):
    boxes = []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        name = obj.find("name").text.strip()
        if name not in LABEL_MAP:
            continue
        bb   = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        cx   = ((xmin + xmax) / 2) / img_w
        cy   = ((ymin + ymax) / 2) / img_h
        bw   = (xmax - xmin) / img_w
        bh   = (ymax - ymin) / img_h
        boxes.append((LABEL_MAP[name], cx, cy, bw, bh))
    return boxes


def ensure_yolo_labels(split_dir):
    """Create labels/ with YOLO .txt files; convert from XMLs if needed."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if lbl_dir.exists() and any(lbl_dir.glob("*.txt")):
        n = len(list(lbl_dir.glob("*.txt")))
        print(f"    labels/ already has {n} files – skipping conversion.")
        return lbl_dir

    lbl_dir.mkdir(parents=True, exist_ok=True)

    xml_dir = None
    for sub in ("annotations/xmls", "annotations/xml", "xmls", "xml"):
        candidate = split_dir / sub
        if candidate.exists():
            xml_dir = candidate
            break

    if xml_dir is None:
        print(f"    No XML folder found – writing empty labels.")
        for img in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
            (lbl_dir / f"{img.stem}.txt").write_text("")
        return lbl_dir

    converted = skipped = 0
    for xml_file in sorted(xml_dir.glob("*.xml")):
        stem    = xml_file.stem
        txt_out = lbl_dir / f"{stem}.txt"
        if txt_out.exists():
            skipped += 1
            continue
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{stem}.png"
        if not img_path.exists():
            skipped += 1
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        boxes = parse_voc_xml(xml_file, w, h)
        with open(txt_out, "w") as f:
            for cls_id, cx, cy, bw, bh in boxes:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        converted += 1

    print(f"    converted={converted}  skipped={skipped}")
    return lbl_dir


# ─────────────────────────────────────────────────────────────
# STEP 2 – POOL CHINA-ONLY IMAGES & BUILD CATEGORY INDEX
# ─────────────────────────────────────────────────────────────
def get_dominant_category(label_path):
    if not label_path.exists():
        return -1
    lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
    if not lines:
        return -1
    counts = Counter(int(l.split()[0]) for l in lines)
    return counts.most_common(1)[0][0]


def pool_china_images(rdd_split, splits):
    """
    Walk each split folder, keep ONLY files starting with 'China_'.
    Returns cat_index, img_map, lbl_map.
    """
    cat_index = defaultdict(list)
    img_map   = {}
    lbl_map   = {}

    for split in splits:
        img_dir = rdd_split / split / "images"
        lbl_dir = rdd_split / split / "labels"

        if not img_dir.exists():
            print(f"  ⚠  images/ not found in {split}/ – skipping.")
            continue

        all_imgs   = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

        # ── CHINA FILTER ──────────────────────────────────────
        china_imgs = [p for p in all_imgs
                      if p.name.lower().startswith("china_")]
        filtered   = len(all_imgs) - len(china_imgs)

        print(f"  {split:6s}  total={len(all_imgs):5d}  "
              f"china={len(china_imgs):5d}  filtered_out={filtered}")

        for img_path in sorted(china_imgs):
            stem     = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            dom_cat  = get_dominant_category(lbl_path)
            cat_index[dom_cat].append(stem)
            img_map[stem] = img_path
            lbl_map[stem] = lbl_path

    return cat_index, img_map, lbl_map


# ─────────────────────────────────────────────────────────────
# STEP 3 – STRATIFIED SPLIT  70 / 20 / 10
# ─────────────────────────────────────────────────────────────
def stratified_split(cat_index, train_r=TRAIN_RATIO, val_r=VAL_RATIO,
                     max_per_cat=None):
    train_s, val_s, test_s = [], [], []

    for cat_id, stems in cat_index.items():
        stems = list(stems)
        random.shuffle(stems)
        if max_per_cat and len(stems) > max_per_cat:
            stems = stems[:max_per_cat]

        n       = len(stems)
        n_test  = max(1, int(n * (1 - train_r - val_r))) if n >= 3 else 0
        n_val   = max(1, int(n * val_r))                 if n >= 2 else 0
        n_train = n - n_val - n_test

        train_s.extend(stems[:n_train])
        val_s.extend(stems[n_train : n_train + n_val])
        test_s.extend(stems[n_train + n_val:])

    return train_s, val_s, test_s


# ─────────────────────────────────────────────────────────────
# STEP 4 – COPY INTO OUTPUT YOLO STRUCTURE
# ─────────────────────────────────────────────────────────────
def copy_split(stems, img_map, lbl_map, out_root, split_name):
    img_out = out_root / "images" / split_name
    lbl_out = out_root / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    missing_img = missing_lbl = 0
    for stem in stems:
        src_img = img_map.get(stem)
        if src_img and src_img.exists():
            shutil.copy2(src_img, img_out / src_img.name)
        else:
            missing_img += 1

        src_lbl = lbl_map.get(stem)
        if src_lbl and src_lbl.exists():
            shutil.copy2(src_lbl, lbl_out / f"{stem}.txt")
        else:
            (lbl_out / f"{stem}.txt").write_text("")
            missing_lbl += 1

    print(f"  [copy]  {split_name:<6}  n={len(stems)}"
          f"  missing_img={missing_img}  empty_lbl={missing_lbl}")


# ─────────────────────────────────────────────────────────────
# STEP 5 – dataset.yaml
# ─────────────────────────────────────────────────────────────
def write_yaml(out_root):
    cfg = {
        "path"  : str(out_root.resolve()),
        "train" : "images/train",
        "val"   : "images/val",
        "test"  : "images/test",
        "nc"    : len(CATEGORY_NAMES),
        "names" : list(CATEGORY_NAMES.values()),
    }
    yaml_path = out_root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  [yaml]  → {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────
# STEP 6 – PLOT DISTRIBUTION
# ─────────────────────────────────────────────────────────────
def count_instances(labels_dir, n_classes=4):
    counts = Counter()
    for lbl in labels_dir.glob("*.txt"):
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1
    return [counts.get(i, 0) for i in range(n_classes)]


def plot_distribution(out_root):
    splits     = ("train", "val", "test")
    split_pcts = ("70%",   "20%", "10%")
    colors     = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    x_labels   = list(CATEGORY_NAMES.values())
    x_pos      = list(range(len(x_labels)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, split, pct in zip(axes, splits, split_pcts):
        counts = count_instances(out_root / "labels" / split)
        bars   = ax.bar(x_pos, counts, color=colors,
                        edgecolor="white", linewidth=0.8)
        ax.set_title(f"{split.capitalize()} split  ({pct})",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Instance count")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=9)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 3, str(cnt),
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle("RDD2022 China – Category Sampling Distribution  (70/20/10)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = out_root / "category_distribution.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [plot]  → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 58)
    print("  RDD2022 China only  |  Category Sampling  |  70/20/10")
    print("═" * 58)
    print(f"\n  RDD_SPLIT  : {RDD_SPLIT}")
    print(f"  Output     : {OUTPUT_ROOT}\n")

    if not RDD_SPLIT.exists():
        print(f"  ✗ Not found: {RDD_SPLIT}")
        print("    Check BASE_DIR points to your  dml\\  folder.")
        return

    # 1. Ensure YOLO labels
    print("[1] Ensuring YOLO labels exist …")
    for split in SPLITS_IN:
        sd = RDD_SPLIT / split
        if not sd.exists():
            print(f"  ⚠  {split}/ not found – skipping.")
            continue
        print(f"  {split}/ …")
        ensure_yolo_labels(sd)

    # 2. Pool China-only images
    print("\n[2] Pooling China_ images & building category index …")
    cat_index, img_map, lbl_map = pool_china_images(RDD_SPLIT, SPLITS_IN)

    total = sum(len(v) for v in cat_index.values())
    print(f"\n  Category distribution  (total China images = {total}):")
    for cat_id, stems in sorted(cat_index.items()):
        name = CATEGORY_NAMES.get(cat_id, "Background / None")
        print(f"    [{cat_id:>2}] {name:<35}  n = {len(stems)}")

    if total == 0:
        print("\n  ✗ No China_ images found. Check your folder paths.")
        return

    # 3. Stratified re-split
    print("\n[3] Stratified split (70 / 20 / 10) …")
    train_s, val_s, test_s = stratified_split(cat_index,
                                               max_per_cat=MAX_PER_CATEGORY)
    print(f"  train={len(train_s)}  val={len(val_s)}  test={len(test_s)}"
          f"  total={len(train_s)+len(val_s)+len(test_s)}")

    # 4. Copy
    print("\n[4] Copying into YOLO output structure …")
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    copy_split(train_s, img_map, lbl_map, OUTPUT_ROOT, "train")
    copy_split(val_s,   img_map, lbl_map, OUTPUT_ROOT, "val")
    copy_split(test_s,  img_map, lbl_map, OUTPUT_ROOT, "test")

    # 5. YAML
    print("\n[5] Writing dataset.yaml …")
    write_yaml(OUTPUT_ROOT)

    # 6. Plot
    print("\n[6] Plotting category distributions …")
    plot_distribution(OUTPUT_ROOT)

    print(f"\n✅  Done!  →  {OUTPUT_ROOT.resolve()}")
    print("    Pass  dataset.yaml  to  train_yolov8s.py\n")


if __name__ == "__main__":
    main()