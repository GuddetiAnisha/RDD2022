# ============================================================
# LOAD TRAINED RT-DETR MODEL (NO RETRAINING)
# Uses your actual folder structure
# ============================================================

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path
from ultralytics import RTDETR

# ============================================================
# CONFIG
# ============================================================

RUN_DIR = Path(
    "runs/detect/runs_rdd2022/rtdetr_r50_50epochs"
)

DATA_YAML = "rdd2022_china_yolo/dataset.yaml"

BEST_WEIGHTS = RUN_DIR / "weights" / "best.pt"
LAST_WEIGHTS = RUN_DIR / "weights" / "last.pt"

# ============================================================
# LOAD RESULTS CSV
# ============================================================

def load_results_csv(run_dir):

    csv_path = run_dir / "results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"results.csv not found:\n{csv_path}"
        )

    history = {}

    with open(csv_path, newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            for k, v in row.items():

                k = k.strip()

                history.setdefault(k, [])

                try:
                    history[k].append(float(v.strip()))

                except:
                    history[k].append(0.0)

    print(f"\nLoaded results.csv from:\n{csv_path}")

    return history

# ============================================================
# SAVE LOSS CURVE
# ============================================================

def save_loss_curve(history, save_dir):

    # RT-DETR losses
    giou = history.get("train/giou_loss", [])
    cls  = history.get("train/cls_loss", [])
    l1   = history.get("train/l1_loss", [])

    if not giou:
        print("No RT-DETR loss data found.")
        return

    total_loss = [
        g + c + l
        for g, c, l in zip(giou, cls, l1)
    ]

    epochs = range(1, len(total_loss) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        total_loss,
        linewidth=2,
        label="Train Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("RT-DETR Training Loss")
    plt.grid(True)
    plt.legend()

    save_path = save_dir / "loss_curve.png"

    plt.savefig(save_path, dpi=150)

    plt.close()

    print(f"Saved:\n{save_path}")

# ============================================================
# MAIN
# ============================================================

def main():

    print("\n============================================")
    print(" LOADING TRAINED RT-DETR MODEL")
    print("============================================\n")

    # ========================================================
    # FIND AVAILABLE WEIGHTS
    # ========================================================

    if BEST_WEIGHTS.exists():

        WEIGHTS = BEST_WEIGHTS
        print("Using best.pt")

    elif LAST_WEIGHTS.exists():

        WEIGHTS = LAST_WEIGHTS
        print("best.pt not found")
        print("Using last.pt instead")

    else:

        raise FileNotFoundError(
            f"\nNo weights found.\n"
            f"Checked:\n"
            f"{BEST_WEIGHTS}\n"
            f"{LAST_WEIGHTS}"
        )

    print(f"\nLoading weights:\n{WEIGHTS}")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = RTDETR(str(WEIGHTS))

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\nRunning validation...")

    metrics = model.val(
        data=DATA_YAML,
        verbose=False
    )

    precision = metrics.box.mp
    recall    = metrics.box.mr
    map50     = metrics.box.map50
    map5095   = metrics.box.map

    f1 = 2 * precision * recall / (
        precision + recall + 1e-16
    )

    print("\n========== FINAL METRICS ==========")

    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"mAP50       : {map50:.4f}")
    print(f"mAP50-95    : {map5095:.4f}")
    print(f"F1 Score    : {f1:.4f}")

    print("====================================")

    # ========================================================
    # LOAD TRAIN HISTORY
    # ========================================================

    history = load_results_csv(RUN_DIR)

    # ========================================================
    # CREATE PLOTS FOLDER
    # ========================================================

    PLOTS_DIR = RUN_DIR / "custom_plots"

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE PLOTS
    # ========================================================

    save_loss_curve(history, PLOTS_DIR)

    print("\n============================================")
    print(" DONE")
    print("============================================")

    print(f"\nPlots saved to:\n{PLOTS_DIR}")

# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()