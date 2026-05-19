# =============================================================================
# COMPLETE YOLOv8m RESEARCH EVALUATION PIPELINE
# =============================================================================
#
# USE AFTER TRAINING COMPLETES
#
# This script generates:
#
# ✓ Precision / Recall / F1 / mAP
# ✓ Per-class metrics
# ✓ Confusion matrix
# ✓ PR curves
# ✓ F1 curves
# ✓ Loss curves
# ✓ Dataset statistics
# ✓ Category distribution
# ✓ Overfitting analysis
# ✓ Prediction visualizations
# ✓ Failure case analysis
# ✓ Hyperparameter logging
# ✓ CSV summaries
# ✓ IEEE-report-ready plots
#
# NO RETRAINING
# EVALUATION ONLY
#
# =============================================================================

import csv
import random
import warnings
import yaml

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from ultralytics import YOLO

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH = Path(
    "runs/detect/runs_rdd2022/custom_yolov8m_50epochs/weights/best.pt"
)

RUN_DIR = Path(
    "runs/detect/runs_rdd2022/custom_yolov8m_50epochs"
)

DATA_YAML = "rdd2022_china_yolo/dataset.yaml"

SAVE_DIR = RUN_DIR / "complete_evaluation"

DEVICE = 0
CONF_THRESH = 0.25
NUM_VIS = 12

SAVE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# VERIFY PATHS
# =============================================================================

def verify_paths():

    print("=" * 70)
    print("VERIFYING PATHS")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nModel file not found:\n{MODEL_PATH.absolute()}"
        )

    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(
            f"\nDataset YAML not found:\n{Path(DATA_YAML).absolute()}"
        )

    print(f"\n✓ MODEL FOUND:")
    print(MODEL_PATH)

    print(f"\n✓ DATA YAML FOUND:")
    print(DATA_YAML)


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model():

    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    model = YOLO(str(MODEL_PATH))

    print("\n✓ Model loaded successfully.")

    return model


# =============================================================================
# DATASET ANALYSIS
# =============================================================================

def analyze_dataset():

    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)

    with open(DATA_YAML, "r") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg.get("path", "."))

    names = data_cfg.get("names", [])

    splits = ["train", "val", "test"]

    dataset_stats = {}

    class_counts = {name: 0 for name in names}

    for split in splits:

        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split

        img_files = list(img_dir.glob("*.jpg")) + \
                    list(img_dir.glob("*.png"))

        lbl_files = list(lbl_dir.glob("*.txt"))

        total_boxes = 0

        for lbl in lbl_files:

            lines = lbl.read_text().splitlines()

            for line in lines:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                cls_id = int(float(parts[0]))

                if cls_id < len(names):
                    class_counts[names[cls_id]] += 1

                total_boxes += 1

        dataset_stats[split] = {
            "images": len(img_files),
            "labels": len(lbl_files),
            "boxes": total_boxes
        }

    print("\nDATASET SUMMARY\n")

    for split, stats in dataset_stats.items():

        print(
            f"{split.upper():<10}"
            f"Images: {stats['images']:<6} "
            f"Labels: {stats['labels']:<6} "
            f"Boxes: {stats['boxes']}"
        )

    # Category Distribution
    plt.figure(figsize=(10, 5))

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    bars = plt.bar(classes, counts)

    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            str(int(bar.get_height())),
            ha="center",
            fontsize=9
        )

    plt.title("Category Distribution")
    plt.ylabel("Number of Instances")

    plt.tight_layout()

    save_path = SAVE_DIR / "category_distribution.png"

    plt.savefig(save_path, dpi=150)

    plt.close()

    print(f"\n✓ Category distribution saved:")
    print(save_path)

    return dataset_stats


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model):

    results = {}

    for split in ["val", "test"]:

        print("\n" + "=" * 70)
        print(f"EVALUATING {split.upper()} SET")
        print("=" * 70)

        metrics = model.val(
            data=DATA_YAML,
            split=split,
            device=DEVICE,
            plots=True,
            save_json=True,
            project=str(SAVE_DIR),
            name=split,
            exist_ok=True
        )

        precision = metrics.box.mp
        recall = metrics.box.mr
        map50 = metrics.box.map50
        map5095 = metrics.box.map

        f1 = 2 * precision * recall / (precision + recall + 1e-16)

        print("\nOVERALL METRICS\n")

        print(f"Precision     : {precision:.4f}")
        print(f"Recall        : {recall:.4f}")
        print(f"F1 Score      : {f1:.4f}")
        print(f"mAP@50        : {map50:.4f}")
        print(f"mAP@50-95     : {map5095:.4f}")

        per_class = {}

        if hasattr(metrics.box, "ap_class_index"):

            print("\nPER-CLASS PERFORMANCE\n")

            for i, cls_idx in enumerate(metrics.box.ap_class_index):

                cls_name = metrics.names[cls_idx]

                cls_map50 = float(metrics.box.ap50[i])
                cls_map95 = float(metrics.box.ap[i])

                per_class[cls_name] = {
                    "map50": cls_map50,
                    "map5095": cls_map95
                }

                print(
                    f"{cls_name:<15}"
                    f"mAP@50={cls_map50:.4f}   "
                    f"mAP@50-95={cls_map95:.4f}"
                )

        results[split] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": map50,
            "map5095": map5095,
            "per_class": per_class
        }

    return results


# =============================================================================
# OVERALL METRICS PLOT
# =============================================================================

def plot_overall_metrics(results):

    labels = ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]

    val_values = [
        results["val"]["precision"],
        results["val"]["recall"],
        results["val"]["f1"],
        results["val"]["map50"],
        results["val"]["map5095"]
    ]

    test_values = [
        results["test"]["precision"],
        results["test"]["recall"],
        results["test"]["f1"],
        results["test"]["map50"],
        results["test"]["map5095"]
    ]

    x = np.arange(len(labels))

    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(
        x - width/2,
        val_values,
        width,
        label="Validation"
    )

    bars2 = ax.bar(
        x + width/2,
        test_values,
        width,
        label="Test"
    )

    for bars in [bars1, bars2]:

        for bar in bars:

            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center",
                fontsize=9
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 1.1)

    ax.set_ylabel("Score")

    ax.set_title("YOLOv8m Performance Comparison")

    ax.legend()

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    save_path = SAVE_DIR / "overall_metrics.png"

    plt.savefig(save_path, dpi=150)

    plt.close()

    print(f"\n✓ Overall metrics plot saved:")
    print(save_path)


# =============================================================================
# PER-CLASS MAP PLOT
# =============================================================================

def plot_per_class_map(results, split="test"):

    per_class = results[split]["per_class"]

    class_names = list(per_class.keys())

    map50_vals = [per_class[c]["map50"] for c in class_names]
    map95_vals = [per_class[c]["map5095"] for c in class_names]

    x = np.arange(len(class_names))

    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x - width/2,
        map50_vals,
        width,
        label="mAP@50"
    )

    ax.bar(
        x + width/2,
        map95_vals,
        width,
        label="mAP@50-95"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(class_names)

    ax.set_ylim(0, 1.1)

    ax.set_ylabel("mAP")

    ax.set_title(f"Per-Class Performance ({split.upper()})")

    ax.legend()

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    save_path = SAVE_DIR / f"per_class_map_{split}.png"

    plt.savefig(save_path, dpi=150)

    plt.close()

    print(f"\n✓ Per-class mAP plot saved:")
    print(save_path)


# =============================================================================
# LOSS CURVES
# =============================================================================

def plot_training_curves():

    print("\n" + "=" * 70)
    print("GENERATING TRAINING CURVES")
    print("=" * 70)

    results_csv = RUN_DIR / "results.csv"

    if not results_csv.exists():

        print("\nWARNING: results.csv not found.")
        return

    df = pd.read_csv(results_csv)

    df.columns = [c.strip() for c in df.columns]

    plt.figure(figsize=(12, 6))

    loss_found = False

    if "train/box_loss" in df.columns:
        plt.plot(df["train/box_loss"], label="Train Box Loss")
        loss_found = True

    if "val/box_loss" in df.columns:
        plt.plot(df["val/box_loss"], label="Val Box Loss")
        loss_found = True

    if "train/cls_loss" in df.columns:
        plt.plot(df["train/cls_loss"], label="Train Cls Loss")
        loss_found = True

    if "val/cls_loss" in df.columns:
        plt.plot(df["val/cls_loss"], label="Val Cls Loss")
        loss_found = True

    if not loss_found:
        print("No loss columns found in results.csv")
        return

    plt.title("Training & Validation Loss Curves")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    save_path = SAVE_DIR / "loss_curves.png"

    plt.savefig(save_path, dpi=150)

    plt.close()

    print(f"\n✓ Loss curves saved:")
    print(save_path)


# =============================================================================
# OVERFITTING ANALYSIS
# =============================================================================

def overfitting_analysis(results):

    labels = ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]

    val_values = [
        results["val"]["precision"],
        results["val"]["recall"],
        results["val"]["f1"],
        results["val"]["map50"],
        results["val"]["map5095"]
    ]

    test_values = [
        results["test"]["precision"],
        results["test"]["recall"],
        results["test"]["f1"],
        results["test"]["map50"],
        results["test"]["map5095"]
    ]

    gaps = [v - t for v, t in zip(val_values, test_values)]

    plt.figure(figsize=(10, 5))

    bars = plt.bar(labels, gaps)

    plt.axhline(0, color="black")

    for bar in bars:

        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{bar.get_height():.3f}",
            ha="center",
            fontsize=9
        )

    plt.title("Generalization Gap Analysis")
    plt.ylabel("Validation - Test")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    save_path = SAVE_DIR / "overfitting_analysis.png"

    plt.savefig(save_path, dpi=150)

    plt.close()

    print(f"\n✓ Overfitting analysis saved:")
    print(save_path)


# =============================================================================
# PREDICTION VISUALIZATIONS
# =============================================================================

def visualize_predictions(model):

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    with open(DATA_YAML, "r") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg.get("path", "."))

    names = data_cfg.get("names", [])

    test_dir = dataset_root / "images" / "test"

    images = list(test_dir.glob("*.jpg")) + \
             list(test_dir.glob("*.png"))

    if len(images) == 0:

        print("No test images found.")
        return

    vis_dir = SAVE_DIR / "visualizations"

    vis_dir.mkdir(exist_ok=True)

    selected = random.sample(
        images,
        min(NUM_VIS, len(images))
    )

    for idx, img_path in enumerate(selected):

        result = model.predict(
            source=str(img_path),
            conf=CONF_THRESH,
            device=DEVICE,
            verbose=False
        )[0]

        img = result.orig_img[:, :, ::-1]

        plt.figure(figsize=(10, 8))

        plt.imshow(img)

        ax = plt.gca()

        if result.boxes is not None:

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                cls_id = int(box.cls[0].cpu().numpy())

                conf = float(box.conf[0].cpu().numpy())

                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    fill=False
                )

                ax.add_patch(rect)

                label = f"{names[cls_id]} {conf:.2f}"

                plt.text(
                    x1,
                    y1 - 5,
                    label,
                    fontsize=8,
                    bbox=dict(facecolor="white")
                )

        plt.axis("off")

        plt.title(img_path.name)

        save_path = vis_dir / f"prediction_{idx}.png"

        plt.savefig(save_path, dpi=120, bbox_inches="tight")

        plt.close()

    print(f"\n✓ Prediction visualizations saved:")
    print(vis_dir)


# =============================================================================
# FAILURE CASE ANALYSIS
# =============================================================================

def analyze_failure_cases(model):

    print("\n" + "=" * 70)
    print("ANALYZING FAILURE CASES")
    print("=" * 70)

    with open(DATA_YAML, "r") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg.get("path", "."))

    names = data_cfg.get("names", [])

    test_dir = dataset_root / "images" / "test"

    images = list(test_dir.glob("*.jpg")) + \
             list(test_dir.glob("*.png"))

    fail_dir = SAVE_DIR / "failure_cases"

    fail_dir.mkdir(exist_ok=True)

    failure_count = 0

    for idx, img_path in enumerate(images[:100]):

        result = model.predict(
            source=str(img_path),
            conf=CONF_THRESH,
            device=DEVICE,
            verbose=False
        )[0]

        num_preds = len(result.boxes) if result.boxes is not None else 0

        if num_preds == 0:

            img = result.orig_img[:, :, ::-1]

            plt.figure(figsize=(10, 8))

            plt.imshow(img)

            plt.title(f"Missed Detection: {img_path.name}")

            plt.axis("off")

            save_path = fail_dir / f"failure_{failure_count}.png"

            plt.savefig(save_path, dpi=120)

            plt.close()

            failure_count += 1

        if failure_count >= 10:
            break

    print(f"\n✓ Failure cases saved:")
    print(fail_dir)


# =============================================================================
# SAVE HYPERPARAMETERS
# =============================================================================

def save_hyperparameters():

    hyperparams = {

        "Model": "YOLOv8m",
        "Image Size": 640,
        "Batch Size": 12,
        "Epochs": 50,
        "Optimizer": "AdamW",
        "Learning Rate": 0.001,
        "Weight Decay": 0.0005,
        "Confidence Threshold": CONF_THRESH,
        "Device": DEVICE,
        "Augmentation": "Mosaic, MixUp, CopyPaste, HSV"
    }

    save_path = SAVE_DIR / "hyperparameters.csv"

    pd.DataFrame(
        hyperparams.items(),
        columns=["Parameter", "Value"]
    ).to_csv(save_path, index=False)

    print(f"\n✓ Hyperparameters saved:")
    print(save_path)


# =============================================================================
# SAVE RESULTS CSV
# =============================================================================

def save_results_csv(results):

    save_path = SAVE_DIR / "evaluation_summary.csv"

    rows = []

    for split in ["val", "test"]:

        r = results[split]

        rows.append({

            "Split": split,
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "mAP50": r["map50"],
            "mAP50-95": r["map5095"]
        })

    pd.DataFrame(rows).to_csv(save_path, index=False)

    print(f"\n✓ Evaluation summary saved:")
    print(save_path)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("COMPLETE YOLOv8m RESEARCH EVALUATION PIPELINE")
    print("=" * 70)

    verify_paths()

    model = load_model()

    analyze_dataset()

    results = evaluate_model(model)

    plot_overall_metrics(results)

    plot_per_class_map(results, split="val")

    plot_per_class_map(results, split="test")

    plot_training_curves()

    overfitting_analysis(results)

    visualize_predictions(model)

    analyze_failure_cases(model)

    save_hyperparameters()

    save_results_csv(results)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(f"\nAll outputs saved in:\n{SAVE_DIR}\n")

    print("Generated outputs:")
    print("-------------------")
    print("✓ category_distribution.png")
    print("✓ overall_metrics.png")
    print("✓ per_class_map_val.png")
    print("✓ per_class_map_test.png")
    print("✓ loss_curves.png")
    print("✓ overfitting_analysis.png")
    print("✓ evaluation_summary.csv")
    print("✓ hyperparameters.csv")
    print("✓ prediction visualizations/")
    print("✓ failure_cases/")
    print("✓ confusion matrices")
    print("✓ PR/F1 curves")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()