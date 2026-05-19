"""
rtdetr_rdd2022_full.py
──────────────────────
RT-DETR (ResNet-50) training on RDD2022 China dataset.
batch=12 to match YOLOv8s/m and Faster R-CNN for fair comparison.

Generates ALL required plots:
  - loss_curve.png
  - component_losses.png
  - mAP_curve.png
  - overfitting_analysis.png
  - BoxF1_BoxP_BoxR_curves.png
  - per_class_metrics.png
  - confusion_matrix.png
  - confusion_matrix_normalized.png
  - val_batch{0,1,2}_labels.jpg
  - val_batch{0,1,2}_pred.jpg
  - overall_metrics_summary.png
  - final_metrics_report.txt
  - epoch_metrics_table.txt

Usage:
    python rtdetr_rdd2022_full.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import yaml

from pathlib import Path
from ultralytics import RTDETR

# ============================================================
# CONFIG
# ============================================================

DATA_YAML   = "rdd2022_china_yolo/dataset.yaml"
PROJECT     = "runs_rdd2022"
RUN_NAME    = "rtdetr_r50_50epochs"
CLASS_NAMES = ["D00", "D10", "D20", "D40"]
EPOCHS      = 50

RUN_DIR   = Path(PROJECT) / RUN_NAME
PLOTS_DIR = RUN_DIR / "custom_plots"
VAL_DIR   = PLOTS_DIR / "val"

for d in [PLOTS_DIR, VAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD results.csv
# ============================================================

def load_results_csv(run_dir):
    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found at: {csv_path}")
    history = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                k = k.strip()
                history.setdefault(k, [])
                try:
                    history[k].append(float(v.strip()))
                except ValueError:
                    history[k].append(0.0)
    print(f"  Loaded results.csv — {len(next(iter(history.values())))} epochs")
    return history

# ============================================================
# PLOT 1 — loss_curve.png
# ============================================================

def save_loss_curve(history, save_dir):
    box   = history.get("train/box_loss", [])
    cls   = history.get("train/cls_loss", [])
    dfl   = history.get("train/dfl_loss", [])
    map50 = history.get("metrics/mAP50(B)", [])

    if not box:
        print("  [SKIP] loss_curve"); return

    train_total = [b+c+d for b, c, d in zip(box, cls, dfl)]
    epochs = range(1, len(train_total) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_total, color="#E63946", lw=2, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="#E63946")
    ax1.tick_params(axis="y", labelcolor="#E63946")

    if len(map50) == len(train_total):
        ax2 = ax1.twinx()
        ax2.plot(epochs, map50, color="#457B9D", lw=2,
                 linestyle="--", label="Val mAP@50")
        ax2.set_ylabel("Val mAP@50", color="#457B9D")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis="y", labelcolor="#457B9D")
        l1, n1 = ax1.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax1.legend(l1+l2, n1+n2, loc="upper right", fontsize=9)

    ax1.set_title("Training Loss vs Val mAP@50", fontsize=14, fontweight="bold")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    print("  Saved: loss_curve.png")

# ============================================================
# PLOT 2 — component_losses.png
# ============================================================

def save_component_losses(history, save_dir):
    if not history.get("train/box_loss"):
        print("  [SKIP] component_losses"); return

    components = [
        ("train/box_loss", "val/box_loss", "Box Loss",   "#E63946"),
        ("train/cls_loss", "val/cls_loss", "Class Loss", "#2A9D8F"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss",   "#E9C46A"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (tk, vk, title, color) in zip(axes, components):
        t_vals = history.get(tk, [])
        v_vals = history.get(vk, [])
        ax.plot(range(1, len(t_vals)+1), t_vals, color=color, lw=2, label="Train")
        if v_vals:
            ax.plot(range(1, len(v_vals)+1), v_vals, color=color,
                    lw=2, linestyle="--", alpha=0.6, label="Val")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Component Losses (Train vs Val)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "component_losses.png", dpi=150)
    plt.close(fig)
    print("  Saved: component_losses.png")

# ============================================================
# PLOT 3 — mAP_curve.png
# ============================================================

def save_map_curve(history, save_dir):
    map50   = history.get("metrics/mAP50(B)", [])
    map5095 = history.get("metrics/mAP50-95(B)", [])
    if not map50:
        print("  [SKIP] mAP_curve"); return

    epochs = range(1, len(map50) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, map50,   label="mAP@50",    color="#E63946", lw=2)
    ax.plot(epochs, map5095, label="mAP@50-95", color="#457B9D", lw=2, linestyle="--")
    ax.set_title("mAP Curves", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "mAP_curve.png", dpi=150)
    plt.close(fig)
    print("  Saved: mAP_curve.png")

# ============================================================
# PLOT 4 — overfitting_analysis.png
# ============================================================

def save_overfitting_analysis(history, save_dir):
    box   = history.get("train/box_loss", [])
    cls   = history.get("train/cls_loss", [])
    dfl   = history.get("train/dfl_loss", [])
    map50 = history.get("metrics/mAP50(B)", [])
    prec  = history.get("metrics/precision(B)", [])
    rec   = history.get("metrics/recall(B)", [])

    if not box:
        print("  [SKIP] overfitting_analysis"); return

    train_total = [b+c+d for b, c, d in zip(box, cls, dfl)]
    f1_arr      = np.array([2*p*r/(p+r+1e-16) for p, r in zip(prec, rec)])
    epochs      = range(1, len(train_total) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1 — dual axis
    ax1 = axes[0]
    ax1.plot(epochs, train_total, color="#E63946", lw=2, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="#E63946")
    ax1.tick_params(axis="y", labelcolor="#E63946")

    ax1b = ax1.twinx()
    ax1b.plot(epochs, map50, color="#457B9D", lw=2,
              linestyle="--", label="Val mAP@50")
    ax1b.set_ylabel("Val mAP@50", color="#457B9D")
    ax1b.tick_params(axis="y", labelcolor="#457B9D")
    ax1b.set_ylim(0, 1)

    best_ep  = int(np.argmax(map50)) + 1
    best_map = float(max(map50))
    ax1b.axvline(best_ep, color="#2A9D8F", linestyle=":", lw=1.5)
    ax1b.annotate(
        f"Best ep {best_ep}\n{best_map:.3f}",
        xy=(best_ep, best_map),
        xytext=(min(best_ep+2, len(train_total)), best_map-0.06),
        fontsize=8, color="#2A9D8F",
        arrowprops=dict(arrowstyle="->", color="#2A9D8F"),
    )
    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax1b.get_legend_handles_labels()
    ax1.legend(l1+l2, n1+n2, loc="upper right", fontsize=8)
    ax1.set_title("Overfitting Analysis:\nTrain Loss vs Val mAP@50", fontweight="bold")
    ax1.grid(alpha=0.3)

    # Panel 2 — smoothed F1 trend
    ax2 = axes[1]
    ax2.plot(epochs, f1_arr, color="#2A9D8F", lw=1.5, alpha=0.5, label="Val F1")
    if len(f1_arr) >= 5:
        window   = max(3, len(f1_arr) // 10)
        smoothed = np.convolve(f1_arr, np.ones(window)/window, mode="valid")
        ax2.plot(range(window, len(f1_arr)+1), smoothed,
                 color="#264653", lw=2.5, label=f"Smoothed (w={window})")
    ax2.set_title("Generalisation: Val F1 Trend", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("Overfitting Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "overfitting_analysis.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: overfitting_analysis.png  (best epoch: {best_ep})")

# ============================================================
# PLOT 5 — BoxF1_BoxP_BoxR_curves.png
# ============================================================

def save_metric_curves(history, save_dir):
    precision = history.get("metrics/precision(B)", [])
    recall    = history.get("metrics/recall(B)", [])
    if not precision:
        print("  [SKIP] metric curves"); return

    f1     = [2*p*r/(p+r+1e-16) for p, r in zip(precision, recall)]
    epochs = range(1, len(precision) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(epochs, precision, color="#E63946", lw=2)
    axes[0].set_title("BoxP Curve", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Precision")
    axes[0].set_ylim(0, 1); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, recall, color="#457B9D", lw=2)
    axes[1].set_title("BoxR Curve", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Recall")
    axes[1].set_ylim(0, 1); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, f1, color="#2A9D8F", lw=2)
    axes[2].set_title("BoxF1 Curve", fontweight="bold")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("F1 Score")
    axes[2].set_ylim(0, 1); axes[2].grid(alpha=0.3)

    fig.suptitle("Precision / Recall / F1 over Epochs", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "BoxF1_BoxP_BoxR_curves.png", dpi=150)
    plt.close(fig)
    print("  Saved: BoxF1_BoxP_BoxR_curves.png")

# ============================================================
# PLOT 6 — per_class_metrics.png
# ============================================================

def save_per_class_bar(metrics, save_dir):
    try:
        p_per_cls = list(metrics.box.p)
        r_per_cls = list(metrics.box.r)
    except AttributeError:
        print("  [SKIP] per_class_metrics"); return

    f1_per_cls = [2*p*r/(p+r+1e-16) for p, r in zip(p_per_cls, r_per_cls)]
    cls_names  = CLASS_NAMES[:len(p_per_cls)]

    x, w = np.arange(len(cls_names)), 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x-w, p_per_cls,  w, label="Precision", color="#E63946")
    ax.bar(x,   r_per_cls,  w, label="Recall",    color="#457B9D")
    ax.bar(x+w, f1_per_cls, w, label="F1",        color="#2A9D8F")
    ax.set_xticks(x); ax.set_xticklabels(cls_names)
    ax.set_ylim(0, 1)
    ax.set_title("Per-Class Precision / Recall / F1", fontweight="bold")
    ax.set_ylabel("Score"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "per_class_metrics.png", dpi=150)
    plt.close(fig)
    print("  Saved: per_class_metrics.png")

# ============================================================
# PLOT 7 — confusion_matrix.png + normalized
# ============================================================

def save_confusion_matrix(metrics, save_dir):
    try:
        cm = metrics.confusion_matrix.matrix.astype(int)
    except AttributeError:
        print("  [SKIP] confusion_matrix"); return

    labels = CLASS_NAMES[:cm.shape[0]-1] + ["background"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Ground Truth")

    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    norm_cm  = np.divide(cm, row_sums,
                         out=np.zeros_like(cm, dtype=float),
                         where=row_sums != 0)
    sns.heatmap(norm_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalised)", fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Ground Truth")

    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(norm_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_title("Confusion Matrix Normalised", fontweight="bold")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Ground Truth")
    fig2.tight_layout()
    fig2.savefig(save_dir / "confusion_matrix_normalized.png", dpi=150)
    plt.close(fig2)
    print("  Saved: confusion_matrix.png + confusion_matrix_normalized.png")

# ============================================================
# PLOT 8 — val batch images
# ============================================================

def save_val_batch_images(model, data_yaml, save_dir, num_batches=3):
    from PIL import Image as PILImage

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    base        = Path(cfg.get("path", "."))
    val_rel     = cfg.get("val", "images/val")
    val_img_dir = base / val_rel
    if not val_img_dir.exists():
        val_img_dir = Path(data_yaml).parent / val_rel
    if not val_img_dir.exists():
        print(f"  [SKIP] val images not found: {val_img_dir}"); return

    all_imgs = sorted(list(val_img_dir.glob("*.jpg")) +
                      list(val_img_dir.glob("*.png")))
    if not all_imgs:
        print("  [SKIP] no val images found"); return

    colors = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D"]

    for batch_idx in range(num_batches):
        batch_imgs = all_imgs[batch_idx*4 : batch_idx*4+4]
        if not batch_imgs:
            break

        n = len(batch_imgs)
        fig_lbl,  axes_lbl  = plt.subplots(1, n, figsize=(5*n, 5))
        fig_pred, axes_pred = plt.subplots(1, n, figsize=(5*n, 5))
        if n == 1:
            axes_lbl  = [axes_lbl]
            axes_pred = [axes_pred]

        results = model.predict(
            [str(p) for p in batch_imgs],
            imgsz=640, conf=0.25, verbose=False,
        )

        for i, (img_path, result) in enumerate(zip(batch_imgs, results)):
            img_np   = np.array(
                PILImage.open(img_path).convert("RGB").resize((640, 640))
            )
            lbl_path = (img_path.parent.parent.parent / "labels" /
                        img_path.parent.name / (img_path.stem + ".txt"))

            # Ground truth
            axes_lbl[i].imshow(img_np)
            axes_lbl[i].set_title(f"Img {batch_idx*4+i}", fontsize=9)
            axes_lbl[i].axis("off")
            if lbl_path.exists():
                h, w = img_np.shape[:2]
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, cx, cy, bw, bh = map(float, parts)
                    if int(cls_id) >= len(CLASS_NAMES):
                        continue
                    x1=(cx-bw/2)*w; y1=(cy-bh/2)*h
                    x2=(cx+bw/2)*w; y2=(cy+bh/2)*h
                    color = colors[int(cls_id) % len(colors)]
                    axes_lbl[i].add_patch(
                        patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                          linewidth=2, edgecolor=color,
                                          facecolor="none"))
                    axes_lbl[i].text(x1, y1-2, CLASS_NAMES[int(cls_id)],
                                     fontsize=7, color="white",
                                     bbox=dict(facecolor=color, alpha=0.7, pad=1))

            # Predictions
            axes_pred[i].imshow(img_np)
            axes_pred[i].set_title(f"Img {batch_idx*4+i}", fontsize=9)
            axes_pred[i].axis("off")
            for box, lbl, score in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box
                color = colors[lbl % len(colors)]
                axes_pred[i].add_patch(
                    patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                      linewidth=2, edgecolor=color,
                                      facecolor="none"))
                name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else str(lbl)
                axes_pred[i].text(x1, y1-2, f"{name} {score:.2f}",
                                  fontsize=7, color="white",
                                  bbox=dict(facecolor=color, alpha=0.7, pad=1))

        for fig, suffix in [(fig_lbl, "labels"), (fig_pred, "pred")]:
            fig.suptitle(f"Batch {batch_idx} — {suffix}", fontweight="bold")
            fig.tight_layout()
            fig.savefig(save_dir / f"val_batch{batch_idx}_{suffix}.jpg", dpi=120)
            plt.close(fig)

    print(f"  Saved val batch images -> {save_dir}")

# ============================================================
# FINAL REPORT  (encoding="utf-8" — no cp1252 crash on Windows)
# ============================================================

def save_final_report(metrics, history, save_dir):
    precision = metrics.box.mp
    recall    = metrics.box.mr
    map50     = metrics.box.map50
    map5095   = metrics.box.map
    f1        = 2 * precision * recall / (precision + recall + 1e-16)

    # Per-class
    try:
        p_cls = list(metrics.box.p)
        r_cls = list(metrics.box.r)
    except AttributeError:
        p_cls = r_cls = []

    summary_lines = [
        "",
        "=" * 60,
        "   OVERALL METRICS SUMMARY — RT-DETR R50 (best weights)",
        "=" * 60,
        f"  Overall Precision       : {precision:.4f}",
        f"  Overall Recall          : {recall:.4f}",
        f"  Overall F1 Score        : {f1:.4f}",
        f"  mAP@50                  : {map50:.4f}",
        f"  mAP@50-95               : {map5095:.4f}",
    ]

    if history:
        best_ep = int(np.argmax(history.get("metrics/mAP50(B)", [0]))) + 1
        summary_lines += [
            "-" * 60,
            f"  Best Epoch (mAP@50)     : {best_ep}",
            f"  Best mAP@50             : {max(history.get('metrics/mAP50(B)', [0])):.4f}",
            f"  Best mAP@50-95          : {max(history.get('metrics/mAP50-95(B)', [0])):.4f}",
        ]

    if p_cls:
        summary_lines += [
            "-" * 60,
            "  PER-CLASS METRICS",
            f"  {'Class':<10} {'P':>6}  {'R':>6}  {'F1':>6}",
            "  " + "-" * 32,
        ]
        for i, cls in enumerate(CLASS_NAMES[:len(p_cls)]):
            f1c = 2*p_cls[i]*r_cls[i]/(p_cls[i]+r_cls[i]+1e-16)
            summary_lines.append(
                f"  {cls:<10} {p_cls[i]:>6.3f}  {r_cls[i]:>6.3f}  {f1c:>6.3f}"
            )

    summary_lines += ["=" * 60, ""]
    report = "\n".join(summary_lines)
    print(report)

    # ✅ encoding="utf-8" — prevents cp1252 crash on Windows
    report_path = save_dir / "final_metrics_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")

    # Epoch table
    if history:
        map50_h   = history.get("metrics/mAP50(B)", [])
        map5095_h = history.get("metrics/mAP50-95(B)", [])
        prec_h    = history.get("metrics/precision(B)", [])
        rec_h     = history.get("metrics/recall(B)", [])
        box_h     = history.get("train/box_loss", [])
        cls_h     = history.get("train/cls_loss", [])
        dfl_h     = history.get("train/dfl_loss", [])

        table_lines = [
            "\nEpoch-by-Epoch Validation Metrics:",
            f"{'Epoch':>6}  {'Loss':>8}  {'Prec':>6}  "
            f"{'Recall':>6}  {'F1':>6}  {'mAP50':>6}  {'mAP50-95':>8}",
            "-" * 62,
        ]
        for ep in range(len(map50_h)):
            loss = (box_h[ep]+cls_h[ep]+dfl_h[ep]
                    if ep < len(box_h) else 0.0)
            p    = prec_h[ep] if ep < len(prec_h) else 0.0
            r    = rec_h[ep]  if ep < len(rec_h)  else 0.0
            f1e  = 2*p*r/(p+r+1e-16)
            table_lines.append(
                f"{ep+1:>6}  {loss:>8.4f}  {p:>6.4f}  "
                f"{r:>6.4f}  {f1e:>6.4f}  "
                f"{map50_h[ep]:>6.4f}  {map5095_h[ep]:>8.4f}"
            )

        # ✅ encoding="utf-8" here too
        table_path = save_dir / "epoch_metrics_table.txt"
        table_path.write_text("\n".join(table_lines), encoding="utf-8")
        print(f"  Saved: {table_path}")

    # Summary bar chart
    if p_cls:
        f1_cls = [2*p*r/(p+r+1e-16) for p, r in zip(p_cls, r_cls)]
        names  = CLASS_NAMES[:len(p_cls)]
        x, w   = np.arange(len(names)), 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x-w, p_cls,  w, label="Precision", color="#E63946", alpha=0.85)
        ax.bar(x,   r_cls,  w, label="Recall",    color="#457B9D", alpha=0.85)
        ax.bar(x+w, f1_cls, w, label="F1",        color="#2A9D8F", alpha=0.85)
        ax.axhline(precision, color="#E63946", linestyle=":",
                   lw=1.5, label=f"Overall P={precision:.3f}")
        ax.axhline(recall,    color="#457B9D", linestyle=":",
                   lw=1.5, label=f"Overall R={recall:.3f}")
        ax.axhline(f1,        color="#2A9D8F", linestyle=":",
                   lw=1.5, label=f"Overall F1={f1:.3f}")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Overall Metrics Summary — RT-DETR R50",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "overall_metrics_summary.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: overall_metrics_summary.png")

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n============================================")
    print(" RT-DETR R50 Training  |  RDD2022 China")
    print(" batch=12  (fair comparison with YOLOv8/FRCNN)")
    print("============================================\n")

    model = RTDETR("rtdetr-l.pt")

    # ── Train ────────────────────────────────────────────────────
    model.train(
        data       = DATA_YAML,
        epochs     = EPOCHS,
        imgsz      = 640,
        batch      = 12,        # matches YOLOv8s/m and Faster R-CNN
        workers    = 0,
        device     = 0,
        optimizer  = "AdamW",
        lr0        = 0.0001,
        weight_decay = 0.0001,
        warmup_epochs = 3,
        patience   = 20,
        pretrained = True,
        verbose    = True,
        project    = PROJECT,
        name       = RUN_NAME,
    )

    # ── Validation ───────────────────────────────────────────────
    print("\nRunning final validation...")
    metrics = model.val(data=DATA_YAML, verbose=False)

    precision = metrics.box.mp
    recall    = metrics.box.mr
    map50     = metrics.box.map50
    map5095   = metrics.box.map
    f1        = 2 * precision * recall / (precision + recall + 1e-16)

    print("\n========== FINAL METRICS ==========")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"mAP50       : {map50:.4f}")
    print(f"mAP50-95    : {map5095:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print("====================================\n")

    # ── Load history ─────────────────────────────────────────────
    history = load_results_csv(RUN_DIR)

    # ── Save all plots ────────────────────────────────────────────
    print("Saving plots...")
    save_loss_curve(history, PLOTS_DIR)
    save_component_losses(history, PLOTS_DIR)
    save_map_curve(history, PLOTS_DIR)
    save_overfitting_analysis(history, PLOTS_DIR)
    save_metric_curves(history, PLOTS_DIR)
    save_per_class_bar(metrics, PLOTS_DIR)
    save_confusion_matrix(metrics, PLOTS_DIR)
    save_val_batch_images(model, DATA_YAML, VAL_DIR)
    save_final_report(metrics, history, PLOTS_DIR)

    print(f"\nAll plots -> {PLOTS_DIR}")
    print(f"Val images -> {VAL_DIR}")
    print("\n============================================")
    print(" Training Complete!")
    print("============================================")


if __name__ == "__main__":
    main()