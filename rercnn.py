"""
rerun_report_only.py
────────────────────
Loads your saved fasterrcnn_best.pth + re-runs ONE val pass
to regenerate the final metrics report and summary plot.

No retraining. Runs in ~2-3 minutes.

Usage:
    python rerun_report_only.py
"""

import torch
import torchvision
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from collections import defaultdict

from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from PIL import Image
from torchvision.transforms import functional as F

# ============================================================
# CONFIG — must match your original fastrcnn.py
# ============================================================

DATASET_ROOT = "rdd2022_china_yolo"
NUM_CLASSES  = 5
IMG_SIZE     = 640
IOU_THRESH   = 0.5
SCORE_THRESH = 0.5
EPOCHS       = 50
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES  = ["background", "D00", "D10", "D20", "D40"]

WEIGHTS_DIR  = Path("weights")
PLOTS_DIR    = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATASET  (same as original)
# ============================================================

class RDDDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="val"):
        self.img_dir = Path(root) / "images" / split
        self.lbl_dir = Path(root) / "labels" / split
        self.images  = sorted(list(self.img_dir.glob("*.jpg")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"

        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        w, h = img.size
        boxes, labels = [], []

        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = map(float, parts)
                if int(cls_id) > 3:
                    continue
                xmin = (cx - bw/2)*w; ymin = (cy - bh/2)*h
                xmax = (cx + bw/2)*w; ymax = (cy + bh/2)*h
                if xmax <= xmin or ymax <= ymin:
                    continue
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(cls_id) + 1)

        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        return F.to_tensor(img), {
            "boxes": boxes, "labels": labels,
            "image_id": torch.tensor([idx])
        }

def collate_fn(batch):
    return tuple(zip(*batch))

# ============================================================
# MODEL
# ============================================================

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ============================================================
# BOX IoU
# ============================================================

def box_iou_single(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-16)

# ============================================================
# EVALUATE  (same logic as original)
# ============================================================

def evaluate(model, loader):
    model.eval()
    metric = MeanAveragePrecision(
        backend="faster_coco_eval",
        iou_thresholds=[IOU_THRESH],
    )

    all_preds_by_cls = defaultdict(list)
    gt_count_by_cls  = defaultdict(int)
    conf_matrix      = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    global_tp = global_fp = global_fn = 0

    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(DEVICE) for img in images]
            outputs = model(images)
            preds_batch, gts_batch = [], []

            for output, target in zip(outputs, targets):
                boxes_pred  = output["boxes"].cpu().numpy()
                scores_pred = output["scores"].cpu().numpy()
                labels_pred = output["labels"].cpu().numpy()
                boxes_gt    = target["boxes"].numpy()
                labels_gt   = target["labels"].numpy()

                preds_batch.append({
                    "boxes":  torch.tensor(boxes_pred),
                    "scores": torch.tensor(scores_pred),
                    "labels": torch.tensor(labels_pred),
                })
                gts_batch.append({
                    "boxes":  target["boxes"],
                    "labels": target["labels"],
                })

                keep          = scores_pred >= SCORE_THRESH
                boxes_pred_k  = boxes_pred[keep]
                scores_pred_k = scores_pred[keep]
                labels_pred_k = labels_pred[keep]

                matched_gt = set()
                order = np.argsort(-scores_pred_k)

                for cls in range(1, NUM_CLASSES):
                    gt_count_by_cls[cls] += int((labels_gt == cls).sum())

                for i in order:
                    pred_cls = labels_pred_k[i]
                    pred_box = boxes_pred_k[i]
                    best_iou = IOU_THRESH - 1e-6
                    best_j   = -1

                    for j, (gt_box, gt_cls) in enumerate(
                            zip(boxes_gt, labels_gt)):
                        if j in matched_gt:
                            continue
                        iou = box_iou_single(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou; best_j = j

                    if best_j >= 0:
                        matched_gt.add(best_j)
                        gt_cls = int(labels_gt[best_j])
                        conf_matrix[gt_cls][int(pred_cls)] += 1
                        tp = 1 if pred_cls == gt_cls else 0
                    else:
                        tp = 0
                        conf_matrix[0][int(pred_cls)] += 1

                    all_preds_by_cls[int(pred_cls)].append(
                        (scores_pred_k[i], tp))
                    if tp:
                        global_tp += 1
                    else:
                        global_fp += 1

                global_fn += len(boxes_gt) - len(matched_gt)

            metric.update(preds_batch, gts_batch)

    results  = metric.compute()
    map50    = results["map_50"].item()
    map5095  = results["map"].item()

    precision = global_tp / (global_tp + global_fp + 1e-16)
    recall    = global_tp / (global_tp + global_fn + 1e-16)
    f1        = 2 * precision * recall / (precision + recall + 1e-16)

    per_class_stats = {}
    for cls in range(1, NUM_CLASSES):
        preds_cls = all_preds_by_cls[cls]
        tp = sum(t for _, t in preds_cls)
        fp = len(preds_cls) - tp
        fn = gt_count_by_cls[cls] - tp
        p  = tp / (tp + fp + 1e-16)
        r  = tp / (tp + fn + 1e-16)
        f  = 2*p*r / (p+r+1e-16)
        per_class_stats[CLASS_NAMES[cls]] = {
            "P": p, "R": r, "F1": f,
            "TP": tp, "FP": fp, "FN": fn
        }

    return (precision, recall, f1, map50, map5095,
            per_class_stats, conf_matrix,
            global_tp, global_fp, global_fn)

# ============================================================
# REPORT  (fixed: encoding="utf-8", no box-drawing chars)
# ============================================================

def print_and_save_final_metrics(precision, recall, f1,
                                  map50, map5095,
                                  final_per_cls,
                                  final_tp, final_fp, final_fn,
                                  save_dir):

    summary_lines = [
        "",
        "=" * 60,
        "      OVERALL METRICS SUMMARY (best.pth evaluation)",
        "=" * 60,
        f"  Overall Precision       : {precision:.4f}",
        f"  Overall Recall          : {recall:.4f}",
        f"  Overall F1 Score        : {f1:.4f}",
        f"  mAP@50                  : {map50:.4f}",
        f"  mAP@50-95               : {map5095:.4f}",
        "-" * 60,
        f"  Global TP               : {final_tp}",
        f"  Global FP               : {final_fp}",
        f"  Global FN               : {final_fn}",
        "-" * 60,
        "  PER-CLASS METRICS",
        f"  {'Class':<10} {'P':>6}  {'R':>6}  {'F1':>6}  "
        f"{'TP':>5}  {'FP':>5}  {'FN':>5}",
        "  " + "-" * 52,
    ]

    for cls_name, st in final_per_cls.items():
        summary_lines.append(
            f"  {cls_name:<10} {st['P']:>6.3f}  {st['R']:>6.3f}  "
            f"{st['F1']:>6.3f}  {int(st['TP']):>5}  "
            f"{int(st['FP']):>5}  {int(st['FN']):>5}"
        )

    summary_lines += ["=" * 60, ""]
    report = "\n".join(summary_lines)
    print(report)

    # ✅ encoding="utf-8" — fixes the cp1252 crash
    report_path = save_dir / "final_metrics_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")

    # ── Summary bar chart ────────────────────────────────────────
    cls_names  = list(final_per_cls.keys())
    p_vals  = [final_per_cls[c]["P"]   for c in cls_names]
    r_vals  = [final_per_cls[c]["R"]   for c in cls_names]
    f1_vals = [final_per_cls[c]["F1"]  for c in cls_names]

    x, w = np.arange(len(cls_names)), 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w, p_vals,  w, label="Precision", color="#E63946", alpha=0.85)
    ax.bar(x,     r_vals,  w, label="Recall",    color="#457B9D", alpha=0.85)
    ax.bar(x + w, f1_vals, w, label="F1",        color="#2A9D8F", alpha=0.85)

    # overall line markers
    ax.axhline(precision, color="#E63946", linestyle=":",
               lw=1.5, label=f"Overall P={precision:.3f}")
    ax.axhline(recall,    color="#457B9D", linestyle=":",
               lw=1.5, label=f"Overall R={recall:.3f}")
    ax.axhline(f1,        color="#2A9D8F", linestyle=":",
               lw=1.5, label=f"Overall F1={f1:.3f}")

    ax.set_xticks(x); ax.set_xticklabels(cls_names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Overall Metrics Summary — Faster R-CNN (best.pth)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "overall_metrics_summary.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir / 'overall_metrics_summary.png'}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n============================================")
    print(" Faster R-CNN — Report Generator (no retrain)")
    print("============================================")
    print(f"Device  : {DEVICE}")
    print(f"Weights : {WEIGHTS_DIR / 'fasterrcnn_best.pth'}\n")

    # ── Load model ──────────────────────────────────────────────
    model = get_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(
        torch.load(WEIGHTS_DIR / "fasterrcnn_best.pth",
                   map_location=DEVICE)
    )
    print("  Model loaded successfully.\n")

    # ── Val loader ──────────────────────────────────────────────
    val_dataset = RDDDataset(DATASET_ROOT, split="val")
    val_loader  = DataLoader(val_dataset, batch_size=1,
                             shuffle=False, num_workers=0,
                             collate_fn=collate_fn)
    print(f"  Val images: {len(val_dataset)}")
    print("  Running evaluation (this takes ~2-3 min)...\n")

    # ── Evaluate ────────────────────────────────────────────────
    (precision, recall, f1, map50, map5095,
     per_cls, conf_mat,
     gtp, gfp, gfn) = evaluate(model, val_loader)

    # ── Save report + chart ─────────────────────────────────────
    print_and_save_final_metrics(
        precision    = precision,
        recall       = recall,
        f1           = f1,
        map50        = map50,
        map5095      = map5095,
        final_per_cls = per_cls,
        final_tp     = gtp,
        final_fp     = gfp,
        final_fn     = gfn,
        save_dir     = PLOTS_DIR,
    )

    print("\n============================================")
    print("Done! Files saved to:", PLOTS_DIR)
    print("============================================")


if __name__ == "__main__":
    main()