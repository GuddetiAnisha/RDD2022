import os
import torch
import torchvision
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from PIL import Image
from pathlib import Path
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT  = "rdd2022_china_yolo"
NUM_CLASSES   = 5        # 4 damage classes + background
BATCH_SIZE    = 4
ACCUMULATION_STEPS = 3   # effective batch = 12
EPOCHS        = 50
IMG_SIZE      = 640
IOU_THRESH    = 0.5      # for precision / recall / F1
SCORE_THRESH  = 0.5      # confidence threshold for predictions
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES   = ["background", "D00", "D10", "D20", "D40"]

# Output directories
WEIGHTS_DIR = Path("weights")
PLOTS_DIR   = Path("plots")
VAL_DIR     = PLOTS_DIR / "val"

for d in [WEIGHTS_DIR, PLOTS_DIR, VAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATASET
# ============================================================

class RDDDataset(Dataset):

    def __init__(self, root, split="train"):
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
                xmin = (cx - bw / 2) * w
                ymin = (cy - bh / 2) * h
                xmax = (cx + bw / 2) * w
                ymax = (cy + bh / 2) * h
                if xmax <= xmin or ymax <= ymin:
                    continue
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(cls_id) + 1)   # background = 0

        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
        }

        return F.to_tensor(img), target

# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):
    return tuple(zip(*batch))

# ============================================================
# MODEL
# ============================================================

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    loss_cls_total = loss_box_total = loss_obj_total = loss_rpn_total = 0.0
    optimizer.zero_grad()
    step = 0

    for step, (images, targets) in enumerate(loader):
        images  = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values()) / ACCUMULATION_STEPS
        losses.backward()

        if (step + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss     += losses.item()
        loss_cls_total += loss_dict.get("loss_classifier",  torch.tensor(0.0)).item()
        loss_box_total += loss_dict.get("loss_box_reg",     torch.tensor(0.0)).item()
        loss_obj_total += loss_dict.get("loss_objectness",  torch.tensor(0.0)).item()
        loss_rpn_total += loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()

        if step % 10 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}] Step [{step}/{len(loader)}] "
                  f"Loss: {losses.item():.4f}")

    # flush remaining accumulated gradients
    if (step + 1) % ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    n = max(len(loader), 1)
    return (total_loss / n,
            loss_cls_total / n,
            loss_box_total / n,
            loss_obj_total / n,
            loss_rpn_total / n)

# ============================================================
# EVALUATION
# ============================================================

def box_iou_single(boxA, boxB):
    """IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-16)


def evaluate(model, loader):
    model.eval()

    metric = MeanAveragePrecision(
        backend="faster_coco_eval",
        iou_thresholds=[IOU_THRESH],
    )

    all_preds_by_cls = defaultdict(list)   # cls -> [(score, tp_flag)]
    gt_count_by_cls  = defaultdict(int)
    conf_matrix      = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    # Global TP / FP / FN counters (across all classes, for overall metrics)
    global_tp = 0
    global_fp = 0
    global_fn = 0

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

                # per-class matching
                keep = scores_pred >= SCORE_THRESH
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

                    for j, (gt_box, gt_cls) in enumerate(zip(boxes_gt, labels_gt)):
                        if j in matched_gt:
                            continue
                        iou = box_iou_single(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_j   = j

                    if best_j >= 0:
                        matched_gt.add(best_j)
                        gt_cls = int(labels_gt[best_j])
                        conf_matrix[gt_cls][int(pred_cls)] += 1
                        tp = 1 if pred_cls == gt_cls else 0
                    else:
                        tp = 0
                        conf_matrix[0][int(pred_cls)] += 1

                    all_preds_by_cls[int(pred_cls)].append((scores_pred_k[i], tp))

                    # Global counts
                    if tp:
                        global_tp += 1
                    else:
                        global_fp += 1

                # Unmatched GTs = false negatives
                global_fn += len(boxes_gt) - len(matched_gt)

            metric.update(preds_batch, gts_batch)

    results   = metric.compute()
    map50     = results["map_50"].item()
    map5095   = results["map"].item()

    # Proper overall precision / recall / F1 from raw TP/FP/FN
    precision = global_tp / (global_tp + global_fp + 1e-16)
    recall    = global_tp / (global_tp + global_fn + 1e-16)
    f1        = 2 * precision * recall / (precision + recall + 1e-16)

    # per-class P / R / F1
    per_class_stats = {}
    for cls in range(1, NUM_CLASSES):
        preds_cls = all_preds_by_cls[cls]
        tp = sum(t for _, t in preds_cls)
        fp = len(preds_cls) - tp
        fn = gt_count_by_cls[cls] - tp
        p  = tp / (tp + fp + 1e-16)
        r  = tp / (tp + fn + 1e-16)
        f  = 2 * p * r / (p + r + 1e-16)
        per_class_stats[CLASS_NAMES[cls]] = {"P": p, "R": r, "F1": f,
                                              "TP": tp, "FP": fp, "FN": fn}

    return (precision, recall, f1, map50, map5095,
            per_class_stats, conf_matrix,
            global_tp, global_fp, global_fn)

# ============================================================
# PLOTTING HELPERS
# ============================================================

def save_loss_curves(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss",  color="#E63946", lw=2)
    ax.plot(epochs, history["val_map50"],  label="Val mAP@50", color="#457B9D", lw=2, linestyle="--")
    ax.set_title("Training Loss vs Val mAP@50", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Value")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # Component losses
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    components = [
        ("loss_cls", "Classification Loss", "#E63946"),
        ("loss_box", "Box Regression Loss",  "#2A9D8F"),
        ("loss_obj", "Objectness Loss",      "#E9C46A"),
        ("loss_rpn", "RPN Box Reg Loss",     "#264653"),
    ]
    for ax, (key, title, color) in zip(axes.flat, components):
        ax.plot(epochs, history[key], color=color, lw=2)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
    fig.suptitle("Component Losses", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "component_losses.png", dpi=150)
    plt.close(fig)


def save_metric_curves(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(epochs, history["precision"], color="#E63946", lw=2)
    axes[0].set_title("BoxP Curve", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Precision")
    axes[0].set_ylim(0, 1); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["recall"], color="#457B9D", lw=2)
    axes[1].set_title("BoxR Curve", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Recall")
    axes[1].set_ylim(0, 1); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["f1"], color="#2A9D8F", lw=2)
    axes[2].set_title("BoxF1 Curve", fontweight="bold")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("F1 Score")
    axes[2].set_ylim(0, 1); axes[2].grid(alpha=0.3)

    fig.suptitle("Precision / Recall / F1 over Epochs", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "BoxF1_BoxP_BoxR_curves.png", dpi=150)
    plt.close(fig)


def save_map_curve(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["val_map50"],   label="mAP@50",    color="#E63946", lw=2)
    ax.plot(epochs, history["val_map5095"], label="mAP@50-95", color="#457B9D", lw=2, linestyle="--")
    ax.set_title("mAP Curves", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "mAP_curve.png", dpi=150)
    plt.close(fig)


def save_overfitting_analysis(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1   = axes[0]
    c1, c2 = "#E63946", "#457B9D"
    ax1.plot(epochs, history["train_loss"], color=c1, lw=2, label="Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)

    ax1b = ax1.twinx()
    ax1b.plot(epochs, history["val_map50"], color=c2, lw=2,
              linestyle="--", label="Val mAP@50")
    ax1b.set_ylabel("Val mAP@50", color=c2)
    ax1b.tick_params(axis="y", labelcolor=c2)
    ax1b.set_ylim(0, 1)

    best_ep  = int(np.argmax(history["val_map50"])) + 1
    best_map = max(history["val_map50"])
    ax1b.axvline(best_ep, color="#2A9D8F", linestyle=":", lw=1.5)
    ax1b.annotate(f"Best ep {best_ep}\n{best_map:.3f}",
                  xy=(best_ep, best_map),
                  xytext=(min(best_ep + 2, len(history["train_loss"])), best_map - 0.06),
                  fontsize=8, color="#2A9D8F",
                  arrowprops=dict(arrowstyle="->", color="#2A9D8F"))

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=8)
    ax1.set_title("Overfitting Analysis:\nTrain Loss vs Val mAP@50", fontweight="bold")
    ax1.grid(alpha=0.3)

    ax2    = axes[1]
    f1_arr = np.array(history["f1"])
    ax2.plot(epochs, f1_arr, color="#2A9D8F", lw=1.5, alpha=0.5, label="Val F1")

    if len(f1_arr) >= 5:
        window   = max(3, len(f1_arr) // 10)
        smoothed = np.convolve(f1_arr, np.ones(window) / window, mode="valid")
        ax2.plot(range(window, len(f1_arr) + 1), smoothed,
                 color="#264653", lw=2.5, label=f"Smoothed (w={window})")

    ax2.set_title("Generalisation: Val F1 Trend", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("Overfitting Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "overfitting_analysis.png", dpi=150)
    plt.close(fig)
    print(f"  Saved overfitting_analysis.png  (best epoch: {best_ep})")


def save_confusion_matrix(conf_matrix, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Ground Truth")

    row_sums = conf_matrix.sum(axis=1, keepdims=True).astype(float)
    norm_cm  = np.divide(conf_matrix, row_sums,
                         out=np.zeros_like(conf_matrix, dtype=float),
                         where=row_sums != 0)
    sns.heatmap(norm_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalised)", fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Ground Truth")

    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(norm_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax2)
    ax2.set_title("Confusion Matrix Normalised", fontweight="bold")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Ground Truth")
    fig2.tight_layout()
    fig2.savefig(save_dir / "confusion_matrix_normalized.png", dpi=150)
    plt.close(fig2)


def save_per_class_bar(per_class_stats, save_dir):
    classes = list(per_class_stats.keys())
    P  = [per_class_stats[c]["P"]  for c in classes]
    R  = [per_class_stats[c]["R"]  for c in classes]
    F1 = [per_class_stats[c]["F1"] for c in classes]

    x, w = np.arange(len(classes)), 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, P,  w, label="Precision", color="#E63946")
    ax.bar(x,     R,  w, label="Recall",    color="#457B9D")
    ax.bar(x + w, F1, w, label="F1",        color="#2A9D8F")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylim(0, 1)
    ax.set_title("Per-Class Precision / Recall / F1", fontweight="bold")
    ax.set_ylabel("Score"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "per_class_metrics.png", dpi=150)
    plt.close(fig)


def save_val_batch_images(model, dataset, save_dir, num_batches=3):
    model.eval()
    colors = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D"]

    for batch_idx in range(num_batches):
        start = batch_idx * 4
        n_imgs = min(4, len(dataset) - start)
        if n_imgs <= 0:
            break

        fig_lbl,  axes_lbl  = plt.subplots(1, n_imgs, figsize=(5 * n_imgs, 5))
        fig_pred, axes_pred = plt.subplots(1, n_imgs, figsize=(5 * n_imgs, 5))

        if n_imgs == 1:
            axes_lbl  = [axes_lbl]
            axes_pred = [axes_pred]

        for i in range(n_imgs):
            img_tensor, target = dataset[start + i]
            img_np = img_tensor.permute(1, 2, 0).numpy()

            axes_lbl[i].imshow(img_np)
            axes_lbl[i].set_title(f"Img {start+i}", fontsize=9)
            axes_lbl[i].axis("off")
            for box, lbl in zip(target["boxes"].numpy(), target["labels"].numpy()):
                x1, y1, x2, y2 = box
                color = colors[(lbl - 1) % len(colors)]
                axes_lbl[i].add_patch(
                    patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                      linewidth=2, edgecolor=color, facecolor="none"))
                axes_lbl[i].text(x1, y1-2, CLASS_NAMES[lbl], fontsize=7,
                                  color="white",
                                  bbox=dict(facecolor=color, alpha=0.7, pad=1))

            with torch.no_grad():
                out = model([img_tensor.to(DEVICE)])[0]

            axes_pred[i].imshow(img_np)
            axes_pred[i].set_title(f"Img {start+i}", fontsize=9)
            axes_pred[i].axis("off")
            keep = out["scores"].cpu() >= SCORE_THRESH
            for box, lbl, score in zip(
                out["boxes"].cpu()[keep].numpy(),
                out["labels"].cpu()[keep].numpy(),
                out["scores"].cpu()[keep].numpy(),
            ):
                x1, y1, x2, y2 = box
                color = colors[(lbl - 1) % len(colors)]
                axes_pred[i].add_patch(
                    patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                      linewidth=2, edgecolor=color, facecolor="none"))
                axes_pred[i].text(x1, y1-2, f"{CLASS_NAMES[lbl]} {score:.2f}",
                                   fontsize=7, color="white",
                                   bbox=dict(facecolor=color, alpha=0.7, pad=1))

        for fig, suffix in [(fig_lbl, "labels"), (fig_pred, "pred")]:
            fig.suptitle(f"Batch {batch_idx} — {suffix}", fontweight="bold")
            fig.tight_layout()
            fig.savefig(save_dir / f"val_batch{batch_idx}_{suffix}.jpg", dpi=120)
            plt.close(fig)

    print(f"  Saved val batch images -> {save_dir}")


# ============================================================
# FINAL OVERALL METRICS SUMMARY  (called once after all epochs)
# ============================================================

def print_and_save_final_metrics(history, best_map50, best_epoch,
                                  final_per_cls, final_conf_mat,
                                  final_tp, final_fp, final_fn,
                                  save_dir):
    """
    Computes and prints a comprehensive summary of overall metrics
    accumulated over the entire training run, then saves plots and
    a text report.
    """

    # ── Overall metrics at final epoch ──────────────────────────────────
    precision = final_tp / (final_tp + final_fp + 1e-16)
    recall    = final_tp / (final_tp + final_fn + 1e-16)
    f1        = 2 * precision * recall / (precision + recall + 1e-16)

    best_precision  = max(history["precision"])
    best_recall     = max(history["recall"])
    best_f1         = max(history["f1"])
    best_map5095    = max(history["val_map5095"])
    mean_precision  = float(np.mean(history["precision"]))
    mean_recall     = float(np.mean(history["recall"]))
    mean_f1         = float(np.mean(history["f1"]))
    mean_map50      = float(np.mean(history["val_map50"]))

    summary_lines = [
        "",
        "╔══════════════════════════════════════════════════════════╗",
        "║          OVERALL METRICS SUMMARY (All Epochs)            ║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║  Total Epochs Trained      : {EPOCHS:<28}║",
        f"║  Best Epoch (mAP@50)       : {best_epoch:<28}║",
        "╠══════════════════════════════════════════════════════════╣",
        "║  FINAL EPOCH METRICS  (from raw TP/FP/FN counts)        ║",
        f"║    Global TP               : {final_tp:<28}║",
        f"║    Global FP               : {final_fp:<28}║",
        f"║    Global FN               : {final_fn:<28}║",
        f"║    Overall Precision       : {precision:<28.4f}║",
        f"║    Overall Recall          : {recall:<28.4f}║",
        f"║    Overall F1 Score        : {f1:<28.4f}║",
        f"║    mAP@50 (final epoch)    : {history['val_map50'][-1]:<28.4f}║",
        f"║    mAP@50-95 (final epoch) : {history['val_map5095'][-1]:<28.4f}║",
        "╠══════════════════════════════════════════════════════════╣",
        "║  BEST VALUES ACROSS ALL EPOCHS                          ║",
        f"║    Best mAP@50             : {best_map50:<28.4f}║",
        f"║    Best mAP@50-95          : {best_map5095:<28.4f}║",
        f"║    Best Precision          : {best_precision:<28.4f}║",
        f"║    Best Recall             : {best_recall:<28.4f}║",
        f"║    Best F1 Score           : {best_f1:<28.4f}║",
        "╠══════════════════════════════════════════════════════════╣",
        "║  MEAN VALUES ACROSS ALL EPOCHS                          ║",
        f"║    Mean mAP@50             : {mean_map50:<28.4f}║",
        f"║    Mean Precision          : {mean_precision:<28.4f}║",
        f"║    Mean Recall             : {mean_recall:<28.4f}║",
        f"║    Mean F1 Score           : {mean_f1:<28.4f}║",
        "╠══════════════════════════════════════════════════════════╣",
        "║  PER-CLASS METRICS (Final Epoch)                        ║",
        f"║  {'Class':<10} {'P':>6}  {'R':>6}  {'F1':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5} ║",
        "║  " + "─" * 54 + " ║",
    ]

    for cls_name, st in final_per_cls.items():
        line = (f"║  {cls_name:<10} {st['P']:>6.3f}  {st['R']:>6.3f}  "
                f"{st['F1']:>6.3f}  {int(st['TP']):>5}  "
                f"{int(st['FP']):>5}  {int(st['FN']):>5} ║")
        summary_lines.append(line)

    summary_lines += [
        "╚══════════════════════════════════════════════════════════╝",
        "",
    ]

    report = "\n".join(summary_lines)
    print(report)

    # Save text report
    report_path = save_dir / "final_metrics_report.txt"
    report_path.write_text(report)
    print(f"  Saved final metrics report -> {report_path}")

    # ── Summary bar chart ────────────────────────────────────────────────
    metrics_labels = ["Precision", "Recall", "F1", "mAP@50", "mAP@50-95"]
    final_values   = [precision, recall, f1,
                      history["val_map50"][-1], history["val_map5095"][-1]]
    best_values    = [best_precision, best_recall, best_f1,
                      best_map50, best_map5095]

    x = np.arange(len(metrics_labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - w/2, final_values, w, label="Final Epoch", color="#E63946", alpha=0.85)
    bars2 = ax.bar(x + w/2, best_values,  w, label="Best Epoch",  color="#457B9D", alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Overall Metrics Summary — Final vs Best Epoch", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "overall_metrics_summary.png", dpi=150)
    plt.close(fig)
    print(f"  Saved overall_metrics_summary.png -> {save_dir}")

    # ── Epoch-by-epoch table saved as text ──────────────────────────────
    table_lines = [
        "\nEpoch-by-Epoch Validation Metrics:",
        f"{'Epoch':>6}  {'Loss':>8}  {'Prec':>6}  {'Recall':>6}  "
        f"{'F1':>6}  {'mAP50':>6}  {'mAP50-95':>8}",
        "─" * 60,
    ]
    for ep in range(len(history["train_loss"])):
        table_lines.append(
            f"{ep+1:>6}  {history['train_loss'][ep]:>8.4f}  "
            f"{history['precision'][ep]:>6.4f}  {history['recall'][ep]:>6.4f}  "
            f"{history['f1'][ep]:>6.4f}  {history['val_map50'][ep]:>6.4f}  "
            f"{history['val_map5095'][ep]:>8.4f}"
        )
    table_path = save_dir / "epoch_metrics_table.txt"
    table_path.write_text("\n".join(table_lines))
    print(f"  Saved epoch-by-epoch table -> {table_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n====================================")
    print(" Faster R-CNN Training  |  RDD2022")
    print("====================================")
    print(f"Device : {DEVICE}")

    train_dataset = RDDDataset(DATASET_ROOT, split="train")
    val_dataset   = RDDDataset(DATASET_ROOT, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=1,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)

    print(f"Train : {len(train_dataset)} images")
    print(f"Val   : {len(val_dataset)}   images")

    model     = get_model(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    history    = defaultdict(list)
    best_map50 = 0.0
    best_epoch = 1

    # Keep final-epoch evaluation outputs for the summary
    final_per_cls  = None
    final_conf_mat = None
    final_tp = final_fp = final_fn = 0

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):

        # ── Train ────────────────────────────────────────────────────────
        loss, loss_cls, loss_box, loss_obj, loss_rpn = train_one_epoch(
            model, train_loader, optimizer, epoch)
        scheduler.step()

        # ── Evaluate ─────────────────────────────────────────────────────
        (precision, recall, f1, map50, map5095,
         per_cls, conf_mat,
         gtp, gfp, gfn) = evaluate(model, val_loader)

        # ── History ──────────────────────────────────────────────────────
        history["train_loss"].append(loss)
        history["loss_cls"].append(loss_cls)
        history["loss_box"].append(loss_box)
        history["loss_obj"].append(loss_obj)
        history["loss_rpn"].append(loss_rpn)
        history["precision"].append(precision)
        history["recall"].append(recall)
        history["f1"].append(f1)
        history["val_map50"].append(map50)
        history["val_map5095"].append(map5095)

        # Always update final-epoch outputs
        final_per_cls  = per_cls
        final_conf_mat = conf_mat
        final_tp, final_fp, final_fn = gtp, gfp, gfn

        print("\n====================================")
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"  Train Loss  : {loss:.4f}")
        print(f"  Precision   : {precision:.4f}")
        print(f"  Recall      : {recall:.4f}")
        print(f"  F1          : {f1:.4f}")
        print(f"  mAP@50      : {map50:.4f}")
        print(f"  mAP@50-95   : {map5095:.4f}")
        print("  Per-class:")
        for cls_name, st in per_cls.items():
            print(f"    {cls_name}: P={st['P']:.3f}  R={st['R']:.3f}  "
                  f"F1={st['F1']:.3f}  TP={int(st['TP'])}  "
                  f"FP={int(st['FP'])}  FN={int(st['FN'])}")
        print("====================================\n")

        # ── Checkpoint ───────────────────────────────────────────────────
        torch.save(model.state_dict(),
                   WEIGHTS_DIR / f"fasterrcnn_epoch_{epoch+1}.pth")

        if map50 > best_map50:
            best_map50 = map50
            best_epoch = epoch + 1
            torch.save(model.state_dict(), WEIGHTS_DIR / "fasterrcnn_best.pth")
            print(f"  *** New best mAP@50: {best_map50:.4f} — saved best model ***")

        # ── Plots every 5 epochs + final ─────────────────────────────────
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print("  Saving plots...")
            save_loss_curves(history, PLOTS_DIR)
            save_metric_curves(history, PLOTS_DIR)
            save_map_curve(history, PLOTS_DIR)
            save_overfitting_analysis(history, PLOTS_DIR)
            save_confusion_matrix(conf_mat, PLOTS_DIR)
            save_per_class_bar(per_cls, PLOTS_DIR)
            save_val_batch_images(model, val_dataset, VAL_DIR)
            print("  Done.\n")

    # ── Final model save ─────────────────────────────────────────────────
    torch.save(model.state_dict(), WEIGHTS_DIR / "fasterrcnn_final.pth")

    # ── OVERALL METRICS SUMMARY after all epochs ─────────────────────────
    print_and_save_final_metrics(
        history      = history,
        best_map50   = best_map50,
        best_epoch   = best_epoch,
        final_per_cls  = final_per_cls,
        final_conf_mat = final_conf_mat,
        final_tp     = final_tp,
        final_fp     = final_fp, 
        final_fn     = final_fn,
        save_dir     = PLOTS_DIR,
    )

    print("\n====================================")
    print("Training Complete!")
    print(f"Best mAP@50 : {best_map50:.4f}  (epoch {best_epoch})")
    print(f"Weights     : {WEIGHTS_DIR}/")
    print(f"Plots       : {PLOTS_DIR}/")
    print("====================================")


if __name__ == "__main__":
    main()