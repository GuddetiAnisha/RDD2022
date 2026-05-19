# =============================================================================
# FASTER R-CNN PREDICTION SCRIPT
# =============================================================================

import cv2
import torch
import yaml
import warnings
import torchvision
import numpy as np

from PIL import Image
from pathlib import Path

from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 640

CONF_THRESHOLD = 0.5

NUM_CLASSES = 5

CLASS_NAMES = {

    1: "D00",
    2: "D10",
    3: "D20",
    4: "D40"
}

# -------------------------------------------------------------------------
# IMPORTANT
# -------------------------------------------------------------------------

MODEL_PATH = "weights/fasterrcnn_best.pth"

DATA_YAML = "rdd2022_china_yolo/dataset.yaml"

OUTPUT_DIR = Path(
    "FINAL_PROJECT_OUTPUTS/Faster_RCNN/predictions"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOAD TEST IMAGES
# =============================================================================

def load_test_images():

    print("\n" + "=" * 70)
    print("LOADING TEST IMAGES")
    print("=" * 70)

    with open(DATA_YAML, "r") as f:

        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg.get("path", "."))

    test_dir = dataset_root / "images" / "test"

    images = list(test_dir.glob("*.jpg"))
    images += list(test_dir.glob("*.png"))
    images += list(test_dir.glob("*.jpeg"))

    if len(images) == 0:

        raise FileNotFoundError(
            f"\nNo test images found in:\n{test_dir}"
        )

    print(f"\n✓ Total test images: {len(images)}")

    return images


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model():

    print("\n" + "=" * 70)
    print("LOADING FASTER R-CNN MODEL")
    print("=" * 70)

    if not Path(MODEL_PATH).exists():

        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}"
        )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(

        in_features,
        NUM_CLASSES
    )

    checkpoint = torch.load(

        MODEL_PATH,

        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    print(f"\n✓ Model loaded:")
    print(MODEL_PATH)

    return model


# =============================================================================
# DRAW PREDICTIONS
# =============================================================================

def draw_predictions(image, boxes, labels, scores):

    for box, label, score in zip(boxes, labels, scores):

        if score < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)

        class_name = CLASS_NAMES.get(

            int(label),
            str(label)
        )

        text = f"{class_name}: {score:.2f}"

        cv2.rectangle(

            image,

            (x1, y1),
            (x2, y2),

            (0, 255, 0),

            2
        )

        cv2.putText(

            image,

            text,

            (x1, y1 - 10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 255, 0),

            2
        )

    return image


# =============================================================================
# SAVE LABELS
# =============================================================================

def save_labels(txt_path, boxes, labels, scores):

    with open(txt_path, "w") as f:

        for box, label, score in zip(boxes, labels, scores):

            if score < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = box

            class_name = CLASS_NAMES.get(

                int(label),
                str(label)
            )

            f.write(

                f"{class_name} "
                f"{score:.4f} "
                f"{x1:.2f} "
                f"{y1:.2f} "
                f"{x2:.2f} "
                f"{y2:.2f}\n"
            )


# =============================================================================
# RUN PREDICTIONS
# =============================================================================

def run_predictions(model, images):

    print("\n" + "=" * 70)
    print("RUNNING PREDICTIONS")
    print("=" * 70)

    for idx, img_path in enumerate(images):

        print(

            f"[{idx+1}/{len(images)}] "
            f"{img_path.name}"
        )

        # -----------------------------------------------------------------
        # LOAD IMAGE
        # -----------------------------------------------------------------

        image_bgr = cv2.imread(str(img_path))

        image_rgb = cv2.cvtColor(

            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        image_rgb = cv2.resize(

            image_rgb,
            (IMG_SIZE, IMG_SIZE)
        )

        pil_image = Image.fromarray(image_rgb)

        image_tensor = F.to_tensor(pil_image).to(DEVICE)

        # -----------------------------------------------------------------
        # INFERENCE
        # -----------------------------------------------------------------

        with torch.no_grad():

            outputs = model([image_tensor])[0]

        boxes = outputs["boxes"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()

        # -----------------------------------------------------------------
        # DRAW PREDICTIONS
        # -----------------------------------------------------------------

        pred_image = draw_predictions(

            image_rgb.copy(),

            boxes,
            labels,
            scores
        )

        # -----------------------------------------------------------------
        # SAVE IMAGE
        # -----------------------------------------------------------------

        save_image_path = OUTPUT_DIR / img_path.name

        cv2.imwrite(

            str(save_image_path),

            cv2.cvtColor(
                pred_image,
                cv2.COLOR_RGB2BGR
            )
        )

        # -----------------------------------------------------------------
        # SAVE LABELS
        # -----------------------------------------------------------------

        txt_path = OUTPUT_DIR / f"{img_path.stem}.txt"

        save_labels(

            txt_path,

            boxes,
            labels,
            scores
        )

    print("\n✓ Predictions completed.")
    print(OUTPUT_DIR)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("FASTER R-CNN PREDICTION PIPELINE")
    print("=" * 70)

    images = load_test_images()

    model = load_model()

    run_predictions(

        model,
        images
    )

    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETED")
    print("=" * 70)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()