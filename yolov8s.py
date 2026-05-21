# baseline_yolov8s_aug.py

from ultralytics import YOLO

def main():

    # Load pretrained YOLOv8s model
    model = YOLO("yolov8s.pt")

    # Train model
    model.train(
        data="rdd2022_china_yolo/dataset.yaml",

        # Training settings
        epochs=50,
        imgsz=640,
        batch=12,
        workers=0,
        device=0,

        # Optimizer
        optimizer="SGD",

        # Standard augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,

        fliplr=0.5,
        flipud=0.0,

        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,

        # Training behavior
        pretrained=True,
        patience=20,
        verbose=True,

        # Output
        project="runs_rdd2022",
        name="baseline_yolov8s_aug"
    )

    # Validation
    metrics = model.val()

    # Metrics
    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-16)

    # Print results
    print("\n========== FINAL METRICS ==========")

    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"mAP50       : {map50:.4f}")
    print(f"mAP50-95    : {map5095:.4f}")
    print(f"F1 Score    : {f1:.4f}")

    print("===================================\n")


if __name__ == "__main__":
    main()