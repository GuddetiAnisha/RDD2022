# custom_yolov8m_main.py

from ultralytics import YOLO

def main():

    # Load pretrained YOLOv8m model
    model = YOLO("yolov8m.pt")

    # Train model
    model.train(

        # Dataset
        data="rdd2022_china_yolo/dataset.yaml",

        # Fair comparison settings
        epochs=50,
        imgsz=640,
        batch=12,
        workers=0,
        device=0,

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        # Scheduler
        cos_lr=True,
        warmup_epochs=3,

        # Mixed precision
        amp=True,

        # Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,

        fliplr=0.5,
        flipud=0.0,

        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.2,

        # Regularization
        label_smoothing=0.1,
        dropout=0.05,

        # Multi-scale training
        multi_scale=False,

        # Stabilize final epochs
        close_mosaic=10,

        # Early stopping
        patience=20,

        # Validation
        val=True,

        # Save outputs
        save=True,
        save_period=5,

        # Output folder
        project="runs_rdd2022",
        name="custom_yolov8m_50epochs",

        pretrained=True,
        verbose=True
    )

    # Final validation with TTA
    metrics = model.val(
        augment=True
    )

    # Metrics
    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-16)

    # Print metrics
    print("\n========== FINAL METRICS ==========")

    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"mAP50       : {map50:.4f}")
    print(f"mAP50-95    : {map5095:.4f}")
    print(f"F1 Score    : {f1:.4f}")

    print("===================================\n")


if __name__ == "__main__":
    main()