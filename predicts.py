# =============================================================================
# COMPLETE STABLE PREDICTION + OUTPUT ORGANIZER
# =============================================================================
#
# FEATURES
# --------
# ✓ Predicts ALL test images
# ✓ YOLOv8s predictions
# ✓ YOLOv8m predictions
# ✓ RT-DETR predictions
# ✓ RT-DETR CUDA OOM fix
# ✓ Auto-detects correct training folders
# ✓ Organizes plots and CSVs
# ✓ Saves labels/confidence
# ✓ Creates report-ready structure
#
# =============================================================================

import shutil
import warnings
import yaml
import torch

from pathlib import Path
from ultralytics import YOLO

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

DATA_YAML = "rdd2022_china_yolo/dataset.yaml"

FINAL_OUTPUT_DIR = Path("FINAL_PROJECT_OUTPUTS")

FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 0

CONF = 0.25


# =============================================================================
# MODEL CONFIG
# =============================================================================

MODELS = {

    "YOLOv8s": {

        "weights":
            "runs/detect/runs_rdd2022/baseline_yolov8s_aug-3/weights/best.pt",

        "possible_run_dirs": [

            "runs/detect/runs_rdd2022/baseline_yolov8s_aug-3",

            "runs_rdd2022/baseline_yolov8s_aug-3"
        ]
    },

    "YOLOv8m": {

        "weights":
            "runs/detect/runs_rdd2022/custom_yolov8m_50epochs/weights/best.pt",

        "possible_run_dirs": [

            "runs/detect/runs_rdd2022/custom_yolov8m_50epochs",

            "runs_rdd2022/custom_yolov8m_50epochs"
        ]
    },

    "RT-DETR": {

        "weights":
            "runs/detect/runs_rdd2022/rtdetr_r50_50epochs/weights/best.pt",

        "possible_run_dirs": [

            "runs/detect/runs_rdd2022/rtdetr_r50_50epochs",

            "runs_rdd2022/rtdetr_r50_50epochs"
        ]
    }
}


# =============================================================================
# FIND VALID RUN DIRECTORY
# =============================================================================

def find_valid_run_dir(possible_dirs):

    for directory in possible_dirs:

        path = Path(directory)

        if path.exists():

            print(f"✓ Using run directory: {path}")

            return path

    return None


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

    print(f"\n✓ Total test images found: {len(images)}")

    return images


# =============================================================================
# VERIFY PATHS
# =============================================================================

def verify_paths():

    print("\n" + "=" * 70)
    print("VERIFYING MODEL PATHS")
    print("=" * 70)

    for model_name, info in MODELS.items():

        weights_path = Path(info["weights"])

        if not weights_path.exists():

            raise FileNotFoundError(
                f"\nWeights not found:\n{weights_path.absolute()}"
            )

        run_dir = find_valid_run_dir(
            info["possible_run_dirs"]
        )

        if run_dir is None:

            print(f"\nWARNING: No valid run directory found for {model_name}")

        info["run_dir"] = run_dir

        print(f"\n✓ {model_name}")
        print(f"Weights : {weights_path}")


# =============================================================================
# RUN PREDICTIONS
# =============================================================================

def run_predictions(model_name, weights_path, images):

    print("\n" + "=" * 70)
    print(f"RUNNING PREDICTIONS: {model_name}")
    print("=" * 70)

    model = YOLO(weights_path)

    prediction_dir = FINAL_OUTPUT_DIR / model_name / "predictions"

    prediction_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # RT-DETR MEMORY SAFE
    # -------------------------------------------------------------------------

    if model_name == "RT-DETR":

        for idx, img_path in enumerate(images):

            print(
                f"[{idx+1}/{len(images)}] "
                f"{img_path.name}"
            )

            try:

                model.predict(

                    source=str(img_path),

                    conf=CONF,

                    device=DEVICE,

                    imgsz=640,

                    save=True,

                    save_txt=True,

                    save_conf=True,

                    project=str(prediction_dir),

                    name="outputs",

                    exist_ok=True,

                    verbose=False
                )

                torch.cuda.empty_cache()

            except Exception as e:

                print(f"\nERROR processing {img_path.name}")
                print(e)

    # -------------------------------------------------------------------------
    # YOLO MODELS
    # -------------------------------------------------------------------------

    else:

        try:

            model.predict(

                source=[str(img) for img in images],

                conf=CONF,

                device=DEVICE,

                imgsz=640,

                batch=4,

                save=True,

                save_txt=True,

                save_conf=True,

                project=str(prediction_dir),

                name="outputs",

                exist_ok=True,

                verbose=True
            )

        except Exception as e:

            print("\nERROR during prediction:")
            print(e)

    print(f"\n✓ Predictions completed:")
    print(prediction_dir)


# =============================================================================
# SAFE COPY
# =============================================================================

def safe_copy(src, dst):

    src = Path(src)
    dst = Path(dst)

    try:

        if src.exists():

            shutil.copy(src, dst)

            print(f"✓ Copied: {src.name}")

        else:

            print(f"WARNING: File not found -> {src}")

    except Exception as e:

        print(f"ERROR copying {src.name}")
        print(e)


# =============================================================================
# ORGANIZE OUTPUTS
# =============================================================================

def organize_outputs(model_name, run_dir):

    print("\n" + "=" * 70)
    print(f"ORGANIZING OUTPUTS: {model_name}")
    print("=" * 70)

    if run_dir is None:

        print("WARNING: No valid run directory available.")
        return

    output_dir = FINAL_OUTPUT_DIR / model_name

    plots_dir = output_dir / "plots"
    csv_dir = output_dir / "csv"

    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------------------

    plot_files = [

        "results.png",

        "confusion_matrix.png",
        "confusion_matrix_normalized.png",

        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",

        "labels.jpg"
    ]

    for file_name in plot_files:

        safe_copy(

            run_dir / file_name,

            plots_dir / file_name
        )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    safe_copy(

        run_dir / "results.csv",

        csv_dir / "results.csv"
    )

    print(f"\n✓ Outputs organized:")
    print(output_dir)


# =============================================================================
# CREATE PROJECT STRUCTURE
# =============================================================================

def create_project_structure():

    print("\n" + "=" * 70)
    print("CREATING PROJECT STRUCTURE")
    print("=" * 70)

    folders = [

        "comparison_results",

        "report_figures",

        "report_tables",

        "qualitative_analysis",

        "quantitative_analysis"
    ]

    for folder in folders:

        path = FINAL_OUTPUT_DIR / folder

        path.mkdir(exist_ok=True)

    print("\n✓ Folder structure created.")


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("COMPLETE STABLE PREDICTION + OUTPUT ORGANIZER")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # VERIFY PATHS
    # -------------------------------------------------------------------------

    verify_paths()

    # -------------------------------------------------------------------------
    # CREATE PROJECT STRUCTURE
    # -------------------------------------------------------------------------

    create_project_structure()

    # -------------------------------------------------------------------------
    # LOAD TEST IMAGES
    # -------------------------------------------------------------------------

    images = load_test_images()

    # -------------------------------------------------------------------------
    # PROCESS MODELS
    # -------------------------------------------------------------------------

    for model_name, info in MODELS.items():

        print("\n" + "=" * 70)
        print(f"PROCESSING {model_name}")
        print("=" * 70)

        run_predictions(

            model_name,
            info["weights"],
            images
        )

        organize_outputs(

            model_name,
            info["run_dir"]
        )

    # -------------------------------------------------------------------------
    # COMPLETE
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETED")
    print("=" * 70)

    print(f"\n✓ Outputs saved in:")
    print(FINAL_OUTPUT_DIR.absolute())

    print("\nIMPORTANT:")
    print("Copy Faster R-CNN outputs manually into:")
    print("FINAL_PROJECT_OUTPUTS/Faster_RCNN/")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()