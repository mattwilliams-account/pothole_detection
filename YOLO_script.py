"""
YOLOv8 Training Script
Trains a YOLOv8 model on a custom dataset and saves metrics to JSON.

Expected dataset structure:
    datasets/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── data.yaml
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model":        "yolov8n.pt",   # Pretrained weights: n / s / m / l / x
    "data_yaml":    "datasets/data.yaml",
    "epochs":       25,
    "imgsz":        640,
    "batch":        16,
    "device":       "",             # "" = auto-detect (GPU if available)
    "workers":      8,
    "patience":     50,             # Early-stopping patience (epochs)
    "output_folder": "runs/train",
    "project_name": "yolov8_training",
}

# ──────────────────────────────────────────────────────────────────────────────
# Device Resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_device(device_arg: str) -> str:
    if device_arg != "":
        return device_arg
    
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        device = "".join(str(i) for i in range(n))
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        print(f" GPU detected: {', '.join(names)} -> device='{device}'")
        return device
    
    print(" No GPU detected, using CPU.")
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a YOLOv8 model and save metrics.")
    p.add_argument("--model",         default=DEFAULT_CONFIG["model"],
                   help="YOLOv8 model variant or path to custom weights.")
    p.add_argument("--data",          default=DEFAULT_CONFIG["data_yaml"],
                   help="Path to data.yaml.")
    p.add_argument("--epochs",        type=int, default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--imgsz",         type=int, default=DEFAULT_CONFIG["imgsz"])
    p.add_argument("--batch",         type=int, default=DEFAULT_CONFIG["batch"])
    p.add_argument("--device",        default=DEFAULT_CONFIG["device"],
                   help="Device: 0, 0,1, cpu, or '' for auto.")
    p.add_argument("--workers",       type=int, default=DEFAULT_CONFIG["workers"])
    p.add_argument("--patience",      type=int, default=DEFAULT_CONFIG["patience"])
    p.add_argument("--output-folder", default=DEFAULT_CONFIG["output_folder"],
                   help="Root folder where runs and metrics are saved.")
    p.add_argument("--name",          default=DEFAULT_CONFIG["project_name"],
                   help="Run name (used in output paths).")
    return p.parse_args()


def extract_metrics(results) -> dict:
    """
    Pull scalar metrics out of a Ultralytics Results object.
    Returns a flat dict safe for JSON serialisation.
    """
    metrics: dict = {}

    # results.results_dict contains the final val metrics
    if hasattr(results, "results_dict") and results.results_dict:
        for k, v in results.results_dict.items():
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                metrics[k] = str(v)

    # Box metrics (mAP, precision, recall, …)
    if hasattr(results, "box"):
        box = results.box
        for attr in ("map", "map50", "map75", "mp", "mr"):
            if hasattr(box, attr):
                metrics[f"box/{attr}"] = float(getattr(box, attr))

    # Speed dict (preprocess / inference / postprocess ms)
    if hasattr(results, "speed") and results.speed:
        for k, v in results.speed.items():
            metrics[f"speed/{k}"] = float(v)

    return metrics


def save_metrics(metrics: dict, output_folder: str, run_name: str) -> str:
    """Save metrics dict to a timestamped JSON file and return the path."""
    os.makedirs(output_folder, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name   = run_name.replace(" ", "_")
    output_path = os.path.join(output_folder, f"{safe_name}_metrics_{timestamp}.json")

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── Validate paths ──────────────────────────────────────────────────────
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml.resolve()}")

    # ── Resolve Device ────────────────────────────────────────
    device = resolve_device(args.device)

    print(f"\n{'='*60}")
    print("  YOLOv8 Training Script")
    print(f"{'='*60}")
    print(f"  Model       : {args.model}")
    print(f"  Data        : {data_yaml.resolve()}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Image size  : {args.imgsz}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Device      : {device}")
    print(f"  Output      : {args.output_folder}")
    print(f"{'='*60}\n")

    # ── Load model ──────────────────────────────────────────────────────────
    model = YOLO(args.model)

    # ── Train ───────────────────────────────────────────────────────────────
    print("[1/3] Starting training …\n")
    train_results = model.train(
        data      = str(data_yaml),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = args.device if args.device else None,
        workers   = args.workers,
        patience  = args.patience,
        project   = args.output_folder,
        name      = args.name,
        exist_ok  = True,
    )

    # ── Validate on the val split ────────────────────────────────────────────
    print("\n[2/3] Running validation …\n")
    val_results = model.val(
        data    = str(data_yaml),
        split   = "val",
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device if args.device else None,
    )

    # ── Evaluate on the test split ───────────────────────────────────────────
    print("\n[2b/3] Running test evaluation …\n")
    test_results = model.val(
        data    = str(data_yaml),
        split   = "test",
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device if args.device else None,
    )

    # ── Collect & save metrics ───────────────────────────────────────────────
    print("\n[3/3] Saving metrics …\n")

    metrics = {
        "run_name":   args.name,
        "timestamp":  datetime.now().isoformat(),
        "config": {
            "model":   args.model,
            "data":    str(data_yaml),
            "epochs":  args.epochs,
            "imgsz":   args.imgsz,
            "batch":   args.batch,
            "device":  args.device,
            "patience": args.patience,
        },
        "val_metrics":  extract_metrics(val_results),
        "test_metrics": extract_metrics(test_results),
    }

    # Best weights path
    best_weights = Path(args.output_folder) / args.name / "weights" / "best.pt"
    metrics["best_weights"] = str(best_weights.resolve()) if best_weights.exists() else "not found"

    output_path = save_metrics(metrics, args.output_folder, args.name)

    print(f"  ✓ Metrics saved → {output_path}")
    print(f"  ✓ Best weights  → {metrics['best_weights']}")

    # ── Quick summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Training Complete – Summary")
    print(f"{'='*60}")
    for section, label in [("val_metrics", "Validation"), ("test_metrics", "Test")]:
        data = metrics.get(section, {})
        mAP50   = data.get("metrics/mAP50(B)",    data.get("box/map50",  "N/A"))
        mAP5095 = data.get("metrics/mAP50-95(B)", data.get("box/map",    "N/A"))
        prec    = data.get("metrics/precision(B)", data.get("box/mp",     "N/A"))
        rec     = data.get("metrics/recall(B)",    data.get("box/mr",     "N/A"))
        print(f"\n  {label}:")
        print(f"    mAP@0.5       : {mAP50}")
        print(f"    mAP@0.5:0.95  : {mAP5095}")
        print(f"    Precision     : {prec}")
        print(f"    Recall        : {rec}")
    print(f"\n{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)