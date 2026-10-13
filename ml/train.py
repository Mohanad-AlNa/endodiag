from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EndoDiag segmentation model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO dataset YAML.")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Base model checkpoint.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs/endodiag"))
    parser.add_argument("--name", type=str, default="seg-exp")
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        seed=42,
        workers=4,
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.25,
        degrees=3,
        translate=0.03,
        scale=0.1,
        shear=1.0,
        perspective=0.0,
        mosaic=0.3,
        mixup=0.1,
        fliplr=0.5,
    )


if __name__ == "__main__":
    main()
