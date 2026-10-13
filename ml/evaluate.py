from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EndoDiag model.")
    parser.add_argument("--model", type=Path, required=True, help="Trained model path.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--save-json", type=Path, default=Path("runs/endodiag/eval_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    metrics = model.val(data=str(args.data), imgsz=args.imgsz, split=args.split)

    payload = {
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "seg_map50": float(metrics.seg.map50),
        "seg_map50_95": float(metrics.seg.map),
        "speed_ms": metrics.speed,
    }
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
