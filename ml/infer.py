from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one periapical image.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/endodiag/infer"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    results = model.predict(
        source=str(args.image),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=str(args.save_dir.parent),
        name=args.save_dir.name,
    )

    r0 = results[0]
    names = r0.names
    payload = []
    for idx in range(len(r0.boxes)):
        cls_id = int(r0.boxes.cls[idx].item())
        payload.append(
            {
                "label": names.get(cls_id, str(cls_id)),
                "confidence": float(r0.boxes.conf[idx].item()),
                "bbox_xyxy": [float(v) for v in r0.boxes.xyxy[idx].tolist()],
            }
        )

    output_file = args.save_dir / "predictions.json"
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved predictions: {output_file}")


if __name__ == "__main__":
    main()
