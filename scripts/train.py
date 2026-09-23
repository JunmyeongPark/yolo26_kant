#!/usr/bin/env python3
"""
YOLO26(n) pretrained 체크포인트를 기반으로 knob/MASTER custom dataset을 fine-tuning.

사용 예:
  python3 train.py
  python3 train.py --model yolo26s.pt --epochs 150 --imgsz 640 --batch 16

기본값은 프로젝트 루트의 data.yaml, 경량 모델(yolo26n.pt)을 사용합니다.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="knob/MASTER YOLO fine-tuning")
    parser.add_argument("--model", default="yolo26n.pt",
                         help="pretrained 체크포인트 (yolo26n/s/m/l/x.pt)")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None,
                         help="예: 0 (GPU 0번), cpu. 미지정 시 자동 선택")
    parser.add_argument("--name", default="puck_knob_v1",
                         help="runs/detect/<name> 으로 결과 저장")
    args = parser.parse_args()

    model = YOLO(args.model)  # COCO pretrained 가중치 로드 (transfer learning 시작점)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=str(PROJECT_ROOT / "runs" / "detect"),
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
