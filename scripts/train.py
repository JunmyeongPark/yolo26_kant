#!/usr/bin/env python3
"""
YOLO26(n) pretrained 체크포인트를 기반으로 knob/MASTER custom dataset을 fine-tuning.

사용 예:
  python3 train.py
  python3 train.py --model yolo26s.pt --epochs 150 --imgsz 640 --batch 16

--model 생략 시 프로젝트의 .pt 목록을 표시하고 학습 시작 가중치를 입력받습니다.
Enter를 누르면 yolo26n.pt를 사용합니다. data.yaml은 프로젝트 루트 기준입니다.
"""

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def list_checkpoints(root):
    checkpoints = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        checkpoints.extend(Path(folder) / name for name in sorted(files)
                           if name.lower().endswith(".pt"))
    return sorted(checkpoints)


def choose_checkpoint(root):
    checkpoints = list_checkpoints(root)
    print("사용 가능한 .pt 파일 (프로젝트 기준 경로):")
    for checkpoint in checkpoints:
        print(f"  {checkpoint.relative_to(root)}")
    if not checkpoints:
        print("  로컬 .pt 파일이 없습니다.")
    print("학습을 시작할 가중치 파일을 선택하세요. 같은 이름이 여러 개면 경로를 입력하세요.")
    print("기본 모델: yolo26n.pt (로컬에 없으면 Ultralytics에서 다운로드)")
    while True:
        try:
            value = input(".pt 파일명 또는 경로 [Enter: yolo26n.pt]: ").strip() or "yolo26n.pt"
        except EOFError:
            raise ValueError("입력을 받을 수 없습니다. --model 파일명.pt 옵션을 지정하세요.") from None
        path = Path(value).expanduser()
        if path.suffix.lower() != ".pt":
            print(".pt 파일명 또는 경로를 입력하세요.")
            continue
        if not path.is_absolute() and len(path.parts) == 1:
            matches = [p for p in checkpoints if p.name == path.name]
            if len(matches) > 1:
                print("같은 이름이 여러 개입니다. 위 목록의 전체 상대 경로를 입력하세요.")
                continue
            if matches:
                return str(matches[0])
        candidate = path if path.is_absolute() else root / path
        if candidate.is_file():
            return str(candidate)
        if value in {f"yolo26{size}.pt" for size in "nsmlx"}:
            return value
        print(f"파일을 찾을 수 없습니다: {candidate}")


def main():
    parser = argparse.ArgumentParser(description="knob/MASTER YOLO fine-tuning")
    parser.add_argument("--model",
                         help="학습 시작 체크포인트. 생략하면 .pt 목록 표시 후 입력")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None,
                         help="예: 0 (GPU 0번), cpu. 미지정 시 자동 선택")
    parser.add_argument("--name", default="puck_knob_v1",
                         help="runs/detect/<name> 으로 결과 저장")
    args = parser.parse_args()

    try:
        checkpoint = args.model or choose_checkpoint(PROJECT_ROOT)
    except ValueError as error:
        parser.error(str(error))
    print(f"학습 시작 가중치: {checkpoint}")
    from ultralytics import YOLO

    model = YOLO(checkpoint)  # COCO pretrained 가중치 로드 (transfer learning 시작점)

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
