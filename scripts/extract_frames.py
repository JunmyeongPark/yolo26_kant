#!/usr/bin/env python3
"""
record_video.py로 찍은 영상에서 프레임을 뽑아 라벨링용 이미지로 저장.

영상을 프레임 단위로 다 뽑으면 거의 똑같은 사진이 수백~수천 장 생겨서
비효율적이라, --every-n-frames 만큼 건너뛰면서 뽑습니다
(기본 5 = 30fps 영상 기준 초당 6장).

사용 예:
  # 영상 한 개
  python3 scripts/extract_frames.py --video dataset/raw_videos/clip_20260922_153000_001.mp4

  # 폴더 안 영상 전부 (record_video.py 기본 저장 위치)
  python3 scripts/extract_frames.py --video-dir dataset/raw_videos \
      --outdir dataset/raw --every-n-frames 5

추출된 이미지(dataset/raw)를 라벨링 툴(CVAT/LabelImg/Roboflow)에 불러와
라벨링한 뒤, 같은 폴더에 이미지+라벨(.txt) 쌍이 모이면
`scripts/split_dataset.py --src dataset/raw ...`로 train/val 분할하세요.
"""

import argparse
from pathlib import Path
import os

import cv2

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def extract_one(video_path: str, outdir: str, every_n: int) -> int:
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] 열 수 없음: {video_path}")
        return 0

    stem = os.path.splitext(os.path.basename(video_path))[0]
    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % every_n == 0:
            out_name = f"{stem}_f{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(outdir, out_name), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"{video_path}: 총 {frame_idx}프레임 중 {saved}장 저장")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="영상에서 라벨링용 프레임 추출")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="영상 파일 하나 (outdir에 직접 저장)")
    source.add_argument("--video-dir", help="하위 폴더를 재귀 탐색하고 상대 폴더 구조 유지")
    parser.add_argument("--outdir", default="dataset/raw")
    parser.add_argument("--every-n-frames", type=int, default=5,
                         help="이 프레임마다 1장씩 저장 (기본 5)")
    args = parser.parse_args()

    if args.every_n_frames < 1:
        parser.error("--every-n-frames는 1 이상이어야 합니다.")

    os.makedirs(args.outdir, exist_ok=True)

    videos = []
    if args.video:
        videos.append((Path(args.video), Path(args.outdir)))
    if args.video_dir:
        root = Path(args.video_dir)
        for video in sorted(root.rglob("*")):
            if video.is_file() and video.suffix.lower() in VIDEO_EXTS:
                videos.append((video, Path(args.outdir) / video.parent.relative_to(root)))

    if not videos:
        print("추출할 영상을 찾지 못했습니다.")
        return

    total = 0
    for video, outdir in videos:
        total += extract_one(video, outdir, args.every_n_frames)

    print(f"완료: 영상 {len(videos)}개 -> 총 {total}장 저장 ({args.outdir})")


if __name__ == "__main__":
    main()
