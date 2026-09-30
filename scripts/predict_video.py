#!/usr/bin/env python3
"""기존 추론 영상에 val_video의 동명 원본을 붙여 비교 영상을 저장합니다."""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def videos_in(path):
    candidates = path.rglob('*') if path.is_dir() else [path]
    return sorted(p for p in candidates if p.is_file()
                  and p.suffix.lower() in VIDEO_EXTS
                  and not p.stem.endswith('_comparison'))


def combine_videos(original, prediction, target):
    left = cv2.VideoCapture(str(original))
    right = cv2.VideoCapture(str(prediction))
    writer = None
    temporary = target.with_name(target.stem + '.partial' + target.suffix)
    codec = {'.mp4': 'mp4v', '.avi': 'MJPG', '.webm': 'VP80'}[target.suffix.lower()]
    count = 0
    try:
        if not left.isOpened() or not right.isOpened():
            raise ValueError(f'영상을 열 수 없습니다: {original}, {prediction}')
        fps = left.get(cv2.CAP_PROP_FPS)
        other_fps = right.get(cv2.CAP_PROP_FPS)
        if not all(math.isfinite(f) and f > 0 for f in (fps, other_fps)) or not math.isclose(fps, other_fps, rel_tol=1e-4):
            raise ValueError(f'FPS가 다르거나 유효하지 않습니다: {fps}, {other_fps}')
        while True:
            ok1, frame = left.read()
            ok2, result = right.read()
            if ok1 != ok2:
                raise ValueError('원본과 추론 영상의 프레임 수가 다릅니다.')
            if not ok1:
                break
            if frame.shape != result.shape:
                raise ValueError('원본과 추론 영상의 해상도가 다릅니다.')
            height, width = frame.shape[:2]
            header = np.zeros((40, width * 2, 3), dtype=np.uint8)
            for label, x in [('Original', 10), ('Prediction', width + 10)]:
                cv2.putText(header, label, (x, 27), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            comparison = np.concatenate((header, np.concatenate((frame, result), axis=1)), axis=0)
            if writer is None:
                writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*codec), fps, (width * 2, height + 40))
                if not writer.isOpened():
                    raise ValueError(f'출력 영상을 만들 수 없습니다: {target}')
            writer.write(comparison)
            count += 1
        if count == 0:
            raise ValueError('읽을 수 있는 프레임이 없습니다.')
        writer.release()
        writer = None
        temporary.replace(target)
    finally:
        left.release()
        right.release()
        if writer is not None:
            writer.release()
        temporary.unlink(missing_ok=True)
    return count


def main():
    parser = argparse.ArgumentParser(description='기존 추론 결과 + val_video 원본을 좌우로 합성 (재추론 없음)')
    parser.add_argument('--prediction', type=Path, help='추론 영상 또는 폴더. 생략하면 runs/detect의 val_video* 폴더 모두 처리')
    parser.add_argument('--original-dir', type=Path, default=PROJECT_ROOT / 'val_video')
    parser.add_argument('--overwrite', action='store_true', help='기존 비교 MP4 재생성')
    parser.add_argument('--format', choices=['mp4', 'avi', 'webm'], default='mp4', help='비교 영상 형식')
    args = parser.parse_args()
    originals = videos_in(args.original_dir.expanduser().resolve())
    if args.prediction:
        predictions = videos_in(args.prediction.expanduser().resolve())
    else:
        predictions = sorted({video for folder in (PROJECT_ROOT / 'runs/detect').rglob('val_video*')
                              if folder.is_dir() for video in videos_in(folder)})
    if not predictions:
        parser.error('추론 결과 영상을 찾지 못했습니다.')
    jobs = []
    for prediction in predictions:
        matches = [p for p in originals if p.stem == prediction.stem]
        if len(matches) != 1:
            parser.error(f'{prediction.name}: 동명 원본이 {len(matches)}개입니다. --original-dir를 확인하세요.')
        if matches[0].resolve() == prediction.resolve():
            parser.error('원본과 추론 입력이 같은 파일입니다.')
        target = prediction.with_name(prediction.stem + '_comparison.' + args.format)
        if target.exists() and not args.overwrite:
            print(f'기존 비교 영상 건너뜀: {target}')
            continue
        if any(job[2] == target for job in jobs):
            parser.error(f'출력 이름 충돌: {target}. --prediction으로 영상 하나를 지정하세요.')
        jobs.append((matches[0], prediction, target))
    for original, prediction, target in jobs:
        try:
            count = combine_videos(original, prediction, target)
        except ValueError as error:
            parser.error(str(error))
        print(f'저장 완료: {target} ({count} frames)', flush=True)


if __name__ == '__main__':
    main()
