#!/usr/bin/env python3
"""영상 이름으로 기존 프레임/라벨을 raw_videos의 상대 폴더에 정리."""
import argparse
from pathlib import Path
import re


def organize_raw(raw, videos):
    if not raw.is_dir() or not videos.is_dir():
        raise ValueError('raw와 raw_videos 폴더가 모두 필요합니다.')
    lookup = {}
    for video in sorted(videos.rglob('*')):
        if video.is_file() and video.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
            if video.stem in lookup:
                raise ValueError(f'중복 영상 이름: {video.stem}')
            lookup[video.stem] = video.parent.relative_to(videos)
    moves, targets = [], set()
    for source in sorted(raw.rglob('*')):
        if not source.is_file() or source.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.txt'}:
            continue
        match = re.fullmatch(r'(.+)_f\d+', source.stem)
        if not match or match[1] not in lookup:
            raise ValueError(f'원본 영상을 찾을 수 없음: {source}')
        target = raw / lookup[match[1]] / source.name
        if target in targets or (source != target and target.exists()):
            raise ValueError(f'출력 이름 충돌: {target}')
        targets.add(target)
        if source != target:
            moves.append((source, target))
    # 모든 매칭과 충돌을 확인한 뒤 이동. 라벨 내용과 이미지 바이트는 유지.
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    print(f'raw 폴더 정리 완료: {len(moves)}개 파일 이동')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, default=Path('dataset/raw'))
    parser.add_argument('--video-dir', type=Path, default=Path('dataset/raw_videos'))
    args = parser.parse_args()
    organize_raw(args.src.resolve(), args.video_dir.resolve())


if __name__ == '__main__':
    main()
