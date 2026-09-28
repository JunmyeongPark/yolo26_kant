#!/usr/bin/env python3
"""영상 기준 raw 정리 → 임시 위치에서 분할 → 기존 출력 백업 후 교체."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from organize_raw import organize_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, default=Path('dataset/raw'))
    parser.add_argument('--video-dir', type=Path, default=Path('dataset/raw_videos'))
    parser.add_argument('--dst', type=Path, default=Path('dataset'))
    parser.add_argument('--val-every', type=int, default=9)
    args = parser.parse_args()
    src, dst, videos = args.src.resolve(), args.dst.resolve(), args.video_dir.resolve()
    if args.val_every < 2:
        parser.error('--val-every는 2 이상이어야 합니다.')
    if dst == src or src in dst.parents:
        parser.error('--dst는 --src 내부에 둘 수 없습니다.')
    for kind in ('images', 'labels'):
        output = dst / kind
        if output.is_symlink() or (output.exists() and not output.is_dir()):
            parser.error(f'출력은 실제 디렉터리여야 합니다: {output}')
        if src == output or output in src.parents or videos == output or output in videos.parents:
            parser.error('원본 폴더를 출력 내부에 둘 수 없습니다.')
    organize_raw(src, videos)
    dst.mkdir(parents=True, exist_ok=True)
    # 같은 파일시스템에서 생성하여 출력 교체는 rename으로 처리.
    with tempfile.TemporaryDirectory(prefix='.resplit-', dir=dst) as temporary:
        staging = Path(temporary)
        subprocess.run([sys.executable, str(Path(__file__).with_name('split_dataset.py')),
                        '--src', str(src), '--dst', str(staging),
                        '--val-every', str(args.val_every)], check=True)
        for kind in ('images', 'labels'):
            (staging / kind).mkdir(exist_ok=True)
        backup_root = dst / '.split_backups'
        backup_root.mkdir(exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix='split-', dir=backup_root))
        installed, backed_up = [], []
        try:
            for kind in ('images', 'labels'):
                if (dst / kind).exists():
                    (dst / kind).rename(backup / kind)
                    backed_up.append(kind)
                (staging / kind).rename(dst / kind)
                installed.append(kind)
        except Exception:
            for kind in reversed(installed):
                (dst / kind).rename(staging / kind)
            for kind in backed_up:
                (backup / kind).rename(dst / kind)
            raise
        print(f'기존 분할 백업: {backup}')
        print(f'재분할 완료: {dst}')


if __name__ == '__main__':
    main()
