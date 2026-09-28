#!/usr/bin/env python3
"""라벨링 완료 폴더에서 라벨 없는 이미지에 빈 YOLO txt를 생성합니다."""
import argparse
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_empty_labels(root, dry_run=False):
    created = existing = 0
    for image in sorted(root.rglob('*')):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
            continue
        label = image.with_suffix('.txt')
        if label.is_file():
            existing += 1
            continue
        if label.exists() or label.is_symlink():
            raise ValueError(f'라벨 경로가 일반 파일이 아닙니다: {label}')
        if not dry_run:
            # 기존 라벨은 절대 덮어쓰지 않습니다.
            try:
                with label.open('x'):
                    pass
            except FileExistsError:
                if not label.is_file():
                    raise
                existing += 1
                continue
        print(f'{"[생성 예정]" if dry_run else "[생성]"} {label}')
        created += 1
    return created, existing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, default=PROJECT_ROOT / 'dataset/raw',
                        help='라벨링 완료 이미지 폴더 (하위 폴더 포함, 기본 dataset/raw)')
    parser.add_argument('--dry-run', action='store_true', help='생성 대상만 표시')
    args = parser.parse_args()
    root = args.src.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'폴더가 없습니다: {root}')
    try:
        created, existing = create_empty_labels(root, args.dry_run)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f'완료: 빈 라벨 {"생성 예정" if args.dry_run else "생성"} {created}개, 기존 라벨 유지 {existing}개')


if __name__ == '__main__':
    main()
