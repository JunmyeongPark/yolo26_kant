#!/usr/bin/env python3
"""하위 폴더별 파일명 순서로 매 N번째 이미지/라벨 쌍을 val에 복사.

python3 scripts/split_dataset.py --src dataset/raw --dst dataset --val-every 9
나머지는 train에 저장하며 test는 새로 만들지 않습니다.
출력에도 원본의 상대 폴더 구조를 유지합니다. 기존 출력은 자동 삭제하지 않습니다.
"""

import argparse
from collections import defaultdict
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def find_pairs(src_dir: Path):
    pairs = []
    for img_path in sorted(src_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = img_path.with_suffix(".txt")
        if not label_path.is_file():
            print(f"[경고] 라벨 없음, 건너뜀: {img_path.relative_to(src_dir)}")
            continue
        pairs.append((img_path, label_path))
    return pairs


def split_pairs(pairs, val_every=9):
    if val_every < 2:
        raise ValueError("val_every는 2 이상이어야 합니다.")
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair[0].parent].append(pair)
    splits = {"train": [], "val": []}
    for folder in sorted(groups):
        group = sorted(groups[folder])
        for index, pair in enumerate(group, start=1):
            splits["val" if index % val_every == 0 else "train"].append(pair)
        print(f"{folder}: train {len(group) - len(group) // val_every}, val {len(group) // val_every}")
    return splits


def copy_split(split_name, pairs, dst_root: Path, src_root: Path, move=False):
    op = shutil.move if move else shutil.copy2
    for img_path, label_path in pairs:
        for source, kind in ((img_path, "images"), (label_path, "labels")):
            target = dst_root / kind / split_name / source.relative_to(src_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            op(str(source), str(target))


def main():
    parser = argparse.ArgumentParser(description="폴더별 매 N번째 쌍을 val로 분할 (나머지는 train)")
    parser.add_argument("--src", required=True, help="이미지와 같은 이름의 txt가 있는 루트 폴더")
    parser.add_argument("--dst", required=True, help="dataset 루트 폴더")
    parser.add_argument("--val-every", type=int, default=9, help="매 N번째 쌍을 val에 배정 (기본 9)")
    parser.add_argument("--move", action="store_true", help="복사 대신 이동(원본 삭제)")
    args = parser.parse_args()
    src_dir = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    if args.val_every < 2:
        parser.error("--val-every는 2 이상이어야 합니다.")
    if not src_dir.is_dir():
        parser.error("--src 폴더가 없습니다.")
    if dst_root == src_dir or src_dir in dst_root.parents:
        parser.error("--dst는 --src 내부에 둘 수 없습니다.")
    for kind in ("images", "labels"):
        root = dst_root / kind
        if root.exists() and (not root.is_dir() or any(p.is_file() for p in root.rglob("*"))):
            parser.error(f"기존 출력이 있습니다: {root}. images/labels를 먼저 백업·이동하거나 빈 --dst를 지정하세요.")
    pairs = find_pairs(src_dir)
    if not pairs:
        parser.error("이미지/라벨 쌍을 찾지 못했습니다.")
    # 서로 다른 이미지 확장자가 같은 라벨을 공유하면 --move 및 출력이 모호해집니다.
    labels = [label for _, label in pairs]
    if len(set(labels)) != len(labels):
        parser.error("같은 폴더에 이름이 같고 확장자만 다른 이미지가 있습니다.")
    splits = split_pairs(pairs, args.val_every)
    for name, subset in splits.items():
        copy_split(name, subset, dst_root, src_dir, move=args.move)
        print(f"{name}: {len(subset)}장")
    print(f"완료. 총 {len(pairs)}쌍 -> {dst_root}")


if __name__ == "__main__":
    main()
