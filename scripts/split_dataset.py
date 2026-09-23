#!/usr/bin/env python3
"""
라벨링이 끝난 (이미지, YOLO txt 라벨) 쌍을 train/val/test로 랜덤 분할해서
dataset/images/{train,val,test}, dataset/labels/{train,val,test} 에 복사합니다.

사용 예:
  python3 split_dataset.py --src ./raw_labeled --dst ../dataset \
      --train 0.8 --val 0.1 --test 0.1 --seed 42

--src 폴더 구조 가정 (라벨링 툴 export 결과를 한 폴더에 모아둔 상태):
  raw_labeled/
    img001.jpg
    img001.txt
    img002.jpg
    img002.txt
    ...

이미지 확장자는 .jpg .jpeg .png 를 지원합니다.
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def find_pairs(src_dir: Path):
    pairs = []
    for img_path in sorted(src_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            print(f"[경고] 라벨 없음, 건너뜀: {img_path.name}")
            continue
        pairs.append((img_path, label_path))
    return pairs


def split_pairs(pairs, train_ratio, val_ratio, test_ratio, seed):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train+val+test 비율의 합은 1.0이어야 합니다."
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train:n_train + n_val]
    test_set = shuffled[n_train + n_val:]
    return {"train": train_set, "val": val_set, "test": test_set}


def copy_split(split_name, pairs, dst_root: Path, move=False):
    img_dir = dst_root / "images" / split_name
    lbl_dir = dst_root / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    op = shutil.move if move else shutil.copy2
    for img_path, label_path in pairs:
        op(str(img_path), str(img_dir / img_path.name))
        op(str(label_path), str(lbl_dir / label_path.name))


def main():
    parser = argparse.ArgumentParser(description="YOLO 데이터셋 train/val/test 분할")
    parser.add_argument("--src", required=True, help="라벨링 완료된 이미지+txt가 모여있는 폴더")
    parser.add_argument("--dst", required=True, help="dataset 루트 폴더 (images/, labels/ 하위 생성)")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true",
                         help="복사 대신 이동(원본 삭제). 기본값은 복사.")
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()

    pairs = find_pairs(src_dir)
    if not pairs:
        print("이미지/라벨 쌍을 찾지 못했습니다. --src 경로를 확인하세요.")
        return

    splits = split_pairs(pairs, args.train, args.val, args.test, args.seed)

    for split_name, split_pairs_list in splits.items():
        copy_split(split_name, split_pairs_list, dst_root, move=args.move)
        print(f"{split_name}: {len(split_pairs_list)}장")

    print(f"\n완료. 총 {len(pairs)}쌍 -> {dst_root} 에 분할 저장됨.")


if __name__ == "__main__":
    main()
