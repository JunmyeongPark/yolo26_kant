#!/usr/bin/env python3
"""Rotate raw/sign_off images 90° counterclockwise and update YOLO boxes."""

import argparse
from pathlib import Path

from PIL import Image


DATASET_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def rotate_label(text: str, path: Path) -> str:
    lines = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: YOLO 라벨은 5개 필드여야 합니다")
        class_id = fields[0]
        x, y, width, height = map(float, fields[1:])
        lines.append(
            f"{class_id} {y:.10g} {1 - x:.10g} {height:.10g} {width:.10g}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def rotate_dataset(source: Path, output: Path, limit: int | None = None) -> int:
    if limit is not None and limit < 1:
        raise ValueError("--limit는 1 이상이어야 합니다")
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"원본 폴더가 없습니다: {source}")
    if output == source or source in output.parents:
        raise ValueError("출력 폴더는 원본 폴더 바깥에 있어야 합니다")
    if output.exists():
        raise FileExistsError(f"출력 폴더가 이미 있습니다: {output}")

    images = sorted(
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit is not None:
        images = images[:limit]
    if not images:
        raise ValueError(f"이미지 파일이 없습니다: {source}")

    labels = {}
    for image_path in images:
        label_path = image_path.with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"짝이 되는 라벨이 없습니다: {label_path}")
        labels[image_path] = rotate_label(label_path.read_text(), label_path)

    output.mkdir(parents=True)
    for image_path in images:
        rotated_stem = f"{image_path.stem}"
        with Image.open(image_path) as image:
            rotated = image.transpose(Image.Transpose.ROTATE_90)
            save_options = {"quality": 95} if image_path.suffix.lower() in {".jpg", ".jpeg"} else {}
            rotated.save(output / f"{rotated_stem}{image_path.suffix}", **save_options)
        (output / f"{rotated_stem}.txt").write_text(labels[image_path])
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DATASET_DIR / "raw/sign_off")
    parser.add_argument("--dst", type=Path, default=DATASET_DIR / "rotated/sign_off")
    parser.add_argument("--limit", type=int, help="처리할 이미지와 라벨 쌍의 최대 개수")
    args = parser.parse_args()
    count = rotate_dataset(args.src, args.dst, args.limit)
    print(f"회전 완료: {count}개 이미지와 라벨 → {args.dst}")


if __name__ == "__main__":
    main()
