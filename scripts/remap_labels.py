#!/usr/bin/env python3
"""
YOLO 라벨(.txt) 파일들의 class_id를 일괄 재매핑.

makesense.ai 등에서 프로젝트를 새로 만들 때마다 라벨을 추가하는 순서가
달라지면, 같은 물체인데도 export된 class_id가 배치마다 다르게 나올 수
있습니다 (class_id는 그 프로젝트 안에서 라벨을 추가한 순서일 뿐, 전역
고정값이 아님). dataset/raw에 여러 배치를 합치기 전에 이 스크립트로
class_id를 data.yaml 기준(0=puck, 1=knob)에 맞춰 통일하세요.

사용 예:
  # 이 폴더 라벨의 0<->1을 맞바꾸기 (순서가 반대로 나온 경우)
  python3 scripts/remap_labels.py --dir path/to/exported_labels --map 0:1,1:0

  # 이 폴더가 knob 하나만 있던 프로젝트(class 0)였던 걸 knob=1로 맞추기
  python3 scripts/remap_labels.py --dir path/to/exported_labels --map 0:1

  # 먼저 실제로 바꾸지 않고 몇 줄이 바뀔지만 확인
  python3 scripts/remap_labels.py --dir path/to/exported_labels --map 0:1 --dry-run
"""

import argparse
import glob
import os


def parse_map(s: str) -> dict:
    mapping = {}
    for pair in s.split(","):
        old, new = pair.split(":")
        mapping[int(old)] = int(new)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 라벨 class_id 재매핑")
    parser.add_argument("--dir", required=True, help="txt 라벨 파일들이 있는 폴더")
    parser.add_argument("--map", required=True,
                         help="old:new 쌍, 콤마로 구분 (예: 0:1,1:0)")
    parser.add_argument("--dry-run", action="store_true",
                         help="실제로 고치지 않고 몇 줄이 바뀔지만 출력")
    args = parser.parse_args()

    mapping = parse_map(args.map)
    txt_files = glob.glob(os.path.join(args.dir, "*.txt"))
    if not txt_files:
        print(f"'{args.dir}'에서 .txt 파일을 찾지 못했습니다.")
        return

    changed_files = 0
    changed_lines = 0
    for path in txt_files:
        with open(path) as f:
            lines = f.readlines()

        new_lines = []
        file_changed = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                new_lines.append(line)
                continue
            cls = int(parts[0])
            if cls in mapping and mapping[cls] != cls:
                parts[0] = str(mapping[cls])
                file_changed = True
                changed_lines += 1
            new_lines.append(" ".join(parts) + "\n")

        if file_changed:
            changed_files += 1
            if not args.dry_run:
                with open(path, "w") as f:
                    f.writelines(new_lines)

    mode = " (dry-run, 실제로 저장 안 함)" if args.dry_run else ""
    print(f"완료{mode}: 파일 {changed_files}개, 총 {changed_lines}줄 class_id 변경")


if __name__ == "__main__":
    main()
