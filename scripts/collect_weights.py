#!/usr/bin/env python3
"""최신 학습의 best.pt를 weights/ 아래 지정한 .pt 파일명으로 보관합니다."""
import argparse
from datetime import datetime
from pathlib import Path
import pickletools
import shutil
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = PROJECT_ROOT / 'weights'


def checkpoint_date(path):
    # pickle을 역직렬화하거나 모델 코드를 실행하지 않고 날짜만 읽습니다.
    with zipfile.ZipFile(path) as archive:
        data = archive.read(next(n for n in archive.namelist() if n.endswith('/data.pkl')))
    strings = [arg for op, arg, _ in pickletools.genops(data)
               if op.name in ('BINUNICODE', 'SHORT_BINUNICODE')]
    for index, value in enumerate(strings[:-1]):
        if value == 'date':
            return datetime.fromisoformat(strings[index + 1]).timestamp()
    raise ValueError(f'체크포인트 날짜가 없습니다: {path}')


def output_filename(value):
    value = value.strip()
    if not value or Path(value).name != value or "\\" in value or Path(value).suffix.lower() != ".pt":
        raise ValueError("경로 없이 .pt 파일명을 입력하세요. 예: v2.pt")
    return value


def choose_output_filename(value=None):
    if value is not None:
        return output_filename(value)
    while True:
        try:
            value = input("학습 결과 저장 파일명 (예: v2.pt): ")
        except EOFError:
            raise ValueError("입력을 받을 수 없습니다. --output v2.pt 옵션을 지정하세요.") from None
        try:
            return output_filename(value)
        except ValueError as error:
            print(error)


def archive_weights(run_dir, destination=DEFAULT_DEST, *, filename):
    filename = output_filename(filename)
    source = Path(run_dir) / 'weights/best.pt'
    if not source.is_file():
        raise ValueError(f'가중치가 없습니다: {source}')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename
    temporary = destination / f".{filename}.tmp"
    try:
        shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'최신 가중치 보관: {source} -> {target}')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', type=Path, default=PROJECT_ROOT / 'runs/detect')
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    parser.add_argument("--output", help="저장할 .pt 파일명. 생략하면 입력받음")
    args = parser.parse_args()
    try:
        filename = choose_output_filename(args.output)
    except ValueError as error:
        parser.error(str(error))
    checkpoints = list(args.runs_dir.expanduser().resolve().glob('*/weights/best.pt'))
    if not checkpoints:
        parser.error('학습 가중치가 없습니다.')
    latest = max(checkpoints, key=checkpoint_date)
    archive_weights(latest.parent.parent, args.dest.expanduser().resolve(), filename=filename)


if __name__ == '__main__':
    main()
