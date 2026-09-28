#!/usr/bin/env bash
# bash scripts/split_dataset.sh [--dst 빈_출력_폴더] [--val-every 9]
# 기존 dataset/images와 dataset/labels는 먼저 별도 위치로 백업·이동하세요.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
"$PYTHON" scripts/split_dataset.py --src dataset/raw --dst dataset --val-every 9 "$@"
