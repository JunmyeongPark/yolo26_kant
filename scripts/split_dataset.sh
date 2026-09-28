#!/usr/bin/env bash
# bash scripts/split_dataset.sh [--dst 빈_출력_폴더] [--val-every 9]
# 영상 기준 raw 폴더 정리 후 train/val 재생성. 기존 출력은 자동 백업.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
"$PYTHON" scripts/prepare_dataset.py "$@"
