#!/usr/bin/env bash
# bash scripts/extract_frames.sh [추가 Python CLI 옵션]
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
"$PYTHON" scripts/extract_frames.py --video-dir dataset/raw_videos --outdir dataset/raw --every-n-frames 5 "$@"
