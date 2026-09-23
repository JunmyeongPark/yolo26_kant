#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# 학습된 best.pt를 라즈베리파이 배포용 NCNN 포맷으로 변환
#
# 반드시 학습이 이뤄진 x86_64 머신(이 노트북)에서 실행하세요.
# Ultralytics 공식 벤치마크 기준, 라즈베리파이에서는 NCNN이
# ONNX/MNN 등보다 가장 빠른 추론 성능을 보입니다.
#
# 사용법:
#   ./export_ncnn.sh                                # 기본 경로 사용
#   ./export_ncnn.sh runs/detect/puck_knob_v1-2/weights/best.pt
#   (주의: --name 폴더가 이미 있으면 ultralytics가 puck_knob_v1-2, -3...으로
#    자동 증가시키니, 실제 존재하는 run 폴더명을 ls로 확인 후 넘기세요)
# ==========================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

WEIGHTS="${1:-runs/detect/puck_knob_v1-2/weights/best.pt}"

if [ ! -f "$WEIGHTS" ]; then
  echo "가중치 파일을 찾을 수 없습니다: $WEIGHTS"
  echo "사용법: ./export_ncnn.sh [학습된 .pt 경로]"
  exit 1
fi

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python3 -c "
from ultralytics import YOLO
model = YOLO('${WEIGHTS}')
model.export(format='ncnn')
"

OUT_DIR="$(dirname "$WEIGHTS")/$(basename "$WEIGHTS" .pt)_ncnn_model"

echo ""
echo "완료. 변환된 모델: ${OUT_DIR}"
echo ""
echo "라즈베리파이로 복사 예시:"
echo "  scp -r ${OUT_DIR} pi@<라즈베리파이IP>:~/craft/custom_yolo26/models/"
