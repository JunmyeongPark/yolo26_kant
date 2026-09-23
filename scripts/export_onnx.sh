#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# 학습된 best.pt를 C++(ONNX Runtime) 추론용 ONNX로 변환
#
# YOLO26은 NMS-free라서 정적 shape로 export하면 출력 텐서가
# 고정 [1, 300, 6] (x1,y1,x2,y2,score,class_id) 형태로 나옵니다.
# 별도 NMS 코드 없이 score threshold + 좌표 rescale만 하면 됩니다.
#
# 반드시 학습이 이뤄진 x86_64 머신(이 노트북)에서 실행하세요.
#
# 사용법:
#   ./export_onnx.sh                                          # 기본 경로/설정
#   ./export_onnx.sh runs/detect/puck_knob_v1-2/weights/best.pt 640
#   (주의: --name 폴더가 이미 있으면 ultralytics가 puck_knob_v1-2, -3...으로
#    자동 증가시키니, 실제 존재하는 run 폴더명을 ls로 확인 후 넘기세요)
# ==========================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

WEIGHTS="${1:-runs/detect/puck_knob_v1-2/weights/best.pt}"
IMGSZ="${2:-640}"

if [ ! -f "$WEIGHTS" ]; then
  echo "가중치 파일을 찾을 수 없습니다: $WEIGHTS"
  echo "사용법: ./export_onnx.sh [학습된 .pt 경로] [imgsz]"
  exit 1
fi

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# 정적 shape(dynamic=False) + opset 12: C++/OpenCV/ONNX Runtime 호환성 위해 고정
python3 -c "
from ultralytics import YOLO
model = YOLO('${WEIGHTS}')
model.export(format='onnx', imgsz=${IMGSZ}, dynamic=False, opset=12)
"

OUT_FILE="$(dirname "$WEIGHTS")/$(basename "$WEIGHTS" .pt).onnx"

echo ""
echo "완료. 변환된 모델: ${OUT_FILE}"
echo "cpp/onnx_infer 에서 이 파일 경로를 --model 인자로 넘겨서 추론 테스트하세요."
