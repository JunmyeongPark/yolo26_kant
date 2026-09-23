#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# custom_yolo26 환경 자동 설정 스크립트
#
# 아키텍처를 감지해서 역할을 나눔:
#   - x86_64  (예: 개발용 노트북) -> 학습(training)용 전체 환경
#   - aarch64 (라즈베리파이 64bit) -> 추론(inference) 전용 경량 환경
#
# YOLO 계열은 학습을 라즈베리파이에서 돌리는 걸 권장하지 않습니다
# (Ultralytics 공식 가이드도 RPi는 inference 전용으로만 다룸).
# 학습은 이 노트북에서, 학습된 모델은 export_ncnn.sh로 NCNN 변환 후
# 라즈베리파이로 옮겨서 추론에 사용하는 흐름입니다.
#
# 사용법:
#   chmod +x setup_env.sh
#   ./setup_env.sh
# ==========================================================

ARCH=$(uname -m)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "감지된 아키텍처: $ARCH"
echo "프로젝트 루트: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

case "$ARCH" in
  x86_64)
    echo ""
    echo "[x86_64] 학습용 환경을 설정합니다."
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt

    if command -v nvidia-smi &> /dev/null; then
      echo "NVIDIA GPU 감지됨. CUDA 사용 가능 여부 확인 중..."
      python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" || \
        echo "(torch import 실패 - 설치가 아직 안 됐거나 문제가 있을 수 있음)"
    else
      echo "GPU 미감지. CPU로 학습됩니다(속도가 느릴 수 있음)."
    fi

    echo ""
    echo "완료. 다음부터는 아래로 가상환경 활성화 후 작업하세요:"
    echo "  source .venv/bin/activate"
    ;;

  aarch64)
    echo ""
    echo "[aarch64] 라즈베리파이 추론 전용 환경을 설정합니다."
    echo "주의: 64bit OS(Raspberry Pi OS Bookworm 이상) 기준입니다."

    sudo apt update
    sudo apt install -y python3-pip python3-venv

    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -U pip
    pip install ultralytics ncnn

    echo ""
    echo "완료. 학습은 이 장치에서 하지 않습니다."
    echo "노트북(x86_64)에서 학습 -> scripts/export_ncnn.sh 로 NCNN 변환 ->"
    echo "변환된 *_ncnn_model 폴더를 이 라즈베리파이의 models/ 로 복사해서 추론에 사용하세요."
    ;;

  armv7l|arm*)
    echo "32bit ARM($ARCH)은 지원 범위 밖입니다."
    echo "Raspberry Pi OS 64bit(Bookworm)로 재설치를 권장합니다."
    exit 1
    ;;

  *)
    echo "지원하지 않는 아키텍처입니다: $ARCH"
    exit 1
    ;;
esac
