#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# RealSense D435 - 스트림별(Depth/Infrared/Color) udev symlink 고정 스크립트
#
# v4l2-ctl --all 로 실측 확인한 D435의 video4linux 노드 구조:
#
#   인터페이스 0 (bInterfaceNumber=00, Depth/Stereo 모듈)
#     index0 -> Depth      (Z16, 예: 256x144)
#     index1 -> Depth Metadata (포맷 없음)
#     index2 -> Infrared 1 (GREY/Y8, 예: 424x240)
#     index3 -> Infrared 1 Metadata (포맷 없음)
#
#   인터페이스 3 (bInterfaceNumber=03, Color 모듈)
#     index0 -> Color      (YUYV, 예: 320x180)
#     index1 -> Color Metadata (포맷 없음)
#
#   * 순수 V4L2(UVC)로는 Infrared 2(우안)가 별도 캡처 노드로 노출되지
#     않는 케이스가 일반적입니다(펌웨어/커널에 따라 다를 수 있음).
#
# %n(minor 번호)처럼 그냥 순서로 붙이는 게 아니라, 시리얼번호 +
# bInterfaceNumber + index 조합으로 "의미 있는" 이름을 고정합니다.
#
# 사용법:
#   chmod +x setup_realsense_udev_symlink.sh
#   sudo ./setup_realsense_udev_symlink.sh
#
# 실행 후 확인:
#   ls -l /dev/realsense_*
# ==========================================================

RULE_FILE="/etc/udev/rules.d/99-realsense-d435-streams.rules"

if [ "$(id -u)" -ne 0 ]; then
  echo "이 스크립트는 udev 규칙 파일 생성을 위해 sudo 권한이 필요합니다."
  echo "예: sudo $0"
  exit 1
fi

echo "[1/4] RealSense video4linux 노드 탐색 중..."
REALSENSE_DEV=""
for f in /sys/class/video4linux/video*; do
  [ -e "$f/name" ] || continue
  name=$(cat "$f/name" 2>/dev/null || true)
  if [[ "$name" == *RealSense* ]]; then
    REALSENSE_DEV=$(basename "$f")
    break
  fi
done

if [ -z "$REALSENSE_DEV" ]; then
  echo "RealSense video4linux 장치를 찾지 못했습니다. 연결 상태를 확인하세요."
  exit 1
fi

echo "  -> 발견: /dev/${REALSENSE_DEV}"

echo "[2/4] 시리얼 번호 조회 중..."
SERIAL=$(udevadm info --query=property --name="/dev/${REALSENSE_DEV}" | sed -n 's/^ID_SERIAL_SHORT=//p')

if [ -z "$SERIAL" ]; then
  echo "ID_SERIAL_SHORT 값을 가져오지 못했습니다."
  echo "udevadm info --query=property --name=/dev/${REALSENSE_DEV} 결과를 확인하세요:"
  udevadm info --query=property --name="/dev/${REALSENSE_DEV}"
  exit 1
fi

echo "  -> 시리얼: ${SERIAL}"

echo "[3/4] udev 규칙 파일 생성 중: $RULE_FILE"
cat > "$RULE_FILE" <<EOF
# RealSense D435 (S/N: ${SERIAL}) - 스트림별 고정 symlink
# 자동 생성됨: $(date '+%Y-%m-%d %H:%M:%S')
#
# 인터페이스 0 = Depth/Stereo 모듈, 인터페이스 3 = Color 모듈
# (v4l2-ctl --all 로 실측 확인된 D435 기본 구조. 다른 개체/펌웨어에서는
#  udevadm info -a -n /dev/videoN 으로 재확인 권장)

SUBSYSTEM=="video4linux", ENV{ID_SERIAL_SHORT}=="${SERIAL}", ATTRS{bInterfaceNumber}=="00", ATTR{index}=="0", SYMLINK+="realsense_depth",        MODE="0666", GROUP="video"
SUBSYSTEM=="video4linux", ENV{ID_SERIAL_SHORT}=="${SERIAL}", ATTRS{bInterfaceNumber}=="00", ATTR{index}=="1", SYMLINK+="realsense_depth_meta",   MODE="0666", GROUP="video"
SUBSYSTEM=="video4linux", ENV{ID_SERIAL_SHORT}=="${SERIAL}", ATTRS{bInterfaceNumber}=="00", ATTR{index}=="2", SYMLINK+="realsense_infra1",       MODE="0666", GROUP="video"
SUBSYSTEM=="video4linux", ENV{ID_SERIAL_SHORT}=="${SERIAL}", ATTRS{bInterfaceNumber}=="00", ATTR{index}=="3", SYMLINK+="realsense_infra1_meta",  MODE="0666", GROUP="video"
SUBSYSTEM=="video4linux", ENV{ID_SERIAL_SHORT}=="${SERIAL}", ATTRS{bInterfaceNumber}=="03", ATTR{index}=="0", SYMLINK+="realsense_color",        MODE="0666", GROUP="video"
SUBSYSTEM=="video4linux", ENV{ID_SERIAL_SHORT}=="${SERIAL}", ATTRS{bInterfaceNumber}=="03", ATTR{index}=="1", SYMLINK+="realsense_color_meta",   MODE="0666", GROUP="video"

# USB 장치 자체 권한(펌웨어 접근 등)
SUBSYSTEM=="usb", ENV{ID_SERIAL_SHORT}=="${SERIAL}", MODE="0666", GROUP="video"
EOF

echo "[4/4] udev 규칙 재적용 중..."
udevadm control --reload-rules
udevadm trigger

echo ""
echo "완료. 확인:"
echo "  ls -l /dev/realsense_*"
echo ""
echo "주의: 위 인터페이스/index 매핑은 이번에 v4l2-ctl로 실측한 값 기준입니다."
echo "      만약 결과가 다르면(symlink가 안 생기거나 개수가 다르면)"
echo "      udevadm info -a -n /dev/videoN 으로 실제 ATTRS{bInterfaceNumber}, ATTR{index} 값을"
echo "      다시 확인해서 규칙을 맞춰야 합니다."
