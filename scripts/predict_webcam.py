#!/usr/bin/env python3
"""
RealSense(또는 아무 V4L2 카메라) 스트림으로 YOLO 실시간 추론 테스트.

주의: `yolo predict model=... source=/dev/realsense_color` 처럼 CLI에
장치 경로를 바로 넘기면 ultralytics가 숫자 인덱스/URL이 아닌 문자열은
webcam으로 인식하지 못해 실패할 수 있습니다. 그래서 여기서는
cv2.VideoCapture로 직접 장치를 열어서 프레임 단위로 model.predict()에
넘기는 방식을 씁니다 - udev symlink(/dev/realsense_color 등)를 그대로
쓸 수 있어서 안정적입니다.

사용 예:
  # pretrained 그대로 (custom dataset 없이 파이프라인/환경 점검용)
  python3 predict_webcam.py --model yolo26n.pt --source /dev/realsense_color

  # fine-tuning된 모델로
  python3 predict_webcam.py --model runs/detect/puck_knob_v1-2/weights/best.pt \
      --source /dev/realsense_color
"""

import argparse

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO 실시간 웹캠/RealSense 추론")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--source", default="/dev/realsense_color",
                         help="udev symlink 장치 경로(예: /dev/realsense_color) 또는 숫자 인덱스")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    model = YOLO(args.model)

    # 숫자 문자열이면 정수 인덱스로, 아니면(장치 경로 등) 문자열 그대로 사용
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {args.source}")

    print(f"모델: {args.model} / 소스: {args.source} (ESC로 종료)")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했습니다.")
            break

        results = model.predict(frame, conf=args.conf, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("puck_knob_detect", annotated)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
