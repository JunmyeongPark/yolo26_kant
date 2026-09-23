#!/usr/bin/env python3
"""
puck/knob 커스텀 데이터셋 구축용 영상 촬영 스크립트.

한 장씩 사진을 찍는 것보다, 물체를 들고 다양한 각도/거리로 천천히 움직이면서
영상으로 쭉 찍어두고 나중에 extract_frames.py로 프레임을 솎아내는 게 훨씬
빠르고 각도/거리 다양성도 자연스럽게 확보됩니다.

사용 예:
  # RealSense로 촬영 (SPACE로 녹화 시작/정지, 여러 번 반복 가능, q/ESC로 종료)
  python3 scripts/record_video.py --source /dev/realsense_color

  # 해상도/저장 위치 지정
  python3 scripts/record_video.py --source /dev/realsense_color \
      --outdir dataset/raw_videos --width 1280 --height 720 --fps 30

조작:
  SPACE : 녹화 시작 / 정지 (정지할 때마다 파일 하나가 저장되고,
          다시 누르면 새 파일로 이어서 촬영 가능 - 여러 컷으로 나눠 찍고 싶을 때 유용)
  q/ESC : 프로그램 종료
"""

import argparse
import datetime
import os

import cv2


def open_capture(source: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {source}")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description="puck/knob 데이터셋용 영상 촬영")
    parser.add_argument("--source", default="/dev/realsense_color",
                         help="카메라 장치 경로(예: /dev/realsense_color) 또는 숫자 인덱스")
    parser.add_argument("--outdir", default="dataset/raw_videos")
    parser.add_argument("--width", type=int, default=0, help="캡처 해상도 너비 (0=카메라 기본값)")
    parser.add_argument("--height", type=int, default=0, help="캡처 해상도 높이 (0=카메라 기본값)")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cap = open_capture(args.source, args.width, args.height, args.fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = None
    recording = False
    clip_idx = 0

    print(f"소스: {args.source} / 해상도: {actual_w}x{actual_h} / {args.fps}fps")
    print(f"저장 위치: {args.outdir}")
    print("SPACE: 녹화 시작/정지, q 또는 ESC: 종료")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했습니다.")
            break

        display = frame.copy()
        if recording:
            cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
        cv2.imshow("record_video (SPACE: rec on/off, q: quit)", display)

        if recording and writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if not recording:
                clip_idx += 1
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(args.outdir, f"clip_{ts}_{clip_idx:03d}.mp4")
                writer = cv2.VideoWriter(out_path, fourcc, args.fps, (actual_w, actual_h))
                recording = True
                print(f"[REC 시작] {out_path}")
            else:
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                print("[REC 정지]")
        elif key in (27, ord("q")):
            break

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
