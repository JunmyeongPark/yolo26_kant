# C++ 추론 (knob / 신호등 / puck)

Python 학습 결과(.pt)를 export한 모델을 C++로 추론하기 위한 코드입니다.
export된 모델 파일은 Python/C++ 어느 쪽 바인딩으로 열어도 동일하게 동작합니다 —
export를 다시 할 필요 없이, 같은 파일을 언어만 바꿔서 씁니다.

사용할 클래스는 `knob`, `sign_red`, `sign_yellow`, `sign_green`, `puck_red`, `puck_green`, `puck_blue`로 한정합니다.
현재 데이터는 앞의 4개 클래스이며, 클래스 번호 및 확장 절차는 [루트 README](../README.md#2-2-라벨링)를 따릅니다.

두 경로를 준비해뒀습니다.

## 1. `onnx_infer/` — ONNX Runtime (완성, 권장 시작점)

YOLO26은 NMS-free라서 정적 shape로 export하면 출력이 고정 `[1, 300, 6]`
(x1, y1, x2, y2, score, class_id) 형태로 나옵니다(공식 문서 기준). 이 스펙에
맞춰 전처리(letterbox) → 추론 → 후처리(score threshold + 좌표 복원)까지
구현되어 있습니다.

**빌드 준비물**
- OpenCV (`sudo apt install libopencv-dev` 로 충분, OpenCV 5 소스 빌드 불필요)
- ONNX Runtime prebuilt 바이너리 (소스 빌드 불필요)
  - x86_64(노트북, 테스트용): https://github.com/microsoft/onnxruntime/releases 에서
    `onnxruntime-linux-x64-<version>.tgz` 다운로드 후 원하는 경로에 압축 해제
  - aarch64(라즈베리파이): 같은 releases 페이지에서
    `onnxruntime-linux-aarch64-<version>.tgz` 다운로드 (없으면 `pip install onnxruntime`로
    설치된 패키지 내 `onnxruntime/capi/` 밑의 .so + 헤더를 사용하거나 소스 빌드 필요)

**빌드**

```bash
cd cpp
mkdir build && cd build
cmake -DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime-linux-aarch64-<version> ..
make -j$(nproc)
```

**실행**

```bash
# 이미지 한 장 테스트
./onnx_infer/detect_onnx ../../models/best.onnx ../../dataset/images/test/sample.jpg 0.5

# 카메라(예: RealSense color가 /dev/realsense_color 등으로 잡혀있는 경우) 실시간
./onnx_infer/detect_onnx ../../models/best.onnx 0 0.5
```

처음 실행하면 콘솔에 실제 output shape이 찍힙니다. `[1, 300, 6]`이 아니면
`src/detect_onnx.cpp`의 후처리 부분을 실제 shape에 맞게 고쳐야 합니다.

## 2. `ncnn_infer/` — NCNN (스텁, 미완성)

라즈베리파이에서 가장 빠른 추론 성능(Ultralytics 공식 벤치마크 기준)을 내는
경로지만, YOLO26 + NCNN 조합의 C++ 출력 텐서 스펙이 아직 검증되지 않아서
로직을 비워뒀습니다. `src/detect_ncnn.cpp` 상단 주석에 채우는 순서가 적혀있고,
핵심은:

1. `scripts/export_ncnn.sh`로 실제 모델 export
2. 생성된 `*_ncnn_model/model.ncnn.param` 텍스트 파일 열어서 input/output 블롭 이름 확인
3. 그 이름으로 `ex.input(...)`/`ex.extract(...)` 채우고, 출력 shape 찍어서 확인 후
   `onnx_infer` 와 동일한 방식(NMS-free, score+rescale)으로 후처리 구현

실제 학습된 모델을 export한 뒤 `.param` 파일 내용을 공유해주시면 여기 코드도
마저 채워드릴 수 있습니다.

## 성능 참고 (Ultralytics 공식 벤치마크, YOLO26n / RPi 5 기준)

| 포맷 | 추론 시간 |
|---|---|
| NCNN | 67.03 ms |
| MNN | 91.87 ms |
| ONNX | 125.99 ms |

지금은 ONNX로 먼저 파이프라인 전체(카메라 입력 → 추론 → 로봇 제어 로직 연동)를
검증하고, 이후 속도가 부족하면 NCNN으로 교체하는 순서를 권장합니다.
