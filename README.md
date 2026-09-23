# custom_yolo26

Waffle Pi 형태 이동로봇 - knob / MASTER 인식용 YOLO custom dataset 학습 프로젝트.

RGB 기반, **pretrained YOLO + custom dataset fine-tuning(transfer learning)** 방식으로 진행합니다.
IR 스트림은 색상 구분(신호등 미션 포함)이 필요해서 사용하지 않음.

**학습은 노트북(x86_64)에서, 추론은 라즈베리파이(ARM)에서** 돌리는 구조입니다.
YOLO 계열은 라즈베리파이에서의 학습을 권장하지 않고(Ultralytics 공식 가이드도 RPi는
inference 전용으로 다룸), 학습된 모델을 NCNN 포맷으로 변환해서 배포하는 흐름을 씁니다.

## 모델 선택: YOLO26

Ultralytics의 최신(2026) 모델인 **YOLO26**을 추천합니다.

- YOLO11 대비 CPU 추론 속도가 크게 향상되어(공식 벤치마크 기준 최대 43% 개선) 로봇 온보드 컴퓨트에서 유리
- NMS-free 구조라 후처리가 단순해져 임베디드 배포에 적합
- Small-Target-Aware Label Assignment(STAL) 도입으로 knob/MASTER처럼 작게 찍히는 물체 탐지에 강함
- 학습 API는 이전 버전(YOLOv8/YOLO11)과 동일한 `ultralytics` 패키지로 그대로 사용 가능

가볍게 시작하려면 `yolo26n.pt`(nano), 정확도를 더 올리고 싶으면 `yolo26s.pt`(small)부터 시도해보세요.
로봇 온보드에서 실시간으로 돌려야 하니 n/s 사이즈를 우선 권장합니다.

## 폴더 구조

```
custom_yolo26/
├── data.yaml               # 클래스 정의(knob, MASTER) + 데이터 경로
├── requirements.txt
├── dataset/
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/   # YOLO txt 포맷 라벨
├── models/                  # pretrained/학습된 가중치(.pt), NCNN 변환 모델 보관
├── scripts/
│   ├── setup_env.sh         # 아키텍처 자동 감지 환경 설정 (x86_64=학습용 / aarch64=추론용)
│   ├── record_video.py      # 데이터셋용 영상 촬영 (SPACE로 녹화 시작/정지)
│   ├── extract_frames.py    # 촬영된 영상에서 라벨링용 프레임 추출
│   ├── split_dataset.py     # 라벨링된 데이터 train/val/test 분할
│   ├── train.py             # fine-tuning 실행 스크립트
│   ├── export_ncnn.sh       # 학습된 모델을 라즈베리파이 배포용 NCNN으로 변환
│   ├── export_onnx.sh       # 학습된 모델을 C++(ONNX Runtime) 추론용 ONNX로 변환
│   ├── predict_webcam.py    # RealSense 등 V4L2 카메라로 실시간 추론 테스트
│   └── setup_realsense_udev_symlink.sh  # RealSense 스트림별 /dev/realsense_* symlink 고정
├── cpp/                     # C++ 추론 (Python보다 오버헤드 적음, 로봇 제어 노드용)
│   ├── onnx_infer/          # ONNX Runtime 기반 (완성, 권장 시작점)
│   └── ncnn_infer/          # NCNN 기반 (스텁 - 실제 export 후 채워야 함)
└── runs/                    # 학습 결과(자동 생성, ultralytics 기본 출력 위치)
```

## 0. RealSense 장치 경로 고정 (최초 1회)

부팅/재연결 시 `/dev/videoN` 번호가 바뀌는 문제가 있어서, udev symlink로 고정해둡니다.
노트북/라즈베리파이 각각에서 RealSense를 처음 연결했을 때 한 번만 실행하면 됩니다.

```bash
chmod +x scripts/setup_realsense_udev_symlink.sh
sudo ./scripts/setup_realsense_udev_symlink.sh
ls -l /dev/realsense_*   # realsense_color, realsense_depth, realsense_infra1 등이 보이면 성공
```

이후 모든 스크립트/코드에서 카메라 소스는 `/dev/realsense_color`(RGB)로 고정해서 씁니다.
숫자 인덱스(`0`, `1` 등)는 노트북 내장 웹캠일 수 있으니 사용하지 마세요.

## 1. 환경 설치

`setup_env.sh`가 `uname -m`으로 아키텍처를 감지해서 알아서 역할을 나눕니다.
**노트북(x86_64)과 라즈베리파이(aarch64) 양쪽 모두에서 그대로 실행**하면 됩니다.

```bash
cd ~/craft/custom_yolo26   # 라즈베리파이에서는 이 폴더를 동일 경로로 복사/clone 해서 실행
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

- x86_64(노트북): venv 생성 + `requirements.txt` 설치 + GPU(CUDA) 감지
- aarch64(라즈베리파이): venv 생성 + `ultralytics`, `ncnn` 추론용 경량 설치만 진행 (학습 관련 패키지는 설치 안 함)

## (선택) Pretrained 그대로 파이프라인 점검

custom dataset 라벨링 전에, 환경/카메라 파이프라인이 제대로 도는지 먼저 확인하고 싶으면
pretrained 가중치(yolo26n.pt) 그대로 실행해볼 수 있습니다. COCO 80개 클래스 기준이라
knob/MASTER을 정확히 맞추진 못하지만, 카메라 연결/추론 자체가 도는지 점검하기엔 충분합니다.

`yolo predict model=... source=/dev/realsense_color` 처럼 CLI에 장치 경로를 바로 넘기면
ultralytics가 이를 webcam으로 인식하지 못해 실패할 수 있어서, `predict_webcam.py`가
`cv2.VideoCapture`로 직접 장치를 열도록 만들어뒀습니다. RealSense udev symlink
(`setup_realsense_udev_symlink.sh`로 고정해둔 경로)를 그대로 사용하세요.

```bash
python3 scripts/predict_webcam.py --model yolo26n.pt --source /dev/realsense_color
```

## 2. 데이터 수집 & 라벨링

### 2-1. 영상 촬영 + 프레임 추출

한 장씩 사진 찍는 것보다, knob/MASTER을 들고 다양한 각도·거리로 천천히 움직이며
영상으로 쭉 찍은 뒤 프레임을 솎아내는 게 훨씬 빠르고 각도/거리 다양성도 자연스럽게
확보됩니다.

**촬영 방식 (체크리스트)**

- 클래스/상황별로 클립을 나눠서 촬영 (knob만, MASTER만, 배경만, 여러 개 겹친 상황 등) —
  나중에 뭘 찍은 영상인지 헷갈리지 않게, 촬영 세션마다 파일명이나 메모로 구분해두기
- 각도: 물체를 들고 천천히 360도 돌려가며 / 카메라를 정면·측면·위에서 내려다보는 각도로 각각 촬영
- 거리: 로봇이 처음 인식하는 먼 거리(~1m)부터 그리퍼로 집기 직전 가까운 거리(~10~20cm)까지 다양하게
- 조명: 실내조명 on/off, 그림자 있는/없는 상태 등 대회 환경과 비슷한 조명 변화를 섞어서 촬영
- 배경만 나오는 클립(물체 없음)도 최소 1개 이상 — false positive 방지용 negative sample
- 일부러 여러 물체(knob/MASTER)를 겹치거나 일부만 보이게 가리는 장면도 조금 섞기 — 실전 가림 상황에 강해짐
- 카메라/손을 너무 빨리 움직이면 모션 블러로 프레임이 흐려져서 라벨링하기 어려우니 천천히 이동

**명령어**

```bash
cd ~/craft/custom_yolo26
source .venv/bin/activate

# 1) 영상 촬영 (SPACE로 녹화 시작/정지, 여러 번 반복해서 여러 클립 촬영 가능, q/ESC로 종료)
python3 scripts/record_video.py --source /dev/realsense_color
# -> dataset/raw_videos/clip_<타임스탬프>_<번호>.mp4 로 저장됨

# 2) dataset/raw_videos 안의 영상들에서 프레임 추출 (기본: 15프레임마다 1장 = 30fps 기준 초당 2장)
python3 scripts/extract_frames.py --video-dir dataset/raw_videos --outdir dataset/raw --every-n-frames 15
# -> dataset/raw/<영상이름>_f<프레임번호>.jpg 로 저장됨 (이 폴더가 라벨링 대상)
```

- `--every-n-frames`를 너무 작게 잡으면 거의 똑같은 사진이 쌓여서 비효율적이니 5~30 사이에서 조절
- 영상을 여러 개 나눠 찍었다면 `--video-dir`가 폴더 안 영상을 전부 처리하니 한 번에 돌리면 됨
- 특정 영상 하나만 다시 뽑고 싶으면 `--video-dir` 대신 `--video dataset/raw_videos/파일명.mp4` 사용

### 2-2. 라벨링

- 라벨링 툴: [CVAT](https://www.cvat.ai/), [LabelImg](https://github.com/heartexlab/labelImg), [Roboflow](https://roboflow.com/) 등에서 `dataset/raw`의 이미지를 불러와 YOLO 포맷으로 export
  - YOLO 라벨 포맷 한 줄: `<class_id> <x_center> <y_center> <width> <height>` (모두 0~1 정규화, 이미지 크기 기준)
  - class_id: `0 = knob`, `1 = MASTER` (data.yaml과 일치해야 함)
- 라벨링 결과(이미지 + 같은 이름의 .txt)가 `dataset/raw`에 모이면 `scripts/split_dataset.py`로 분할:

```bash
python3 scripts/split_dataset.py --src dataset/raw --dst dataset --train 0.8 --val 0.1 --test 0.1
```

## 3. 학습(Fine-tuning)

```bash
python3 scripts/train.py --model yolo26n.pt --epochs 100 --imgsz 640
```

- `model=yolo26n.pt`처럼 COCO pretrained 체크포인트로 시작 → backbone/neck은 pretrained 유지, detection head만 새 클래스 수(2개)에 맞춰 재학습되면서 전체 fine-tune됨
- Data augmentation(mosaic, HSV jitter, flip, scale 등)은 ultralytics 학습 파이프라인에 기본 내장되어 있어 별도 코드 없이 적용됨
- 조명 변화 대응력을 더 높이고 싶으면 `--hsv-h/--hsv-s/--hsv-v` 등 augmentation 하이퍼파라미터를 조정 가능 (`yolo cfg` 문서 참고)
- 학습 결과(가중치, 로그, 그래프)는 `runs/detect/puck_knob_v1/` 에 저장됨
  - 주의: 같은 이름의 폴더가 이미 있으면(예: 이전에 중단된 학습) ultralytics가 자동으로 `puck_knob_v1-2`, `puck_knob_v1-3`처럼 뒤에 번호를 붙여 새 폴더를 만듭니다. 아래 예시 경로 그대로 복붙하지 말고, 학습이 끝나면 터미널에 출력되는 실제 저장 경로(또는 `ls runs/detect/`)를 먼저 확인하세요. (현재 이 프로젝트의 최신 학습 결과는 `runs/detect/puck_knob_v1-2/weights/best.pt` 입니다.)

## 4. 검증 / 추론 테스트 (노트북에서)

```bash
yolo detect val model=runs/detect/puck_knob_v1-2/weights/best.pt data=data.yaml
yolo detect predict model=runs/detect/puck_knob_v1-2/weights/best.pt source=dataset/images/test
```

## 5. 라즈베리파이 배포

1. 노트북(x86_64)에서 학습된 모델을 NCNN으로 변환:

   ```bash
   ./scripts/export_ncnn.sh runs/detect/puck_knob_v1-2/weights/best.pt
   ```

2. 변환된 `best_ncnn_model` 폴더를 라즈베리파이로 복사(scp 등):

   ```bash
   scp -r runs/detect/puck_knob_v1-2/weights/best_ncnn_model \
       pi@<라즈베리파이IP>:~/craft/custom_yolo26/models/
   ```

3. 라즈베리파이에서 `setup_env.sh`로 추론 환경 설정 후 실행:

   ```bash
   python3 scripts/predict_webcam.py --model models/best_ncnn_model --source /dev/realsense_color
   ```

- Ultralytics 공식 벤치마크 기준(YOLO26n, RPi 5) NCNN이 ONNX/MNN보다 유의미하게 빠름(약 67ms vs 92~126ms/이미지)
- 32bit OS(armv7l)는 지원 범위 밖 — 64bit Raspberry Pi OS(Bookworm 이상) 필요

## 6. C++ 추론

Python(`ultralytics`)뿐 아니라 export된 모델 파일 그대로 C++에서도 추론할 수 있습니다
(export를 다시 할 필요 없음 - 같은 파일을 언어만 바꿔서 로드). 로봇 제어 노드가
ROS2 C++ 기반이거나 Python 오버헤드를 줄이고 싶을 때 사용하세요.

- `cpp/onnx_infer/` — ONNX Runtime 기반, **완성됨**. `export_onnx.sh`로 만든 `.onnx`를 그대로 사용
- `cpp/ncnn_infer/` — NCNN 기반, **스텁 상태**. 실제 모델 export 후 출력 스펙 확인하고 채워야 함

자세한 빌드/실행 방법은 `cpp/README.md` 참고.

## TODO / 다음 단계

- [ ] 데이터 수집량 목표 설정 (클래스당 수백~1000장 권장 범위에서 시작)
- [ ] 라벨링 완료 후 `split_dataset.py` 실행
- [ ] 1차 학습 후 val 성능 확인, 오탐/미탐 케이스 위주로 데이터 보강
- [ ] 라즈베리파이 실물에서 NCNN 모델 추론 속도 실측
