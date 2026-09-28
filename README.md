# custom_yolo26

## Shortcut · 명령어 모음

**저장소 루트에서 실행하세요.** 최초 설치 후 새 터미널마다 `source .venv/bin/activate`로 환경을 활성화합니다.
아래 명령은 필요한 작업만 선택해서 실행하며, `knob_sign_v2` 모델 경로는 실제 학습 결과 경로로 바꾸세요.

| 작업 | 명령어 |
|---|---|
| 최초 환경 설치 | `bash scripts/setup_env.sh` |
| 가상환경 활성화 | `source .venv/bin/activate` |
| 영상 촬영 | `python3 scripts/record_video.py --source /dev/realsense_color` |
| 프레임 추출 (기본 5프레임마다) | `bash scripts/extract_frames.sh` |
| 라벨링 완료 후 빈 라벨 생성 | `python3 scripts/create_empty_labels.py` |
| 데이터 정리 및 train/val 분할 | `bash scripts/split_dataset.sh` |
| 학습 (.pt 목록 표시 후 입력) | `python3 scripts/train.py --epochs 100 --imgsz 640 --batch 16 --name knob_sign_v2` |
| 모델 지정 학습 | `python3 scripts/train.py --model yolo26n.pt --epochs 100 --name knob_sign_v2` |
| 검증 | `yolo detect val model=runs/detect/knob_sign_v2/weights/best.pt data=data.yaml` |
| 검증 영상 추론 결과 생성 | `yolo detect predict model=runs/detect/knob_sign_v2/weights/best.pt source=val_video project=runs/detect name=val_video_knob_sign_v2_best` |
| 기존 추론 영상 + 원본 합치기 | `python3 scripts/predict_video.py` |
| 비교 영상 다시 만들기 | `python3 scripts/predict_video.py --overwrite` |
| 카메라 실시간 추론 | `python3 scripts/predict_webcam.py --model runs/detect/knob_sign_v2/weights/best.pt --source /dev/realsense_color` |
| NCNN 변환 | `bash scripts/export_ncnn.sh runs/detect/knob_sign_v2/weights/best.pt` |
| ONNX 변환 | `bash scripts/export_onnx.sh runs/detect/knob_sign_v2/weights/best.pt` |

추출 전 `raw_videos`를 클래스별 폴더로 정리하고, 분할 전 이미지와 같은 폴더에 동명의 `.txt` 라벨을 준비하세요.
`split_dataset.sh`는 기존 분할을 자동 백업한 뒤 재생성합니다. 비교 영상 합치기는 기존 추론 결과를 사용합니다.

**상세 설명 바로가기:** [환경 설치](#quick-setup) · [촬영/추출](#quick-extract) · [라벨링](#quick-label) · [분할](#quick-split) · [학습](#quick-train) · [검증](#quick-val) · [영상 합치기](#quick-compare) · [배포](#quick-deploy)

---

Waffle Pi 형태 이동로봇 - knob, sign_red, sign_yellow, sign_green, puck_red, puck_green, puck_blue 인식용 YOLO custom dataset 학습 프로젝트.

RGB 기반, **pretrained YOLO + custom dataset fine-tuning(transfer learning)** 방식으로 진행합니다.
IR 스트림은 색상 구분(신호등 미션 포함)이 필요해서 사용하지 않음.

**학습은 노트북(x86_64)에서, 추론은 라즈베리파이(ARM)에서** 돌리는 구조입니다.
YOLO 계열은 라즈베리파이에서의 학습을 권장하지 않고(Ultralytics 공식 가이드도 RPi는
inference 전용으로 다룸), 학습된 모델을 NCNN 포맷으로 변환해서 배포하는 흐름을 씁니다.

## 모델 선택: YOLO26

Ultralytics의 최신(2026) 모델인 **YOLO26**을 추천합니다.

- YOLO11 대비 CPU 추론 속도가 크게 향상되어(공식 벤치마크 기준 최대 43% 개선) 로봇 온보드 컴퓨트에서 유리
- NMS-free 구조라 후처리가 단순해져 임베디드 배포에 적합
- Small-Target-Aware Label Assignment(STAL) 도입으로 knob, 신호등, puck처럼 작게 찍히는 물체 탐지에 강함
- 학습 API는 이전 버전(YOLOv8/YOLO11)과 동일한 `ultralytics` 패키지로 그대로 사용 가능

가볍게 시작하려면 `yolo26n.pt`(nano), 정확도를 더 올리고 싶으면 `yolo26s.pt`(small)부터 시도해보세요.
로봇 온보드에서 실시간으로 돌려야 하니 n/s 사이즈를 우선 권장합니다.

## 폴더 구조

```
custom_yolo26/
├── data.yaml               # 현재 학습 클래스 정의 + 데이터 경로
├── requirements.txt
├── dataset/
│   ├── raw_videos/{knob,sign_red,sign_yellow,sign_green}/ # 클래스별 영상
│   ├── raw/{knob,sign_red,sign_yellow,sign_green}/        # 추출 이미지 + 같은 이름의 txt
│   ├── images/{train,val}/<클래스폴더>/
│   └── labels/{train,val}/<클래스폴더>/   # YOLO txt 포맷 라벨
├── models/                  # pretrained/학습된 가중치(.pt), NCNN 변환 모델 보관
├── scripts/
│   ├── setup_env.sh         # 아키텍처 자동 감지 환경 설정 (x86_64=학습용 / aarch64=추론용)
│   ├── record_video.py      # 데이터셋용 영상 촬영 (SPACE로 녹화 시작/정지)
│   ├── extract_frames.py    # 영상 하위 폴더 구조를 유지하며 프레임 추출
│   ├── extract_frames.sh    # 기본 경로로 프레임 추출 실행
│   ├── split_dataset.py     # 폴더별 매 9번째 쌍은 val, 나머지는 train
│   ├── split_dataset.sh     # 기본 경로로 데이터 분할 실행
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

## 명령 실행 위치

아래 명령은 모두 **이 저장소의 루트**(`README.md`, `requirements.txt`, `scripts/`가 있는 폴더)에서 실행합니다.
VS Code에서 이 저장소 폴더를 열고 터미널의 현재 위치를 확인하세요.
저장소는 원하는 위치에 clone해도 되며, 노트북과 라즈베리파이의 경로가 같을 필요는 없습니다.
`.venv`는 다른 장치로 복사하거나 저장소 이동 후 재사용하지 말고, 각 위치에서 다시 생성하세요.

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

<a id="quick-setup"></a>

## 1. 환경 설치

`setup_env.sh`가 `uname -m`으로 아키텍처를 감지해서 알아서 역할을 나눕니다.
**노트북(x86_64)과 라즈베리파이(aarch64) 양쪽 모두에서 그대로 실행**하면 됩니다.

```bash
bash scripts/setup_env.sh
```

- x86_64(노트북): venv 생성 + `requirements.txt` 설치 + GPU(CUDA) 감지
- aarch64(라즈베리파이): 시스템의 `python3-pip`, `python3-venv` 설치 + venv 생성 + `ultralytics`, `ncnn` 설치

스크립트는 `.venv/bin/python -m pip`로 해당 환경에 패키지를 설치합니다.
**설치가 성공한 다음, 현재 Bash 터미널에서 아래 블록을 별도로 실행하세요.**

```bash
source .venv/bin/activate
python -c 'import sys; print(sys.executable); print("venv:", sys.prefix != sys.base_prefix)'
```

출력 경로가 이 저장소의 `.venv/bin/python`이고 `venv: True`이면 적용된 것입니다.
`bash scripts/setup_env.sh`는 자식 셸에서 실행되므로, 그 안에서 `source`를 실행해도
부모인 현재 터미널에는 적용되지 않습니다. 새 터미널을 열 때마다 활성화가 필요합니다.
설치 스크립트는 Bash용이므로 `sh`가 아닌 위의 `bash` 명령으로 실행하세요.

VS Code의 Python 실행은 **Python: Select Interpreter**에서 이 저장소의
`.venv/bin/python`을 선택하세요. Jupyter/노트북 셀은 별도로 해당 venv를 커널로 선택해야 합니다
(커널 실행에 `ipykernel`이 필요하면 `.venv/bin/python -m pip install ipykernel`로 설치).
셀의 `!bash ...` 또는 `!source ...`는 이미 실행 중인 Python 커널을 바꾸지 않습니다.

터미널 활성화 없이 실행하려면 Python 경로를 직접 지정할 수도 있습니다:

```bash
.venv/bin/python scripts/predict_webcam.py --model yolo26n.pt --source /dev/realsense_color
```

설치 도중 `ensurepip is not available` 또는 venv 생성 오류가 발생하면,
Ubuntu/Debian에서는 `sudo apt install python3-venv`로 venv 지원을 설치한 뒤 스크립트를 다시 실행하세요.
패키지 다운로드나 설치가 실패했다면 활성화 여부와는 별개의 오류이므로, 설치 로그의 오류를 먼저 해결해야 합니다.

## (선택) Pretrained 그대로 파이프라인 점검

custom dataset 라벨링 전에, 환경/카메라 파이프라인이 제대로 도는지 먼저 확인하고 싶으면
pretrained 가중치(yolo26n.pt) 그대로 실행해볼 수 있습니다. COCO 80개 클래스 기준이라
knob, 신호등, puck을 정확히 맞추진 못하지만, 카메라 연결/추론 자체가 도는지 점검하기엔 충분합니다.

`yolo predict model=... source=/dev/realsense_color` 처럼 CLI에 장치 경로를 바로 넘기면
ultralytics가 이를 webcam으로 인식하지 못해 실패할 수 있어서, `predict_webcam.py`가
`cv2.VideoCapture`로 직접 장치를 열도록 만들어뒀습니다. RealSense udev symlink
(`setup_realsense_udev_symlink.sh`로 고정해둔 경로)를 그대로 사용하세요.

```bash
python3 scripts/predict_webcam.py --model yolo26n.pt --source /dev/realsense_color
```

## 2. 데이터 수집 & 라벨링

<a id="quick-extract"></a>

### 2-1. 영상 촬영 + 프레임 추출

한 장씩 사진 찍는 것보다, knob, 신호등, puck을 들고 다양한 각도·거리로 천천히 움직이며
영상으로 쭉 찍은 뒤 프레임을 솎아내는 게 훨씬 빠르고 각도/거리 다양성도 자연스럽게
확보됩니다.

**촬영 방식 (체크리스트)**

- 클래스/상황별로 클립을 나눠서 촬영 (knob만, 신호등만, puck만, 배경만, 여러 개 겹친 상황 등) —
  나중에 뭘 찍은 영상인지 헷갈리지 않게, 촬영 세션마다 파일명이나 메모로 구분해두기
- 각도: 물체를 들고 천천히 360도 돌려가며 / 카메라를 정면·측면·위에서 내려다보는 각도로 각각 촬영
- 거리: 로봇이 처음 인식하는 먼 거리(~1m)부터 그리퍼로 집기 직전 가까운 거리(10~20cm)까지 다양하게
- 조명: 실내조명 on/off, 그림자 있는/없는 상태 등 대회 환경과 비슷한 조명 변화를 섞어서 촬영
- 배경만 나오는 클립(물체 없음)도 최소 1개 이상 — false positive 방지용 negative sample
- 일부러 여러 물체(knob, 신호등, puck)를 겹치거나 일부만 보이게 가리는 장면도 조금 섞기 — 실전 가림 상황에 강해짐
- 카메라/손을 너무 빨리 움직이면 모션 블러로 프레임이 흐려져서 라벨링하기 어려우니 천천히 이동

**명령어**

```bash
source .venv/bin/activate

# 1) 영상 촬영 (SPACE로 녹화 시작/정지, 여러 번 반복해서 여러 클립 촬영 가능, q/ESC로 종료)
python3 scripts/record_video.py --source /dev/realsense_color
# -> dataset/raw_videos/clip_<타임스탬프>_<번호>.mp4 로 저장됨

# 2) 촬영된 영상을 raw_videos/knob, sign_red, sign_yellow, sign_green 등으로 정리한 뒤 추출
bash scripts/extract_frames.sh
# 기본: 5프레임마다 1장 (30fps 영상 기준 초당 6장)
# raw_videos/sign_yellow/clip.mp4 -> raw/sign_yellow/clip_f000000.jpg

# 추출 간격 변경 예시
# bash scripts/extract_frames.sh --every-n-frames 10
```

- `--video-dir`는 하위 폴더까지 재귀 탐색하고, `raw`에도 같은 상대 폴더 구조를 만듭니다.
- 위 셸 명령은 `.venv/bin/python`으로 다음 Python 명령을 실행합니다:

```bash
python3 scripts/extract_frames.py --video-dir dataset/raw_videos --outdir dataset/raw --every-n-frames 5
```

- 특정 영상 하나만 추출할 때는 출력할 클래스 폴더를 직접 지정합니다:

```bash
python3 scripts/extract_frames.py --video dataset/raw_videos/sign_yellow/clip.mp4 --outdir dataset/raw/sign_yellow --every-n-frames 5
```

- 기존 라벨링 데이터의 폴더만 다시 정리하려면 아래 명령을 실행합니다. 영상 이름과 `_f프레임번호`를 대조해 이미지와 `.txt`를 함께 이동하며, 내용을 다시 생성하거나 라벨 번호를 바꾸지 않습니다. 매칭 실패·중복 영상 이름·파일 충돌은 이동 전에 중단합니다.

```bash
python3 scripts/organize_raw.py
```

- 새 영상의 프레임 생성은 `extract_frames.sh`, 기존 이미지·라벨의 폴더 정리는 `organize_raw.py`가 담당합니다. 기존 라벨을 재사용할 때는 프레임을 다시 추출할 필요가 없습니다.

<a id="quick-label"></a>

### 2-2. 라벨링

- 라벨링 툴: [CVAT](https://www.cvat.ai/), [LabelImg](https://github.com/heartexlab/labelImg), [Roboflow](https://roboflow.com/) 등에서 `dataset/raw`의 이미지를 불러와 YOLO 포맷으로 export
  - YOLO 라벨 포맷 한 줄: `<class_id> <x_center> <y_center> <width> <height>` (모두 0~1 정규화, 이미지 크기 기준)
  - 허용 클래스는 아래 7개만 사용하며, 번호와 이름을 고정합니다.

    | class_id | 이름 |
    |---|---|
    | 0 | knob |
    | 1 | sign_red |
    | 2 | sign_yellow |
    | 3 | sign_green |
    | 4 | puck_red |
    | 5 | puck_green |
    | 6 | puck_blue |

  - 기존 랜덤 분할 데이터는 `0~3`의 4개 클래스입니다 (train 3,908장 / val 488장 / test 489장).
    `data.yaml`도 현재는 이 4개 클래스를 정의합니다. puck 데이터 추가 시 기존 번호를 유지하고
    `4: puck_red`, `5: puck_green`, `6: puck_blue`를 `names`에 추가한 뒤 새로 학습하세요.
  - 라벨링 도구의 export 순서도 위 번호와 일치시켜야 합니다. 배경 이미지는 별도 클래스를 만들지 않고 빈 라벨로 둡니다.
- 이미지와 같은 이름의 `.txt`를 **같은 클래스 하위 폴더**에 저장합니다. 예: `raw/sign_yellow/clip_f000000.jpg`와 `raw/sign_yellow/clip_f000000.txt`.
- 폴더 이름은 정리 기준이며 라벨의 class_id를 자동 지정하지 않습니다. 각 이미지에 보이는 대상은 클래스 번호에 맞게 모두 라벨링하세요.
- 라벨 파일이 없는 이미지는 분할에서 제외됩니다. 배경 이미지는 빈 `.txt` 파일을 만들어 포함하세요.

#### makesense.ai export 후 빈 라벨 생성

makesense.ai에서 라벨을 export하고 이미지 옆에 배치한 뒤, 탐지 대상이 없는 이미지의 빈 `.txt`를 다음 명령으로 보완합니다.
**선택한 폴더의 모든 이미지에 대한 라벨링과 export 배치를 완료한 뒤 실행하세요.** 이 스크립트는 라벨링 미완료 이미지와 배경 이미지를 구분하지 않으며, `.txt`가 없으면 모두 배경 라벨을 만듭니다.

```bash
# 생성 대상 확인 (파일 변경 없음)
python3 scripts/create_empty_labels.py --dry-run

# dataset/raw 하위 폴더 전체에서 누락된 .txt 생성
python3 scripts/create_empty_labels.py

# 이후 이미지와 라벨을 함께 train/val로 분할
bash scripts/split_dataset.sh
```

일부 폴더만 라벨링을 마쳤다면 `python3 scripts/create_empty_labels.py --src dataset/raw/sign_yellow`처럼 범위를 지정하세요.
`.jpg`, `.jpeg`, `.png`를 재귀 탐색하며 기존 `.txt`는 내용이 있거나 비어 있거나 그대로 유지합니다.
예: `raw/sign_yellow/clip_f000000.jpg`에 라벨이 없으면 같은 위치에 0바이트 `clip_f000000.txt`를 생성합니다.
이렇게 만든 이미지·빈 라벨 쌍도 분할과 학습에 포함됩니다. 탐지할 물체가 있는 이미지는 먼저 해당 박스를 라벨링해야 합니다.

<a id="quick-split"></a>

### 2-3. train/val 분할

각 하위 폴더에서 라벨이 있는 이미지 쌍을 **파일명 순서로 정렬**하고, **9번째·18번째·27번째…는 val**, 나머지는 train에 복사합니다. 폴더마다 순번을 다시 셉니다. 9장 미만인 폴더에서는 val이 생기지 않습니다.
랜덤 분할이 아니며 `--train`, `--val`, `--test`, `--seed` 옵션은 더 이상 사용하지 않습니다. 새 test 데이터는 만들지 않습니다.

재분할 명령은 **`raw_videos`의 영상 이름 기준으로 기존 `raw` 이미지·라벨 정리 → 폴더별 train/val 생성 → 기존 출력 백업 후 교체**를 한 번에 수행합니다. 새 프레임 추출이나 라벨 생성은 하지 않습니다. 실행 중인 학습을 마친 뒤 사용하세요.

```bash
bash scripts/split_dataset.sh
# 검증 간격 변경: bash scripts/split_dataset.sh --val-every 9
```

기존 images/labels와 캐시는 `dataset/.split_backups/split-*/`에 보관하며 Git에서 제외됩니다. 새 분할 생성에 성공한 뒤 기존 출력을 교체합니다.

raw 정리 없이 이미 정리된 원본을 빈 출력 위치로 분할하는 기존 Python 명령도 사용할 수 있습니다:

```bash
python3 scripts/split_dataset.py --src dataset/raw --dst dataset_resplit --val-every 9
```

출력 예: `dataset/images/val/sign_yellow/clip_f000120.jpg`와 `dataset/labels/val/sign_yellow/clip_f000120.txt`.
백업 폴더는 보관용이며 학습 입력이나 새 커밋에 포함하지 마세요.

별도 빈 위치에 미리 분할하려면 `bash scripts/split_dataset.sh --dst dataset_resplit`을 사용합니다.
그 결과로 학습하려면 별도 YAML에서 `path`를 해당 폴더의 **절대경로**로 지정하고 `--data`로 전달하세요. 기본 `data.yaml`은 계속 `dataset`을 사용합니다.

현재 `data.yaml`의 `train: images/train`, `val: images/val`은 그대로 사용합니다.
새 분할에는 test가 없으므로 `test: images/test` 항목은 제거하거나 주석 처리하세요. 별도 test를 수집한 경우에만 해당 경로를 설정합니다.
같은 영상의 인접 프레임이 train과 val에 들어가므로, 최종 성능은 별도로 촬영한 영상에서도 확인하세요.

<a id="quick-train"></a>

## 3. 학습(Fine-tuning)

```bash
source .venv/bin/activate
python3 scripts/train.py --epochs 100 --imgsz 640 --batch 16 --name knob_sign_v2
```

실행하면 프로젝트 안의 `.pt` 파일 경로 목록을 먼저 표시하고, **학습 시작 가중치**의 파일명 또는 경로를 입력받습니다.
Enter만 누르면 `yolo26n.pt`를 사용합니다. 파일명이 중복되는 `best.pt` 등은 목록에 표시된 프로젝트 기준 상대 경로 또는 절대경로를 입력하세요.
로컬에 없는 기본 YOLO26 모델(`yolo26n/s/m/l/x.pt`)은 Ultralytics가 다운로드합니다.

입력 절차 없이 모델을 지정하려면:

```bash
python3 scripts/train.py --model yolo26n.pt --epochs 100 --imgsz 640 --batch 16 --name knob_sign_v2

# 기존 학습 가중치를 시작점으로 새로 fine-tuning하는 예시
# python3 scripts/train.py --model runs/detect/knob_sign_v2/weights/best.pt --epochs 100 --name knob_sign_v3
```

- `--model`은 시작 가중치이고, `--name`은 학습 결과 폴더 이름입니다. 기존 `.pt`를 선택해도 중단된 학습을 resume하는 방식은 아닙니다.
- `--data` 기본값은 프로젝트의 `data.yaml`입니다. 실제 라벨 번호와 `names` 정의를 일치시키세요.
- GPU 지정은 `--device 0`, CPU 지정은 `--device cpu`를 추가합니다.
- 위 예시 결과는 `runs/detect/knob_sign_v2/`에 저장됩니다. 같은 이름이 있으면 새 이름이 자동 부여될 수 있으므로 터미널에 표시된 실제 저장 경로를 확인하세요.
- 최종 추론에 사용할 가중치는 결과 폴더의 `weights/best.pt`입니다. 입력한 `.pt`의 이름이 결과 파일명으로 사용되지는 않습니다.
- 배포 장치에는 학습 데이터를 복사할 필요가 없습니다. Python 추론에는 학습된 `.pt`를 사용하고, 아래 NCNN 배포 절차를 사용할 때는 변환된 모델 폴더를 가져갑니다.

<a id="quick-val"></a>

## 4. 검증 / 추론 테스트 (노트북에서)

아래 모델 경로는 실제 학습 출력 경로로 바꾸세요. 새 분할은 test를 생성하지 않으므로 val 이미지로 확인하는 예시입니다.

```bash
yolo detect val model=runs/detect/knob_sign_v2/weights/best.pt data=data.yaml
yolo detect predict model=runs/detect/knob_sign_v2/weights/best.pt source=dataset/images/val
```

<a id="quick-compare"></a>

### 검증 영상: 원본과 추론 결과 동시 재생

```bash
# runs/detect 아래 val_video* 폴더의 기존 추론 영상에 원본 붙이기
python3 scripts/predict_video.py

# 특정 결과 폴더 또는 영상만 처리
python3 scripts/predict_video.py --prediction runs/detect/val_video_puck_knob_folder_split_best
```

재추론 없이 기존 영상 두 개를 합칩니다. `val_video`에서 확장자를 제외한 파일명이 같은 원본을 찾아,
왼쪽에 원본·오른쪽에 추론 결과를 배치한 `<영상이름>_comparison.mp4`를 추론 결과와 같은 폴더에 저장합니다.
따라서 기존 `val_video_<pt이름>` 결과 폴더 이름이 유지됩니다. `.pt` 파일이나 GPU는 필요하지 않습니다.
원본과 추론 영상은 첫 프레임부터 대응하는 전체 영상이어야 하며 FPS·프레임 수·해상도가 다르면 중단합니다.
원본 FPS를 유지하고 음성은 포함하지 않습니다.
기존 `_comparison.mp4`는 입력에서 제외하고, 출력이 이미 있으면 건너뜁니다. 재생성하려면 `--overwrite`를 추가하세요.
원본 위치를 바꾸려면 `--original-dir 경로`를 지정합니다. 동명 원본이 여러 개면 해당 원본의 하위 폴더를 지정하세요.

<a id="quick-deploy"></a>

## 5. 라즈베리파이 배포

1. 노트북(x86_64)에서 학습된 모델을 NCNN으로 변환:

   ```bash
   ./scripts/export_ncnn.sh runs/detect/puck_knob_v1-2/weights/best.pt
   ```

2. 라즈베리파이에 저장소를 원하는 위치로 clone하고, 그 저장소 루트에서
   `mkdir -p models`와 `pwd`를 실행합니다. 노트북에서는 아래 두 변수에
   실제 접속 정보와 방금 확인한 라즈베리파이의 절대경로를 넣어 복사하세요:

   ```bash
   PI_HOST='사용자@라즈베리파이IP'
   PI_PROJECT_DIR='/라즈베리파이에서/pwd로/확인한/저장소경로'
   scp -r runs/detect/puck_knob_v1-2/weights/best_ncnn_model \
       "${PI_HOST}:${PI_PROJECT_DIR}/models/"
   ```

3. 라즈베리파이의 저장소 루트에서 추론 환경을 설치하고 활성화한 뒤 실행:

   ```bash
   bash scripts/setup_env.sh  # 최초 설치 시
   source .venv/bin/activate
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
- [x] knob 및 신호등 3색 라벨링 완료, train/val/test 분할
- [ ] puck_red, puck_green, puck_blue 데이터 수집·라벨링 및 설정 확장
- [ ] 1차 학습 후 val 성능 확인, 오탐/미탐 케이스 위주로 데이터 보강
- [ ] 라즈베리파이 실물에서 NCNN 모델 추론 속도 실측
