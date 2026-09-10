# UOD: BFE / IFE / NFE / IFD 전체 학습·추론 코드

## 먼저 구분할 점

이 프로젝트는 Fan et al., **Unifying RGB and thermal object detection in one detector**, Pattern Recognition 179 (2026), 113902의 **Section 3에 제시된 공유 가중치 구조와 특징 분리 손실을 구현한 연구용 reference implementation**입니다.

**공식 구현이나 UOD-v5 / UOD-v8의 동일 재현물이 아닙니다.** 논문에서는 BFE / IFE / IFD를 YOLOv5의 Backbone / Neck / Head에 대응시킵니다(4.2절). 기본 설정은 내부를 직접 수정하기 쉬운 **torchvision ResNet + 단순 PAFPN + decoupled anchor-free head**이고, 별도 설정으로 **YOLOv5-style CSPDarknet + concat/C3 PAFPN**도 제공합니다. 두 경우 모두 공식 YOLO 학습 코드/anchor loss/Detect head와 동일하지 않으므로 공식 YOLO 성능표와 같은 설정의 결과로 비교하면 안 됩니다.

논문에 명시된 부분과 코드를 동작시키기 위해 추가한 부분을 아래에서 분리합니다. 이 프로젝트에는 원본 PDF, 학습된 가중치, 비공개 데이터가 포함되지 않습니다.

## 1. 구현 구조

| 역할 | 이번 코드의 실제 구현 | RGB/Thermal 사이 가중치 |
|---|---|---|
| BFE | ResNet-50/18 또는 YOLOv5n/s/m/l-style CSPDarknet. C3/C4/C5 출력 | 하나의 모듈을 공유 |
| IFE | `simple_pafpn` 또는 `yolov5_pafpn`; FPN top-down + PAN bottom-up | 하나의 모듈을 공유 |
| NFE_RGB | IFE와 동일한 FeatureNeck 구조 | 독립 파라미터 |
| NFE_Thermal | IFE와 동일한 FeatureNeck 구조 | 독립 파라미터 |
| IFD | 서로 독립적인 classification tower / regression tower | 하나의 head를 두 모달리티가 공유 |

입력 640×640, ResNet-50, neck 채널 256인 경우:

```text
BFE C3: [B, 512, 80, 80]
BFE C4: [B, 1024, 40, 40]
BFE C5: [B, 2048, 20, 20]

IFE와 각 NFE 출력:
P3: [B, 256, 80, 80]
P4: [B, 256, 40, 40]
P5: [B, 256, 20, 20]
```

`BFE`, `FeatureNeck`, `YOLOv5PAFPN`, `IFD`, `UOD`, `UODDetector`는 `uod/model.py`에 있습니다.

### YOLOv5-style 옵션

`configs/uod_yolov5s.yaml`은 YOLOv5 계열에 더 가까운 실험용 설정입니다.

```yaml
model:
  backbone: yolov5s
  neck: yolov5_pafpn
  pretrained: false
  normalization: none
```

이 경로는 YOLOv5의 CSP/C3/SPPF backbone과 concat+C3 PAN-FPN 흐름을 코드 안에 직접 구현합니다. 다만 출력 feature channel을 공통 `channels`로 맞춰 기존 decoupled IFD를 그대로 사용하므로, Ultralytics YOLOv5의 anchor-based `Detect` head와 loss를 그대로 가져온 것은 아닙니다.

### 채널 복제와 가중치 공유는 서로 다른 일

Thermal `[B,1,H,W]`를 `repeat(1,3,1,1)`로 `[B,3,H,W]`로 만듭니다. 이후 RGB와 동일한 normalization을 적용합니다. ImageNet normalization을 선택하면 두 모달리티 모두 같은 mean/std를 사용합니다.

**채널 복제만으로 가중치가 공유되는 것은 아닙니다.** `self.bfe`, `self.ife`, `self.ifd` 객체가 각각 하나만 존재합니다. RGB용/thermal용으로 복사한 두 모델의 초기값만 맞추는 방식이 아닙니다.

학습 forward에서는 RGB B장과 thermal B장을 배치 축으로 합쳐 2B장을 공유 경로에 한 번 통과시킵니다. 채널 축에 붙이는 early fusion이 아닙니다. 이 설계는 공유 BatchNorm이 두 모달리티를 같은 batch에서 보도록 합니다. mixed-batch 실행은 이 구현의 선택이며 논문의 별도 필수 구현 규칙은 아닙니다.

### NFE는 실제 뺄셈 장치가 아닙니다

`intrinsic = backbone - non_discriminative`를 계산하지 않습니다. 같은 BFE 출력이 IFE와 NFE에 각각 들어가고 손실로 출력을 분리합니다. 원문 식 (1)은 개념적인 분해이며, 실제 뺄셈이나 재구성 손실이 요구되지 않습니다.

기본 `nfe_init: perturbed_copy`는 IFE 값을 독립 NFE에 복사한 뒤 Conv 가중치에 작은 서로 다른 잡음을 더합니다. **파라미터 저장 공간은 공유하지 않습니다.** 초기 특징이 완전히 같아 제곱거리 미분이 0인 상태를 피하려는 구현 선택입니다. 독립 무작위 초기화는 `nfe_init: independent`로 설정할 수 있습니다. 논문은 이 초기화 방식을 명시하지 않습니다.

## 2. 논문 특징 대조 손실

`uod/losses.py`의 `FeatureSeparationLoss`는 식 (10)~(12)의 항을 구현합니다.

각 공간 위치의 채널 벡터를 L2 정규화한 뒤:

```python
d_pos = (intrinsic_rgb - intrinsic_thermal).square().sum(dim=1)
d_r   = (intrinsic_rgb - non_rgb).square().sum(dim=1)
d_t   = (intrinsic_thermal - non_thermal).square().sum(dim=1)
d_n   = (non_rgb - non_thermal).square().sum(dim=1)

L_C_RGB = mean(d_pos) + mean(relu(margin - d_r))
L_C_T   = mean(d_pos) + mean(relu(margin - d_t))
L_N     = mean(relu(margin - d_n))
L_FS    = L_C_RGB + L_C_T + L_N
```

중요한 세부사항:

- 일반 `TripletMarginLoss`로 대체하지 않습니다. 원문은 positive 항과 negative hinge 항이 분리되어 있습니다.
- `d_pos`는 RGB 손실과 thermal 손실에 각각 들어가므로 최종 합에서 **두 번** 더해집니다.
- `margin=1`은 **제곱 거리**에 적용됩니다. 단위 정규화 벡터의 제곱 거리 최대값은 4입니다.
- 특징맵 전체를 flatten해서 하나의 벡터로 normalize하지 않습니다. `[B,C,H,W]`의 `dim=1`에서 normalize합니다.
- 거리는 채널 방향 **합**입니다. 단순히 전체 원소 MSE를 평균하면 채널 수만큼 손실 스케일이 달라집니다.
- Feature loss에는 `.detach()`를 넣지 않습니다. BFE, IFE, NFE 양쪽으로 gradient가 흐릅니다. Detection head 입력 자체는 L2 정규화하지 않습니다.
- 배경까지 포함한 공간 위치 대조가 기본입니다. 객체 박스만 잘라 대조하는 ROI/instance loss를 임의로 추가하지 않았습니다.

원문에서 다중 해상도 특징의 평균 규칙과 N의 구체적인 집계가 충분히 명시되지 않아 기본값은 **전체 pyramid의 B×H×W 위치 수에 따른 평균(`reduction: pixel`)**으로 정했습니다. 레벨별 동일 가중 평균은 `reduction: level`로 선택할 수 있습니다.

`mask_padding: false`가 기본입니다. letterbox padding을 대조에서 제외하는 선택적 변형은 `true`로 설정합니다. 이 옵션은 원문 명시 사항이 아닌 추가 기능입니다.

### Intrinsic positive는 “같은 클래스의 아무 이미지”가 아닙니다

같은 시점/장면의 정렬된 RGB와 thermal에서 대응하는 공간 위치를 비교합니다. RGB의 사람 A와 다른 장면의 사람 B를 같은 클래스라는 이유만으로 positive로 묶지 않습니다. 픽셀 수준 정렬이 잘못되면 배경과 객체를 가깝게 학습하는 문제가 생깁니다.

### NFE에 별도의 잡음 정답은 없습니다

`RGB noise=0`, `thermal noise=1` 같은 분류 라벨을 부여하지 않습니다. NFE는 detection loss를 직접 받지 않고 위의 특징 분리 손실로 학습합니다. “색은 항상 불필요하고 온도는 항상 불필요하다”와 같은 수작업 규칙도 없습니다.

**거리 제약만으로 NFE의 객체 정보가 완전히 사라졌거나 상호정보량이 0임을 보장하지는 않습니다.** 이는 원문의 목표/가설에 해당하며, 실제 학습 결과는 별도의 feature probe와 실험으로 확인해야 합니다. 임의의 adversarial loss, GRL, reconstruction loss를 추가하지 않았습니다.

## 3. Detection head와 Detection loss

`IFD.cls_tower`와 `IFD.reg_tower`의 Conv 파라미터는 서로 독립입니다. 각 모달리티가 별도 head를 가지는 것은 아니며 **이 동일한 decoupled head를 공유**합니다.

Classification 출력은 foreground C개 + background 1개의 **C+1 softmax logits**입니다. Regression 출력은 각 grid center에서 박스 경계까지의 양수 l/t/r/b 거리입니다. softplus로 양수를 만들고 stride를 곱해 픽셀 좌표로 복원합니다. 별도 objectness/centerness branch는 두지 않았습니다.

최종 손실:

```text
L_D_RGB = CE_RGB + L1_RGB
L_D_T   = CE_T   + L1_T
L_total = L_D_RGB + L_D_T + alpha * L_FS
```

기본 `alpha=1`입니다. 이 코드는 원문의 CE+L1 형태를 사용하며 YOLOv8의 BCE+IoU+DFL 손실을 같은 것처럼 대체하지 않습니다.

다만 **식 (8)만으로 실행 가능한 dense detector의 모든 세부사항이 정해지지는 않습니다.** 다음은 이번 구현의 선택입니다.

| 추가 선택 | 기본값/규칙 |
|---|---|
| GT assignment | FCOS-inspired point assignment: 박스 내부 + center sampling + 크기 범위 |
| 박스 중첩 시 GT | 가장 작은 면적의 유효 GT |
| 해상도별 크기 범위 | `[0,64)`, `[64,128)`, `[128,∞)` |
| 배경 불균형 | background weight 0.1, hard-negative ratio 3, 최소 negative 32 |
| L1 좌표 단위 | `box_normalization: stride`; 각 점의 stride로 나눈 박스 좌표 |
| 정규화 대안 | `image`=입력 W/H로 나눔, `none`=픽셀 좌표 그대로 |
| batch reduction | 모달리티별 batch mean을 구한 뒤 RGB/Thermal 합 |

이 detector는 공식 FCOS 구현도 아닙니다. 해당 발상의 dense point assignment를 사용한 별도 기준 모델입니다. 원문 재현 실험에서는 이러한 선택을 반드시 공개해야 합니다.

너무 작은 GT는 stride 8의 점을 하나도 포함하지 않을 수 있습니다. `rgb/unmatched_gt_per_image`, `thermal/unmatched_gt_per_image`를 기록하므로 확인하세요. 작은 객체가 많으면 입력 크기, P2 추가, assignment 수정 등을 별도 실험해야 합니다.

## 4. 데이터 준비

### 입력 계약

RGB와 thermal은 **시간·공간적으로 정렬된 같은 장면**이어야 합니다. 본 코드의 shared label 파일은 두 이미지가 같은 좌표계와 GT를 사용한다는 전제입니다. 모달리티별 박스 좌표가 다르면 먼저 registration과 annotation 정리가 필요합니다.

동일 해상도인지 검사하지만, 동일 해상도가 정렬을 보증하지는 않습니다. calibration/registration 알고리즘은 포함하지 않았습니다.

```text
data/
  train.jsonl
  val.jsonl
  rgb/scene_0001.png
  thermal/scene_0001.png
  labels/scene_0001.txt
```

JSONL은 한 줄에 한 쌍입니다. 경로는 **해당 manifest 파일 위치 기준**입니다.

```json
{"id":"scene_0001","scene_id":"sequence_01","rgb":"rgb/scene_0001.png","thermal":"thermal/scene_0001.png","label":"labels/scene_0001.txt"}
```

YOLO 형식 라벨은 한 객체당 한 줄입니다.

```text
class_id center_x center_y width height
```

좌표는 원본 이미지에 대해 0~1 정규화합니다. 예를 들어:

```text
0 0.5000 0.4500 0.2000 0.4000
1 0.7500 0.6000 0.3000 0.2500
```

`class_names: [person, car]`면 0은 person, 1은 car입니다. 클래스 순서가 양쪽 모달리티, train/val, 체크포인트에서 같아야 합니다. 객체가 없는 이미지는 **빈 txt**를 만들고, 파일이 없는 경우를 negative 이미지로 묵인하지 않습니다.

`scene_id`는 선택 사항입니다. 사용할 때는 모든 sample에 제공해야 하며 train/val scene_id 중복도 검사합니다. scene_id를 생략하면 파일 경로 중복은 검사하지만 인접 프레임의 장면 누출까지 판별하지 못합니다.

### Thermal 범위

Thermal은 single-channel 8-bit/16-bit 이미지를 지원합니다. 단일 채널을 단순 복제한 3채널 파일도 허용하지만 false-color RGB thermal은 거부합니다.

- 8-bit 기본: `thermal_divisor: 255`
- full-range uint16 예: `65535`
- 14-bit 센서 범위 예: `16383`

실제 센서/파일의 유효 범위에 맞춰 설정해야 합니다. uint16을 uint8로 강제 변환하거나 이미지마다 min-max normalize하지 않습니다. 온도 단위 radiometric calibration을 수행하는 코드는 아닙니다.

### 동기화된 augmentation

두 이미지에 같은 letterbox와 같은 좌우 반전 결정을 적용하며 정답 박스와 valid mask도 함께 변환합니다. RGB brightness augmentation은 선택 사항이고 기본 0입니다. 서로 다른 crop, 독립 mosaic, 독립 random flip은 사용하지 않습니다.

## 5. 설치와 실행

CPU/CUDA 환경에 맞는 **서로 호환되는 torch/torchvision 빌드**를 먼저 준비하세요. 가능하면 기존 연구 프로젝트와 분리된 환경을 사용하세요.

```bash
cd uod_reference
python -m pip install -r requirements.txt
```

로컬 검증 환경은 Python 3.13.5 / torch 2.10.0+cpu / torchvision 0.25.0+cpu입니다. requirements의 모든 과거 버전 조합을 검증한 것은 아닙니다.

### 외부 데이터/가중치 없이 먼저 검사

```bash
python -m unittest discover -s tests -v
python smoke_test.py
```

`smoke_test.py`는 임시 toy 데이터 생성, 3 epoch 학습, epoch 중간이 아닌 **epoch 경계에서의 정확한 resume**, NFE-free export, export 전후 validation 비교, RGB/Thermal/RGBT 추론을 수행합니다. 결과는 `smoke_report.json`에 남기고 큰 임시 가중치는 삭제합니다. toy 이미지 결과는 실데이터 성능 증거가 아닙니다.

단계를 직접 실행하면서 파일을 남기려면:

```bash
python make_demo_data.py --output demo_data
python train.py --config configs/smoke.yaml --device cpu
```

CPU 스레드가 지나치게 많아 작은 테스트가 느린 Linux 환경에서는 앞에 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`를 붙여 실행할 수 있습니다.

### 실제 데이터 검사

`configs/uod_resnet50.yaml`의 manifest 경로와 class_names를 수정한 뒤:

```bash
python audit_data.py --config configs/uod_resnet50.yaml --output data_audit
python inspect_model.py --config configs/uod_resnet50.yaml --size 128
```

`data_audit/`의 paired preview에서 **같은 박스가 두 이미지의 같은 물체를 감싸는지** 확인하세요. 파일 검사만 통과했다고 정렬이 확인된 것은 아닙니다.

### LLVIP / M3FD manifest 준비

논문 성능과 비교하려면 논문이 사용한 데이터 버전, split, image size, epoch, backbone 크기, metric을 먼저 맞춰야 합니다. 이 저장소는 데이터 다운로드를 자동화하지 않으므로 LLVIP와 M3FD는 공식 배포처에서 받은 뒤 manifest만 생성합니다.

LLVIP처럼 split directory가 있는 경우:

```bash
python3 prepare_paired_manifest.py --root data/LLVIP \
  --rgb-dir visible --thermal-dir infrared --label-dir Annotations \
  --label-format voc --class-names person \
  --split train --split test --output data/llvip_uod
```

M3FD/TarDAL처럼 `meta/train.txt`, `meta/val.txt`가 있는 경우:

```bash
python3 prepare_paired_manifest.py --root data/m3fd \
  --rgb-dir vi --thermal-dir ir --label-dir labels \
  --split train:meta/train.txt --split val:meta/val.txt \
  --output data/m3fd_uod
```

그 다음 `configs/llvip_yolov5s.yaml` 또는 `configs/m3fd_yolov5s.yaml`의 `class_names`, `image_size`, `train_manifest`, `val_manifest`를 실제 논문 프로토콜에 맞춰 확인하세요.

### 학습

#### Docker + GPU 1 환경

이 프로젝트의 `compose.yaml`은 호스트의 물리 GPU 1만 컨테이너에 노출합니다. 컨테이너 안에서는 노출된 장치가 다시 0번부터 번호가 매겨지므로 학습 인자는 `cuda:0`이 맞습니다.

```bash
docker compose build uod-train

# GPU 격리와 PyTorch CUDA 연결 확인
docker compose run --rm uod-train python -c \
  "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"

# 실제 LLVIP 한 batch의 FP16 forward/backward/optimizer step 확인
docker compose run --rm uod-train \
  python gpu_smoke_test.py --config configs/llvip_yolov5s.yaml --device cuda:0

# LLVIP 학습
docker compose run --rm uod-train \
  python train.py --config configs/llvip_yolov5s.yaml --device cuda:0
```

현재 LLVIP manifest에는 이 작업공간의 절대 경로가 들어 있으므로 프로젝트를 컨테이너에서도 `/workspace/Hari/uod_reference`에 마운트합니다. 저장소 위치를 옮기면 `prepare_paired_manifest.py`로 manifest를 다시 생성해야 합니다. 체크포인트와 로그는 bind mount를 통해 호스트의 `runs/llvip_yolov5s/`에 그대로 남습니다.

```bash
python train.py --config configs/uod_resnet50.yaml --device cuda:0
```

YOLOv5-style CSPDarknet/PAFPN으로 실험하려면:

```bash
python train.py --config configs/uod_yolov5s.yaml --device cuda:0
```

`batch_size`는 이미지 수가 아닌 **pair 수**입니다. 기본 4면 공유 backbone을 통과하는 이미지는 8장입니다. OOM이면 pair batch를 줄이고 `accum_steps`를 늘리세요. 실제 메모리 사용량은 환경/입력 크기에 따라 달라집니다.

기본 ResNet 설정의 `pretrained: true`는 torchvision ImageNet backbone만 초기화합니다. 첫 실행에는 가중치 다운로드가 필요하며, neck/head/NFE가 사전 학습된 UOD 모델이라는 의미가 아닙니다. YOLOv5-style 경로에는 공식 YOLOv5 checkpoint loader를 연결하지 않았으므로 `pretrained: false`를 사용합니다.

일반 예제의 AMP는 기본 false이고, `configs/llvip_yolov5s.yaml`은 RTX GPU 학습을 위해 `amp: true`, `amp_dtype: float16`으로 설정했습니다. feature distance/CE/L1은 FP32로 계산합니다.

출력 파일:

```text
runs/uod_resnet50/
  config.resolved.json
  history.csv
  last.pt                 # model + NFE + optimizer + scheduler + scaler + RNG
  best.pt                 # RGB/Thermal AP50 평균 기준; GT가 있는 val 필요
  epoch_005.pt            # save_every마다 보존
  val_epoch_001.json
```

best selection은 **이번 구현의 선택**으로, RGB와 Thermal의 평균 AP50를 사용합니다. 논문에서 이 기준을 지정했다고 주장하지 않습니다.

### 이어서 학습

```bash
python train.py --config configs/uod_resnet50.yaml \
  --resume runs/uod_resnet50/last.pt --device cuda:0
```

동일한 모델 구조와 클래스 순서가 필요합니다. epoch 수를 바꾸면 남은 cosine LR schedule도 달라질 수 있습니다. GPU 연산의 bitwise 결정성은 보장하지 않습니다. 단일 CPU toy 실행의 epoch-boundary resume는 직접 비교했습니다.

### 검증

```bash
python evaluate.py --config configs/uod_resnet50.yaml \
  --checkpoint runs/uod_resnet50/best.pt --device cuda:0 \
  --output results/evaluation.json
```

같은 checkpoint로 RGB, thermal, RGBT를 각각 평가합니다. AP는 0~1 단위입니다.

포함된 AP evaluator는 일반적인 **fully-labeled, non-crowd boxes**에 대해 IoU 0.50~0.95, 101 recall samples, 클래스 평균을 계산합니다. COCO crowd/ignore-region/area-range/difficult-object 규칙은 구현하지 않았습니다. 논문 수치와 비교하거나 공식 benchmark를 보고할 때는 데이터셋 공식 평가기로 교체하세요.

### NFE ablation

NFE에 객체 탐지 정보가 남아 있는지 빠르게 확인하려면 **training checkpoint**로 실행합니다. deploy checkpoint에는 NFE가 제거되어 있어 사용할 수 없습니다.

```bash
python nfe_ablation.py --config configs/uod_resnet50.yaml \
  --checkpoint runs/uod_resnet50/best.pt --device cuda:0 \
  --output results/nfe_ablation.json
```

출력에는 정상 `BFE -> IFE -> IFD` 기준 AP와, `BFE -> NFE_RGB/NFE_Thermal -> IFD`로 바로 넣은 direct-head AP가 함께 저장됩니다. NFE AP가 높으면 NFE branch에 task-discriminative 정보가 많이 남아 있을 가능성이 있습니다. NFE AP가 낮아도 상호정보량이 0이라는 수학적 증명은 아니므로, 논문 보고용으로는 별도 frozen-feature probe나 adversarial probe를 추가해 검증하는 편이 좋습니다.

### NFE를 실제로 제외한 배포 파일

```bash
python export_deploy.py --checkpoint runs/uod_resnet50/best.pt \
  --output runs/uod_resnet50/deploy.pt
```

`deploy.pt`는 `bfe.*`, `ife.*`, `ifd.*`만 저장합니다. NFE와 optimizer state는 제거합니다. **`.eval()`만 호출한 원본 학습 모델을 그대로 저장하는 방식이 아닙니다.** 배포 모델 클래스 `UODDetector`에도 NFE가 없습니다.

이 export는 PyTorch state_dict 형식입니다. ONNX/TensorRT 변환 기능은 포함하지 않았습니다. 배포 파일로 학습을 resume할 수 없습니다.

### 추론

```bash
python predict.py --checkpoint runs/uod_resnet50/deploy.pt --mode rgb \
  --rgb data/rgb/scene_0001.png --device cuda:0 --output results/rgb

python predict.py --checkpoint runs/uod_resnet50/deploy.pt --mode thermal \
  --thermal data/thermal/scene_0001.png --device cuda:0 --output results/thermal

python predict.py --checkpoint runs/uod_resnet50/deploy.pt --mode rgbt \
  --rgb data/rgb/scene_0001.png --thermal data/thermal/scene_0001.png \
  --device cuda:0 --output results/rgbt
```

결과는 원본 좌표계의 box/class/score JSON과 box를 그린 PNG입니다. RGBT는 두 영상의 후보를 **모달리티별 NMS 이전에 concatenate**한 뒤 클래스별 NMS를 한 번 수행합니다. Feature fusion이나 box 좌표 평균을 추가하지 않았습니다. RGBT도 같은 원본 좌표계를 사용해야 합니다.

## 6. 무엇을 확인했고 무엇을 확인하지 않았는가

18개 unit test: 수식의 양성 항 중복, 채널별 정규화, margin, 독립 NFE 파라미터, 공유 모듈 호출 횟수, decoupled tower, 각 모듈 gradient, 빈 GT, paired flip, uint16 처리, 좌표 복원, NMS, AP, export/reload 일치를 확인합니다.

추가로 ResNet-50/ResNet-18의 CPU forward/backward, toy 학습, checkpoint resume, NFE 제거 전후 추론 및 검증 결과 일치, 3가지 추론 CLI를 실행했습니다. 자세한 기록은 `VALIDATION_REPORT.md`를 보세요.

**실제 LLVIP/M3FD/FLIR 학습 성능, pretrained weight 다운로드, CUDA/AMP/MPS 동작, DDP, ONNX/TensorRT는 이 환경에서 검증하지 않았습니다.** 현재 trainer는 단일 장치용입니다. 사용자 데이터셋과 라벨링 규칙에 맞춘 검증이 추가로 필요합니다.

## 7. 핵심 파일

```text
uod/model.py       BFE, FeatureNeck(IFE/NFE), decoupled IFD, train/deploy graph
uod/losses.py      논문 feature separation + CE/L1 + dense GT assignment
uod/data.py        paired dataset, YOLO labels, synchronized transforms
uod/geometry.py    decoding, class-aware NMS, RGBT fusion, coordinate inversion
uod/metrics.py     범위가 명시된 AP evaluator
uod/engine.py      loaders, evaluation, checkpoint I/O, export
train.py          single-device training, checkpoint resume
predict.py        RGB / thermal / RGBT inference
smoke_test.py     offline end-to-end execution test
```

## 근거 자료와 구현 선택의 출처

- 주 논문: DOI `10.1016/j.patcog.2026.113902`, Section 3.1~3.4 / Eq. (1)~(13), 구현 모듈 대응은 Section 4.2. PDF의 식 (6) thermal 항은 RGB 특징으로 인쇄되어 있으나, 이 코드는 주변 본문과 Fig. 2의 흐름에 따라 thermal 특징을 IFE에 넣습니다. 이는 원문 표기와 구분한 문맥상 해석입니다.
- torchvision ResNet-50 문서: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
- torchvision detection operators: https://docs.pytorch.org/vision/stable/ops.html
- dense point-assignment 발상 참고: Tian et al., FCOS, https://arxiv.org/abs/1904.01355

이 저장소의 concrete backbone/neck/head, target assignment 세부값, NFE 초기화, optimizer/augmentation/평가 정책은 위에 공개한 구현 선택이며, 원문에서 모두 지정한 것으로 간주하면 안 됩니다.
