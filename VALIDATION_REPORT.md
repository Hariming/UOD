# 실행 검증 기록

검증 환경: Python 3.13.5, torch 2.10.0+cpu, torchvision 0.25.0+cpu. CPU 사용.

## 자동 unit test

`python -m unittest discover -s tests -v`

**19 / 19 PASS**. 상세 이름/결과는 `test_logs/unit_tests.txt`.

검증 항목: 논문 대조식의 양성 항 두 번 포함, 채널별 L2 정규화, 제곱거리 margin, padding 옵션, 공유 모듈의 한 번 실행, NFE 동일 구조/독립 파라미터, classification/regression tower 분리, YOLOv5-style backbone/neck shape, 모든 5개 모듈로 gradient 전달, 빈 GT, Thermal 1→3 반복 입력 일치, NFE-free export 및 strict reload, 동기화된 flip, 16-bit 입력, box 좌표 round-trip, 클래스별 NMS, AP의 기본 사례.

## 통합 실행

`python smoke_test.py`

**PASS**. `test_logs/integration_report.json` 참조.

3 epoch toy 학습, epoch 1 저장점에서 재개한 결과와 중단 없이 epoch 3까지 진행한 결과의 **전체 model state tensor가 정확히 일치**함을 확인했습니다. NFE/optimizer를 제외한 deploy checkpoint를 생성했고, 배포 전후 validation JSON도 일치했습니다. RGB, Thermal, RGBT inference를 통과했습니다.

별도로 `predict.py`를 세 모드에서 실행하여 JSON 및 PNG가 생성됨과 `NFE modules loaded: []`를 확인했습니다.

## 기본 ResNet-50 구조 점검

사전 학습 weight 없이, 입력 128×128, 1 pair, 2 classes 설정의 forward/backward에서 finite loss를 확인했습니다.

| 모듈 | 파라미터 수 |
|---|---:|
| BFE | 23,508,032 |
| IFE | 5,051,392 |
| IFD | 2,377,482 |
| NFE_RGB | 5,051,392 |
| NFE_Thermal | 5,051,392 |
| 학습 모델 합계 | 41,039,690 |
| 배포 모델 합계 | 30,936,906 |

같은 설정에서 IFE/NFE 특징 크기는 `[1,256,16,16]`, `[1,256,8,8]`, `[1,256,4,4]`였습니다. 클래스 수/설정을 변경하면 파라미터 수가 달라집니다.

## 검증 범위의 한계

toy 데이터는 실행 오류와 텐서/손실 연결을 확인하기 위한 인공 사각형 이미지입니다. 정확도 개선이나 논문 재현 성능을 입증하지 않습니다. 실제 데이터, pretrained weight 다운로드, CUDA/AMP/MPS, multi-GPU 학습을 검증하지 않았습니다. 제공 AP evaluator는 crowd/ignore/area 규칙이 없는 일반 박스용이며 공식 benchmark evaluator가 아닙니다.
