# 중-일간 산수화 분류 웹 서비스

남기산 일본학과 2024072924 - 일본 산수화 수집 및 웹 서비스 구축
김찬영 중국학과 0000000000 - 중국 산수화 수집 및 모델 학습

## I. Proposal

동아시아의 산수화는 각기 다른 역사적, 문화적 배경을 바탕으로 독특한 미적 특성을 지니고 있습니다. 그러나 외형적으로 유사해 보이는 경우도 많아, 비전문가가 구분하기 어려운 경우가 많습니다. 본 프로젝트는 ResNet18 기반의 딥러닝 모델을 활용하여 중일 산수화를 분류하고, 이를 웹 서비스를 통해 사용자에게 제공함으로써 다음과 같은 목적을 달성하고자 합니다.

1. 문화 예술 감상의 새로운 접근: 인공지능을 활용하여 전통 예술 작품을 감상하고 이해하는 새로운 방법을 제시합니다.

2. 딥러닝 기술의 실용적 응용: 실제 데이터를 기반으로 딥러닝 모델을 학습시키고, 이를 웹 서비스에 적용하여 기술의 실용성을 탐색합니다.

3. 웹 개발과 AI의 융합 경험: 팀원 간의 협업을 통해 웹 개발과 딥러닝 모델 개발을 통합한 서비스를 구현합니다.

## II. Data Collection
### 1. 중일 산수화 데이터셋 구축
중국과 일본의 산수화 작품을 수집하여 데이터셋을 구축합니다. 각 작품은 jp/cn 폴더에 저장됩니다. 중국과 일본 산수화 각각 200장씩 수집합니다. 메타데이터는 포함하지 않습니다.

- 중국 산수화 데이터셋: [chinese-landscape-painting-dataset](https://www.kaggle.com/datasets/myzhang1029/chinese-landscape-painting-dataset)
- 일본 산수화 데이터셋: [WikiArt Dataset (Refined)](https://www.kaggle.com/datasets/trungit/wikiart30k/data?select=Ukiyo_e) (Ukiyo-e 작품 중 산수화에 해당하는 작품만 사용)

### 2. 데이터셋 구조
```
data/
├── cn/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── jp/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── ...
```

## III. Model Training
### 1. 모델 구조
ResNet18을 기반으로 한 딥러닝 모델을 사용합니다. ResNet18은 잔차 연결을 통해 깊은 신경망의 학습을 용이하게 하며, 이미지 분류 작업에 효과적입니다. (추가 요망)

### 2. 학습 환경
- **프레임워크**: PyTorch
- **모델**: ResNet18
- **환경**: Vagon Computer Flame 인스턴스(like as aws ec2 g5.2xlarge, NVIDIA A10G GPU)

### 2. 학습 과정
모델은 PyTorch를 사용하여 학습합니다. 데이터셋을 불러오고, 전처리 과정을 거친 후, 모델을 학습합니다. 

학습 과정은 다음과 같습니다:
1. 라이브러리 임포트

```python
import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
```
- `os`: 파일 및 디렉토리 작업을 위한 모듈
- `torch`: PyTorch 라이브러리로 딥러닝 모델을 구축하고 학습하는 데 사용
- `torchvision`: 이미지 데이터셋과 변환을 위한 모듈
- `torch.utils.data`: 데이터 로더를 위한 유틸리티 모듈
- `torch.nn`: 신경망 모듈을 위한 PyTorch의 하위 모듈
- `torch.optim`: 최적화 알고리즘을 위한 모듈
- `matplotlib.pyplot`: 데이터 시각화를 위한 라이브러리
- `numpy`: 수치 계산을 위한 라이브러리
- `tqdm`: 진행 상황 표시를 위한 라이브러리
- `PIL`: 이미지 처리 라이브러리

2. 데이터셋 준비
```python
data_dir = 'data'

# 데이터셋 경로 설정
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

```
(추가 예정)

