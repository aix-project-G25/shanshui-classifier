# 중-일간 산수화 분류 웹 서비스

남기산 일본학과 2024072924 - 일본 산수화 수집 및 웹 서비스 구축
김찬영 중국학과 2021023181 - 중국 산수화 수집 및 모델 학습

## I. Proposal
### 1. 프로젝트 개요
동아시아의 산수화는 각기 다른 역사적, 문화적 배경을 바탕으로 독특한 미적 특성을 지니고 있습니다. 그러나 외형적으로 유사해 보이는 경우도 많아, 비전문가가 구분하기 어려운 경우가 많습니다. 본 프로젝트는 ResNet-18 기반의 딥러닝 모델을 활용하여 중일 산수화를 분류하고, 이를 웹 서비스를 통해 사용자에게 제공합니다.

### 2. 목표
1. 문화 예술 감상의 새로운 접근: 인공지능을 활용하여 전통 예술 작품을 감상하고 이해하는 새로운 방법을 제시합니다.

2. 딥러닝 기술의 실용적 응용: 실제 데이터를 기반으로 딥러닝 모델을 학습시키고, 이를 웹 서비스에 적용하여 기술의 실용성을 탐색합니다.

3. 웹 개발과 AI의 융합 경험: 팀원 간의 협업을 통해 웹 개발과 딥러닝 모델 개발을 통합한 서비스를 구현합니다.

## II. Data Collection
### 1. 중일 산수화 데이터셋 구축
중국과 일본의 산수화 작품을 수집하여 데이터셋을 구축합니다. 각 작품은 jp/cn 폴더에 저장됩니다. 중국과 일본 산수화 각각 200장씩 수집합니다. 메타데이터는 포함하지 않습니다.

- 중국 산수화 데이터셋: [chinese-landscape-painting-dataset](https://www.kaggle.com/datasets/myzhang1029/chinese-landscape-painting-dataset)
- 일본 산수화 데이터셋: [WikiArt Dataset (Refined)](https://www.kaggle.com/datasets/trungit/wikiart30k/data?select=Ukiyo_e) (Ukiyo-e 작품 중 산수화에 해당하는 작품만 사용)

## III. Model Development
### 1. 모델 선택 이유
본 프로젝트에서는 ResNet-18 모델을 채택하였습니다. 
ResNet은 일반적인 딥러닝 모델이 층이 깊어질수록 학습이 잘 되지 않는 문제(기울기 소실)를 해결하기 위해 등장한 구조입니다. 핵심 아이디어는 "입력값을 그대로 출력으로 보내는 지름길(잔차 연결)"을 추가하는 것입니다. 이를 통해 모델이 학습 과정에서 꼭 필요한 정보만 새롭게 학습하고, 나머지는 그대로 전달할 수 있어 불필요한 계산을 줄이고 안정적으로 학습할 수 있습니다.

ResNet 계열 중에서도 본 프로젝트는 ResNet-18을 선택하였습니다. 이는 비교적 얕은 구조(총 18개 층)를 가지고 있어 다음과 같은 장점을 갖습니다:
- 학습 속도가 빠르고
- 과적합 위험이 낮으며
- 복잡한 구조 없이도 작품의 세밀한 스타일 차이를 잘 구분할 수 있는 성능을 보입니다.
따라서, 중일 산수화처럼 미묘한 시각적 차이가 존재하는 분류 문제에 ResNet-18은 효율적이면서도 안정적인 선택이라 판단하였습니다.

### 2. 개발 환경 및 사용 기술 
- **개발 프레임워크**: PyTorch
- **모델 구조**: ResNet-18
- **하드웨어**: Vagon Computer Flame 인스턴스(like as aws ec2 g5.2xlarge, NVIDIA A10G GPU)
- **데이터 전처리 및 증강 기법**: ~~Resize, Random Horizontal Flip, Normalize~~
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

### 3. 데이터 구성 및 전처리
1. 디렉토리 구조
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
2. 데이터 전처리
```
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
```

### 4. 데이터 로딩 
```
train_dir = os.path.join('data', 'train')
val_dir = os.path.join('data', 'val')

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### 5. 모델 구성 및 학습 설정 
```
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 중/일 2클래스 분류

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

### 6. 학습 
```
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
...
```
## IV. Result 
### 1. 모델 성능
1. 최종 정확도
2. 손실 값 변화


### 2. 성능 시각화
1. Loss 변화 시각화
2. Accuracy 변화 시각화

### 3. 주요 오분류 사례 

### 4. 한계점 및 개선 방향 

(추가 예정)

