---
title: "훈민정음 화자인식 공모전 후기"
date: 2022-6-6
author: jieun
math: True
categories: [Audio]
tags: [Audio, Meta-learning]
typora-root-url: ..
---



## 대회 주제

- 매칭된 두 개의 음성파일을 읽어 같은 발화자인지 다른 발화자인지 추론할 수 있는 인공지능 모델 개발
  - 외부 데이터 및 사전 학습 모델 사용 불가
- 평가 지표: EER(Equal Error Rate)
- 학습/테스트 데이터(AI허브 오디오 데이터)
  - 전체 크기: 42.2GB
  - 파일 수
    - train_data: 239,378
    - test_data: 1,221

## feature engineering

1. spectrogram
2. melspectrogram
   1. max length보다 길면 자르기 & max length보다 짧으면 zero padding
   2. max length보다 길면 자르고 랜덤하게 순서 섞기 & max legnth보다 짧으면 반복해서 max length 채우기
   3. **max length보다 길면 자르고 랜덤하게 순서 섞기 & 짧으면 zero padding** (최종 모델에 사용)
3. mfcc

## 모델 구조

resnet18로 512 차원의 embedding vector 추출 → dropout(0.2) → fully-connected layer(512, 화자 수)

```
class Net(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet18()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        embeddings = self.resnet(x)
        x = self.dropout(embeddings)
        out = self.fc(x)

        return embeddings, out
```

## 학습 방법

두 개의 loss를 더해서 사용

1. triplet + arcface
2. arcface + cross entropy
3. **triplet + cross entropy(+ label smoothing=0.1)** (최종 모델에 사용)

### label smoothing

- softmax는 정답 라벨에 지나치게 높은 확률 값을 부여(overconfident)하여 모델이 학습한 적 없는 클래스(unseen class)를 예측해야 할 때 문제가 됨
- 라벨을 hard target에서 soft target으로 바꿔서 **일반화 성능** 향상

## 하이퍼파라미터 튜닝

- feature engineering 관련 파라미터
  - max_length = 50000
  - resample = False
  - n_fft = 1024
  - win_length = 200
  - hop_length = 100
- batch 구성
  - triplet set을 생성하기 위한 class당 sample 개수를 8 정도로 조정했을 때 가장 좋은 성능을 보임.
- learning rate schedule
  - 초기 learning rate = 0.001
  - LambdaLR scheduler 사용