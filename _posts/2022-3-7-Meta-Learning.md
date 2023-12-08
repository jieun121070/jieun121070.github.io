---
title: "Meta Learning"
date: 2022-3-7
author: jieun
math: True
categories: [Meta-Learning]
tags: [Meta-Learning]
typora-root-url: ..
---



# Few shot learning

전통적인 supervised learning은 아래와 같은 학습 데이터를 사용해 모델을 학습시킨 다음, 학습 데이터에서 등장한 적 없는 새로운 테스트 데이터로 모델을 평가합니다. 이 때, 테스트 데이터는 학습 데이터에 포함된 class여야 합니다. 예를 들어 아래 학습 데이터로 학습한 분류 모델은 토끼 이미지를 분류해 낼 수 없을 것입니다. 허스키, 코끼리, 호랑이, 앵무새, 차 외에 다른 class의 정보는 모델이 전혀 가지고 있지 않기 때문입니다.

![](/assets/img/meta/few-shot1.jpg)

하지만 모델에게 아래와 같은 힌트를 준다면 상황은 달라질 것입니다. 이처럼 대용량 데이터셋으로 학습한 모델에 **Support Set**이라는 힌트를 주고, 학습 데이터에 포함되어 있지 않은 unknown class의 **Query Sample**을 제시하는 학습 문제가 **few shot learning**입니다. 조금 더 구체적으로는, Support Set의 class 개수($N$)와 클래스별 데이터 개수($K$)에 따라 N-way K-shot 문제라고 부릅니다. 아래와 같은 경우는 6-way 1-shot 문제입니다.

![](/assets/img/meta/few-shot2.jpg)

Support Set이 주어진다고 해도 모델이 unknown class를 분류하는 것은 쉬운 일이 아닙니다. 딥러닝 모델이 우수한 성능을 달성할 수 있도록 하는 전제 조건은 대용량 데이터셋을 충분히 학습하는 것인데, 이 데이터가 부족하기 때문입니다. 이러한 상황에서 적용할 수 있는 방법 중 하나가 바로 meta learning입니다.



# meta learning

meta learning은 적은 양의 데이터만으로도 효율적으로 학습할 수 있도록 모델이 학습 방법을 스스로 학습하는 것입니다. meta learning의 종류는 다음과 같이 나누어 볼 수 있습니다. 

- 모델 기반 meta learning
- 최적화 기반 meta learning
- 메트릭 기반 meta learning



# Reference

- [Stanford CS330: Deep Multi-Task & Meta Learning](https://www.youtube.com/watch?v=dYmJd_fJLW0&list=PLoROMvodv4rMIJ-TvblAIkw28Wxi27B36)
- [Few-Shot Learning](https://www.youtube.com/watch?v=hE7eGew4eeg)