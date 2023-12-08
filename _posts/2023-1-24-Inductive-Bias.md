---
title: "Inductive Bias"
date: 2023-1-24
author: jieun
math: True
categories: [Vision]
tags: [Vision-Transformer, inductive-bias]
---

머신러닝 분야에서 inductive bias란 학습 시 보지 못했던 주어진 입력에 대해 모델이 출력을 예측할 때 사용하는 일련의 가정을 의미합니다.

대부분의 vision task들은 CNN의 sliding window 방식으로 인해 내재된 inductive bias에 오랫동안 의존해 왔는데요.  Vision Transformer는 이 부분이 결여돼 있어서 image classification 외 vision task들을 모두 아우르는 backbone network가 되기에는 한계가 있었습니다. 그래서 [Transformer](https://jieun121070.github.io/posts/paper-review-Attention-is-All-You-Need/)와 CNN의 hybrid model로써 [SWIN Transformer](https://jieun121070.github.io/posts/paper-review-Swin-Transformer-Hierarchical-Vision-Transformer-using-Shifted-Windows/)가 제안되었습니다. SWIN Transformer는 CNN의 translation invariance 가정을 반영하기 위해 shifted window 구조를 적용했습니다. 

# Inductive biases in image data

- Stationarity in image dataset
  - 데이터의 통계량이 시공간에 대해 불변하다는 가정
- Locality
  - 시공간상 가까이 위치한 데이터끼리 높은 상관관계를 갖는다는 가정
- Translation invariance
  - 물체의 시공간상 절대적 위치가 함수 output에 영향을 미치지 않는다는 가정
  - convolution, pooling, activation function이 input의 일부 범위에만 적용되므로 상대적인 위치 좌표에 의존
- Translation equivariance
  - 물체의 시공간상 절대적 위치가 함수 output에 영향을 미친다는 가정

# Reference
- [CNN의 inductive bias - YouTube](https://www.youtube.com/watch?v=2VCRoO_d5Go)
- [Principles and applications of relational inductive biases in deep learning - YouTube](https://www.youtube.com/watch?v=sTGKOUzIpaQ)
- [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/pdf/1806.01261.pdf)
- [The Inductive Bias of ML Models, and Why You Should Care About It](https://towardsdatascience.com/the-inductive-bias-of-ml-models-and-why-you-should-care-about-it-979fe02a1a56)
- [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases (arxiv.org)](https://arxiv.org/pdf/2103.10697.pdf)
