---
title: "[Paper Review] ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
date: 2022-6-6
author: jieun
math: True
categories: [Vision]
tags: [Face-Recognition]
typora-root-url: ..
---

이번 포스트에서는 얼굴 인식 분야에서 큰 주목을 받은 모델 중 하나인 **[ArcFace(2019)](https://arxiv.org/abs/1801.07698)**에 대해 알아보겠습니다. ArcFace는 얼굴 인식에서 **margin-based softmax loss**를 이용해 기존보다 훨씬 높은 분류 성능과 discriminative power를 달성한 모델로 평가받고 있습니다. ArcFace의 핵심 아이디어는 무엇인지, 그리고 기존 방법과 어떤 차이가 있는지 자세히 살펴보겠습니다.

## ArcFace의 핵심 아이디어

![](/assets/img/arcface/arcface.png)

ArcFace의 핵심은 softmax 계산에서 각도($\theta$)에 margin $m$ 을 더하는 것입니다.

- 기존 softmax logits:

$$z_i = \|W_{y_i}\| \cdot \|x_i\| \cdot \cos(\theta_{y_i})$$

- ArcFace의 수정된 logits:

$$z_i = \|x_i\| \cdot \cos(\theta_{y_i} + m)$$

weight와 feature를 L2 norm으로 정규화하고, $\cos(\theta)$에 margin $m$을 더함으로써 **angular decision boundary**를 더 넓게 만듭니다.

![](/assets/img/arcface/arcfaceloss.png)

얼굴 인식의 핵심은 같은 사람은 feature space에서 가깝고, 다른 사람은 멀리 배치하는 것입니다. 하지만 기존 softmax(a)는 feature 간의 각도를 충분히 구분하지 못해 decision boundary가 명확하지 않다는 문제가 있었습니다. ArcFace(b)는 여기에 **angular margin**을 추가해 **같은 클래스는 더 촘촘히 모이고, 다른 클래스는 더 멀어지도록 유도**하여 구분력을 극적으로 높였습니다.

## FaceNet과의 차이점

| 구분      | FaceNet                              | ArcFace                        |
| --------- | ------------------------------------ | ------------------------------ |
| 학습 방식 | Contrastive Learning                 | Classification                 |
| Loss      | Triplet Loss                         | Margin-based Softmax Loss      |
| 학습 단위 | Triplet (anchor, positive, negative) | Single sample + label          |
| 목적      | 임베딩 거리 직접 학습                | Softmax decision boundary 개선 |