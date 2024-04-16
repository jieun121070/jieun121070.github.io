---
title: "GANs for Single Image Super-Resolution"
date: 2023-8-14
author: jieun
math: True
categories: [Vision]
tags: [GAN, SinGAN]
typora-root-url: ..
---

**Single Image Super-Resolution(SISR)**은 한 장의 저해상도 이미지를 고해상도 이미지로 변환하는 방법을 연구하는 분야입니다. 일반적으로, upsampling을 통해 저해상도 이미지의 width, height를 키운 다음, Neural Network에 통과시켜 세밀한 정보를 추가한 고해상도 이미지를 생성합니다.

- Externally Trained Network (Supervised SISR)
  - 외부의 다수의 이미지를 학습 데이터로 사용
  - 고해상도 이미지로 저해상도 이미지를 만든 다음, 저해상도 이미지를 Neural Network에 입력해서 고해상도 이미지로 복원할 수 있도록 학습 진행
- 이미지에 자기 반복성(internal recurrence)이 존재하는 경우라면?
  - 반복되는 패치 중 작은 패치의 해상도를 높이고자 할 때, 유사하게 생긴 큰 패치를 참고하면 효과적으로 고해상도 변환 가능

> It cannot be found in any external database of examples, no matter how large this dataset is!

- Internally Trained Network (Unsupervised Zero-Shot SISR)
  - **한 장의 이미지만 학습 데이터로 사용**
  - 한 장의 이미지에 특화된 CNN을 학습해서, 그 이미지에 내재된 feature 정보를 토대로 고해상도 결과를 예측
  - 한계점
    - 학습에 사용된 이미지 외의 다른 이미지에 적용하기 어려움
    - 자기 반복성이 떨어지는 경우에도 적용하기 어려움 ex) 인간의 얼굴

## 1. [SinGAN](https://arxiv.org/pdf/1905.01164.pdf) (2019)

![](/assets/img/gan/singan.png)

- [official code](https://github.com/tamarott/SinGAN)
- 한 장의 이미지만을 사용해 GAN 네트워크를 학습
- 총 $N+1$개의 가벼운 개별 GAN 학습 (PGGAN은 하나의 네트워크)
  - 생성자
    - Residual Learning을 이용해 초반에는 대략적인(coarse) 정보 > 점차 세밀한(fine) 정보를 추가
      - 생성자는 noise $z_n$와 이전 단계 output $\tilde{x}_{n+1}$을 입력으로 받게 됨
      - 깊은 네트워크를 사용할 수 있음

        $$\tilde{x} \ast n=G_n(z_n, (\tilde{x} \ast {n+1}) \uparrow^r)$$

        $$\tilde{x} \ast n=(\tilde{x} \ast {n+1}) \uparrow^r+\psi_n(z_n+(\tilde{x}_{n+1}) \uparrow^r)$$

    - $\psi_n$은 5개의 conv-block으로 이루어진 **fully convolutional net**
      - conv-block 구조: Conv(3X3)-BatchNorm-LeakyReLU
      - conv-block별 kernel 개수: 32, 64, 128, 256, 512
      - fully convolutional net이기 때문에 테스트 시에 noise map의 모양을 바꿔가면서 다양한 크기와 가로세로비를 갖는 이미지를 생성할 수 있음
  - 판별자는 전체 이미지가 아니라 **패치 단위** 판별
    
    - 초반에는 패치의 크기가 크고, 점점 패치의 크기가 작아짐 > 점진적으로 자세한 정보를 추가
- 목적함수
  - $\underset{G_n}{\min}\,\underset{D_n}{\max}\,\mathcal{L} \ast {adv}(G_n, D_n)+\alpha \mathcal{L} \ast {rec}(G_n)$
  - Adversarial loss $\mathcal{L}_{adv}(G_n, D_n)$
    - 실제 이미지 $x_n$ 내 패치들의 분포와 $G_n$이 생성한 가짜 이미지 $\tilde x_{n}$ 내 패치들의 분포 사이의 **wasserstein 거리**가 가깝도록 학습
    - Markovian discriminator $D_n$: overlapping된 패치가 real인지 fake인지 분류
  - Reconstruction loss $\mathcal{L}_{rec}(G_n)$
    - noise map $z$로 실제 이미지 $x$를 정확히 생성하도록 학습

## Reference
- [From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)
- [GAN 구조 개요](https://developers.google.com/machine-learning/gan/gan_structure?hl=ko)
