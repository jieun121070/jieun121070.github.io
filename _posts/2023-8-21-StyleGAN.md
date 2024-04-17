---
title: "[Paper Review] A Style-Based Generator Architecture for Generative Adversarial Networks"
date: 2023-8-21
author: jieun
math: True
categories: [Vision]
tags: [GAN, ProGAN, StyleGAN]
typora-root-url: ..
---

## 1. [Progressive Growing of GANs](https://arxiv.org/pdf/1710.10196.pdf) (2017)

![](/assets/img/gan/pggan.gif)
- [official code](https://github.com/tkarras/progressive_growing_of_gans)
- 처음부터 복잡한 네트워크를 학습하는 것이 아니라, 저해상도에서 고해상도로 학습을 진행하는 과정에서 점진적으로 네트워크의 레이어를 붙여 나감
- 한계점
  - 이미지의 특징들이 분리되지 않아 특징 제어가 어려움

## 2. [StyleGAN](https://arxiv.org/pdf/1701.07875.pdf) (2018)

- [official code](https://github.com/NVlabs/stylegan)

- Disentanglement 특성을 향상시켜 이미지의 특징 제어가 어려운 ProGAN의 한계점을 개선함
    - **Disentanglement**: 생성된 이미지의 특정 attribute를 분리하여 조절할 수 있는 능력
    - 생성 모델의 Disentanglement 성능이 낮다면, 얼굴 이미지에서 안경만 제거하고 싶거나 피부 색상만 변경하려고 할 때 의도하지 않은 attribute 변경이 발생할 수 있음
    
- 고해상도 얼굴 데이터셋(FFHQ)을 발표

- 모델 구조

    ![](/assets/img/gan/stylegan.png)

    - Mapping Network

        - 8개의 layer로 구성된 Mapping Network $f$를 사용해 domain $\mathcal{Z}$에서 domain $\mathcal{W}$로 매핑
        - 가우시안 분포에서 샘플링한 $z$ vector를 사용하지 않고, $w$ vector를 사용하면 linear space에서 특징들을 분리할 수 있음

    - Adaptive Instance Normalization (ADaIN)

        - [official code](https://github.com/xunhuang1995/AdaIN-style)

        - feed-forward 방식의 style transfer network에 사용된 구조

        - ADaIN을 사용하면 외부 데이터로부터 style 정보를 가져와 적용할 수 있음

        - 학습시킬 파라미터가 없음

            - content input $x$, style input $y$가 주어졌을 때 ADaIN layer의 output은 다음과 같음

                $$\text{AdaIN}(x,y)=\sigma(y)(\frac{x-\mu(x)}{\sigma(x)})+\mu(y)$$

        - 모델 구조

            ![](/assets/img/gan/adain.png)

            - `encoder`(fixed VGG-19)가 이미지의 content와 style을 인코딩하고, `ADaIN layer`는 encoder의 output을 입력받아 feature space에서 style transfer를 수행함
            - `decoder`는 `ADaIN layer`의 output을 image space로 변환함
            - content loss $\mathcal{L}_c$와 style loss $\mathcal{L}_s$를 계산하기 위해 동일한 `encoder` 사용

    - style을 입혀서 high-lavel attribute를 변경할 뿐만 아니라, 다양한 stochastic variation을 구현할 수 있도록 Noise를 함께 입력함

## 3. [StyleGAN2](https://arxiv.org/pdf/1912.04958.pdf) (2019)

- [official code](https://github.com/NVlabs/stylegan2-ada-pytorch)

## Reference
- [From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)
- [GAN 구조 개요](https://developers.google.com/machine-learning/gan/gan_structure?hl=ko)
- [ProGAN: How NVIDIA Generated Images of Unprecedented Quality](https://towardsdatascience.com/progan-how-nvidia-generated-images-of-unprecedented-quality-51c98ec2cbd2)
