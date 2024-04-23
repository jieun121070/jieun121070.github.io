---
title: "Generative Adversarial Networks"
date: 2023-8-7
author: jieun
math: True
categories: [Vision]
tags: [GAN, WGAN]
typora-root-url: ..
---

본 포스트에서는 GAN부터 WGAN까지 GAN 계열 모델의 발전 과정을 훑어보면서 각 모델의 한계점과 이전 모델보다 나아진 점들을 짚어보려고 합니다. GAN 모델에 대해 소개하기에 앞서 지난 포스트에서 다루었던, GAN과 같은 생성 모델의 일종인 [VAE](https://jieun121070.github.io/posts/Variational-Autoencoder(VAE)/)와 GAN의 차이점에 대해 알아보겠습니다.

- 모델링 대상
  - VAE는 explicit distribution을 모델링
    - VAE는 입력 데이터의 **잠재 공간에 대한 확률 분포를 명시적으로 모델링**
    - encoder는 입력 데이터를 기반으로 잠재 공간에서의 평균(mean)과 분산(variance)을 출력하고, 이 평균과 분산을 사용해 잠재 공간에서 샘플링을 수행하여 새로운 데이터를 생성
  - GAN은 implicit distribution을 모델링 (VAE 보다 좀 더 practical)
    - GAN에서 생성기는 임의의 noise 벡터를 받아 실제 데이터와 유사한 데이터를 생성하는 함수를 학습
    - 판별기는 실제 데이터와 생성된 데이터를 구분하는 기능을 학습
    - 생성기의 목적은 판별기를 속이는 것이므로, 생성기는 결국 실제 데이터 분포를 모방하는 데이터를 생성할 수 있는 **함수를 간접적으로 학습**하게 됨
- [분포 사이의 유사성을 측정하는 metric](https://jieun121070.github.io/posts/%EB%B6%84%ED%8F%AC-%EA%B0%84%EC%9D%98-%EA%B1%B0%EB%A6%AC%EB%A5%BC-%EC%B8%A1%EC%A0%95%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95%EB%93%A4/)
  - VAE는 KL divergence 사용
    - 두 분포 $p$와 $q$가 주어졌을 때, $p$가 $q$에 대해 얼마나 다른지 측정
    - KL divergence는 비대칭(asymmetric)
  - (Vanila) GAN은 JS divergence 사용
    - 두 분포 $p$과 $q$가 주어졌을 때, 두 분포의 중간 지점과의 차이를 측정
    - JS divergence 는 대칭(symmetric)

## 1. [GAN](https://arxiv.org/pdf/1406.2661.pdf) (2014)

![](/assets/img/gan/gan.png)
_GAN architecture_

- 모델 구조
  
  - 생성기와 판별기 두 개의 네트워크 학습
  - 생성기의 역할은 판별기가 구분하기 어려운, 진짜같은 이미지를 생성하는 것
  - 판별기의 output은 Real(1), Fake(0)
  - loss function
    - $\underset{G}{\min}\,\underset{D}{\max}\,V(D, G)=E_{x \sim p_{data}(x)}[logD(x)]+E_{z \sim p_{z}(z)}[log(1-D(G(z)))]$
      - 판별기 학습 시 생성기 고정 $\underset{D}{\max}\,V(D, G)=E_{x \sim p_{data}(x)}[logD(x)]+E_{z \sim p_{z}(z)}[log(1-D(G(z)))]$
      - 생성기 학습 시 판별기 고정
        $\underset{G}{\min}\,V(D, G)=E_{z \sim p_{z}(z)}[log(1-D(G(z)))]$
      - 위 과정을 반복하는 과정에서 생성기와 판별기가 서로 경쟁하면서 학습
    - 생성기가 만든 이미지가 real인지 fake인지 판별기가 구분할 수 없어서($p_{data}=p_g$) 판별기의 output이 0.5에 가까워지는 것이 목표
  
- 실험 결과
  - 학습 데이터를 단순히 암기한 것이 아님
  
  - 흐릿하지 않고 또렷한 이미지를 생성할 수 있음
  
    ![](/assets/img/gan/gan1.png)
  
    ![](/assets/img/gan/gan2.png)
  
  - latent vector 사이의 interpolation으로도 있을 법한 이미지를 생성할 수 있음

```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
```

## 2. [Conditional GAN](https://arxiv.org/pdf/1411.1784.pdf) (2014)

- 생성하고자 하는 label $y$를 조건으로 입력

$$ \underset{G}{\min}\,\underset{D}{\max}\,V(D, G)=E_{x \sim p_{data}(x)}[logD(x \vert y)]+E_{z \sim p_{z}(z)}[log(1-D(G(z \vert y)))] $$

## 3. [Deep Convolutional GAN](https://arxiv.org/pdf/1511.06434.pdf) (2016)

![](/assets/img/gan/dcgan.png)
_DCGAN generator architecture_

- GAN은 생성기와 판별기 학습할 때 MLP 구조를 사용했는데, DCGAN은 CNN 구조를 사용

- GAN보다 고해상도의 이미지 생성 가능

- 모델 구조

    - 생성기
        - 표준 정규분포로부터 랜덤 샘플링된 100 차원의 vector $z$를 입력받음
        - fully connected layer를 통과시켜서 $4 \times 4 \times 1024$ 차원의 vector를 생성
        - activation map 형태(height 4, width 4, channel 1024)로 Reshape
        - **transposed convolution**를 사용해 너비와 높이를 증가시킴 (upsampling)
        - 최종적으로 RGB 형태(height 64, width 64, channel 3)의 output 산출
        - activation function
            - 출력 layer에는 Tanh, 그 외에는 ReLU 사용
            - -1과 1사이의 출력 layer output을 0과 255 사이의 값으로 변환

    - 판별기
      - 생성기의 output(가짜 이미지)과 실제 이미지를 입력받음
      - **strided convolution**를 사용해 너비와 높이를 감소시킴 (downsampling)
      - activation function
        - 출력 layer에서는 Sigmoid, 그 외에는 Leaky ReLU 사용
        - 출력 layer의 최종 output은 Real(1), Fake(0)

- 실험 결과

![](/assets/img/gan/dcgan2.png)

- 임의의 두 vector z_1, z_2를 interpolation한 vector들을 생성기에 입력해 결과 이미지를 나열했을 때, 부드러운 transition이 나타나는 것을 확인할 수 있음

![](/assets/img/gan/dcgan1.png)

- 사람의 얼굴에서 나타나는 다양한 특성을 적절하게 인코딩함
  - 웃는 여성 이미지를 만드는 vector들의 평균 값(smiling woman)에서 무표정한 여성 이미지를 만드는 vector들의 평균 값(neutral woman)을 빼면, 여성이라는 특성은 지워지고 웃는다는 특성만 남음
  - 여기에 무표정한 남성 이미지를 만드는 vector들의 평균 값(neutral man)을 더해 생성기에 입력하면 웃는 남성 이미지가 생성됨

```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
```

## 4. [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf) (2017)

- GAN의 문제점 중 하나인 학습의 불안정성을 개선하기 위해 JS divergence 대신 Wasserstein 거리(Earth-Mover 거리)를 사용
- WGAN은 weight clipping을 이용해 1-Lipshichtz 조건을 만족하도록 함으로써 안정적인 학습을 유도
- [WGAN-GP(2017)](https://arxiv.org/pdf/1704.00028.pdf)는 gradient penalty를 이용하여 WGAN의 성능을 개선

## Reference
- [From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)
- [GAN 구조 개요](https://developers.google.com/machine-learning/gan/gan_structure?hl=ko)
