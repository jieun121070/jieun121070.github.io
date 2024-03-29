---
title: "GAN부터 SinGAN까지"
date: 2023-6-12
author: jieun
math: True
categories: [Vision]
tags: [GAN]
typora-root-url: ..
---

본 포스트에서는 GAN부터 SinGAN까지 GAN 모델들을 간략하게 훑어보면서 각 모델의 한계점과 이전 모델보다 나아진 점들을 짚어보려고 합니다. GAN 모델에 대해 소개하기에 앞서 지난 포스트에서 다루었던, GAN과 같은 생성 모델의 일종인 [VAE](https://jieun121070.github.io/posts/Variational-Autoencoder(VAE)/)와 GAN의 차이점에 대해 알아보겠습니다.

- 모델링 대상
  - VAE는 explicit distribution을 모델링
    - VAE는 입력 데이터의 **잠재 공간에 대한 확률 분포를 명시적으로 모델링**
    - encoder는 입력 데이터를 기반으로 잠재 공간에서의 평균(mean)과 분산(variance)을 출력하고, 이 평균과 분산을 사용해 잠재 공간에서 샘플링을 수행하여 새로운 데이터를 생성
  - GAN은 implicit distribution을 모델링 (VAE 보다 좀 더 practical)
    - GAN에서 생성자는 임의의 noise 벡터를 받아 실제 데이터와 유사한 데이터를 생성하는 함수를 학습
    - 판별자는 실제 데이터와 생성된 데이터를 구분하는 기능을 학습
    - 생성자의 목적은 판별자를 속이는 것이므로, 생성자는 결국 실제 데이터 분포를 모방하는 데이터를 생성할 수 있는 **함수를 간접적으로 학습**하게 됨
- 분포 사이의 유사성을 측정하는 metric
  - VAE는 KL divergence 사용
    - 두 분포 $p$와 $q$가 주어졌을 때, $p$가 $q$에 대해 얼마나 다른지 측정
    - 확률 분포 $p$와 확률 분포 $q$가 모든 point에서 같을 때 최솟값 0을 가짐
    - KL divergence는 비대칭(asymmetric)
      - 확률 분포 $p$가 0에 가깝고, 확률 분포 $q$는 0이 아닐 때 q의 효과는 무시됨
      - 동등하게 중요한 두 분포 사이의 유사도를 측정하고 싶을 때 적합하지 않음
  - (Vanila) GAN은 JS divergence 사용
    - 두 분포 $p$과 $q$가 주어졌을 때, 두 분포의 중간 지점과의 차이를 측정
    - JS divergence 는 대칭(symmetric)

## 1. [GAN](https://arxiv.org/pdf/1406.2661.pdf) (2014)

![](/assets/img/gan/gan.png)
_GAN architecture_

- 생성자와 판별자 두 개의 네트워크 학습
  - 생성자의 역할은 판별자가 구분하기 어려운, 진짜같은 이미지를 생성하는 것
  - 판별자의 output은 Real(1), Fake(0)
  - loss function
    - $\underset{G}{\min}\,\underset{D}{\max}\,V(D, G)=E_{x \sim p_{data}(x)}[logD(x)]+E_{z \sim p_{z}(z)}[log(1-D(G(z)))]$
      - 판별자 학습 시 생성자 고정 $\underset{D}{\max}\,V(D, G)=E_{x \sim p_{data}(x)}[logD(x)]+E_{z \sim p_{z}(z)}[log(1-D(G(z)))]$
      - 생성자 학습 시 판별자 고정
        $\underset{G}{\min}\,V(D, G)=E_{z \sim p_{z}(z)}[log(1-D(G(z)))]$
      - 위 과정을 반복하는 과정에서 생성자와 판별자가 서로 경쟁하면서 학습
    - 생성자가 만든 이미지가 real인지 fake인지 판별자가 구분할 수 없어서 $D(G(z))$가 0.5에 가까워지는 것이 목표
- 실험 결과
  - 학습 데이터를 단순히 암기한 것이 아님
  - 흐릿하지 않고 또렷한 이미지를 생성할 수 있음
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

- GAN은 생성자와 판별자 학습할 때 MLP 구조를 사용했는데, DCGAN은 CNN 구조를 사용

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

## 5. [Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf) (2017)

- [official code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- Image-to-Image translation는 주어진 이미지의 어떤 attribute를 다른 값으로 바꾸는 것 ex) 성별을 여성에서 남성으로 바꾸는 등
- conditional GAN을 활용한 image-to-image translation
  - conditional GAN의 일종으로, **이미지 $x$ 자체를 조건으로 입력**
  - 픽셀을 입력받아 픽셀을 예측
- 노이즈 $z$를 사용하지 않기 때문에 거의 deterministic한 결과 생성
- U-Net 사용
  - skip-connection
  - input과 output은 많은 양의 low-level information을 공유함. encoder에서 추출한 information을 decoder에서 활용하도록 함으로써 decoder에서는 추가적인 information을 학습하도록 함 + 학습 난이도를 낮춤
- conditional GAN loss와 L1 loss를 함께 사용
  - L2 loss이 이미지 간 비교에 적용되면 blurry한 결과가 나올 수 있음([참고](https://velog.io/@sjinu/L2-norm-vs-L1-norm))
- 다양한 Task에 공통적으로 적용할 수 있는 generic approach
- 한계점
  - 서로 다른 두 도메인 $X$, $Y$의 데이터들을 한쌍으로 묶은 **paired** dataset 필요

## 6. [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf) (2017)

- [official code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- (데이터셋 측면에서의 한계점을 개선) paired dataset이 필요하다는 Pix2Pix의 한계점을 개선함 > **unpaired** image-to-image translation
- 원본 이미지 $x$의 content를 유지한 상태로 translation이 가능하다는 보장이 없음
  - 어떤 입력 $x$가 주어져도 taget domain $Y$에 해당하는 하나의 이미지만 출력하면 판별자를 충분히 속일 수 있기 때문 
  - 추가적인 제약 조건 필요 - $G(x)$가 다시 원본 이미지 $x$로 재구성될 수 있도록 함
  - $F(G(x)) \approx x$, $G(F(y)) \approx y$ ($F$와 $G$는 역함수 관계)
- 한계점
  - shape 정보를 포함한 content의 변경이 필요한 경우
  - 학습 데이터에 포함되지 않은 사물을 처리하는 경우

## 7. [Progressive Growing of GANs](https://arxiv.org/pdf/1710.10196.pdf) (2017)

- [official code](https://github.com/tkarras/progressive_growing_of_gans)
- 처음부터 복잡한 네트워크를 학습하는 것이 아니라, 학습을 진행하는 과정에서 점진적으로 네트워크의 레이어를 붙여 나감
- 저해상도 > 고해상도
- 한계점
  - 이미지의 특징 제어가 어려움

## 8. [StarGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.pdf) (2018)

- [official code](https://github.com/yunjey/stargan)
- 다중 도메인에서 효율적인 image-to-image translation

## 9. [SinGAN](https://arxiv.org/pdf/1905.01164.pdf) (2019)

### Single Image Super-Resolution (SISR)

- 한 장의 저해상도 이미지를 고해상도 이미지로 변환하는 방법을 연구하는 분야
- 일반적으로, upsampling을 통해 저해상도 이미지의 width, height를 키우고 > Neural Network > 세밀한 정보를 추가한 고해상도 이미지 생성
- Externally Trained Network (Supervised SISR)
  - 외부의 다수의 이미지를 학습 데이터로 사용
  - 고해상도 이미지로 저해상도 이미지를 만든 다음, 저해상도 이미지를 Neural Network에 입력해서 고해상도 이미지로 복원할 수 있도록 학습 진행
- 이미지에 자기 반복성(internal recurrence)이 존재하는 경우라면?
  - 반복되는 패치 중 작은 패치의 해상도를 높이고자 할 때, 유사하게 생긴 큰 패치를 참고하면 효과적으로 고해상도 변환 가능
  - It cannot be found in any external database of examples, no matter how large this dataset is!
- Internally Trained Network (Unsupervised Zero-Shot SISR)
  - **한 장의 이미지만 학습 데이터로 사용**
    - 한 장의 이미지에 특화된 CNN을 학습해서, 그 이미지에 내재된 feature 정보를 토대로 고해상도 결과를 예측
  - 한계점
    - 학습에 사용된 이미지 외의 다른 이미지에 적용하기 어려움
    - 자기 반복성이 떨어지는 경우에도 적용하기 어려움 ex) 인간의 얼굴

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
