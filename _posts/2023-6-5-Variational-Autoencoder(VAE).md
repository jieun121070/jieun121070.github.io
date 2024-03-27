---
title: "Variational Autoencoder(VAE)"
date: 2023-6-5
author: jieun
math: True
categories: [Vision]
tags: [VAE]
typora-root-url: ..
---

본 포스트에서는 GAN에 대해 알아보기에 앞서, GAN과 같은 생성 모델의 일종인 Variational Autoencoder(VAE)에 대해서 다뤄보려고 합니다. 생성 모델은 실존하지는 않지만 현실에 존재할 법한 새로운 데이터를 생성하는 모델인데요. classification 모델처럼 클래스를 구분짓는 decision boundary를 학습하는 것이 아니라 클래스의 확률 분포를 학습합니다. VAE와 깊이 연관된 모델인 Autoencoder에 대해서 먼저 알아보겠습니다.

## 1. Autoencoder

![](/assets/img/gan/ae.png)

Autoencoder는 encoder과 decoder로 구성된 모델이며, input과 output이 동일합니다. 고차원의 input을 저차원으로 압축했다가 다시 원래 input으로 복원하는 과정을 거칩니다. input을 다시 원래대로 잘 복원하기 위해서는 저차원으로 압축한 feature(code)가 핵심적인 정보를 담고 있어야 합니다. Autoencoder는 feature를 최대한 잘 압축하는 방향으로 학습됩니다. 즉, Autoencoder의 목적은 새로운 데이터를 생성하는 것이 아니라 차원 축소(manifold learning)입니다.

### Manifold learning

manifold learning은 고차원 데이터를 저차원에 매핑하는 차원 축소 방법입니다. 차원 축소 기법은 데이터를 선형 공간에 매핑하는지, 비선형 공간에 매핑하는지에 따라 아래와 같이 분류할 수 있는데요.

- 선형
  - [PCA](https://jieun121070.github.io/posts/PCA/)
  - LDA
- 비선형
  - Autoencoder
  - t-SNE
  - LLE
  - Isomap
  - [Kernel PCA](https://jieun121070.github.io/posts/Kernel-PCA/)

선형 차원 축소 기법은 데이터를 선형 공간에 projection 하여 차원을 축소하는 반면, manifold learning은 비선형 구조를 찾고자 합니다. 따라서 데이터가 본질적으로 비선형 구조를 가질 때 선형 차원 축소 기법보다 manifold learning이 더 효과적일 수 있습니다. manifold를 잘 찾았다면, manifold 좌표들이 조금씩 변할 때 데이터도 유의미하게 조금씩 변하는 것을 확인할 수 있습니다.

## 2. Variational Autoencoder

앞서 설명한 바와 같이, VAE는 생성 모델의 일종으로 Autoencoder와는 목적이 다릅니다. Autoencoder는 input을 고정된 vector로 매핑하지만 VAE는 input을 분포 $p_{\theta}$로 매핑하고자 합니다. input을 $x$라 하고 latent encoding vector를 $z$라 하면 둘 사이의 관계는 다음과 같이 나타낼 수 있습니다.

- Prior $p_{\theta}(z)$
- Likelihood $p_{\theta}(x \vert z)$
- Posterior $p_{\theta}(z \vert x)$

최적의 파라미터 $\theta^*$를 알고 있다고 가정했을 때, 아래와 같은 순서를 따라 새로운 데이터 $x$를 생성할 수 있습니다.

- Prior $p_{\theta^*}(z)$에서 $z^{(i)}$를 샘플링
- 조건부 확률 $p_{\theta^*}(x \vert z=z^{(i)})$에서 real data point처럼 보이는 $x^{(i)}$를 생성

이렇게 구성된 모델을 학습시키기 위해서는 training data의 likelihood $p_{\theta}(x)=\int{p_{\theta}(z)p_{\theta}(x \vert z)}dz$를 최대화하는 파라미터 $\theta$를 찾아야 합니다. 그런데 모든 $z$에 대해서 $p_{\theta}(z)p_{\theta}(x \vert z)$를 계산해서 더하는 것은 너무 많은 비용이 듭니다. Posterior density $p_{\theta}(z \vert x)=p_{\theta}(x \vert z)p_{\theta}(z)/p_{\theta}(x)$도 계산하는 것이 불가능합니다. 새로운 함수 $q_{\theta}(z \vert x)$를 도입해 $p_{\theta}(z \vert x)$에 근사시키는 방법으로 이러한 문제를 해결하는 모델이 바로 VAE입니다. 여기에서 $q_{\theta}(z \vert x)$가 encoder이고, $p_{\theta}(x \vert z)$가 decoder입니다.

![](/assets/img/gan/vae.png)

## Reference

- [How to Use Autoencoders for Image Denoising](https://www.omdena.com/blog/denoising-autoencoders)
- [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)
- [오토인코더의 모든 것](https://www.youtube.com/watch?v=o_peo6U7IRM)
