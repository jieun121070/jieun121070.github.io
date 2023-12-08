---
title: "[Paper Review] Towards Total Recall in Industrial Anomaly Detection"
date: 2023-2-27
author: jieun
math: True
categories: [Vision]
tags: [Anomaly-Detection, PatchCore]
typora-root-url: ..
---

PatchCore는 SPADE와 PaDiM의 특징을 합친 모델입니다. SPADE의 `Gallery`와 비슷한 역할을 하는 `Memory Bank`를 사용하고, PaDiM처럼 patch level 접근법을 사용합니다. 특히 이웃한 pixel feature들을 묶어서 patch feature를 구성하는 것이 특징입니다. 본 포스트에서는 PatchCore가 이상탐지를 진행하는 과정과 함께 MVTec 데이터셋으로 모델 성능을 평가한 결과를 살펴보겠습니다.



## 이상탐지 진행 과정

![](/assets/img/ad/patchcore.jpg)
_PatchCore 이상탐지 진행 과정_

### 1. Locally aware patch features

첫 번째 단계는 **정상 이미지들의 feature를 추출**하는 단계입니다. 위 이미지의 'Pretrained Encoder' 부분에 나타나 있는데요. ImageNet 데이터셋으로 pre-train한 CNN을 $\phi$라고 하면, 입력이미지 $x_i$를 pre-trained CNN에 입력해서 얻은 $j$번째 block의 output은 다음과 같이 나타낼 수 있습니다. $c^* $는 depth(channel 수), $h^* $와 $w^* $는 각각 높이와 너비를 의미합니다.

$$\phi_{i,j}=\phi_j(x_i), j=\{ 1,2,3,4 \}$$ 

$$\phi_{i,j}= \in \mathbb{R}^{h^* \times w^* \times c^*}$$

본 논문에서는 지나치게 추상적이거나 ImageNet 데이터셋에 편향된 feature가 추출되는 것을 막기 위해 중간 layer의 feature들을 추출해 합쳤는데요. ($j=[2,3]$)  $j$번째 block의 output $\phi_{i,j}$과 $j+1$번째 block의 output $\phi_{i,j+1}$은 크기가 다르기 때문에 $\phi_{i,j+1}$에 bilinear interpolation을 취해서 크기를 맞춰준 다음 concatenate 합니다.

다음으로는 이 feature들로 `locally aware patch feature`를 만듭니다. 특정 위치 $(h,w)$를 중심으로 patch size만큼의 주변 feature vector들을 neighborhood로 묶은 다음, **adaptive average pooling**을 취해서 만드는데요. 논문에서 사용한 patch size $p=3$이고, striding parameter $s=1$입니다. 이는 **개별 feature map들에 대한 local smoothing** 과정이라고 볼 수 있습니다. 위 이미지의 'locally aware patch features' 부분을 보면 $3 \times 3$ 크기의 feature vector들을 묶어서 adaptive average pooling을 거쳐 하나의 patch feature를 만드는 과정이 표현되어 있습니다. <u>이렇게 하면 resolution을 유지하면서도 sliding window를 사용한 것처럼 주변 context를 고려한 feature map을 만들 수 있다고 합니다.</u>

### 2. Coreset-reduced patch-feature memory bank

![](/assets/img/ad/mb.jpg)

두 번째 단계는 `Memory Bank` $\mathcal{M}$를 구성하는 단계입니다. `Memory Bank`는 정상 이미지로부터 추출한 patch feature들 중 일부(논문에서는 25%, 10%, 1%로 실험)를 sampling하여 구성하는데요. sampling에는 `greedy coreset selection` 방법을 사용했습니다. 이를 의미하는 아래 수식을 좀 더 자세히 살펴보겠습니다.

$$m_i \leftarrow \arg max_{m \in {\mathcal{M} - \mathcal{M_C}}}\min_{n \in \mathcal{M_C}} ||\psi(m)-\psi(n)||_2$$

극단적인 예로, 아래와 같이 현재까지 sampling한 $\mathcal{M}_C= \{ n_1, n_2, n_3 \}$이고, $m_1, m_2, m_3, m_4$ 중에서 다음 feature를 sampling 한다고 가정해 보겠습니다. 먼저 $m_1, m_2, m_3, m_4$ 각각을 기준으로 $n_1, n_2, n_3$과의 거리를 구한 뒤 최솟값을 찾습니다. 파란색으로 표시한 각각의 최소 거리 중 최댓값을 갖는 것은 $m_4$이므로, $m_4$를 sampling 합니다. $m_1, m_2, m_3$ 보다 $m_4$를 sampling 했을 때 `Memory Bank` 내 **feature들이 고르게 분포**함을 확인할 수 있습니다.

![](/assets/img/ad/mb2.jpeg)

### 3. Anomaly Detection with PatchCore

마지막 단계는 이전 단계에서 구한 `Memory Bank` $\mathcal{M}$을 사용하여 image-level anomaly score $s$를 구하는 것입니다.

- `Step 1` test 이미지 $x^{test}$의 patch feature들 $m^{test} \in \mathcal{P}(x^{test})$과 `Memory Bank` $\mathcal{M}$에 속한 정상 patch feature들 $m \in \mathcal{M}$ 사이의 거리를 모두 구합니다.
- `Step 2` $m^{test}$별 거리 최솟값을 찾습니다.
- `Step 3` Step 2를 최대화하는 $m^{test}$와 $m$을 찾습니다. ($m^{test, \ast}, m^\ast$)
- `Step 4` Step 3에서 구한 $m^{test, \ast}, m^\ast$을 통해 maximum distance score $s^*$을 구합니다.

$$s^*=||m^{test, *}-m^*||_2$$

- `Step 5` 최종 image-level anomaly score $s$를 구하기 위해 $s^\ast$을 scaling 합니다. scaling $w$가 의미하는 것은, anomaly score가 작아지려면($w$가 작아지려면) $m^\ast$뿐만 아니라,  $m^\ast$ 근처에 위치한 $b$개의 정상 patch feature들과의 거리도 가까워야 한다는 것입니다.

$$s=w \cdot s^* = (1-\frac{exp||m^{test, *}-m^*||_2}{\sum_{m \in \mathcal{N}_b(m^*)}exp||m^{test, *}-m||_2}) \cdot s^*$$



## 모델 성능 평가

![](/assets/img/ad/patchcore_res1.jpg)
_PatchCore 실험 결과: image level anomaly detection_

![](/assets/img/ad/patchcore_res2.jpg)
_PatchCore 실험 결과: anomaly segmentation(AUROC)_

![](/assets/img/ad/patchcore_res3.jpg)
_PatchCore 실험 결과: anomaly segmentation(PRO)_
