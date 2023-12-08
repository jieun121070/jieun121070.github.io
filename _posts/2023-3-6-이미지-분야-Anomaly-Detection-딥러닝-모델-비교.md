---
title: "이미지 분야 Anomaly Detection 딥러닝 모델 비교"
date: 2023-3-13
author: jieun
math: True
categories: [Vision]
tags: [Anomaly-Detection, MahalanobisAD, SPADE, PaDiM, PatchCore, CFA]
typora-root-url: ..
---

본 포스트에서는 이미지 분야에서 **Anomaly Detection**과 **Anomaly Localization**을 위해 고안된 딥러닝 모델들을 비교해 보려고 합니다. Anomaly Detection이 Anomaly Localization을 포괄하는 용어로써 자주 사용되고 있는 것 같긴 하지만 좀 더 구체적으로 용어를 정의해 본다면,  Anomaly Detection은 이미지 전체에 anomaly score를 부여하여 이미지 자체의 정상, 비정상을 분류합니다.  Anomaly Localization은 이미지 내 각각의 픽셀에 anomaly score를 부여하여 이미지의 어느 부분에서 비정상성이 나타나는지까지 보여줍니다. Anomaly Detection 보다 좀 더 까다롭지만 더 정확하고, 해석 가능한 결과를 도출한다는 장점이 있습니다.

모델들을 비교하기에 앞서, 먼저 Anomaly Detection 문제 해결이 어려운 이유를 살펴보면 다음과 같습니다.

- 비정상 이미지가 극히 적은 데이터 불균형 문제
- 비정상 패턴이 모호하고, 테스트 시에 새로운 패턴이 나타날 가능성이 있음
- 특히 산업 이미지 데이터는 비정상 패턴이 다양해서 까다로움

위와 같은 문제 때문에 많은 Anomaly Detection 모델들이 정상 데이터만을 학습에 사용하는 `one class classification` 방식을 사용하는데요. 오늘 비교 대상 모델들도 모두 one class classification 방식을 사용합니다. 또한 ImageNet 데이터 기반의 pre-trained CNN 모델을 사용해 이미지의 feature를 추출하고, 이 때 **여러 layer로부터 구한 feature를 종합적으로 고려**하는 방식을 사용했다는 공통점이 있습니다. 여러 level의 feature를 사용한 이유는 detection 및 localization 성능을 향상시키기 위함인데요. CNN 모델의 마지막 layer로부터 구한 feature만 사용하면 아래와 같은 문제가 발생합니다.

- 입력 이미지는 convolution layer를 거치면서 점차 위치 정보를 잃게 됩니다. 마지막 layer에서는 위치 정보가 거의 존재하지 않는 추상적인 feature를 생성합니다.
- 마지막 layer에서는 ImageNet 데이터 기반의 classification task에 편향된 feature를 생성합니다.

| <center>모델 명</center>                                     | <center>Official<br />Code</center>                          | <center>분석 단위</center>               | <center>Anomaly Score<br />산출 방식</center>    | <center>Memory Bank<br />크기</center>                       | <center>Pre-trained<br />CNN</center>                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [SPADE](https://jieun121070.github.io/posts/paper-review-Sub-Image-Anomaly-Detection-with-Deep-Pyramid-Correspondences/) |                                                              | <center>weaker<br />image level</center> | <center>KNN</center>                             | <center>$\mathcal{G} \in \mathbb{R}^{\left\vert \mathcal{X} \right\vert \times H \times W \times D}$</center> | <center>wide ResNet50</center>                               |
| [Mahalanobis AD](https://jieun121070.github.io/posts/paper-review-Modeling-the-Distribution-of-Normal-Data-in-Pre-Trained-Deep-Features-for-Anomaly-Detection/) | [link](https://github.com/ORippler/gaussian-ad-mvtec)        | <center>image level</center>             | <center>MVG +<br />Mahalanobis distance</center> | <center>$\mathcal{N}(\mu, \Sigma) \in \mathbb{R}^{H \times W \times D^2}$</center> | <center>EfficientNet<br />ResNet</center>                    |
| [PaDiM](https://jieun121070.github.io/posts/paper-review-PaDiM-a-Patch-Distribution-Modeling-Framework-for-Anomaly-Detection-and-Localization/) |                                                              | <center>patch level</center>             | <center>MVG +<br />Mahalanobis distance</center> | <center>$\mathcal{N}(\mu, \Sigma) \in \mathbb{R}^{H \times W \times D^2}$</center> | <center>ResNet18<br />wide ResNet50<br />EfficientNet-B5</center> |
| [PatchCore](https://jieun121070.github.io/posts/paper-review-Towards-Total-Recall-in-Industrial-Anomaly-Detection/) | [link](https://github.com/amazon-science/patchcore-inspection) | <center>patch level</center>             |                                                  | <center>$\mathcal{M} \in \mathbb{R}^{\left\vert \mathcal{X} \right\vert \times \gamma(H \times W) \times D^\prime}$</center> | <center>wide ResNet50</center>                               |
| [CFA](https://jieun121070.github.io/posts/Paper-Review-CFA-Coupled-hypersphere-based-Feature-Adaptation/) | [link](https://github.com/sungwool/CFA_for_anomaly_localization) | <center>patch level</center>             |                                                  | <center>$\mathcal{C} \in \mathbb{R}^{\gamma(H \times W \times D)}$</center> | <center>ResNet18<br />wide ResNet50<br />EfficientNet-B5</center> |

