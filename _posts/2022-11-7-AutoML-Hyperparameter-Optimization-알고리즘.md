---
title: "[AutoML] Hyperparameter Optimization 알고리즘 (1)"
date: 2022-11-7
author: jieun
math: True
categories: [AutoML]
tags: [Hyperparameter-Optimization, Grid-Search, Random-Search, Bayesian-Optimization, Population-based-training, Hyperband, BOHB]
typora-root-url: ..
---

AutoML은 machine learning으로 machine learning을 설계하는 것이라고 정의할 수 있습니다. 기계가 학습을 통해 스스로 최적의 모델을 찾도록 만드는 분야인데요. AutoML은 크게 Architecture search와 Hyperparameter Optimization로 나뉩니다. Architecture search는 학습을 통해 최적의 Neural Network 구조를 추정하는 것으로, 대표적인 모델은 아래와 같습니다.

**Architecture search**

- NAS - Neural Architecture Search with reinforcement learning(2017)
- NASNet - Learning Transferable Architectures for Scalable Image Recognition(2018)
- EnvelopeNets - Fast Neural Architecture Construction using EnvelopeNets(2018)
- EfficientNet - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks(2019)

Hyperparameter Optimization은 학습을 통해 최적의 하이퍼파라미터 조합을 추정하는 것이고, 아래와 같은 세부 분야로 나눌 수 있습니다.

**Hyperparameter Optimization(HPO)**

- Exhaust search of the search space: Grid Search, Random Search
- [Use of surrogate model](https://jieun121070.github.io/posts/AutoML-Hyperparameter-Optimization-알고리즘-2/): Bayesian Optimization, Tree-structured Parzen Estimators(TPE)
- [Algorithms dedicated to hyper-parameter tuning](https://jieun121070.github.io/posts/AutoML-Hyperparameter-Optimization-알고리즘-3/): Population-based training(PBR), Hyperband, BOHB



# Exhaust search of the search space

![](/assets/img/hpo/hpo1.jpg){: width="700"}

## Grid Search
- 하이퍼파라미터의 값을 일정한 간격으로 나눈 다음, 가능한 모든 조합을 적용하여 학습. 특정 구간에서 좋은 성능을 보이면, 해당 구간을 좀 더 세밀하게 나누어 적용하고, 이를 반복
- Grid Search는 범위 내에서 일정한 간격으로 탐색하므로최적의 하이퍼파라미터 값을 놓칠 가능성이 큼

## Random Search
- 하이퍼파라미터 조합을 랜덤하게 선택하여 학습
- Random Search는 $n$번의 학습을 수행할 때, 서로 다른 $n$개의 하이퍼파라미터 값을 적용할 수 있으므로 Grid Search보다 최적의 하이퍼파라미터 조합을 찾을 가능성이 높음



# Reference

- [하이퍼 파라미터 최적화 알고리즘 전반](https://www.yadavsaurabh.com/hyperparameter-optimisation-at-scale/)
- [Population-based training](https://www.deepmind.com/blog/population-based-training-of-neural-networks)
