---
title: "[AutoML] Hyperparameter Optimization 알고리즘 (2)"
date: 2022-11-8
author: jieun
math: True
categories: [AutoML]
tags: [Hyperparameter-Optimization, Grid-Search, Random-Search, Bayesian-Optimization, Population-based-training, Hyperband, BOHB]
typora-root-url: ..
---

**Hyperparameter Optimization(HPO)**

- [Exhaust search of the search space](https://jieun121070.github.io/posts/AutoML-Hyperparameter-Optimization-알고리즘/): Grid Search, Random Search
- Use of surrogate model: Bayesian Optimization, Tree-structured Parzen Estimators(TPE)
- [Algorithms dedicated to hyper-parameter tuning](https://jieun121070.github.io/posts/AutoML-Hyperparameter-Optimization-알고리즘-3/): Population-based training(PBR), Hyperband, BOHB



# Use of surrogate model

## Bayesian Optimization
![](/assets/img/hpo/hpo2.jpg){: width="550"}
- 목적: 미지의 목적함수 $f$를 상정하여 그 함수 값 $f(x)$를 최대로 만드는 최적해 $x^*$를 효율적으로 찾는 것
- Grid Search, Random Search 방법의 최적화 과정이 병렬적으로 동시에(parallel) 진행되는 것과 달리, 순차적으로(sequentially) 진행됨 (이전 시점에서 얻은 정보를 활용)
- 진행 과정
  - `step 1` 최초에 $n$개의 입력 값을 랜덤 샘플링
  - `step 2` $n$개의 입력 값 $x_1, x_2,...,x_n$에 대하여 모델을 학습하여 함수 값 $f(x_1),f(x_2),...,f(x_n)$을 도출
  - `step 3` 입력 값-함수 값 set $(x_1, f(x_1)), (x_2, f(x_2)),...,(x_n, f(x_n))$에 대하여 surrogate model(GP)를 이용하여 목적함수의 분포에 대한 확률적 추정 수행
  - `step 4` 아래 과정을 반복
    

![](/assets/img/hpo/hpo3.jpg){: width="300"}

![](/assets/img/hpo/bo.gif)

### Acquisition function
- 최적해 $x^*$를 찾기 위해 현재 시점에서 가장 유용할 만한, 다음 $x$를 선택하는 함수
- 다음 $x$를 선택하는 두 가지 전략
  - `Exploitation`: 이전 관측 값들을 토대로 Gaussian Process에 의해 추정된 목적함수 값이 큰 지점을 다음 $x$로 샘플링
  - `Exploration`: 불확실성이 큰, 탐색이 많이 이루어지지 않은 지점을 다음 $x$로 샘플링
- 시간이 제한되어 있기 때문에 `Exploitation`과 `Exploration`은 서로 trade-off 관계에 있으며, `Exploitation`-`Exploration` 간의 상대적 강도를 어떻게 조절하느냐에 따라 Acquisition function이 달라짐 (다음으로 샘플링할 $x$가 달라짐)

  ![](/assets/img/hpo/hpo4.jpg){: width="600"}



# Reference

- [하이퍼 파라미터 최적화 알고리즘 전반](https://www.yadavsaurabh.com/hyperparameter-optimisation-at-scale/)
- [Population-based training](https://www.deepmind.com/blog/population-based-training-of-neural-networks)
