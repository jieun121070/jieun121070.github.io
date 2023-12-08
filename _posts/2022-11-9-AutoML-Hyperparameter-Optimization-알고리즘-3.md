---
title: "[AutoML] Hyperparameter Optimization 알고리즘 (3)"
date: 2022-11-9
author: jieun
math: True
categories: [AutoML]
tags: [Hyperparameter-Optimization, Grid-Search, Random-Search, Bayesian-Optimization, Population-based-training, Hyperband, BOHB]
typora-root-url: ..
---

**Hyperparameter Optimization(HPO)**

- [Exhaust search of the search space](https://jieun121070.github.io/posts/AutoML-Hyperparameter-Optimization-알고리즘/): Grid Search, Random Search
- [Use of surrogate model](https://jieun121070.github.io/posts/AutoML-Hyperparameter-Optimization-알고리즘-2/): Bayesian Optimization, Tree-structured Parzen Estimators(TPE)
- Algorithms dedicated to hyper-parameter tuning: Population-based training(PBR), Hyperband, BOHB



# Algorithms dedicated to hyper-parameter tuning

## Exhaust Search, Bayesian Optimization과의 차이점
- Exhaust Search와 Bayesian Optimization 방법을 이용하면 많은 시간을 투자할수록 최적의 하이퍼파라미터 조합을 찾을 가능성은 높아지지만, 현실적으로 제한된 시간을 투자할 수밖에 없음.
- 제한된 시간 내에서 수행 가능한 모든 하이퍼파라미터 조합의 학습이 종료되어야 최종적으로 가장 성능이 높은 조합을 선택할 수 있으며, 이 때 선택된 조합이 정말 최적의 조합이 맞는지 판단하기 어려움.
- Population-based training(PBT)와 Hyperband는 성능이 낮은 조합의 학습을 early-stopping함으로써 주어진 resource를 효율적으로 사용하고, 소요 시간을 단축할 수 있다는 장점이 있음.

## Population-based training(PBT) 
![](/assets/img/hpo/pbt.jpg)
- Random Search와 Hand-tuning 방법을 결합한 것으로, 유전 알고리즘을 적용.
- Population-based training 진행과정
  - `step 1` n개의 하이퍼파라미터 조합을 랜덤 샘플링하고, 병렬적으로 동시에(parallel) 학습
  - `step 2` 성능이 높은 모델의 하이퍼파라미터를 복제(exploit)
  - `step 3` 복제한 하이퍼파라미터를 그대로 적용하는 것이 아니라, 변형시켜서 새로운 하이퍼파라미터 조합을 얻고, 이를 적용하여 학습을 이어 나감(explore) (반복)
![](/assets/img/hpo/pbt.gif)

## Hyperband
![](/assets/img/hpo/hyperband.jpg){: width="600"}

- Hyperband는 Successive Halving 기법을 확장하여 제안된 기법.
- Successive Halving 진행과정
  - `step 1` $n$개의 하이퍼파라미터 조합을 랜덤 샘플링.
  - `step 2` 전체 $resource(B)/n$ 만큼의 resource를 $n$개의 하이퍼파라미터 조합에 각각 할당하여 학습.
  - `step 3` 절반 혹은 일정 비율 이상의 상위 조합만 남기고 나머지는 버림. (반복)
- Successive Halving은 성능이 높은 조합에 resource를 집중시킬 수 있다는 장점이 있지만, 최초에 랜덤 샘플링하는 하이퍼파라미터 조합의 개수 $n$을 결정해야 함.
- $n$이 크면 하이퍼파라미터 조합별 학습 시간이 짧아지기 때문에, 성능이 높지만 수렴 속도가 느린 조합을 놓칠 수 있음. 반대로 $n$이 작으면 성능이 낮은 조합에도 지나치게 많은 resource를 할당할 수 있으므로 비효율적.
- $n$을 결정할 사전정보가 없는 한 이러한 trade-off를 감수해야 한다는 것이 Successive Halving의 한계점.
- 반면에 Hyperband는 주어진 resource 하에서 가능한 몇 개의 $n$에 대하여 grid search를 수행하기 때문에 $n$을 선택할 필요가 없음.

![](/assets/img/hpo/hyperband2.jpg){: width="600"}

## BOHB
- Bayesian Optimization과 Hyperband 방법을 결합한 것.
- 랜덤 샘플링을 통해 하이퍼파라미터 조합을 초기화하고, 이전 시점에 샘플링한 조합으로부터 얻을 수 있는 정보를 활용하지 않는다는 Hyperband 기법의 한계점을 개선하고자 함.
- n개의 하이퍼파라미터 조합을 랜덤 샘플링하지 않고 Bayesian Optimization을 사용하여 샘플링.
- n개의 하이퍼파라미터 조합에 resource를 할당하는 과정은 hyperband와 동일

![](/assets/img/hpo/bohb.jpg){: width="450"}



# Reference

- [하이퍼 파라미터 최적화 알고리즘 전반](https://www.yadavsaurabh.com/hyperparameter-optimisation-at-scale/)
- [Population-based training](https://www.deepmind.com/blog/population-based-training-of-neural-networks)
