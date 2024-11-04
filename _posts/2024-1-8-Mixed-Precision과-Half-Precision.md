---
title: "Mixed Precision과 Half Precision"
date: 2024-1-8
author: jieun
math: True
categories: [Model-Serving]
tags: [Mixed-Precision, Half-Precision]
typora-root-url: ..
---

**Mixed Precision**와 **Half Precision**는 데이터 타입과 연산 방식을 최적화하여 메모리 사용량을 줄이고 연산 속도를 높이기 위해 사용하는 기법입니다. 그러나 두 가지는 사용 방식과 적용 범위에서 차이가 있습니다.

## 1. Half Precision (FP16)

Half Precision은 **FP16(16-bit floating point)**를 사용하여 모든 연산과 데이터 저장을 16비트로 처리하는 방식을 의미합니다. 모델의 모든 가중치, 활성화값(activations), 연산을 FP16 형식으로 변환하여 메모리와 계산 속도를 최적화할 수 있습니다.

Half Precision 방식을 사용하면 메모리 사용량이 절반으로 줄어들며, FP16 연산이 최적화된 GPU에서 연산 속도가 크게 향상된다는 장점이 있습니다. 하지만 **정밀도 손실**이 있을 수 있으며, 특히 FP16은 작은 값을 처리할 때 underflow가 발생할 수 있습니다. 수렴 과정이 불안정해지거나 성능이 저하될 가능성이 있습니다.

## 2. Mixed Precision

Mixed Precision은 **FP32와 FP16을 혼합하여 사용**하는 방식입니다. GPU에서 Mixed Precision Training을 수행할 때, 중요한 계산(가중치 업데이트, 손실 계산 등)은 FP32로 처리하고, 가중치와 활성화값 등 메모리 사용량이 큰 부분은 FP16으로 처리하여 메모리 절약과 속도 향상을 동시에 노립니다.

ResNet 계열의 모델들은 일반적으로 FP32로 학습되지만, PyTorch 공식 문서에 따르면 `wide_resnet50_2`와 `wide_resnet101_2` 모델은 Mixed Precision으로 학습되었습니다. 배치 정규화(Batch Normalization) 파라미터는 FP32로, 나머지 가중치는 FP16으로 저장되는데요. 배치 정규화에서 평균과 분산을 계산할 때 작은 정밀도 차이가 모델의 수렴에 영향을 줄 수 있기 때문에 FP32로 유지합니다. 

Mixed Precision 방식은 FP16과 FP32의 장점을 결합하여 **정밀도 손실 없이** 속도와 메모리 효율을 높일 수 있다는 장점이 있습니다. 하지만 혼합 정밀도 처리를 위한 추가적인 설정이 필요하고, 모든 하드웨어에서 완벽히 지원되지 않을 수 있다는 단점도 있는데요. PyTorch에서 `torch.cuda.amp`를 사용하면 Mixed Precision을 비교적 쉽게 구현할 수 있습니다.