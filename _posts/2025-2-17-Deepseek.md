---
title: "[Paper Review] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
date: 2025-2-17
author: jieun
math: True
categories: [Language-Model]
tags: [LLM, DeepSeek, Qwen]
typora-root-url: ..
---

작년 9월, OpenAI는 추론 모델 o1을 발표했는데요. OpenAI 연구팀은 강화 학습을 늘리고(train-time compute) 생각을 더 오래 할수록(test-time compute) o1의 성능이 일관적으로 향상하는 것을 확인했습니다. 하지만 모델이 생각하는 시간을 무한정 늘릴 수는 없기 때문에 어떻게 하면 이 test-time을 효율적으로 사용할 것인지가 중요한 문제로 대두되었습니다.

![](/assets/img/llm/gpt_reasoning.png)

DeepSeek 연구팀은 이 문제를 강화 학습을 통해 풀고자 했습니다. 논문 제목에서도 알 수 있듯이 DeepSeek-R1은 **강화 학습을 통해 추론 능력을 향상**시킨 모델로, GPT-4o, Claude 3.5-Sonnet과 같은 최신 모델에 필적하는 성능을 달성했습니다. 게다가 이 고성능 모델의 weight를 공개해 학술적 연구뿐만 아니라 상업적 용도로 자유롭게 사용할 수 있도록 했다는 점에서 출시와 동시에 AI 업계에 큰 반향을 일으켰습니다. 이번 포스트에서는 DeepSeek-R1이 경쟁 모델들과 어떤 차이가 있는지, 구체적으로 어느 정도의 성능을 달성한 것인지 자세히 살펴보겠습니다. 

## 1. DeepSeek-V3 모델 구조

DeepSeek-R1은 작년에 발표된 **DeepSeek-V3**를 기반으로 강화 학습 Post-Training을 적용해 추론 능력을 한 층 끌어올린 모델입니다. DeepSeek-R1에 대해 알아보기에 앞서, 베이스 모델인 DeepSeek-V3의 모델 구조를 살펴보겠습니다. DeepSeek-V3 모델 구조에서 주목할 만한 점은 **MoE Transformer**와 **MLA**를 사용했다는 것입니다.

![](/assets/img/llm/deepseek.png)

### Mixture of Experts (MoE)

[지난 포스트](https://jieun121070.github.io/posts/Qwen/)에서 설명했듯이, MoE Transformer를 사용하면 전체 파라미터를 항상 사용하는 것이 아니라 필요한 부분만 활성화하기 때문에 계산 비용을 절감하면서도 강력한 성능을 유지할 수 있습니다. DeepSeek-V3의 전체 파라미터 수는 6,710억 개에 달하지만, 각 토큰을 처리할 때 **실제로 활성화되는 파라미터는 370억 개**에 불과합니다.

DeepSeek-V3의 MoE 구조에는 두 가지 유형의 전문가(expert)가 있는데요. 위 그림에서 초록색 부분에 해당하는 **Shared Expert**와 하늘색 부분에 해당하는 **Routed Expert**입니다. Shared Expert는 모든 토큰에 대해 항상 활성화되는 전문가입니다. 주로 공통적인 언어 규칙이나 기본적인 문법과 같은 일반적인 지식을 처리하는 역할을 담당합니다. 모든 토큰이 거쳐야하는 기본 처리 과정을 담당하기 때문에, 모델의 안정성과 효율성을 높이는 데 기여합니다. 반면, Routed Expert는 토큰별로 다르게 선택되어 활성화됩니다. 여기에서는 코딩, 수학, 특정 도메인 지식 등 특화된 영역에 대한 정보를 처리합니다. 논문에 따르면, MoE layer당 1개의 Shared Expert와 256개의 Routed Expert가 있습니다. 그리고 각 토큰에 대해 256개의 Routed Expert 중 8개의 전문가가 선택되어 활성화됩니다.

### Multi-head Latent Attention (MLA)

![](/assets/img/llm/mha.png)
_Multi-head Attention_

![](/assets/img/llm/gqa_.png)
_Grouped-query Attention_

![](/assets/img/llm/mla.png)
_Multi-head Latent Attention_

MLA는 KV를 저차원 공간으로 투영했다가 복원하는 방식입니다.

## 2. Post-Training

[이전 포스트](https://jieun121070.github.io/posts/LLaMA3/)에서 다룬 Llama 3와 같은 일반적인 LLM은 Pre-training 후에 Supervised Fine-Tuning (SFT)과 강화 학습(RLHF)을 거쳐 성능을 개선합니다. DeekSeek 연구팀은 라벨링된 데이터 없이, **순수 강화 학습만으로 추론 능력을 향상**시킬 수 있을지 LLM의 잠재력을 탐구해보고자 했습니다. 이를 위해 학습된 모델이 **DeepSeek-R1-Zero** 입니다. 먼저, 프롬프트는 아래와 같이 \<think>...\</think>와 \<answer>...\</answer> 형식만 지시합니다.

![](/assets/img/llm/deepseek_r1_prompt.png)

## 3. Knowledge Distillation

DeepSeek-R1로 800K 데이터를 생성해서 Qwen 7B 모델로 SFT한 다음, DeepSeek-R1의 추론 패턴(Chain-of-Thought)을 Knowledge Distillation으로 이식했습니다. 이는 [지난 포스트](https://jieun121070.github.io/posts/Alpaca/)에서 다룬 적 있는 Alpaca와 유사한 방식입니다.

## Reference

- [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)
- [LLM으로 추론하는 법 배우기](https://openai.com/ko-KR/index/learning-to-reason-with-llms/)
