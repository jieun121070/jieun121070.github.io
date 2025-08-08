---
title: "[Paper Review] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
date: 2025-2-17
author: jieun
math: True
categories: [Language-Model]
tags: [LLM, DeepSeek, Qwen]
typora-root-url: ..
---

지난달 DeepSeek-R1은 출시와 동시에 AI 업계에 큰 반향을 일으켰습니다. 무엇보다도 GPT-4o, Claude 3.5-Sonnet과 같은 최신 모델에 필적하는 고성능 모델의 weight를 공개했다는 점에서 큰 주목을 받았습니다. DeepSeek-R1은 논문 제목에서도 알 수 있듯이, **강화 학습을 통해 추론 능력을 향상**시킨 모델입니다. 좀 더 구체적으로 설명하자면, 작년에 발표된 **DeepSeek-V3**를 기반으로 강화 학습 Post-Training을 적용해 추론 능력을 한 층 끌어올렸습니다.

작년 9월, OpenAI는 추론 모델 o1을 발표했는데요. OpenAI 연구팀은 강화 학습을 늘리고(train-time compute) 생각을 더 오래 할수록(test-time compute) o1의 성능이 일관적으로 향상하는 것을 확인했습니다.

![](/assets/img/llm/gpt_reasoning.png)

## 1. DeepSeek-V3 모델 구조

DeepSeek-R1에 대해 알아보기에 앞서, 베이스 모델인 DeepSeek-V3의 모델 구조를 살펴보겠습니다.

![](/assets/img/llm/deepseek.png)

### Mixture of Experts (MoE)

전체 파라미터 수는 6,710억 개에 달하지만, **추론 시에는 370억 개의 파라미터만 활성화**하여 효율성을 높였습니다.

### Multi-head Latent Attention (MLA)

![](/assets/img/llm/mha.png)
_Multi-head Attention_

![](/assets/img/llm/gqa_.png)
_Grouped-query Attention_

![](/assets/img/llm/mla.png)
_Multi-head Latent Attention_

MLA는 KV를 저차원 공간으로 투영했다가 복원하는 방식입니다.

## 2. Post-Training

[이전 포스트](https://jieun121070.github.io/posts/LLaMA3/)에서 다룬 Llama 3와 같은 일반적인 LLM은 Pre-training 후에 Supervised Fine-Tuning (SFT)과 강화 학습(RLHF)을 거쳐 성능을 개선합니다. DeepSeek-R1-Zero는 라벨링된 데이터 없이, **순수 강화 학습만으로 추론 능력을 향상**시킬 수 있을지 LLM의 잠재력을 탐구해보고자 했습니다. 먼저, 프롬프트는 아래와 같이 \<think>...\</think>와 \<answer>...\</answer> 형식만 지시합니다.

![](/assets/img/llm/deepseek_r1_prompt.png)

## 3. Knowledge Distillation

DeepSeek-R1로 800K 데이터를 생성해서 Qwen 7B 모델로 SFT한 다음, DeepSeek-R1의 추론 패턴(Chain-of-Thought)을 Knowledge Distillation으로 이식했습니다. 이는 [지난 포스트](https://jieun121070.github.io/posts/Alpaca/)에서 다룬 적 있는 Alpaca와 유사한 방식입니다.

## Reference

- [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)
- [LLM으로 추론하는 법 배우기](https://openai.com/ko-KR/index/learning-to-reason-with-llms/)
