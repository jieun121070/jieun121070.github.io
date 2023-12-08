---
title: "BERT, ELMo, GPT-2 모델 비교"
date: 2023-4-3
author: jieun
math: True
categories: [NLP]
tags: [GPT, ELMo, BERT, Transformer]
typora-root-url: ..
---

[ELMo](https://jieun121070.github.io/posts/Paper-Review-Deep-contextualized-word-representations/), [BERT](https://jieun121070.github.io/posts/BERT/), [GPT-2](https://jieun121070.github.io/posts/Paper-Review-Improving-Language-Understanding/)는 모두 `contextualized word representation`을 생성하는 모델입니다. `contextualized word representation`은 단어들이 등장하는 문맥에 따라 서로 다른 vector를 갖습니다. 다의어의 경우 의미에 따라 여러 개의 vector를 갖게 되는 것인데요. 이러한 특성은 문맥에 관계없이 단어 당 하나의 vector만을 갖는 `static word embedding`과 대조됩니다.

세 모델이 생성하는 word representation의 특성은 동일하지만, 모델 구조 상에는 큰 차이가 있습니다. 세 모델의 차이점을 간단히 비교해 보면 아래와 같습니다.

![](/assets/img/bert/comparison.PNG)

|                    |          ELMo           |                      BERT                      |                   GPT-2                    |
| :----------------: | :---------------------: | :--------------------------------------------: | :----------------------------------------: |
| 기본적인 모델 구조 | bidirectional<br />LSTM |            Transformer<br />encoder            |          Transformer<br />decoder          |
|   pre-trained LM   |  bidirectional<br />LM  | Masked LM<br />& Next Sentence Prediction(NSP) | Multitask Learning<br />방식으로 학습한 LM |

ELMo는 BERT, GPT-2와 달리 [LSTM](https://jieun121070.github.io/posts/Language-Model-n-gram%EC%97%90%EC%84%9C-RNN%EC%9C%BC%EB%A1%9C%EC%9D%98-%EB%B0%9C%EC%A0%84/) 기반 모델입니다.

![](/assets/img/bert/context1.PNG)

![](/assets/img/bert/context2.PNG)

![](/assets/img/bert/context3.PNG)



# Reference

- [How Contextual are Contextualized Word Representations?](https://ai.stanford.edu/blog/contextual/)

- [how-bert-and-gpt-models-change-the-game-for-nlp](https://www.ibm.com/blog/how-bert-and-gpt-models-change-the-game-for-nlp/)

