---
title: "Pytorch 모델 onnx 모델로 변환하기"
date: 2024-1-15
author: jieun
math: True
categories: [Model-Serving]
tags: [ONNX]
typora-root-url: ..
---

ONNX는 클라우드, 모바일, 엣지 디바이스 등에서 모두 지원되어 인프라에 구애받지 않고 동일한 모델을 여러 환경에 배포할 수 있습니다. 또한 `CUDAExecutionProvider`, `TensorRTExecutionProvider` 등 다양한 하드웨어 가속기를 지원하여 여러 하드웨어 환경에서 최적화된 성능으로 추론을 수행할 수 있게 해 줍니다.  
이러한 장점들 때문에 PyTorch, TensorFlow, Keras 등에서 학습된 모델을 ONNX 형식으로 변환하는 경우가 많습니다. 변환 후에도 모델의 구조와 가중치 데이터는 기존 모델과 동일하게 유지됩니다. [Netron](https://netron.app/)에서 ONNX 모델 파일을 업로드하면 모델 구조를 확인해볼 수 있습니다.

```python
# 모델에 대한 입력값
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# 모델 변환
torch.onnx.export(torch_model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "torch_to_onnx.onnx",      # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 Constant Folding을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})
```

## 입출력 텐서의 특정 차원을 동적으로 설정하기

ONNX는 기본적으로 입력 크기가 고정되어 있어야 합니다. 만약 모델이 동적 크기의 입력을 처리해야 한다면, `dynamic_axes`를 설정해야 합니다. `dynamic_axes`는 ONNX로 변환할 때 입력과 출력 텐서의 특정 차원을 동적으로 설정할 수 있는 기능입니다. 즉, 특정 차원의 크기를 고정하지 않고 변할 수 있도록 설정하여 다양한 입력 크기에 대해 모델을 유연하게 사용할 수 있게 합니다. 이를 통해 변환된 ONNX 모델이 다양한 배치 크기(batch size)나 이미지 크기 등을 처리할 수 있게 됩니다.  

어떤 모델이 입력으로 `(batch_size, channels, height, width)` 형태의 이미지를 받을 때, 이미지 크기와 배치 크기를 동적으로 변할 수 있도록 설정하고 싶다면 다음과 같이 작성할 수 있습니다.

```python
dynamic_axes = {
    'input': {0: 'batch_size', 2: 'height', 3: 'width'},  # 0번 축은 batch_size, 2번 축은 height, 3번 축은 width가 동적으로 변함
    'output': {0: 'batch_size'}  # 출력의 0번 축도 batch_size에 따라 변동 가능
}
```

## ONNX에서 지원하지 않는 연산을 사용하는 경우

ONNX는 다양한 프레임워크의 연산을 지원하지만, 모든 연산을 지원하지는 않습니다. 변환하려는 모델에 ONNX에서 지원하지 않는 연산이 포함되어 있다면 변환 시 오류가 발생할 수 있습니다. 예를 들어 ONNX는 `torch.nn.SyncBatchNorm` 연산을 직접 지원하지 않습니다. 이런 경우, ONNX로 모델을 내보내기 위해서는 아래와 같이 `SyncBatchNorm`을 `BatchNorm2d`로 변환해야 합니다.

```python
def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output
```
  
ONNX opset 버전별 지원하는 연산은 [이곳](https://onnx.ai/onnx/operators/)에서 확인할 수 있는데요. ONNX의 opset 버전은 ONNX의 버전과는 다르다는 점을 주의해야 합니다. [이곳](https://onnxruntime.ai/docs/reference/compatibility.html)에서 ONNX 버전과 opset 버전의 호환성을 체크하는 것이 좋습니다.

## 모델 경량화

ONNX로 모델을 변환한 후에는 다음과 같은 방법을 통해 모델을 경량화 할 수 있습니다.

### 1. FP16 변환

- 모델을 [FP16(half precision)](https://jieun121070.github.io/posts/Mixed-Precision%EA%B3%BC-Half-Precision/)으로 변환하면 모델 크기와 메모리 사용량을 절반으로 줄일 수 있습니다.
- `onnxconverter_common.float16.convert_float_to_float16`를 사용하여 ONNX 모델의 가중치를 FP16으로 변환할 수 있습니다. 하지만, 정밀도 손실이 발생할 수 있으므로 성능 저하 여부를 확인해야 합니다.

```python
from onnxconverter_common import float16
import onnx

# 기존 ONNX 모델 불러오기
model = onnx.load("model.onnx")

# FP16으로 변환
model_fp16 = float16.convert_float_to_float16(model)

# 변환된 모델 저장
onnx.save(model_fp16, "model_fp16.onnx")
```

### 2. ONNX 최적화 도구 사용

- ONNX의 `onnxoptimizer`나 `onnxruntime`에서 제공하는 최적화 도구를 사용하여 불필요한 연산을 제거하고, 연산 그래프를 최적화하여 모델 크기를 줄일 수 있습니다. 예를 들어, `Constant Folding`과 같은 최적화를 통해 모델을 조금 더 간결하게 만들 수 있습니다.
- `Constant Folding`은 모델 그래프에 포함된 **상수 연산을 사전 계산하여 모델을 최적화**하는 기법입니다. 모델을 변환할 때, 모델 그래프 내에서 상수값을 가지고 반복되는 계산이 있다면 ONNX는 이를 **한 번 계산하여 상수 값으로 고정**합니다. 이렇게 미리 계산된 상수 값은 그래프 내에 **고정된 상수 노드**로 대체됩니다. 이를 통해 모델의 실행 속도를 높이고, 그래프 내 불필요한 연산을 줄일 수 있습니다.

### 3. 양자화(Quantization)

- ONNX 모델을 INT8로 양자화하면 모델 크기를 줄이고, 특히 추론 속도를 향상할 수 있습니다. 이는 모바일 또는 엣지 디바이스에 모델을 배포할 때 유용합니다.
- `onnxruntime.quantization` 모듈을 사용하여 INT8로 양자화할 수 있습니다.
