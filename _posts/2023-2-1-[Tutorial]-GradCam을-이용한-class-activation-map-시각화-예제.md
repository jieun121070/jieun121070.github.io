---
title: "[Tutorial] GradCam을 이용한 class activation map 시각화 예제"
date: 2023-2-1
author: jieun
math: True
categories: [XAI]
tags: [GradCam, ResNet, Vision-Transformer, Swin-Transformer, ConvNext, Image-Classificaion]
typora-root-url: ..
---

이번 포스트에서는 ResNet, Vision Transformer, SWIN Transformer, ConvNeXt 모델로 이미지를 분류하고, GradCam을 이용해 class activation map을 시각화하는 예제를 공유하고자 합니다. 전체 코드는 [링크](https://github.com/jieun121070/gradcam-tutorial)에서 확인하실 수 있습니다. 각 모델에 대한 자세한 설명은 이전 포스트를 참고해 주세요!

- [ResNet](https://jieun121070.github.io/posts/Resnet/)
- [Vision Transformer](https://jieun121070.github.io/posts/paper-review-An-Image-is-Worth-16x16-Words-Transformers-for-Image-Recognition-at-Scale/)
- [SWIN Transformer](https://jieun121070.github.io/posts/paper-review-Swin-Transformer-Hierarchical-Vision-Transformer-using-Shifted-Windows/)
- [ConvNeXt](https://jieun121070.github.io/posts/paper-review-A-ConvNet-for-the-2020s/) 



먼저, 이번 예제에서는 `timm` 라이브러리를 사용해 pretrained 모델을 불러오려고 하는데요. `timm.list_models`로 `timm`에서 제공하는 모델 리스트를 확인할 수 있습니다. 아래와 같이 실행하면 ConvNeXt 모델 리스트를 확인할 수 있습니다.

```python
timm.list_models('convnext*')
```

다음으로, `get_args` 함수로 테스트 이미지의 경로 및 class activation map을 시각화할 method를 선택합니다. default 값을 변경하면 GradCam 외에 다른 method의 결과도 확인할 수 있습니다.

```python
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='_examples',
        help='Input image folder name')
    parser.add_argument(
        '--image-name',
        type=str,
        default='horses.jpeg',
        help='Input image file name')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam', 'ablationcam',
                 'eigencam', 'eigengradcam', 'layercam', 'fullgrad'])

    args = parser.parse_args(args=[])
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args
```

아래 함수는 tensor를 (B, C, H, W) shape으로 변환하는 함수입니다.

```python
def reshape_transform(tensor):    
    if len(tensor.size()) == 4:
      if tensor.size(1) == tensor.size(2):
        # tensor shape이 아래와 같이 [B, H, W, C]
        # ex) tensor.shape = torch.Size([1, 7, 7, 768])
        result = tensor.transpose(2, 3).transpose(1, 2)
        # result.shape = torch.Size([1, 768, 7, 7])
      elif tensor.size(2) == tensor.size(3):
        # tensor shape이 아래와 같이 [B, C, H, W]이면 그대로 사용
        # ex) tensor.shape = torch.Size([1, 768, 7, 7])
        result = tensor

    elif len(tensor.size()) == 3:
      if math.sqrt(tensor.size(1)) % 1 == 0:
        height = width = int(math.sqrt(tensor.size(1)))
        result = tensor.reshape(tensor.size(0),
                                height, width, tensor.size(2))
      else:
        height = width = int(math.sqrt(tensor.size(1)-1))
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                          height, width, tensor.size(2))
      result = result.transpose(2, 3).transpose(1, 2)
    
    return result
```

입력 이미지를 전처리해서 `input_tensor` 변수에 저장합니다. `org_img`는 결과 이미지를 만들기 위해 원본 이미지를 저장해 두는 변수입니다.

```python
if __name__ == '__main__':
    """ python swinT_example.py -image-path 
    Example usage of using cam-methods on a SwinTransformers network.
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    rgb_img = cv2.imread(os.path.join(args.image_path, args.image_name), 1)
    org_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = rgb_img[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
```

`timm.create_model` 함수로 pretrained model을 불러옵니다. `print(model)`로 모델별 구조를 확인하고 시각화할 `target_layers`를 설정합니다.

```python
    for model_name in ["ResNet50", "ViT", "SwinT", "ConvNext"]:
      if model_name == "ResNet50":
        model = timm.create_model('resnet50', pretrained=True)
        target_layers = [model.layer4]
      elif model_name == "ViT":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        target_layers = [model.blocks[-1].norm1]
      elif model_name == "SwinT":
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        target_layers = [model.layers[-1].blocks[-1].norm2]
      elif model_name == "ConvNext":
        model = timm.create_model('convnext_base', pretrained=True)
        target_layers = [model.stages[-1].blocks[-1].norm]

      model.eval()
```

특정 class $C$에 대한 결과를 확인하려면 targets를 `[ClassifierOutputTarget(C의 class index)]`로 설정합니다. 아래와 같이 targets를 `None`으로 설정하면 classification score가 가장 높은 클래스에 대한 결과를 보여줍니다.

```python
      if args.use_cuda:
          model = model.cuda()
    
      if args.method not in methods:
          raise Exception(f"Method {args.method} not implemented")
    
      if args.method == "ablationcam":
          cam = methods[args.method](model=model,
                                     target_layers=target_layers,
                                     use_cuda=args.use_cuda,
                                     reshape_transform=reshape_transform,
                                     ablation_layer=AblationLayerVit())
      else:
          cam = methods[args.method](model=model,
                                     target_layers=target_layers,
                                     use_cuda=args.use_cuda,
                                     reshape_transform=reshape_transform)
    
      cam.batch_size = 32

      grayscale_cam = cam(input_tensor=input_tensor,
                          targets=None,
                          eigen_smooth=args.eigen_smooth,
                          aug_smooth=args.aug_smooth)

      grayscale_cam = grayscale_cam[0, :]
```

`show_cam_on_image` 함수로 `rgb_img` 위에 class activation map을 시각화하고, 모델별 결과를 이어 붙입니다.

```python
      cam_image = show_cam_on_image(rgb_img, grayscale_cam)
      org_img = np.hstack((org_img, cam_image))
```

결과 이미지를 _results 폴더에 저장합니다.

```python
    cv2.imwrite('_results/result_{}.jpg'.format(args.image_name.split(".")[0]),
                org_img)
```
