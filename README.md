# ADBA: Approximation Decision Boundary Approach for Black-Box Adversarial Attacks

## Overview
ADBA and ADBA-md are methodologies for black-box adversarial attacks, leveraging decision boundary approximations. This README provides instructions for setting up and running these methods.

## Requirements
- **Python**: 3.11.5
- **Libraries**:
  - PyTorch 2.3.0
  - Torchvision 0.18.0

## Installation
1. **Python Setup**: Ensure that you have the correct version of Python installed. If not, download and install it from [Python's official site](https://www.python.org/downloads/release/python-3115/).

2. **Library Installation**:
   ```bash
   pip install torch==2.3.0 torchvision==0.18.0
   ```

## Models
Download the required model files into the `/code/model/` directory using the following links:
- VGG19: [Download](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)
- ResNet50: [Download](https://download.pytorch.org/models/resnet50-11ad3fa6.pth)
- Inception V3: [Download](https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth)
- Vision Transformer (ViT) B_32: [Download](https://download.pytorch.org/models/vit_b_32-d86f8d99.pth)
- EfficientNet B0: [Download](https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth)
- DenseNet161: [Download](https://download.pytorch.org/models/densenet161-8d451a50.pth)

## Dataset Setup
### MNIST
Download and prepare the MNIST dataset:
```python
import torchvision
import torchvision.transforms as transforms
test_dataset = torchvision.datasets.MNIST(root='./data/', download=True, train=False, transform=transforms.ToTensor())
```

### CIFAR-10
Download and prepare the CIFAR-10 dataset:
```python
import torchvision
import torchvision.transforms as transforms
test_dataset = torchvision.datasets.CIFAR10(root='./data/', download=True, train=False, transform=transforms.ToTensor())
```

### ImageNet
Download the ImageNet dataset from the following Kaggle link:
[ImageNet Mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/data)

## Usage
Run ADBA and ADBA-md using the following command structure. Specify the dataset and other parameters such as epsilon, binary mode (0 for ADBA, 1 for ADBA-md), the number of images, and budget.

```bash
python3 ADBA.py --dataset=mnist --epsilon=0.15 --binaryM=1 --imgnum=1000 --budget=10000
python3 ADBA.py --dataset=cifar --epsilon=0.031 --binaryM=1 --imgnum=1000 --budget=10000
python3 ADBA.py --dataset=vgg --epsilon=0.05 --binaryM=1 --imgnum=1000 --budget=10000
```

---

This reformatted README file now provides a more structured and easier to navigate documentation, improving clarity for users setting up and using the methodologies for adversarial attacks.
