# ADBA:Approximation Decision Boundary Approach for Black-Box Adversarial Attacks

1. Download python and pip install python packages:
Python 3.11.5
PyTorch 2.3.0
Torchvision 0.18.0


2. Models 
Download model pt files to /code/model/ from urls:
https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
https://download.pytorch.org/models/resnet50-11ad3fa6.pth
https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth
https://download.pytorch.org/models/vit_b_32-d86f8d99.pth
https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth
https://download.pytorch.org/models/densenet161-8d451a50.pth


3. Dataset
#The mnist dataset is available without Download
Download MNIST dataset to /code/data/MNIST/ using python code:
import torchvision
import torchvision.transforms as transforms
test_dataset = torchvision.datasets.MNIST(root='./data/', download=True, train=False, transform=transforms.ToTensor())

Download cifar-10 dataset to /code/data/cifar10-py/ using python code:
import torchvision
import torchvision.transforms as transforms
test_dataset =torchvision.datasets.CIFAR10(root='./data/', download=True, train=False, transform=transforms.ToTensor())

Download ImageNet dataset to /code/data/imagenet/ from url: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/data


4. Run ADBA and ADBA-md
--dataset={mnist,cifar,vgg,resnet50,inception_v3,vit,efficient,densenet}
--binaryM={0 for ADBA, 1 for ADBA-md}
python3 ADBA.py --dataset=mnist  --epsilon=0.15  --binaryM=1  --imgnum=1000 --budget=10000
python3 ADBA.py --dataset=cifar  --epsilon=0.031  --binaryM=1  --imgnum=1000 --budget=10000
python3 ADBA.py --dataset=vgg  --epsilon=0.05  --binaryM=1  --imgnum=1000 --budget=10000
