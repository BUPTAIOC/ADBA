# coding:utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(1024, 200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 1
        layers += [nn.Conv2d(in_channels, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.Conv2d(32, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(32, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(3200, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 3
        layers += [nn.Conv2d(in_channels, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(64, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.Conv2d(128, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)




class GeneralTorchModel(nn.Module):
    def __init__(self, model, n_class=10, im_mean=None, im_std=None):
        super(GeneralTorchModel, self).__init__()
        self.model = model
        self.model.eval()
        self.num_queries = 0
        self.im_mean = im_mean
        self.im_std = im_std
        self.n_class = n_class

    def forward(self, image):
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        logits = self.model(image)
        return logits

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            processed = torch.from_numpy(image).type(torch.FloatTensor)
        else:
            processed = image

        if self.im_mean is not None and self.im_std is not None:
            im_mean = torch.tensor(self.im_mean).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            im_std = torch.tensor(self.im_std).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            processed = (processed - im_mean) / im_std
        return processed

    def predict_prob(self, image):
        with torch.no_grad():
            if len(image.size()) != 4:
                image = image.unsqueeze(0)
            image = self.preprocess(image)
            logits = self.model(image)
            self.num_queries += image.size(0)
        return logits

    def predict_label(self, image):
        logits = self.predict_prob(image)
        _, predict = torch.max(logits, 1)
        return predict
