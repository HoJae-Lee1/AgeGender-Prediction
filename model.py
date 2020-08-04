import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class AgeGenModel(nn.Module):
    def __init__(self, age_clss):
        super(AgeGenModel, self).__init__()
        self.resNet = models.resnet34(pretrained=True, num_classes=256)

        self.age_pred = nn.Linear(256, age_clss)
        self.gender_pred = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.resNet.forward(x))

        age = F.softmax(self.age_pred(x), dim=1)
        gender = F.softmax(self.gender_pred(x), dim=1)

        return age, gender
