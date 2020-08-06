import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class AgeGenModel(nn.Module):
    def __init__(self, age_clss):
        super(AgeGenModel, self).__init__()
        self.resNet = models.resnet101(pretrained=True)
        self.fc1 = nn.Linear(2048, 256)
        self.age_pred = nn.Linear(256, age_clss)
        self.gender_pred = nn.Linear(256, 2)

    def forward(self, x):
        x = self.resNet.conv1(x)
        x = self.resNet.bn1(x)
        x = self.resNet.relu(x)
        x = self.resNet.maxpool(x)

        x = self.resNet.layer1(x)
        x = self.resNet.layer2(x)
        x = self.resNet.layer3(x)
        x = self.resNet.layer4(x)
        x = self.resNet.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = F.relu(self.fc1(x))
        age = F.softmax(self.age_pred(x), dim=1)
        gender = F.softmax(self.gender_pred(x), dim=1)
        return age, gender
    # def __init__(self, age_clss):
    #     super(AgeGenModel, self).__init__()
    #     self.resNet = models.resnet34(pretrained=True, num_classes=256)
    #
    #     self.age_pred = nn.Linear(256, age_clss)
    #     self.gender_pred = nn.Linear(256, 2)
    #
    # def forward(self, x):
    #     x = F.relu(self.resNet.forward(x))
    #
    #     age = F.softmax(self.age_pred(x), dim=1)
    #     gender = F.softmax(self.gender_pred(x), dim=1)
    #
    #     return age, gender
