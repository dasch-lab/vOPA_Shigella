import torch
import torch.nn as nn
# import models

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

# import weights
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights


class ResNet(nn.Module):
    """ Adapter then encoder"""

    def __init__(self, in_channels=4, out_channels=3, nn_classes=2, size='s'):
        super(ResNet, self).__init__()

        self.nn_classes = nn_classes
        self.adapter = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        # Initialize pretrained model
        if size == '18':
            self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif size == '34':
            self.encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        elif size == '50':
            self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif size == '101':
            self.encoder = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        elif size == '152':
            self.encoder = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Size {size} not implemented")

        in_features = self.encoder.fc.in_features

        # set the encoder.classifier to the identity
        self.encoder.fc = nn.Identity()

        # new classifier with nn_classes output
        self.classifier = nn.Linear(in_features, nn_classes)

    def forward(self, x):
        """ Forward pass of the model """

        adapted = self.adapter(x)
        embedding = self.encoder(adapted)
        x = self.classifier(embedding)

        return x
