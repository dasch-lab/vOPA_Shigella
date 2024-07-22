import torch
import torch.nn as nn
# import models
from torchvision.models import densenet121, densenet161

# import weights
from torchvision.models import DenseNet121_Weights, DenseNet161_Weights


class DenseNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, size='161', nn_classes=2):
        super(DenseNet, self).__init__()

        self.nn_classes = nn_classes
        self.adapter = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        if size == '161':
            self.encoder = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
        elif size == '121':
            self.encoder = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Size {size} not implemented")

        in_features = self.encoder.classifier.in_features

        self.encoder.classifier = nn.Identity()

        self.classifier = nn.Linear(in_features, nn_classes)

    def forward(self, x):
        """ Forward pass of the model """

        adapted = self.adapter(x)
        embedding = self.encoder(adapted)
        x = self.classifier(embedding)

        return x
