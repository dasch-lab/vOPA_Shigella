import torch
import torch.nn as nn
# import models
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

# import weights
from torchvision.models import EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights


class EfficientNet(nn.Module):
    """ Adapter then encoder"""

    def __init__(self, in_channels=4, out_channels=3, nn_classes=2, size='s'):
        super(EfficientNet, self).__init__()

        self.nn_classes = nn_classes
        self.adapter = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        # Initialize pretrained model
        if size == 's':
            self.encoder = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        elif size == 'm':
            self.encoder = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        elif size == 'l':
            self.encoder = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Size {size} not implemented")

        in_features = self.encoder.classifier[1].in_features

        # set the encoder.classifier to the identity
        self.encoder.classifier = nn.Identity()

        # new classifier with nn_classes output
        self.classifier = nn.Linear(in_features, nn_classes)

    def forward(self, x):
        """ Forward pass of the model """

        adapted = self.adapter(x)
        embedding = self.encoder(adapted)
        x = self.classifier(embedding)

        return x
