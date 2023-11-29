import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms


class ResNet50(nn.Module):
    def __init__(self, input_shape=None, num_classes=10):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights="IMAGENET1K_V2")
        for param in self.model.parameters():
            param.requires_grad = False

        for layer in [self.model.layer3, self.model.layer4, self.model.fc]:
            for param in layer.parameters():
                param.requires_grad = True

        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)

    def train_dev_transforms(self):
        train_dev_transforms = {
            "train": transforms.Compose(
                [
                    UnsqueezeTransform(),
                    RepeatTransform(),
                    transforms.Resize(
                        224, antialias=True
                    ),  # Resize the shorter side to 224, maintaining aspect ratio
                    transforms.Pad(
                        (0, 0, 224, 224), fill=0, padding_mode="constant"
                    ),  # Pad to make it 224x224
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    UnsqueezeTransform(),
                    RepeatTransform(),
                    transforms.Resize(
                        224, antialias=True
                    ),  # Resize the shorter side to 224, maintaining aspect ratio
                    transforms.Pad((0, 0, 224, 224), fill=0, padding_mode="constant"),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }
        return train_dev_transforms

    def forward(self, x):
        return self.model(x)


# replacing Lambda transforms with these because of multiprocessing issues
class UnsqueezeTransform:
    def __call__(self, x):
        x = x.unsqueeze(0)
        return x


class RepeatTransform:
    def __call__(self, x):
        x = x.repeat(3, 1, 1)
        return x


if __name__ == "__main__":
    model = ResNet50()
