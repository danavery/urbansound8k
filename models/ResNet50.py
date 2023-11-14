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

        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)

    def train_dev_transforms(self):
        train_dev_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Lambda(lambda x: x.unsqueeze(0).repeat(3, 1, 1)),  # Add channel dimension and repeat
                    transforms.RandomResizedCrop(224, antialias=True),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Lambda(lambda x: x.unsqueeze(0).repeat(3, 1, 1)),  # Add channel dimension and repeat
                    transforms.Resize(256, antialias=True),
                    transforms.CenterCrop(224),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }
        return train_dev_transforms

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ResNet50()
