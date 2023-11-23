import torch.nn as nn
import torchaudio
from torchvision.models import resnet50
from torchvision import transforms


class ResNet50_5(nn.Module):
    def __init__(self, input_shape=None, num_classes=10):
        super(ResNet50_5, self).__init__()
        self.model = resnet50(weights="IMAGENET1K_V2")
        for param in self.model.parameters():
            param.requires_grad = False

        for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4, self.model.fc]:
            for param in layer.parameters():
                param.requires_grad = True

        num_ftrs = self.model.fc.in_features
        print(f"{num_ftrs=}")
        self.fc_layers = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Optional: Dropout for regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Optional: Dropout for regularization
            nn.Linear(256, num_classes)
        )

        self.model.fc = self.fc_layers

    def train_dev_transforms(self):
        train_dev_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Lambda(lambda x: x.unsqueeze(0)),
                    torchaudio.transforms.TimeMasking(time_mask_param=80, iid_masks=True),
                    torchaudio.transforms.FrequencyMasking(freq_mask_param=80, iid_masks=True),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Add channel dimension and repeat
                    transforms.Resize(224, antialias=True),  # Resize the shorter side to 224, maintaining aspect ratio
                    transforms.Pad((0, 0, 224, 224), fill=0, padding_mode='constant'),  # Pad to make it 224x224
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Lambda(lambda x: x.unsqueeze(0).repeat(3, 1, 1)),  # Add channel dimension and repeat
                    transforms.Resize(224, antialias=True),  # Resize the shorter side to 224, maintaining aspect ratio
                    transforms.Pad((0, 0, 224, 224), fill=0, padding_mode='constant'),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }
        return train_dev_transforms

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ResNet50_5()
