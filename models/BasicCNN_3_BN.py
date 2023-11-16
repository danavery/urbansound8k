import torch
import torch.nn as nn


class BasicCNN_3_BN(nn.Module):
    def __init__(self, input_shape=None, num_classes=10):
        super(BasicCNN_3_BN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=1)
        self.conv1a = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1)
        self.conv2a = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1)
        self.conv3a = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        with torch.no_grad():
            dummy_input = torch.zeros(
                1, 1, input_shape[0], input_shape[1]
            )  # Batch size of 1, 1 channel, input_shape[0] x input_shape[1] image
            dummy_output = self.convolute(dummy_input)
            print(
                f"input_shape: {dummy_input.shape} -> post-convolution: {dummy_output.shape}"
            )
            self.flat_features = int(torch.numel(dummy_output) / dummy_output.shape[0])

        self.fc1 = nn.Linear(self.flat_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def train_dev_transforms(self):
        return None

    def convolute(self, x):
        x = self.relu(self.bn1(self.conv1a(self.conv1(x))))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2a(self.conv2(x))))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3a(self.conv3(x))))
        x = self.pool3(x)
        return x

    def forward(self, x):
        x = self.convolute(x)
        x = x.view(-1, self.flat_features)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x
