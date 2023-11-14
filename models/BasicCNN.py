import torch
import torch.nn as nn


class BasicCNN(nn.Module):
    def __init__(self, input_shape=None, num_classes=10):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        with torch.no_grad():
            dummy_input = torch.zeros(
                1, 1, input_shape[0], input_shape[1]
            )  # Batch size of 1, 1 channel, input_shape[0] x input_shape[1] image
            dummy_output = self.convolute(dummy_input)
            self.flat_features = int(torch.numel(dummy_output) / dummy_output.shape[0])
        self.fc1 = nn.Linear(self.flat_features, num_classes)

    def train_dev_transforms(self):
        return None

    def convolute(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        return x

    def preprocess(self, x):
        return x

    def forward(self, x):
        x = self.convolute(x)
        x = x.view(-1, self.flat_features)
        x = self.fc1(x)
        return x
