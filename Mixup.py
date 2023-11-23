import torch
from torch.distributions.beta import Beta


class Mixup:
    def __init__(self, alpha, device):
        self.alpha = alpha
        self.device = device

    def mixup_data(self, x, y):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if self.alpha > 0:
            lam = self.sample_beta_distribution(device=self.device)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def sample_beta_distribution(self, size=1):
        distribution = Beta(
            torch.full((size,), self.alpha, device=self.device),
            torch.full((size,), self.alpha, device=self.device),
        )
        return distribution.sample()

    def mixup_loss(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
