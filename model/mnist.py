import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTModel(nn.Module):
    """ Simple classifier using Pytorch Lightning"""
    def __init__(self):
        super().__init__()
        # Dense layer
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def loss_func(self, y_pred, y):
        return F.cross_entropy(y_pred, y)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))