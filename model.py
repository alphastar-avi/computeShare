import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    """
    A simple neural network with one hidden layer.
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
