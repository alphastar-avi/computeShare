import torch
import torch.nn as nn
import torchvision.models as models

class SimpleNet(nn.Module):
    """
    A lightweight standard PyTorch ResNet-18 model modified for the MNIST dataset.
    Since MNIST is a 1-channel (grayscale) image dataset, we need to adapt 
    the first convolutional layer and the final fully connected layer.
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Load an untrained ResNet18 model
        self.resnet = models.resnet18(weights=None)
        
        # Modify the first Convolutional Layer to accept 1-channel images instead of 3 (RGB)
        # keeping the other original parameters intact.
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final Fully Connected Layer to output exactly 10 classes (digits 0-9)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.resnet(x)

