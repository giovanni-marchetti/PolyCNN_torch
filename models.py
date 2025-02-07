import torch
import numpy as np

from torch import nn


class Polyact(nn.Module):
    def forward(self, x): 
        return 0.169 + 0.5 * x + 0.295 * x**2 - 0.0255 * x**4 + 0.00122 * x**6 - 2.107e-5 * x**8
    
    
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(19,19), stride=1, padding='same'),  
            Polyact(),
            nn.Conv2d(1, 1, kernel_size=(   11,11), stride=1, padding='valid'),  
            Polyact(),
            nn.Conv2d(1, 1, kernel_size=(9,9), stride=1, padding='valid'),  
            Polyact(),
            nn.Conv2d(1, 1, kernel_size=(7,7), stride=1, padding='valid'),  
            Polyact(),
            nn.Conv2d(1, 10, kernel_size=(4,4), stride=1, padding='valid'),  
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        return self.net(x)  .squeeze()
    
    def get_weights(self):
        weight_list = [p.flatten() for p in self.parameters()]
        return torch.cat(weight_list)