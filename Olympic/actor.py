import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class ContinuousActor(nn.Module):
    def __init__(
        self, device
    ):
        """Initialize."""
        super(ContinuousActor, self).__init__()

        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2)
        )
        
        self.cnn_net = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [16, 4, 4]

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [32, 2, 2]

            nn.Flatten()
        )

        self.hidden = nn.Linear(128, 32)

        self.mu_layer = nn.Linear(32, 2)
        self.log_std_layer = nn.Linear(32, 2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # for name, parameter in self.named_parameters():
        #     print(name, "actor parameter")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        x_1 = self.encoder(state)
        x_2 = self.cnn_net(x_1)
        # print(x_2.shape, "!@$%$%#$%#$@")
        x_3 = self.hidden(x_2)
        x_4 = self.relu(x_3)

        x_mu = self.mu_layer(x_4)
        x_log = self.log_std_layer(x_4)

        mu = self.tanh(x_mu)
        log_std = self.tanh(x_log)
        
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.sample()
        #print(mu,"!!!!")
        return action, dist