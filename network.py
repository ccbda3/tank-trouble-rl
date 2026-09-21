import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_dim, hidden=128, actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, actions),
        )

    def forward(self, x):
        return self.net(x)
