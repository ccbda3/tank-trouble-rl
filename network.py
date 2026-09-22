import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, frames=4, size=24, actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(frames, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, actions),
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.fc(torch.flatten(self.net(x), 1))