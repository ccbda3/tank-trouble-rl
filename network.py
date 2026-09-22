import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRESH_KEYS = ("fc.0.",)


class DQN(nn.Module):
    def __init__(self, frames=4, size=32, actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(frames, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(64, 16, 3, 1, padding=3, dilation=3), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * size * size, 512), nn.ReLU(),
            nn.Linear(512, actions),
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.fc(torch.flatten(self.net(x), 1))
