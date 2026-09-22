import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, frames=4, size=64, actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(frames, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )
        conv_out = self._conv_out(frames, size)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, actions),
        )

    def _conv_out(self, frames, size):
        with torch.no_grad():
            x = torch.zeros(1, frames, size, size)
            x = self.net(x)
            return int(x.reshape(1, -1).size(1))

    def forward(self, x):
        x = x.float() / 255.0
        return self.fc(torch.flatten(self.net(x), 1))