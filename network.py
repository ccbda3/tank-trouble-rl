import torchfix

torchfix.apply()

import torch
import torch.nn as nn

import config
import tank_vec

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CTRL = torch.tensor(tank_vec.CTRL, dtype=torch.float32)
ACTIONS = CTRL.shape[0]


class DQN(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        c1, c2, c3 = config.CONV_CH
        self.conv = nn.Sequential(
            nn.Conv2d(tank_vec.OBS_CH, c1, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(c2, c3, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            probe = self.conv(torch.zeros(1, tank_vec.OBS_CH, tank_vec.OBS_H, tank_vec.OBS_W))
        flat = probe.shape[1]
        self.fc = nn.Sequential(nn.Linear(flat + feat_dim, config.HIDDEN), nn.ReLU())
        self.value = nn.Linear(config.HIDDEN, 1)
        self.adv = nn.Linear(config.HIDDEN, ACTIONS)

    def forward(self, obs, feat):
        h = self.fc(torch.cat([self.conv(obs), feat], dim=1))
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def act(self, obs, feat, eps=0.0):
        q = self(obs, feat)
        greedy = q.argmax(dim=1)
        if eps <= 0.0:
            return greedy
        noise = torch.rand_like(greedy, dtype=torch.float32) < eps
        rand = torch.randint(0, ACTIONS, greedy.shape, device=greedy.device)
        return torch.where(noise, rand, greedy)


def make(feat_dim):
    return DQN(feat_dim).to(DEVICE)


def obs_tensor(packed):
    return torch.as_tensor(tank_vec.unpack(packed), dtype=torch.float32, device=DEVICE)


def feat_tensor(feat):
    return torch.as_tensor(feat, dtype=torch.float32, device=DEVICE)
