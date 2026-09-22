from collections import deque

import cv2
import numpy as np

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers.joypad_space import JoypadSpace


class MarioEnv:
    def __init__(self, frames=4, size=84):
        self.gymenv = JoypadSpace(
            gym_super_mario_bros.make("SuperMarioBros-v0", apply_api_compatibility=True),
            SIMPLE_MOVEMENT,
        )
        self.frames = frames
        self.size = size
        self.history = deque(maxlen=frames)
        self.raw = None
        self._score = 0.0
        self._max_x = 0.0
        self._steps = 0

    @property
    def score(self):
        return self._score

    def _pre(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return gray

    def _stack(self):
        return np.stack(self.history, axis=0)

    def reset(self):
        out = self.gymenv.reset()
        info = {}
        if isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], dict):
            obs, info = out
        else:
            obs = out
        self.raw = np.asarray(obs)
        self.history.clear()
        f = self._pre(self.raw)
        for _ in range(self.frames):
            self.history.append(f)
        self._score = 0.0
        self._max_x = float(info.get("x_pos", 0.0))
        self._steps = 0
        return self._stack()

    def step(self, action):
        out = self.gymenv.step(action)
        if len(out) == 5:
            obs, reward_raw, terminated, truncated, info = out
        else:
            obs, reward_raw, terminated, info = out
            truncated = False
        self._steps += 1
        if self._steps > 4000:
            terminated = True
        self.raw = np.asarray(obs)
        x_pos = float(info.get("x_pos", 0.0))
        progress = max(0.0, x_pos - self._max_x)
        self._max_x = max(self._max_x, x_pos)
        self._score = self._max_x
        reward = float(reward_raw) + progress * 0.1
        self.history.append(self._pre(self.raw))
        return self._stack(), reward, bool(terminated or truncated)