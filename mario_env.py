from collections import deque

import cv2
import numpy as np

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers.joypad_space import JoypadSpace

JUMP_ACTIONS = {2, 4, 5}


class MarioEnv:
    def __init__(self, frames=4, size=84, accel=4):
        self.gymenv = JoypadSpace(
            gym_super_mario_bros.make("SuperMarioBros-v0", apply_api_compatibility=True),
            SIMPLE_MOVEMENT,
        )
        self.frames = frames
        self.size = size
        self.accel = accel
        self.history = deque(maxlen=frames)
        self.raw = None
        self._score = 0.0
        self._max_x = 0.0
        self._steps = 0
        self._coins = 0
        self._prev_score = 0
        self._prev_enemy = 0
        self._nes = None
        n = self.gymenv
        while hasattr(n, "env"):
            n = n.env
        self._nes = n

    def _count_enemy(self):
        ram = self._nes.ram
        return sum(1 for a in range(0x16, 0x1B) if ram[a] != 0)

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
        self._coins = int(info.get("coins", 0))
        self._prev_score = int(info.get("score", 0))
        try:
            self._prev_enemy = self._count_enemy()
        except Exception:
            self._prev_enemy = 0
        return self._stack()

    def step(self, action):
        total = 0.0
        done = False
        for _ in range(self.accel):
            r, d = self._step1(action)
            total += r
            if d:
                done = True
                break
        return self._stack(), total, done

    def _step1(self, action):
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

        coins = int(info.get("coins", 0))
        score = int(info.get("score", 0))
        flag_get = bool(info.get("flag_get", False))
        coin_delta = max(0, coins - self._coins)
        cur_enemy = self._prev_enemy
        try:
            cur_enemy = self._count_enemy()
        except Exception:
            cur_enemy = self._prev_enemy
        killed = self._prev_enemy - cur_enemy > 0 and score > self._prev_score

        shaped = coin_delta * 2.0
        if killed:
            shaped += 5.0
        if flag_get:
            shaped += 15.0
        if action in JUMP_ACTIONS:
            shaped -= 0.03
        done_ = bool(terminated or truncated)
        if done_ and not flag_get:
            shaped -= 2.0

        reward = float(reward_raw) + progress * 0.1 + shaped

        self._coins = coins
        self._prev_score = score
        self._prev_enemy = cur_enemy
        self.history.append(self._pre(self.raw))
        return reward, done_