from collections import deque

import numpy as np

import game


class Env:
    def __init__(self, frames=4, size=96):
        self.g = game.Breakout()
        self.frames = frames
        self.size = size
        self.history = deque(maxlen=frames)

    def _stack(self):
        return np.stack(self.history, axis=0)

    @property
    def score(self):
        return self.g.score

    def reset(self):
        self.g.reset()
        f = game.render_bw(self.g, self.size)
        self.history.clear()
        for _ in range(self.frames):
            self.history.append(f)
        return self._stack()

    def step(self, action):
        r, d = self.g.step(action)
        self.history.append(game.render_bw(self.g, self.size))
        return self._stack(), r, d