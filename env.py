from collections import deque

import numpy as np

import snake


class Env:
    def __init__(self, grid=64, frames=4):
        self.g = snake.Snake(grid)
        self.grid = grid
        self.frames = frames
        self.history = deque(maxlen=frames)

    @property
    def score(self):
        return self.g.score

    def _stack(self):
        return np.stack(list(self.history), axis=0)

    def reset(self):
        self.g.reset()
        f = self.g.render()
        self.history.clear()
        for _ in range(self.frames):
            self.history.append(f)
        return self._stack()

    def step(self, action):
        r, done = self.g.step(action)
        self.history.append(self.g.render())
        return self._stack(), r, done