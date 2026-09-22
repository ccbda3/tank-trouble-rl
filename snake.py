import random

import numpy as np

DIRS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}


class Snake:
    def __init__(self, grid):
        self.grid = grid
        self.reset()

    def reset(self):
        cy = self.grid // 2
        self.body = [(cy, cy), (cy, cy - 1), (cy, cy - 2)]
        self.dir = (0, 1)
        self.score = 0
        self._place_apple()

    def _place_apple(self):
        while True:
            p = (random.randrange(self.grid), random.randrange(self.grid))
            if p not in self.body:
                self.apple = p
                return

    def render(self):
        img = np.zeros((self.grid, self.grid), np.uint8)
        img[self.apple[0], self.apple[1]] = 255
        for i, (x, y) in enumerate(self.body):
            img[x, y] = 170 if i == 0 else 80
        return img

    def step(self, action):
        nd = DIRS[action]
        if nd == (-self.dir[0], -self.dir[1]):
            nd = self.dir
        self.dir = nd
        head = self.body[0]
        nh = (head[0] + self.dir[0], head[1] + self.dir[1])
        if not (0 <= nh[0] < self.grid and 0 <= nh[1] < self.grid):
            return -10.0, True
        if nh in self.body[:-1]:
            return -10.0, True
        old_d = abs(head[0] - self.apple[0]) + abs(head[1] - self.apple[1])
        self.body.insert(0, nh)
        if nh == self.apple:
            self.score += 1
            self._place_apple()
            return 10.0, False
        self.body.pop()
        new_d = abs(nh[0] - self.apple[0]) + abs(nh[1] - self.apple[1])
        return (0.1 if new_d < old_d else -0.15), False