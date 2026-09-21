from game import Breakout


class Env:
    def __init__(self):
        self.g = Breakout()

    def reset(self):
        return self.g.reset()

    def step(self, action):
        r, d = self.g.step(action)
        return self.g.state(), r, d
