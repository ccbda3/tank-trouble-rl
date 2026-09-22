import random
import threading
import time

import torch

from network import DEVICE


class Counter:
    def __init__(self, value=0):
        self.value = value
        self.lock = threading.Lock()

    def add(self, n=1):
        with self.lock:
            self.value += n
            return self.value

    def get(self):
        with self.lock:
            return self.value


class Agent:
    def __init__(self, net, buffer, env, counter, config, stats):
        self.net = net
        self.buffer = buffer
        self.env = env
        self.counter = counter
        self.config = config
        self.stats = stats
        self.actions = net.fc[-1].out_features

    def act(self, s):
        if random.random() < self.epsilon():
            return random.randrange(self.actions)
        with torch.no_grad():
            q = self.net(torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0))
            return int(q.argmax().item())

    def epsilon(self):
        steps = self.counter.get()
        ratio = min(1.0, steps / self.config.EPS_DECAY)
        eps = self.config.EPS_START - (self.config.EPS_START - self.config.EPS_END) * ratio
        return max(self.config.EPS_END, eps)

    def run_episode(self):
        s = self.env.reset()
        done = False
        while not done:
            a = self.act(s)
            s2, r, done = self.env.step(a)
            self.buffer.push(s, a, r, s2, done)
            self.counter.add(1)
            s = s2
        return self.env.score

    def loop(self, stop_event):
        while not stop_event.is_set():
            score = self.run_episode()
            with self.stats["lock"]:
                self.stats["recent"].append(score)
                self.stats["episodes"] += 1
            time.sleep(0.001)
