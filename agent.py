import logging
import random
import threading
import time

import numpy as np
import torch

from network import DEVICE

log = logging.getLogger("snake")


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

    def set(self, value):
        with self.lock:
            self.value = value


def epsilon_of(trainer, config):
    ratio = min(1.0, trainer.train_steps / config.EPS_DECAY)
    eps = config.EPS_START - (config.EPS_START - config.EPS_END) * ratio
    return max(config.EPS_END, eps)


class Agent:
    def __init__(self, net, buffer, env, counter, config, stats, trainer):
        self.net = net
        self.buffer = buffer
        self.env = env
        self.counter = counter
        self.config = config
        self.stats = stats
        self.trainer = trainer
        self.actions = net.fc[-1].out_features

    def epsilon(self):
        return epsilon_of(self.trainer, self.config)

    def act(self, s):
        if random.random() < self.epsilon():
            return random.randrange(self.actions)
        with torch.no_grad():
            q = self.net(torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0))
            return int(q.argmax().item())


class EnvWorker:
    def __init__(self, net, buffer, envs, counter, config, stats, trainer):
        self.net = net
        self.buffer = buffer
        self.envs = list(envs)
        self.counter = counter
        self.config = config
        self.stats = stats
        self.trainer = trainer
        self.actions = net.fc[-1].out_features

    def epsilon(self):
        return epsilon_of(self.trainer, self.config)

    def act_batch(self, states, eps):
        n = len(states)
        if random.random() < eps:
            return [random.randrange(self.actions) for _ in range(n)]
        with torch.no_grad():
            x = torch.from_numpy(np.stack(states)).to(DEVICE)
            return self.net(x).argmax(1).cpu().tolist()

    def loop(self, stop_event):
        states = [e.reset() for e in self.envs]
        interval = getattr(self.config, "TICK_INTERVAL", 0.0)
        while not stop_event.is_set():
            try:
                self._tick(states, interval)
            except Exception as e:
                log.exception("采集线程出错，0.5 秒后重试: %s", e)
                time.sleep(0.5)

    def _tick(self, states, interval):
        n = len(self.envs)
        t0 = time.perf_counter()
        acts = self.act_batch(states, self.epsilon())
        for i in range(n):
            s = states[i]
            s2, r, done = self.envs[i].step(acts[i])
            self.buffer.push(s, acts[i], r, s2, done)
            self.counter.add(1)
            if done:
                with self.stats["lock"]:
                    self.stats["recent"].append(self.envs[i].score)
                    self.stats["episodes"] += 1
                states[i] = self.envs[i].reset()
            else:
                states[i] = s2
        if interval > 0:
            rest = interval - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)
