import logging
import random
import threading
import time

import numpy as np
import torch

import env
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
        counts = [0] * self.actions
        for i in range(n):
            s = states[i]
            counts[acts[i]] += 1
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
        with self.stats["lock"]:
            for k in range(self.actions):
                self.stats["acts"][k] += counts[k]
        if interval > 0:
            rest = interval - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)


class EvalWorker:
    def __init__(self, net, config, stats):
        self.net = net
        self.config = config
        self.stats = stats
        self.envs = [env.Env(config.GRID, config.FRAME_STACK)
                     for _ in range(config.EVAL_EPISODES)]

    def loop(self, stop_event):
        while not stop_event.is_set():
            try:
                self.run_once()
            except Exception as e:
                log.exception("贪心评估出错: %s", e)
            if stop_event.wait(self.config.EVAL_INTERVAL):
                break

    def run_once(self):
        envs = self.envs
        states = [e.reset() for e in envs]
        steps = [0] * len(envs)
        alive = [True] * len(envs)
        while any(alive):
            idx = [i for i in range(len(envs)) if alive[i]]
            x = torch.from_numpy(np.stack([states[i] for i in idx])).to(DEVICE)
            with torch.no_grad():
                acts = self.net(x).argmax(1).cpu().tolist()
            for k, i in enumerate(idx):
                s2, r, done = envs[i].step(acts[k])
                steps[i] += 1
                if not done and steps[i] < self.config.EVAL_MAX_STEPS:
                    states[i] = s2
                    continue
                alive[i] = False
                with self.stats["lock"]:
                    self.stats["eval_recent"].append(envs[i].score)
                    self.stats["eval_recent_steps"].append(steps[i])
                    self.stats["eval_all"].append(envs[i].score)
                    self.stats["eval_episodes"] += 1
                    rec = self.stats["eval_recent"]
                    self.stats["eval_avg"] = sum(rec) / len(rec)
                    self.stats["eval_best"] = max(rec)
                    key = envs[i].g.death or "超时"
                    self.stats["deaths"][key] = self.stats["deaths"].get(key, 0) + 1
                log.info("贪心评估第 %d 局：吃 %d 个，走 %d 步，死于%s",
                         self.stats["eval_episodes"], envs[i].score, steps[i], key)
