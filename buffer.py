import threading

import numpy as np

import config


class PrioritizedReplayBuffer:
    def __init__(self, capacity, frames, size):
        self.capacity = capacity
        self.cap2 = 1 << max(1, (capacity - 1).bit_length())
        self.depth = self.cap2.bit_length() - 1
        self.obs = np.zeros((capacity, frames, size, size), np.uint8)
        self.next_obs = np.zeros((capacity, frames, size, size), np.uint8)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.tree = np.zeros(2 * self.cap2, np.float64)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0
        self.steps = 0
        self.lock = threading.Lock()

    def push(self, s, a, r, s2, d):
        with self.lock:
            i = self.pos
            self.obs[i] = s
            self.next_obs[i] = s2
            self.actions[i] = a
            self.rewards[i] = r
            self.dones[i] = 1.0 if d else 0.0
            j = i + self.cap2
            self.tree[j] = self.max_priority
            j >>= 1
            while j >= 1:
                self.tree[j] = self.tree[2 * j] + self.tree[2 * j + 1]
                j >>= 1
            self.pos = (i + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1

    def sample(self, n):
        with self.lock:
            total = float(self.tree[1])
            if total <= 0.0:
                total = 1.0
            seg = total / n
            vals = np.random.uniform(0.0, seg, n) + seg * np.arange(n)
            leaf = np.ones(n, np.int64)
            for _ in range(self.depth):
                left = 2 * leaf
                right = vals > self.tree[left]
                vals = np.where(right, vals - self.tree[left], vals)
                leaf = np.where(right, left + 1, left)
            probs = self.tree[leaf] / total
            beta = min(
                1.0,
                config.BETA_START + (1.0 - config.BETA_START) * (self.steps / config.BETA_FRAMES),
            )
            self.steps += 1
            w = np.power(self.size * probs, -beta)
            w = w / w.max()
            pos = np.minimum(leaf - self.cap2, self.size - 1)
            return (
                pos,
                self.obs[pos],
                self.actions[pos],
                self.rewards[pos],
                self.next_obs[pos],
                self.dones[pos],
                w.astype(np.float32),
            )

    def update_priorities(self, pos, priorities):
        with self.lock:
            p = np.power(np.abs(np.asarray(priorities, np.float64)) + 1e-6, config.ALPHA)
            self.max_priority = max(self.max_priority, float(p.max()))
            j = pos + self.cap2
            self.tree[j] = p
            for _ in range(self.depth):
                j = j >> 1
                self.tree[j] = self.tree[2 * j] + self.tree[2 * j + 1]

    def __len__(self):
        with self.lock:
            return self.size
