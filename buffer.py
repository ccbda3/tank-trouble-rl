import random
import threading

import config


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity)
        self.data = [None] * capacity
        self.pos = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = idx // 2
        self.tree[parent] += change
        if parent != 1:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = idx * 2
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[1]

    def add(self, priority, data):
        idx = self.pos + self.capacity
        self.data[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        self.update(idx, priority)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(1, s)
        return idx, self.data[idx - self.capacity]


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.steps = 0
        self.lock = threading.Lock()

    def push(self, s, a, r, s2, d):
        with self.lock:
            self.tree.add(self.max_priority, (s, a, r, s2, d))

    def sample(self, n):
        with self.lock:
            batch = []
            idxs = []
            probs = []
            seg = self.tree.total() / n
            beta = min(1.0, config.BETA_START + (1.0 - config.BETA_START) * (self.steps / config.BETA_FRAMES))
            self.steps += 1
            for i in range(n):
                s = random.uniform(seg * i, seg * (i + 1))
                idx, data = self.tree.get(s)
                idxs.append(idx)
                probs.append(self.tree.tree[idx])
                batch.append(data)
            total = self.tree.total()
            if total <= 0:
                total = 1.0
            probs = [p / total for p in probs]
            weights = [(self.tree.size * p) ** -beta for p in probs]
            if weights:
                max_w = max(weights)
                weights = [w / max_w for w in weights]
            return batch, weights, idxs

    def update_priorities(self, idxs, priorities):
        with self.lock:
            for idx, p in zip(idxs, priorities):
                p = (p + 1e-6) ** config.ALPHA
                self.max_priority = max(self.max_priority, p)
                self.tree.update(idx, p)

    def __len__(self):
        with self.lock:
            return self.tree.size