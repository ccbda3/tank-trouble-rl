import random
import threading


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        self.pos = 0
        self.lock = threading.Lock()

    def push(self, s, a, r, s2, d):
        item = (s, a, r, s2, d)
        with self.lock:
            if len(self.data) < self.capacity:
                self.data.append(item)
            else:
                self.data[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, n):
        with self.lock:
            return random.sample(self.data, n)

    def __len__(self):
        with self.lock:
            return len(self.data)
