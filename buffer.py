from collections import deque

import numpy as np


class Replay:
    def __init__(self, cap, obs_shape, feat_dim):
        self.cap = cap
        self.obs = np.zeros((cap,) + obs_shape, dtype=np.uint8)
        self.feat = np.zeros((cap, feat_dim), dtype=np.float32)
        self.act = np.zeros(cap, dtype=np.int64)
        self.ret = np.zeros(cap, dtype=np.float32)
        self.next_obs = np.zeros((cap,) + obs_shape, dtype=np.uint8)
        self.next_feat = np.zeros((cap, feat_dim), dtype=np.float32)
        self.done = np.zeros(cap, dtype=np.float32)
        self.prio = np.zeros(cap, dtype=np.float64)
        self.alpha = 0.6
        self.beta = 0.4
        self.eps = 1e-3
        self.max_prio = 1.0
        self.head = 0
        self.size = 0

    def push(self, obs, feat, act, ret, next_obs, next_feat, done):
        i = self.head
        self.obs[i] = obs
        self.feat[i] = feat
        self.act[i] = act
        self.ret[i] = ret
        self.next_obs[i] = next_obs
        self.next_feat[i] = next_feat
        self.done[i] = done
        self.prio[i] = self.max_prio
        self.head = (self.head + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n):
        if self.size < n:
            return None
        p = self.prio[: self.size]
        cs = np.cumsum(p)
        if cs[-1] <= 0.0:
            return None
        masses = np.random.uniform(0.0, cs[-1], size=n)
        slot = np.searchsorted(cs, masses)
        slot = np.clip(slot, 0, self.size - 1)
        w = (p[slot] / (self.max_prio + self.eps)) ** (-self.beta)
        w = w / w.max()
        return (self.obs[slot], self.feat[slot], self.act[slot], self.ret[slot],
                self.next_obs[slot], self.next_feat[slot], self.done[slot], w, slot)

    def update_prio(self, slot, td_error):
        p = np.abs(np.asarray(td_error, dtype=np.float64)) + self.eps
        self.prio[slot] = p ** self.alpha
        self.max_prio = max(self.max_prio, float(self.prio[slot].max()))


class NStep:
    def __init__(self, n_envs, nstep, gamma):
        self.n_envs = n_envs
        self.nstep = nstep
        self.gamma = gamma
        self.q = [deque(maxlen=nstep) for _ in range(n_envs)]

    def push(self, env_idx, obs, feat, act, rew, next_obs, next_feat, done):
        out = []
        q = self.q[env_idx]
        q.append((obs, feat, act, rew, next_obs, next_feat, done))
        if done:
            while q:
                ret = 0.0
                for k, item in enumerate(q):
                    ret += (self.gamma ** k) * item[3]
                first = q[0]
                last = q[-1]
                out.append((first[0], first[1], first[2], ret, last[4], last[5], 1.0))
                q.popleft()
        elif len(q) == self.nstep:
            ret = 0.0
            for k, item in enumerate(q):
                ret += (self.gamma ** k) * item[3]
            first = q[0]
            last = q[-1]
            out.append((first[0], first[1], first[2], ret, last[4], last[5], 0.0))
        return out
