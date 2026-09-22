import time

import numpy as np
import torch

from aimonitor import Monitor, imaging
import config
from network import DEVICE

EDGE_TTL = 5.0
HIST_TTL = 5.0


def _forward(net, obs):
    x = torch.from_numpy(np.asarray(obs)).unsqueeze(0).to(net.fc[0].weight.device)
    x = x.float() / 255.0
    with torch.no_grad():
        c1a = net.net[1](net.net[0](x))
        c2a = net.net[3](net.net[2](c1a))
        c3a = net.net[5](net.net[4](c2a))
        pool = net.net[6](c3a)
        f = net.fc[1](net.fc[0](torch.flatten(pool, 1)))
        out = net.fc[2](f)
    return c1a, c2a, c3a, f, out


def _chan_matrix(weight):
    w = torch.abs(weight)
    while w.dim() > 2:
        w = w.mean(-1)
    return np.round(w.detach().cpu().numpy().astype(np.float32), 5)


def _fc_matrix(weight, chan):
    w = torch.abs(weight).detach().cpu().numpy().astype(np.float32)
    return np.round(w.reshape(w.shape[0], chan, -1).mean(2), 5)


def _nparam(mod):
    return sum(p.numel() for p in mod.parameters())


class SnakeMonitor:
    def __init__(self, net, target_net, buffer, counter, trainer, stats, probe, r_env,
                 start_fn, stop_fn, eps_fn, gpu_fn):
        self.net = net
        self.target_net = target_net
        self.buffer = buffer
        self.counter = counter
        self.trainer = trainer
        self.stats = stats
        self.probe = probe
        self.r_env = r_env
        self.start_fn = start_fn
        self.stop_fn = stop_fn
        self.eps_fn = eps_fn
        self.gpu_fn = gpu_fn
        self._edges = {"t": 0.0, "v": None}
        self._hists = {"t": 0.0, "v": []}
        self._rate = {"steps": None, "t": None, "v": 0.0}
        self._gs = {"t": 0.0, "v": (0.0, 0.0, 0.0)}
        self._maps = None

        zero = np.zeros((config.FRAME_STACK, config.GRID, config.GRID), np.uint8)
        c1a, c2a, c3a, _, _ = _forward(net, zero)
        self._shape = (int(c1a.shape[-1]), int(c2a.shape[-1]), int(c3a.shape[-1]))
        k1 = net.net[0].out_channels
        k2 = net.net[2].out_channels
        k3 = net.net[4].out_channels
        s1, s2, s3 = self._shape
        nf = net.fc[0].out_features
        self._chans = (k1, k2, k3)
        self._width = nf

        self._meta = [
            {"name": "输入", "detail": "%d 帧 × %d×%d" % (config.FRAME_STACK, config.GRID, config.GRID),
             "params": 0, "realCount": config.FRAME_STACK * config.GRID * config.GRID},
            {"name": "C1", "detail": "%d 通道 × %d×%d" % (k1, s1, s1),
             "params": _nparam(net.net[0]), "realCount": k1 * s1 * s1},
            {"name": "C2", "detail": "%d 通道 × %d×%d" % (k2, s2, s2),
             "params": _nparam(net.net[2]), "realCount": k2 * s2 * s2},
            {"name": "C3", "detail": "%d 通道 × %d×%d" % (k3, s3, s3),
             "params": _nparam(net.net[4]), "realCount": k3 * s3 * s3},
            {"name": "FC", "detail": "%d 个神经元 · 全部显示" % nf,
             "params": _nparam(net.fc[0]), "realCount": nf},
            {"name": "输出", "detail": "%d 个动作 · 全部显示" % config.ACTIONS,
             "params": _nparam(net.fc[2]), "realCount": config.ACTIONS},
        ]

        self._hist_regs = []
        for name, p in net.named_parameters():
            self._hist_regs.append((name, name + " · 权重"))
            self._hist_regs.append((name + "@grad", name + " · 梯度"))

    def _edges_cached(self):
        now = time.time()
        if self._edges["v"] is not None and now - self._edges["t"] < EDGE_TTL:
            return self._edges["v"]
        net = self.net
        self._edges["v"] = [
            _chan_matrix(net.net[0].weight),
            _chan_matrix(net.net[2].weight),
            _chan_matrix(net.net[4].weight),
            _fc_matrix(net.fc[0].weight, net.net[4].out_channels),
            _chan_matrix(net.fc[2].weight),
        ]
        self._edges["t"] = now
        return self._edges["v"]

    def _hist_values(self, key):
        if key.endswith("@grad"):
            g = self.trainer.grad_snapshot.get(key[:-5])
            return None if g is None else g.numpy().ravel()
        w = self.trainer.weight_snapshot.get(key)
        return None if w is None else w.numpy().ravel()

    def _rate_of(self, key, value_fn):
        now = time.time()
        v = float(value_fn())
        if key not in self._rate or not isinstance(self._rate[key], dict):
            self._rate[key] = {"v": v, "t": now, "r": 0.0}
            return 0.0
        st = self._rate[key]
        dt = now - st["t"]
        if dt >= 1.0:
            st["r"] = max(0.0, (v - st["v"]) / dt)
            st["v"] = v
            st["t"] = now
        return st["r"]

    def _dps(self):
        return self._rate_of("steps", self.counter.get)

    def _grad_stats(self):
        now = time.time()
        if now - self._gs["t"] < 1.0:
            return self._gs["v"]
        g_sq = 0.0
        w_sq = 0.0
        g_max = 0.0
        for w in self.trainer.weight_snapshot.values():
            wf = w.float()
            w_sq += float(torch.sum(wf * wf))
        for g in self.trainer.grad_snapshot.values():
            gf = g.float()
            g_sq += float(torch.sum(gf * gf))
            m = float(gf.abs().max())
            if m > g_max:
                g_max = m
        self._gs = {"t": now, "v": (g_sq ** 0.5, w_sq ** 0.5, g_max)}
        return self._gs["v"]

    def _graph(self):
        obs = self.r_env._stack()
        c1a, c2a, c3a, f, out = _forward(self.net, obs)
        frames = np.asarray(obs, dtype=float)
        c1 = c1a[0].cpu().numpy()
        c2 = c2a[0].cpu().numpy()
        c3 = c3a[0].cpu().numpy()
        self._maps = (frames, c1, c2, c3)
        nodes = [
            [float(frames[i].mean()) for i in range(frames.shape[0])],
            c1.mean((1, 2)).tolist(),
            c2.mean((1, 2)).tolist(),
            c3.mean((1, 2)).tolist(),
            f[0].cpu().numpy().tolist(),
            out[0].cpu().numpy().tolist(),
        ]
        layers = []
        for i, m in enumerate(self._meta):
            lay = dict(m)
            lay["nodeCount"] = len(nodes[i])
            lay["nodes"] = nodes[i]
            layers.append(lay)
        return {"layers": layers, "edges": self._edges_cached()}

    def _image(self, key):
        if self._maps is None:
            return None
        frames, c1, c2, c3 = self._maps
        if key == "input":
            return frames[0]
        table = {"feat_C1": c1, "feat_C2": c2, "feat_C3": c3}
        arr = table.get(key)
        if arr is None:
            return None
        return imaging.montage(list(arr), cols=8, pad=1, pad_val=float(arr.min()))

    def _avg(self):
        with self.stats["lock"]:
            r = self.stats["recent"]
            return (sum(r) / len(r)) if r else 0.0

    def _best(self):
        with self.stats["lock"]:
            return max(self.stats["recent"]) if self.stats["recent"] else 0

    def _params(self):
        net = self.net
        k1, k2, k3 = self._chans
        return [
            {"title": "网络结构 · Double DQN", "layers": self._meta,
             "rows": [["算力", "<b>%s</b>" % ("GPU 加速" if DEVICE.type == "cuda" else "CPU")],
                      ["总参数量", "<b>%d</b>" % sum(p.numel() for p in net.parameters())]]},
            {"title": "DQN 超参数", "rows": [
                ["Double DQN", "<b>启用</b>"],
                ["优先级经验池", "SumTree · α=%.2f" % config.ALPHA],
                ["折扣因子 gamma", "%.3f" % config.GAMMA],
                ["学习率 lr", "%.1e" % config.LR],
                ["批大小 / 梯度累积", "%d / %d" % (config.BATCH, config.ACCUM)],
                ["目标网络同步", "%d 步" % config.TARGET_SYNC],
                ["经验池容量", "%d" % config.REPLAY_CAP],
                ["探索率区间", "%.2f → %.2f" % (config.EPS_START, config.EPS_END)],
                ["探索衰减更新数", "%d" % config.EPS_DECAY],
                ["当前梯度更新", "%d" % self.trainer.train_steps],
            ]},
            {"title": "运行配置", "rows": [
                ["棋盘", "%d × %d" % (config.GRID, config.GRID)],
                ["堆叠帧 / 动作数", "%d / %d" % (config.FRAME_STACK, config.ACTIONS)],
                ["并行环境 / 采集线程", "%d / %d" % (config.AGENTS, config.WORKERS)],
                ["采集节流", "%.1f ms/批" % (config.TICK_INTERVAL * 1000)],
                ["保存权重间隔", "%d 秒" % config.CHECKPOINT_INTERVAL],
                ["本局蛇长", "%d" % self.r_env.score],
            ]},
        ]

    def build(self, port=config.MONITOR_PORT):
        mon = Monitor(title="贪吃蛇 · DQN 强化学习训练监控", port=port, interval=1.0, history=900)
        k1, k2, k3 = self._chans
        s1, s2, s3 = self._shape

        mon.info(lambda: {
            "设备": str(DEVICE).upper(),
            "棋盘": "%d×%d" % (config.GRID, config.GRID),
            "并行": "%d 局 / %d 线程" % (config.AGENTS, config.WORKERS),
            "参数": "%.2fM" % (sum(p.numel() for p in self.net.parameters()) / 1e6),
        })
        mon.metric("steps", self.counter.get, label="总步数", fmt="int")
        mon.metric("eps", self.eps_fn, label="探索率 ε", fmt="f3", color="#8fb9ff")
        mon.metric("loss", lambda: self.stats["loss"], label="Loss", fmt="f4", color="#ff8f5a")
        mon.metric("avg_score", self._avg, label="平均吃苹果", fmt="f2", color="#3ddc84")
        mon.metric("best_score", self._best, label="最长蛇身", fmt="int", color="#ffd36a")
        mon.metric("train_steps", lambda: self.trainer.train_steps, label="梯度更新", fmt="int")
        mon.metric("buffer", lambda: len(self.buffer), label="经验池", fmt="int")
        mon.metric("dps", self._dps, label="决策/秒", fmt="int", color="#7fd0e0")
        mon.metric("gpu", lambda: (-1 if self.gpu_fn() is None else self.gpu_fn()),
                   label="GPU 利用率", fmt="f0", color="#cf7fff")
        mon.metric("episodes", lambda: self.stats["episodes"], label="完成局数", fmt="int", chart=False)

        mon.health("grad_norm", lambda: self._grad_stats()[0], label="梯度范数", fmt="sci")
        mon.health("weight_norm", lambda: self._grad_stats()[1], label="权重范数", fmt="f2")
        mon.health("grad_max", lambda: self._grad_stats()[2], label="梯度最大绝对值", fmt="sci")
        mon.health("lr", lambda: config.LR, label="学习率", fmt="sci")
        mon.health("replay", lambda: self.trainer.train_steps * config.BATCH / max(1, self.counter.get()),
                   label="训练/采样比", unit="×", fmt="f2")
        mon.health("params", lambda: sum(p.numel() for p in self.net.parameters()),
                   label="可训练参数", fmt="int")

        mon.graph(self._graph)
        mon.surface(lambda: self.probe.snapshot())

        mon.image("input", lambda: self._image("input"),
                  label="输入 %d 帧（真实像素）" % config.FRAME_STACK, cmap="gray",
                  scale=6, group="图片")
        for name, cnt, sp in (("C1", k1, s1), ("C2", k2, s2), ("C3", k3, s3)):
            mon.image("feat_" + name, (lambda k: (lambda: self._image(k)))("feat_" + name),
                      label="%s 全部 %d 通道（每通道 %d×%d）" % (name, cnt, sp, sp),
                      cmap="viridis", scale=3, group="特征图")

        for key, label in self._hist_regs:
            mon.histogram(key, (lambda k: (lambda: self._hist_values(k)))(key), label=label, bins=48)

        mon.params(self._params)
        mon.control(start=self.start_fn, stop=self.stop_fn)
        self.mon = mon
        return mon
