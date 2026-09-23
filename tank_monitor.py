import logging
import subprocess
import threading
import time

import numpy as np
import torch

import config
import network
import tank
import tank_vec
from aimonitor import Monitor, imaging

log = logging.getLogger("tank")

EDGE_TTL = 5.0
HIST_TTL = 5.0
GRAD_EVERY = 200
GPU_EVERY = 5.0

WEIGHT_KEYS = ("conv.0.weight", "conv.2.weight", "conv.4.weight",
               "fc.0.weight", "value.weight", "adv.weight")


def _convs(net):
    return [m for m in net.conv if isinstance(m, torch.nn.Conv2d)]


def _nparam(mod):
    return sum(p.numel() for p in mod.parameters())


def _chan_matrix(weight):
    w = torch.abs(weight)
    while w.dim() > 2:
        w = w.mean(-1)
    return np.round(w.detach().cpu().numpy().astype(np.float32), 6)


def _flat_matrix(weight, n_conv, n_spatial):
    w = torch.abs(weight)[:, :n_conv * n_spatial].detach().cpu().numpy().astype(np.float32)
    return np.round(w.reshape(w.shape[0], n_conv, n_spatial).mean(2), 6)


def _out_matrix(net):
    adv = net.adv.weight.detach()
    m = adv - adv.mean(0, keepdim=True) + net.value.weight.detach()
    return np.round(torch.abs(m).cpu().numpy().astype(np.float32), 6)


class NetProbe:
    def __init__(self, env, net):
        self.env = env
        self.net = net
        self.maps = None
        self._edge_t = 0.0
        self._edge_v = None
        cv = _convs(net)
        with torch.no_grad():
            obs = network.obs_tensor(np.zeros((1,) + (tank_vec.OBS_H, tank_vec.OBS_W), np.uint8))
            feat = network.feat_tensor(np.zeros((1, env.feat(0).shape[1]), np.float32))
            feats, y = [], obs
            for m in net.conv:
                y = m(y)
                if isinstance(m, torch.nn.Conv2d):
                    feats.append(y)
        self.shapes = [int(f.shape[-1]) for f in feats]
        self.chans = [int(f.shape[1]) for f in feats]
        self.flat = int(np.prod(feats[-1].shape[1:]))
        self.nfeat = int(feat.shape[1])
        self.hidden = int(net.value.in_features)
        s1, s2, s3 = self.shapes
        k1, k2, k3 = self.chans
        self.meta = [
            {"name": "观测", "params": 0, "realCount": tank_vec.OBS_CH,
             "detail": "%d 通道 × %d×%d · 按位打包（墙/自机/其余 %d 人合并/子弹/地雷/道具/激光 + 保留位）"
                       % (tank_vec.OBS_CH, tank_vec.OBS_H, tank_vec.OBS_W, config.PLAYERS - 1)},
            {"name": "C1", "params": _nparam(cv[0]), "realCount": k1 * s1 * s1,
             "detail": "%d 通道 × %d×%d · stride 2" % (k1, s1, s1)},
            {"name": "C2", "params": _nparam(cv[1]), "realCount": k2 * s2 * s2,
             "detail": "%d 通道 × %d×%d · stride 2" % (k2, s2, s2)},
            {"name": "C3", "params": _nparam(cv[2]), "realCount": k3 * s3 * s3,
             "detail": "%d 通道 × %d×%d · stride 2" % (k3, s3, s3)},
            {"name": "FC", "params": _nparam(net.fc[0]), "realCount": self.hidden,
             "detail": "%d 神经元 · 视觉 %d 维 + 标量 %d 维（自身状态/对手状态/最近 8 颗子弹）"
                       % (self.hidden, self.flat, self.nfeat)},
            {"name": "输出 Q", "params": _nparam(net.value) + _nparam(net.adv), "realCount": network.ACTIONS,
             "detail": "%d 个动作 · Dueling 价值/优势分解" % network.ACTIONS},
        ]

    def forward(self):
        env = self.env
        obs = network.obs_tensor(env.obs_pack(0)[0][None])
        feat = network.feat_tensor(env.feat(0)[0][None])
        net = self.net
        with torch.no_grad():
            y = obs
            feats = []
            for m in net.conv:
                y = m(y)
                if isinstance(m, torch.nn.Conv2d):
                    feats.append(y)
            flat = torch.flatten(y, 1)
            h = net.fc(torch.cat([flat, feat], dim=1))
            v = net.value(h)
            a = net.adv(h)
            q = v + a - a.mean(dim=1, keepdim=True)
        self.maps = (obs[0], feats, h[0], v[0], a[0], q[0])
        return self.maps

    def graph(self):
        obs, feats, h, v, a, q = self.forward()
        chans = tank_vec.unpack(np.asarray(self.env.obs_pack(0)[:1]))[0]
        nodes = [
            [float(c.mean()) for c in chans],
            [float(c.mean().item()) for c in feats[0][0]],
            [float(c.mean().item()) for c in feats[1][0]],
            [float(c.mean().item()) for c in feats[2][0]],
            [float(x) for x in h],
            [float(x) for x in q],
        ]
        layers = []
        for i, m in enumerate(self.meta):
            lay = dict(m)
            lay["nodeCount"] = len(nodes[i])
            lay["nodes"] = nodes[i]
            layers.append(lay)
        return {"layers": layers, "edges": self._edges()}

    def _edges(self):
        now = time.time()
        if self._edge_v is not None and now - self._edge_t < EDGE_TTL:
            return self._edge_v
        cv = _convs(self.net)
        self._edge_v = [
            _chan_matrix(cv[0].weight),
            _chan_matrix(cv[1].weight),
            _chan_matrix(cv[2].weight),
            _flat_matrix(self.net.fc[0].weight, self.chans[2], self.flat // self.chans[2]),
            _out_matrix(self.net),
        ]
        self._edge_t = now
        return self._edge_v

    def feature_image(self, which):
        if self.maps is None:
            return None
        _, feats, h, _, _, q = self.maps
        if which == "fc":
            return h.reshape(-1, 1).cpu().numpy()
        if which == "q":
            return q.reshape(-1, 1).cpu().numpy()
        c = feats[which]
        arr = c[0].cpu().numpy()
        maps = []
        for k in range(arr.shape[0]):
            a = arr[k]
            lo = float(a.min())
            hi = float(a.max())
            maps.append(np.zeros_like(a) if hi - lo < 1e-9 else (a - lo) / (hi - lo) * 255.0)
        return imaging.montage(maps, cols=8, pad=1, pad_val=0.0)


class LandscapeProbe:
    def __init__(self, net, target, replay, nfeat_dim, size=11, batch=16, interval=60.0, extent=0.5):
        self.net = net
        self.target = target
        self.replay = replay
        self.size = size
        self.batch = batch
        self.interval = interval
        self.extent = extent
        self.probe = network.make(nfeat_dim)
        self.du = None
        self.dv = None
        self.first = None
        self.fixed = None
        self.path = []
        self.count = 0
        self.out = {"ready": 0}
        self.lock = threading.Lock()

    def snapshot(self):
        with self.lock:
            d = dict(self.out)
            d["path"] = list(self.path)
            d["probeCount"] = self.count
            return d

    def _dirs(self, named):
        if self.du is not None:
            return
        gen = torch.Generator().manual_seed(20240922)
        self.du = {}
        self.dv = {}
        for n, p in named:
            a = torch.randn(p.shape, generator=gen)
            a = a / a.norm().clamp_min(1e-12)
            b = torch.randn(p.shape, generator=gen)
            b = b - (b * a).sum() * a
            nb = b.norm()
            if float(nb) < 1e-6:
                b = torch.zeros_like(a)
            else:
                b = b / nb
            self.du[n] = a.to(p.device)
            self.dv[n] = b.to(p.device)

    def _freeze(self):
        if self.fixed is not None or self.replay.size < self.batch:
            return
        batch = self.replay.sample(self.batch)
        if batch is None:
            return
        obs_p, feat_p, act_p, ret_p, nobs_p, nfeat_p, done_p, *_ = batch
        obs = network.obs_tensor(obs_p)
        feat = network.feat_tensor(feat_p)
        nobs = network.obs_tensor(nobs_p)
        nfeat = network.feat_tensor(nfeat_p)
        a = torch.as_tensor(act_p, dtype=torch.long, device=network.DEVICE)[:, None]
        ret = torch.as_tensor(ret_p, dtype=torch.float32, device=network.DEVICE)[:, None]
        done = torch.as_tensor(done_p, dtype=torch.float32, device=network.DEVICE)[:, None]
        with torch.no_grad():
            best = self.net(nobs, nfeat).argmax(dim=1, keepdim=True)
            tgt = ret + (config.GAMMA ** config.NSTEP) * (1.0 - done) * \
                self.target(nobs, nfeat).gather(1, best)
        self.fixed = (obs, feat, a, tgt)

    def _loss(self):
        obs, feat, a, tgt = self.fixed
        q = self.probe(obs, feat).gather(1, a)
        return float(torch.nn.functional.smooth_l1_loss(q, tgt).item())

    def measure(self):
        self._freeze()
        if self.fixed is None:
            return
        named = list(self.net.named_parameters())
        self._dirs(named)
        if self.first is None:
            self.first = {n: p.detach().clone() for n, p in named}
        self.probe.load_state_dict(self.net.state_dict())
        params = list(self.probe.named_parameters())
        alpha = 0.0
        beta = 0.0
        base = {}
        for n, p in params:
            base[n] = p.detach().clone()
        with torch.no_grad():
            for n, p in named:
                dd = p.detach() - self.first[n]
                alpha += float((dd * self.du[n]).sum())
                beta += float((dd * self.dv[n]).sum())

            def at(au, bv):
                for n, p in params:
                    p.copy_(base[n]).add_(self.du[n], alpha=au).add_(self.dv[n], alpha=bv)
                return self._loss()

            cur = at(0.0, 0.0)
            e = self.extent
            zmax = max(at(e, e), at(e, -e), at(-e, e), at(-e, -e))
            ref = cur if cur > 1e-6 else 1e-6
            r = (zmax - cur) / ref
            if r > 1e-9:
                self.extent = max(1e-5, min(1.0, e * max(0.02, min(5.0, (3.0 / r) ** 0.5))))
            ext = self.extent
            res = self.size
            vals = []
            for i in range(res):
                au = -ext + 2 * ext * i / (res - 1)
                for j in range(res):
                    bv = -ext + 2 * ext * j / (res - 1)
                    vals.append(at(au, bv))
            for n, p in params:
                p.copy_(base[n])
        self.path.append({"a": alpha, "b": beta, "z": cur})
        if len(self.path) > 400:
            del self.path[0]
        self.count += 1
        with self.lock:
            self.out = {"ready": 1, "res": res, "ext": ext, "zmin": min(vals),
                        "zmax": max(vals), "values": vals, "center": [alpha, beta], "cur": cur}

    def loop(self, stop):
        while not stop.is_set():
            try:
                self.measure()
            except Exception as e:
                log.warning("损失曲面测量失败: %s", e)
            if stop.wait(self.interval):
                break


def _gpu_util():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class TankMonitor:
    def __init__(self, env, policy, target, replay, stats, steps_fn, args):
        self.env = env
        self.policy = policy
        self.target = target
        self.replay = replay
        self.stats = stats
        self.steps_fn = steps_fn
        self.args = args
        self.probe = NetProbe(env, policy)
        self.land = LandscapeProbe(policy, target, replay, self.probe.nfeat)
        self.updates = 0
        self.act_count = np.zeros(network.ACTIONS, dtype=np.float64)
        self.act_step = np.zeros(network.ACTIONS, dtype=np.float64)
        self.grad = {}
        self.grad_at = 0.0
        self._lock = threading.Lock()
        self._gs = {"t": 0.0, "v": (0.0, 0.0, 0.0)}
        self._gn = {"t": 0.0, "v": []}
        self._gpu = {"t": 0.0, "v": None}
        self.mon = None
        self.stop = threading.Event()

    def tick_update(self, net):
        self.updates += 1
        if self.updates == 1 or self.updates % GRAD_EVERY == 0:
            self.grab_grads(net)

    def grab_grads(self, net):
        out = {}
        for n, p in net.named_parameters():
            if p.grad is not None:
                out[n] = p.grad.detach().abs().cpu().numpy().copy()
        with self._lock:
            self.grad = out

    def count_acts(self, acts):
        self.act_count += np.bincount(np.asarray(acts).ravel(),
                                      minlength=network.ACTIONS)[:network.ACTIONS]
        self.act_step += self.env.n * self.env.p

    def _weights(self):
        with torch.no_grad():
            return {n: p.detach().abs().cpu().numpy() for n, p in self.policy.named_parameters()}

    def _grad_stats(self):
        now = time.time()
        if now - self._gs["t"] < 1.0:
            return self._gs["v"]
        w_sq = g_sq = g_max = 0.0
        with torch.no_grad():
            for p in self.policy.parameters():
                w = p.detach().float()
                w_sq += float(torch.sum(w * w))
        with self._lock:
            grad = dict(self.grad)
        for g in grad.values():
            gf = g.astype(np.float64)
            g_sq += float((gf * gf).sum())
            m = float(gf.max()) if gf.size else 0.0
            if m > g_max:
                g_max = m
        self._gs = {"t": now, "v": (g_sq ** 0.5, w_sq ** 0.5, g_max)}
        return self._gs["v"]

    def _layer_grad_norm(self):
        now = time.time()
        if self._gn["v"] and now - self._gn["t"] < 2.0:
            return self._gn["v"]
        with self._lock:
            grad = dict(self.grad)
        out = []
        for name, g in grad.items():
            gf = g.astype(np.float64)
            out.append([name, float(np.sqrt((gf * gf).sum()))])
        self._gn = {"t": now, "v": out}
        return out

    def _layer_stats(self):
        with self._lock:
            grad = dict(self.grad)
        rows = []
        with torch.no_grad():
            for name, p in self.policy.named_parameters():
                w = p.detach().float()
                g = grad.get(name)
                rows.append([
                    name.split(".")[0], name,
                    "×".join(str(int(s)) for s in tuple(p.shape)),
                    int(p.numel()), float(w.mean()), float(w.std()) if w.numel() > 1 else 0.0,
                    0.0 if g is None else float(np.sqrt((g.astype(np.float64) ** 2).sum())),
                ])
        return rows

    def _hist_values(self, key):
        if key.endswith("@grad"):
            with self._lock:
                g = self.grad.get(key[:-5])
            return None if g is None else g.ravel()
        w = self._weights().get(key)
        return None if w is None else w.ravel()

    def _gpu_util(self):
        now = time.time()
        if now - self._gpu["t"] < GPU_EVERY:
            return self._gpu["v"]
        self._gpu = {"t": now, "v": _gpu_util()}
        return self._gpu["v"]

    def _acts(self):
        total = float(self.act_count.sum())
        if total <= 0:
            return []
        return [[tank.ACTION_NAMES[i], float(v)] for i, v in enumerate(self.act_count)]

    def _buffer_td(self):
        b = self.replay
        if b.size <= 0:
            return None
        p = b.prio[:b.size]
        p = p[p > 0]
        if p.size == 0:
            return None
        return np.power(p, 1.0 / b.alpha) - b.eps

    def _alert_grad(self):
        g = self._grad_stats()[0]
        if g > 200.0:
            return "梯度范数 %.1f，疑似梯度爆炸" % g
        if 0.0 < g < 1e-7:
            return "梯度范数 %.2e，疑似梯度消失" % g
        return None

    def _alert_q(self):
        q = self.stats.get("q_mean")
        if q is None or self.updates < 5000:
            return None
        if abs(q) > 200.0:
            return "平均 Q 值 %.1f 量级异常，价值函数可能发散" % q
        return None

    def _alert_eval(self):
        r = (self.stats.get("eval") or {}).get("随机")
        if r and r["rounds"] >= 400 and r["place"] > 3.2 and self.steps_fn() > 20000000:
            return "对随机平均名次 %.2f（随机水平 2.5），策略可能已退化" % r["place"]
        return None

    def _params(self):
        net = self.policy
        total = sum(p.numel() for p in net.parameters())
        return [
            {"title": "网络结构 · Dueling Double DQN", "layers": self.probe.meta,
             "rows": [["算力", "<b>%s</b>" % ("GPU 加速" if network.DEVICE.type == "cuda" else "CPU")],
                      ["总参数量", "<b>%d</b>" % total]]},
            {"title": "DQN 超参数", "rows": [
                ["Double DQN", "<b>启用</b>（动作由在线网选、价值由目标网给）"],
                ["网络结构", "Dueling：共享卷积 + 价值 V + 优势 A，18 个动作"],
                ["优先经验回放", "α=%.2f（|TD|^α 采样）· β=0.4 重要性权重" % self.replay.alpha],
                ["折扣因子 γ", "%.3f（有效视野约 %d 步）" % (config.GAMMA, int(1 / (1 - config.GAMMA)))],
                ["n-step 回报", "%d 步" % config.NSTEP],
                ["学习率 lr", "%.1e" % config.LR],
                ["批大小", "%d" % config.BATCH],
                ["目标网络同步", "%d 步" % config.TARGET_EVERY],
                ["经验池容量", "%d" % config.REPLAY_CAP],
                ["更新频率", "每 %d 帧一次梯度更新" % config.TRAIN_EVERY],
                ["探索率区间", "%.2f → %.2f（对手方固定 %.2f）" % (config.EPS_START, config.EPS_END, config.OPP_EPS)],
                ["探索衰减步数", "%d" % config.EPS_DECAY],
                ["步惩罚 / 道具奖励", "%.3f / +%.2f" % (config.STEP_PENALTY, config.ITEM_REWARD)],
                ["当前梯度更新", "%d" % self.updates],
            ]},
            {"title": "运行配置", "rows": [
                ["对局形式", "%d 人自由混战 · 活到最后只剩 1 人算赢" % config.PLAYERS],
                ["并行环境", "%d 局（1 组全员自对弈 + 3 组 1 席对 3 席 PFSP 快照）" % self.args.envs],
                ["迷宫", "%d × %d 格 · 墙厚 %.0fpx · 走廊 %.0fpx" % (tank_vec.NC, tank_vec.NR, tank.WALL_W, tank.CORR_W)],
                ["动作数 / 观测", "%d 个动作 / %d 通道 %d×%d" % (network.ACTIONS, tank_vec.OBS_CH, tank_vec.OBS_H, tank_vec.OBS_W)],
                ["回合计时", "上限 %d 帧（%.0f 秒），到点仍多于 1 人存活算平局（双方 0 奖励）" % (
                    tank_vec.ROUND_LIMIT, tank_vec.ROUND_LIMIT / 60.0)],
                ["回合奖励", "最后存活 +1 · 被淘汰 -1 · 每步 %.3f · 拾取道具 +%.2f" % (
                    config.STEP_PENALTY, config.ITEM_REWARD)],
                ["子弹反弹上限", "%d 次" % tank.BOUNCE_LIMIT],
                ["快照池", "每 %d 步存一个，最多 %d 个，PFSP 指数 %.2f" % (
                    config.POOL_SNAPSHOT_EVERY, config.POOL_MAX, config.PFSP_POWER)],
                ["评测频率", "每 %d 步 · %d 局 × %d 帧" % (config.EVAL_EVERY, config.EVAL_ENVS, config.EVAL_FRAMES)],
                ["损失曲面探针", "每 %.0f 秒 · %d×%d 网格 · %d 样本" % (
                    self.land.interval, self.land.size, self.land.size, self.land.batch)],
            ]},
        ]

    def build(self):
        mon = Monitor("坦克动荡 · 自对弈训练监控", port=config.MONITOR_PORT,
                      interval=0.5, history=1200)
        k1, k2, k3 = self.probe.chans
        s1, s2, s3 = self.probe.shapes
        mon.info(lambda: {
            "设备": str(network.DEVICE).upper(),
            "迷宫": "%d×%d" % (tank_vec.NC, tank_vec.NR),
            "并行": "%d 局" % self.args.envs,
            "参数": "%.2fM" % (sum(p.numel() for p in self.policy.parameters()) / 1e6),
        })
        ev = lambda k: (lambda: (self.stats.get("eval") or {}).get(k, {}).get("win"))
        ep = lambda k: (lambda: (self.stats.get("eval") or {}).get(k, {}).get("place"))
        for nm in ("随机", "脚本Lv1", "脚本Lv2", "脚本Lv3", "最早快照"):
            mon.metric("eval_" + nm, ev(nm), label="对" + nm + "胜率", fmt="pct")
        mon.metric("place", lambda: self.stats.get("place"), label="平均名次（随机 2.5）", fmt="f3",
                   color="#3ddc84")
        mon.metric("suicide", lambda: self.stats.get("suicide"), label="自杀占死因比", fmt="pct",
                   color="#e08f7f")
        mon.metric("draw", lambda: self.stats.get("draw"), label="平局率", fmt="pct")
        mon.metric("span", lambda: self.stats.get("span"), label="平均回合秒数", fmt="f2")
        mon.metric("rpm", lambda: self.stats.get("rpm"), label="局/分", fmt="int")
        mon.metric("rew", lambda: self.stats.get("rew"), label="平均回报", fmt="f4")
        mon.metric("loss", lambda: self.stats.get("loss"), label="Loss(Huber)", fmt="f4", color="#ff8f5a")
        mon.metric("q_mean", lambda: self.stats.get("q_mean"), label="平均 Q 值", fmt="f4", color="#c39a5c")
        mon.metric("td_mean", lambda: self.stats.get("td_mean"), label="平均 |TD| 误差", fmt="f4", color="#ff9f7a")
        mon.metric("eps", lambda: self.stats.get("eps"), label="探索率 ε", fmt="f3", color="#8fb9ff")
        mon.metric("spm", lambda: self.stats.get("spm"), label="每秒步数", fmt="int")
        mon.metric("updates", lambda: self.updates, label="梯度更新", fmt="int")
        mon.metric("buf", lambda: self.stats.get("buf"), label="经验池(千)", fmt="f1")
        mon.metric("gpu", self._gpu_util, label="GPU 利用率", fmt="int", color="#cf7fff")

        mon.health("grad_norm", lambda: self._grad_stats()[0], label="梯度范数", fmt="sci")
        mon.health("weight_norm", lambda: self._grad_stats()[1], label="权重范数", fmt="f2")
        mon.health("grad_max", lambda: self._grad_stats()[2], label="梯度最大绝对值", fmt="sci")
        mon.health("lr", lambda: config.LR, label="学习率", fmt="sci")
        mon.health("gamma", lambda: config.GAMMA, label="折扣因子 γ", fmt="f3")
        mon.health("params", lambda: sum(p.numel() for p in self.policy.parameters()),
                   label="可训练参数", fmt="int")
        mon.health("ratio", lambda: self.updates * config.BATCH / max(1.0, self.steps_fn()),
                   label="训练/采样比", unit="×", fmt="f3")
        mon.health("steps", lambda: self.steps_fn(), label="已训练步数", fmt="int")
        mon.health("rounds", lambda: self.stats.get("rounds"), label="累计对局数", fmt="int")
        for nm in ("随机", "脚本Lv1", "脚本Lv2", "脚本Lv3"):
            mon.health("place_" + nm, ep(nm), label="对" + nm + "平均名次", fmt="f2")

        mon.graph(self.probe.graph)
        mon.surface(self.land.snapshot)

        mon.image("battle", lambda: _render(self.env, 0), label="实时对局 1", scale=1, group="战场")
        mon.image("battle2", lambda: _render(self.env, 1), label="实时对局 2", scale=1, group="战场")
        mon.image("obs", lambda: _obs(self.env), label="观测通道（1 号位视角）", scale=3, group="输入与特征")
        for i, (nm, cnt, sp) in enumerate((("C1", k1, s1), ("C2", k2, s2), ("C3", k3, s3))):
            mon.image("feat_" + nm, (lambda k: (lambda: self.probe.feature_image(k)))(i),
                      label="%s 全部 %d 通道（每通道 %d×%d，逐通道归一）" % (nm, cnt, sp, sp),
                      scale=3, group="输入与特征")

        for key in WEIGHT_KEYS:
            mon.histogram(key, (lambda k: (lambda: self._hist_values(k)))(key),
                          label=key + " · 权重", bins=48)
            mon.histogram(key + "@grad", (lambda k: (lambda: self._hist_values(k)))(key + "@grad"),
                          label=key + " · 梯度", bins=48)
        mon.histogram("buffer_td", self._buffer_td, label="经验池 |TD 误差| 分布", bins=40, refresh=5.0)

        mon.bars("layer_grad", self._layer_grad_norm, label="逐层梯度范数")
        mon.bars("acts", self._acts, label="动作使用率（累计）")
        mon.table("layer_stats", self._layer_stats, label="逐层权重与梯度统计",
                  columns=["层", "参数名", "形状", "参数量", "权重均值", "权重标准差", "梯度范数"])
        mon.alert("grad", self._alert_grad, label="梯度", level="warn")
        mon.alert("q", self._alert_q, label="价值量级", level="warn")
        mon.alert("eval", self._alert_eval, label="策略质量", level="warn")
        mon.watchdog("steps", lambda: self.steps_fn(), label="训练步数", timeout=30.0)
        mon.attach_logger(log, key="train_log", label="训练日志（实时镜像）")
        mon.params(self._params)
        self.mon = mon
        threading.Thread(target=self.land.loop, args=(self.stop,), daemon=True).start()
        return mon

    def close(self):
        self.stop.set()


def _render(env, i):
    import tank_view
    return tank_view.render_rgb(env, i)


def _obs(env, i=0, cols=4):
    ch = tank_vec.unpack(env.obs_pack(i)[i:i + 1])[0]
    maps = [(ch[c] * 255).astype(np.uint8) for c in range(tank_vec.OBS_CH)]
    return imaging.montage(maps, cols=cols, pad=1, pad_val=0)
