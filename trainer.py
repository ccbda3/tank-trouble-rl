import threading
import time

import torch

from network import DEVICE, DQN


class Trainer:
    def __init__(self, net, target_net, buffer, optimizer, counter, config, stats, writer=None):
        self.net = net
        self.target_net = target_net
        self.buffer = buffer
        self.optimizer = optimizer
        self.counter = counter
        self.config = config
        self.stats = stats
        self.writer = writer
        self.train_steps = 0
        self.updates = 0
        self.loss = 0.0

    def _hist(self, world, named_params, step):
        for name, p in named_params:
            self.writer.add_histogram(f"{world}/{name}", p.detach().cpu(), step)
            if p.grad is not None:
                self.writer.add_histogram(f"{world}/{name}_grad", p.grad.detach().cpu(), step)

    def save_full(self):
        torch.save({
            "net": self.net.state_dict(),
            "target": self.target_net.state_dict(),
            "opt": self.optimizer.state_dict(),
            "steps": self.counter.get(),
            "train_steps": self.train_steps,
            "updates": self.updates,
            "loss": self.loss,
        }, self.config.FULL_STATE)

    def loop(self, stop_event):
        last_save = time.time()
        self.optimizer.zero_grad()
        while not stop_event.is_set():
            if len(self.buffer) < self.config.BATCH:
                time.sleep(0.05)
                continue
            batch = self.buffer.sample(self.config.BATCH)
            pos, obs, acts, rews, nobs, dones, wts = batch
            s = torch.from_numpy(obs).to(DEVICE)
            a = torch.from_numpy(acts).to(DEVICE).unsqueeze(1)
            r = torch.from_numpy(rews).to(DEVICE).unsqueeze(1)
            s2 = torch.from_numpy(nobs).to(DEVICE)
            d = torch.from_numpy(dones).to(DEVICE).unsqueeze(1)
            w = torch.from_numpy(wts).to(DEVICE).unsqueeze(1)

            with torch.no_grad():
                a_argmax = self.net(s2).max(1, keepdim=True).indices
                target = r + self.config.GAMMA * (1 - d) * self.target_net(s2).gather(1, a_argmax)
            q = self.net(s).gather(1, a)
            td = (q - target).abs()
            loss = (w * torch.nn.functional.smooth_l1_loss(q, target, reduction="none")).mean()
            self.buffer.update_priorities(pos, td.detach().squeeze(1).cpu().numpy())

            loss = loss / self.config.ACCUM
            loss.backward()
            self.train_steps += 1
            with self.stats["lock"]:
                self.stats["loss"] = float(loss.item() * self.config.ACCUM)

            if self.train_steps % self.config.ACCUM == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.updates += 1
                if self.writer is not None:
                    with self.stats["lock"]:
                        cur_loss = self.stats["loss"]
                    self.writer.add_scalar("train/loss", cur_loss, self.updates)
                    self.writer.add_scalar("train/steps", self.counter.get(), self.updates)
                    if self.updates % 25 == 0:
                        self._hist("weights", self.net.named_parameters(), self.updates)
                if self.train_steps % self.config.TARGET_SYNC == 0:
                    self.target_net.load_state_dict(self.net.state_dict())
                if self.train_steps % self.config.SAVE_EVERY == 0:
                    torch.save(self.net.state_dict(), self.config.CHECKPOINT)
                if time.time() - last_save > self.config.CHECKPOINT_INTERVAL:
                    last_save = time.time()
                    torch.save(self.net.state_dict(), self.config.CHECKPOINT)
                    self.save_full()


class LandscapeProbe:
    def __init__(self, net, target_net, buffer, config, size=15, probe_batch=64, interval=20.0, extent=0.6):
        self.net = net
        self.target_net = target_net
        self.buffer = buffer
        self.config = config
        self.size = size
        self.probe_batch = probe_batch
        self.interval = interval
        self.extent = extent
        self.probe = DQN(config.FRAME_STACK, config.GRID, config.ACTIONS).to(DEVICE)
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
            b = torch.randn(p.shape, generator=gen)
            b = b - (b * a).sum() / (a * a).sum() * a
            self.du[n] = (a / a.norm()).to(p.device)
            self.dv[n] = (b / b.norm()).to(p.device)

    def _freeze(self):
        if self.fixed is not None or len(self.buffer) < self.probe_batch:
            return
        pos, obs, acts, rews, nobs, dones, _ = self.buffer.sample(self.probe_batch)
        s = torch.from_numpy(obs).float().to(DEVICE) / 255.0
        a = torch.from_numpy(acts).to(DEVICE).unsqueeze(1)
        r = torch.from_numpy(rews).to(DEVICE).unsqueeze(1)
        s2 = torch.from_numpy(nobs).float().to(DEVICE) / 255.0
        d = torch.from_numpy(dones).to(DEVICE).unsqueeze(1)
        with torch.no_grad():
            am = self.net(s2).max(1, keepdim=True).indices
            tgt = r + self.config.GAMMA * (1 - d) * self.target_net(s2).gather(1, am)
        self.fixed = (s, a, tgt)

    def _loss(self):
        s, a, tgt = self.fixed
        q = self.probe(s).gather(1, a)
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
        alpha = 0.0
        beta = 0.0
        with torch.no_grad():
            for n, p in named:
                dd = p.detach() - self.first[n]
                alpha += float((dd * self.du[n]).sum())
                beta += float((dd * self.dv[n]).sum())
            base = {n: p.detach().clone() for n, p in self.probe.named_parameters()}
            params = list(self.probe.named_parameters())

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
            self.out = {
                "ready": 1,
                "res": res,
                "ext": ext,
                "zmin": min(vals),
                "zmax": max(vals),
                "values": vals,
                "center": [alpha, beta],
                "cur": cur,
            }

    def loop(self, stop_event):
        while not stop_event.is_set():
            try:
                self.measure()
            except Exception:
                pass
            time.sleep(self.interval)