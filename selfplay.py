import argparse
import logging
import sys
import time

import numpy as np
import torch

import buffer
import config
import network
import tank
import tank_vec

DT = 1.0 / 60.0
log = logging.getLogger("tank")
RNG = np.random.default_rng(20240922)


def pack_action(turn, thrust, fire):
    return (fire.astype(np.int64) * 9 + (1 - thrust) * 3 + (turn + 1)).astype(np.int64)


class ScriptedBot:
    def __init__(self, n, level, seed=0):
        self.n = n
        self.level = level
        self.rng = np.random.default_rng(seed)
        self.period = (8, 3, 2)[level - 1]
        self.tol = (0.34, 0.13, 0.05)[level - 1]
        self.lead = (0.0, 0.08, 0.22)[level - 1]
        self.tick = 0
        self.cache = None

    def act(self, env, side, foe):
        if self.cache is None or self.tick % self.period == 0:
            self.cache = self._decide(env, side, foe)
        self.tick += 1
        return self.cache

    def _decide(self, env, side, foe):
        n = 20
        rows = np.arange(env.n)
        tx = env.tx[:, foe] + env.t_vx[:, foe] * self.lead
        ty = env.ty[:, foe] + env.t_vy[:, foe] * self.lead
        want = np.arctan2(ty - env.ty[:, side], tx - env.tx[:, side])
        diff = (want - env.ta[:, side] + np.pi) % (2.0 * np.pi) - np.pi
        turn = np.where(np.abs(diff) < 0.05, 0, np.where(diff > 0, 1, -1))
        f = (np.arange(1, n + 1) / n)[None, :]
        sx = env.tx[:, side][:, None] + (env.tx[:, foe] - env.tx[:, side])[:, None] * f
        sy = env.ty[:, side][:, None] + (env.ty[:, foe] - env.ty[:, side])[:, None] * f
        clear = env.free(np.repeat(rows[:, None], n, axis=1), sx, sy, 1.0).all(axis=1)
        blocked = ~env.free(rows,
                            env.tx[:, side] + np.cos(env.ta[:, side]) * (tank.TANK_H + tank.PITCH * 0.55),
                            env.ty[:, side] + np.sin(env.ta[:, side]) * (tank.TANK_H + tank.PITCH * 0.55),
                            tank.TANK_H)
        thrust = np.where(np.abs(diff) < 1.5, 1, 0)
        thrust = np.where(blocked, np.where(np.abs(diff) < 0.25, -1, 0), thrust)
        fire = (clear & (np.abs(diff) < self.tol)).astype(np.int64)
        return pack_action(turn, thrust, fire)


class RandomBot:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, side, foe):
        return self.rng.integers(0, network.ACTIONS, size=env.n)


class Snapshot:
    def __init__(self, state, feat_dim):
        self.net = network.make(feat_dim)
        self.net.load_state_dict({k: v.clone() for k, v in state.items()})
        self.net.eval()

    def act(self, pack, feat, eps=0.0):
        return self.net.act(network.obs_tensor(pack), network.feat_tensor(feat), eps).cpu().numpy()


def policy_actions(policy, pack, feat, sl, eps):
    with torch.no_grad():
        return policy.act(network.obs_tensor(pack[sl]), network.feat_tensor(feat[sl]), eps).cpu().numpy()


def pick_snapshot(pfsp):
    w = np.asarray(pfsp, dtype=np.float64) ** config.PFSP_POWER
    w = w / w.sum()
    return int(min(np.searchsorted(np.cumsum(w), RNG.random()), len(pfsp) - 1))


def run_eval(policy, opponents, n_envs, frames, seed):
    env = tank_vec.VecEnv(n_envs, 2, seed=seed)
    out = {}
    for name, opp in opponents.items():
        wins = 0.0
        losses = 0.0
        prev = env.rounds.copy()
        for _ in range(frames):
            a0 = policy_actions(policy, env.obs_pack(0), env.feat(0), slice(0, env.n), 0.0)
            if isinstance(opp, Snapshot):
                a1 = opp.act(env.obs_pack(1), env.feat(1), 0.0)
            else:
                a1 = opp.act(env, 1, 0)
            ctrl = np.zeros((env.n, 2, 3), dtype=np.float32)
            ctrl[:, 0] = tank_vec.CTRL[a0]
            ctrl[:, 1] = tank_vec.CTRL[a1]
            env.step(ctrl, DT)
            new = env.rounds - prev
            prev = env.rounds.copy()
            rows = np.nonzero(new > 0)[0]
            if rows.size:
                dead = env.dead[rows]
                wins += float((dead == 1).sum())
                losses += float((dead == 0).sum())
                env._new_round(rows)
        tot = wins + losses
        out[name] = wins / tot if tot > 0 else 0.5
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--envs", type=int, default=config.ENVS)
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(config.LOG, encoding="utf-8")])
    log.info("设备 %s  envs %d  观测 %d 通道 %dx%d  特征 %d 维",
             network.DEVICE, args.envs, tank_vec.OBS_CH, tank_vec.OBS_H, tank_vec.OBS_W,
             tank_vec.VecEnv(1, 2, 0).feat(0).shape[1])

    env = tank_vec.VecEnv(args.envs, 2, seed=config.SEED)
    feat_dim = env.feat(0).shape[1]
    policy = network.make(feat_dim)
    target = network.make(feat_dim)
    target.load_state_dict(policy.state_dict())
    target.eval()
    opt = torch.optim.Adam(policy.parameters(), lr=config.LR)
    replay = buffer.Replay(config.REPLAY_CAP, (tank_vec.OBS_H, tank_vec.OBS_W), feat_dim)
    nstep = buffer.NStep(args.envs, config.NSTEP, config.GAMMA)

    groups = 4
    size = args.envs // groups
    sides = np.zeros(args.envs, dtype=np.int64)
    for g in range(groups):
        sides[g * size:(g + 1) * size] = g % 2
    bots = [ScriptedBot(args.envs, 2, 1), ScriptedBot(args.envs, 3, 2)]
    snapshots = [Snapshot(policy.state_dict(), feat_dim)]
    pfsp = [1.0]

    steps = 0
    it = 0
    rounds_total = 0
    rew_sum = rew_n = 0.0
    loss_sum = loss_n = 0.0
    win_sum = win_n = 0.0
    t0 = time.time()
    t_last = t0
    stats = {"loss": 0.0, "rew": 0.0, "eps": 1.0, "spm": 0.0, "buf": 0.0, "eval": {}}

    monitor = None
    writer = None
    if args.view:
        import tank_view
        writer = tank_view.Writer()
        log.info("观战共享内存已开启：另开一个窗口运行 python tank_view.py 就能看到实时对弈")
    if args.monitor:
        from aimonitor import Monitor
        monitor = Monitor("坦克动荡 · 自对弈", port=config.MONITOR_PORT)
        monitor.metric("loss", lambda: stats["loss"], label="TD 损失")
        monitor.metric("rew", lambda: stats["rew"], label="平均回报")
        monitor.metric("eps", lambda: stats["eps"], label="ε", fmt=".3f")
        monitor.metric("spm", lambda: stats["spm"], label="每秒步数")
        monitor.metric("buf", lambda: stats["buf"], label="回放池(千)")
        monitor.attach_logger(log)
        monitor.start()
        log.info("监控面板 http://127.0.0.1:%d", config.MONITOR_PORT)

    deadline = t0 + args.minutes * 60.0
    while time.time() < deadline:
        it += 1
        packs = [env.obs_pack(0), env.obs_pack(1)]
        feats = [env.feat(0), env.feat(1)]
        ctrl = np.zeros((args.envs, 2, 3), dtype=np.float32)
        eps = max(config.EPS_END, config.EPS_START - steps / config.EPS_DECAY)
        acts = np.zeros(args.envs, dtype=np.int64)
        for g in range(groups):
            sl = slice(g * size, (g + 1) * size)
            ps = int(sides[g * size])
            other = 1 - ps
            a = policy_actions(policy, packs[ps], feats[ps], sl, eps)
            acts[sl] = a
            ctrl[sl, ps] = tank_vec.CTRL[a]
            if g == 0:
                b = policy_actions(policy, packs[other], feats[other], sl, config.OPP_EPS)
            elif g == 1:
                b = snapshots[pick_snapshot(pfsp)].act(packs[other][sl], feats[other][sl], config.OPP_EPS)
            elif g == 2:
                b = bots[0].act(env, other, ps)[sl]
            else:
                b = bots[1].act(env, other, ps)[sl]
            ctrl[sl, other] = tank_vec.CTRL[b]

        prev_rounds = env.rounds.copy()
        env.step(ctrl, DT)
        steps += args.envs
        new = (env.rounds - prev_rounds) > 0
        dead = env.dead
        rew = np.zeros(args.envs, dtype=np.float32)
        rew[new & (dead == sides)] = -1.0
        rew[new & (dead != sides)] = 1.0
        rounds_total += int(new.sum())
        rew_sum += float(rew.sum())
        rew_n += args.envs
        win_sum += float((new & (dead != sides)).sum())
        win_n += int(new.sum())

        packs_next = [env.obs_pack(0), env.obs_pack(1)]
        feats_next = [env.feat(0), env.feat(1)]
        for i in range(args.envs):
            ps = int(sides[i])
            for t in nstep.push(i, packs[ps][i], feats[ps][i], acts[i], rew[i],
                                packs_next[ps][i], feats_next[ps][i], float(new[i])):
                replay.push(t[0], t[1], t[2], t[3], t[4], t[5], t[6])
        rows = np.nonzero(new)[0]
        if rows.size:
            env._new_round(rows)

        if steps >= config.LEARN_START and it % config.TRAIN_EVERY == 0:
            batch = replay.sample(config.BATCH)
            if batch is not None:
                obs_p, feat_p, act_p, ret_p, nobs_p, nfeat_p, done_p, w_p, slot = batch
                obs = network.obs_tensor(obs_p)
                feat = network.feat_tensor(feat_p)
                nobs = network.obs_tensor(nobs_p)
                nfeat = network.feat_tensor(nfeat_p)
                act_t = torch.as_tensor(act_p, dtype=torch.long, device=network.DEVICE)
                ret = torch.as_tensor(ret_p, dtype=torch.float32, device=network.DEVICE)
                done = torch.as_tensor(done_p, dtype=torch.float32, device=network.DEVICE)
                wts = torch.as_tensor(w_p, dtype=torch.float32, device=network.DEVICE)
                q = policy(obs, feat).gather(1, act_t[:, None]).squeeze(1)
                with torch.no_grad():
                    best = policy(nobs, nfeat).argmax(dim=1, keepdim=True)
                    nq = target(nobs, nfeat).gather(1, best).squeeze(1)
                    y = ret + (config.GAMMA ** config.NSTEP) * nq * (1.0 - done)
                td = y - q
                loss = (wts * td.pow(2)).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                opt.step()
                replay.update_prio(slot, td.detach().cpu().numpy())
                loss_sum += float(loss.item())
                loss_n += 1
            if steps % config.TARGET_EVERY < args.envs:
                target.load_state_dict(policy.state_dict())

        if steps % config.POOL_SNAPSHOT_EVERY < args.envs:
            snapshots.append(Snapshot(policy.state_dict(), feat_dim))
            pfsp.append(1.0)
            if len(snapshots) > config.POOL_MAX:
                snapshots.pop(0)
                pfsp.pop(0)

        now = time.time()
        if now - t_last >= 10.0:
            spm = steps / (now - t0)
            stats.update({"loss": loss_sum / max(1, loss_n), "rew": rew_sum / max(1, rew_n),
                          "eps": eps, "spm": spm, "buf": replay.size / 1000.0})
            log.info("步 %d  %.0f 步/秒  ε %.3f  平均回报 %+.3f  胜率 %.1f%%  局/分 %.0f  回放 %.0fk  损失 %.4f",
                     steps, spm, eps, stats["rew"], 100.0 * win_sum / max(1.0, win_n),
                     60.0 * rounds_total / (now - t0), stats["buf"], stats["loss"])
            loss_sum = loss_n = 0.0
            rew_sum = rew_n = 0.0
            win_sum = win_n = 0.0
            t_last = now

        if steps % config.EVAL_EVERY < args.envs:
            opps = {"随机": RandomBot(7), "脚本Lv1": ScriptedBot(64, 1, 3),
                    "脚本Lv2": ScriptedBot(64, 2, 3), "脚本Lv3": ScriptedBot(64, 3, 3),
                    "最早快照": snapshots[0]}
            res = run_eval(policy, opps, config.EVAL_ENVS, config.EVAL_FRAMES, 999)
            stats["eval"] = res
            log.info("评测 %s", "  ".join("%s %.0f%%" % (k, 100.0 * v) for k, v in res.items()))
            torch.save({"model": policy.state_dict(), "steps": steps}, config.CHECKPOINT)

        if writer is not None:
            writer.push(env)
            writer.stats(stats["loss"], stats["rew"], eps, stats["spm"], stats["buf"],
                         win_sum / max(1.0, win_n), steps,
                         [stats["eval"].get(k, 0.0)
                          for k in ("随机", "脚本Lv1", "脚本Lv2", "脚本Lv3", "最早快照")])

    torch.save({"model": policy.state_dict(), "steps": steps}, config.CHECKPOINT)
    log.info("训练结束，共 %d 步", steps)
    if monitor is not None:
        monitor.stop()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
