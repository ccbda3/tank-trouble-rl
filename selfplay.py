import argparse
import logging
import os
import shutil
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
EVAL_NAMES = ("随机", "脚本Lv1", "脚本Lv2", "脚本Lv3", "最早快照")


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

    def act(self, env, side, foes):
        if self.cache is None or self.tick % self.period == 0:
            self.cache = self._decide(env, side, foes)
        self.tick += 1
        return self.cache

    def _target(self, env, side, foes):
        rows = np.arange(env.n)
        best = np.full(env.n, 1e18, dtype=np.float32)
        tx = env.tx[:, side].copy()
        ty = env.ty[:, side].copy()
        vx = np.zeros(env.n, dtype=np.float32)
        vy = np.zeros(env.n, dtype=np.float32)
        for foe in foes:
            d = np.where(env.t_alive[:, foe],
                         (env.tx[:, foe] - tx) ** 2 + (env.ty[:, foe] - ty) ** 2, 1e18)
            better = d < best
            best = np.where(better, d, best)
            tx = np.where(better, env.tx[:, foe], tx)
            ty = np.where(better, env.ty[:, foe], ty)
            vx = np.where(better, env.t_vx[:, foe], vx)
            vy = np.where(better, env.t_vy[:, foe], vy)
        return rows, tx + vx * self.lead, ty + vy * self.lead

    def _decide(self, env, side, foes):
        n = 20
        rows, tx, ty = self._target(env, side, foes)
        want = np.arctan2(ty - env.ty[:, side], tx - env.tx[:, side])
        diff = (want - env.ta[:, side] + np.pi) % (2.0 * np.pi) - np.pi
        turn = np.where(np.abs(diff) < 0.05, 0, np.where(diff > 0, 1, -1))
        f = (np.arange(1, n + 1) / n)[None, :]
        sx = env.tx[:, side][:, None] + (tx - env.tx[:, side])[:, None] * f
        sy = env.ty[:, side][:, None] + (ty - env.ty[:, side])[:, None] * f
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

    def act(self, env, side, foes):
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


def _snap_table(snapshots, snap_at, pfsp, place, rounds):
    rows = []
    for i in range(len(snapshots)):
        avg = place[i] / rounds[i] if rounds[i] > 0 else 0.0
        rows.append(["快照%d" % (i + 1), int(snap_at[i]), round(float(pfsp[i]), 3),
                     int(rounds[i]), round(float(avg), 3)])
    return rows


def run_eval(policy, opponents, n_envs, frames, seed, players):
    env = tank_vec.VecEnv(n_envs, players, seed=seed)
    foes = list(range(1, players))
    out = {}
    for name, opp in opponents.items():
        wins = 0.0
        losses = 0.0
        draws = 0.0
        place_sum = 0.0
        place_n = 0
        span = []
        prev = env.rounds.copy()
        for _ in range(frames):
            ctrl = np.zeros((env.n, players, 3), dtype=np.float32)
            ctrl[:, 0] = tank_vec.CTRL[policy_actions(policy, env.obs_pack(0), env.feat(0),
                                                      slice(0, env.n), 0.0)]
            for j in foes:
                if isinstance(opp, Snapshot):
                    a = opp.act(env.obs_pack(j), env.feat(j), 0.0)
                else:
                    a = opp.act(env, j, [k for k in range(players) if k != j])
                ctrl[:, j] = tank_vec.CTRL[a]
            env.step(ctrl, DT)
            new = env.rounds - prev
            prev = env.rounds.copy()
            rows = np.nonzero(new > 0)[0]
            if rows.size:
                won = rows[env.winner[rows] == 0]
                died = rows[(env.winner[rows] != 0) & (env.place[rows, 0] > 0)]
                wins += float(won.size)
                losses += float(died.size)
                draws += float(rows.size - won.size - died.size)
                p = env.place[rows, 0]
                p = p[p > 0]
                place_sum += float(p.sum())
                place_n += int(p.size)
                span.append(float(np.mean(env.round_frames[rows])))
                env._new_round(rows)
        tot = wins + losses + draws
        out[name] = {"win": wins / max(1.0, wins + losses),
                     "rounds": int(tot),
                     "draws": int(draws),
                     "span": float(np.mean(span)) if span else 0.0,
                     "place": place_sum / place_n if place_n > 0 else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--envs", type=int, default=config.ENVS)
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(config.LOG, encoding="utf-8")])
    log.info("设备 %s  envs %d  %d 人混战  观测 %d 通道 %dx%d  特征 %d 维",
             network.DEVICE, args.envs, config.PLAYERS, tank_vec.OBS_CH,
             tank_vec.OBS_H, tank_vec.OBS_W,
             tank_vec.VecEnv(1, config.PLAYERS, 0).feat(0).shape[1])

    players = config.PLAYERS
    env = tank_vec.VecEnv(args.envs, players, seed=config.SEED)
    feat_dim = env.feat(0).shape[1]
    policy = network.make(feat_dim)
    target = network.make(feat_dim)
    target.load_state_dict(policy.state_dict())
    target.eval()
    opt = torch.optim.Adam(policy.parameters(), lr=config.LR)
    replay = buffer.Replay(config.REPLAY_CAP, (tank_vec.OBS_H, tank_vec.OBS_W), feat_dim)
    nstep = buffer.NStep(args.envs * players, config.NSTEP, config.GAMMA)

    groups = 4
    size = args.envs // groups
    group_snap = np.zeros(groups, dtype=np.int64)
    snapshots = [Snapshot(policy.state_dict(), feat_dim)]
    snap_at = [0]
    pfsp = [1.0]
    snap_place = [0.0]
    snap_rounds = [0.0]

    def drop_first(n):
        for lst in (snapshots, snap_at, pfsp, snap_place, snap_rounds):
            del lst[0:n]

    steps = 0
    last_target = 0
    last_snap = 0
    last_eval = 0
    if args.resume and os.path.exists(config.CHECKPOINT):
        shutil.copyfile(config.CHECKPOINT, config.CHECKPOINT + ".bak")
        ck = torch.load(config.CHECKPOINT, map_location=network.DEVICE)
        policy.load_state_dict(ck["model"])
        target.load_state_dict(policy.state_dict())
        snapshots = [Snapshot(policy.state_dict(), feat_dim)]
        snap_at = [0]
        pfsp = [1.0]
        snap_place = [0.0]
        snap_rounds = [0.0]
        steps = int(ck.get("steps", 0))
        last_target = last_snap = last_eval = steps
        log.info("断点续训：载入 %s，从第 %d 步继续", config.CHECKPOINT, steps)

    it = 0
    rounds_total = 0
    rew_sum = rew_n = 0.0
    loss_sum = loss_n = 0.0
    q_sum = td_sum = td_n = 0.0
    place_sum = place_n = 0.0
    suicide_sum = death_sum = 0.0
    draw_sum = 0.0
    span_sum = 0.0
    span_n = 0
    t0 = time.time()
    t_last = t0
    stats = {"loss": 0.0, "rew": 0.0, "eps": 1.0, "spm": 0.0, "buf": 0.0,
             "place": 2.5, "suicide": 0.0, "draw": 0.0, "span": 0.0, "rpm": 0.0,
             "rounds": 0.0, "q_mean": 0.0, "td_mean": 0.0, "eval": {}}

    monitor = None
    tmon = None
    writer = None
    if args.view:
        import tank_view
        writer = tank_view.Writer()
        log.info("观战共享内存已开启：另开一个窗口运行 python tank_view.py 就能看到实时对弈")
    if args.monitor:
        import tank_monitor
        tmon = tank_monitor.TankMonitor(env, policy, target, replay, stats,
                                        lambda: steps, args)
        monitor = tmon.build()
        monitor.bars("pool", lambda: [("快照%d" % (i + 1), w) for i, w in enumerate(pfsp)],
                     label="对手池采样权重")
        monitor.table("snap", lambda: _snap_table(snapshots, snap_at, pfsp, snap_place, snap_rounds),
                      label="快照池", columns=["快照", "诞生步数", "权重", "对局数", "平均名次"])
        monitor.start()
        log.info("监控面板 http://127.0.0.1:%d", config.MONITOR_PORT)

    deadline = t0 + args.minutes * 60.0
    while time.time() < deadline:
        it += 1
        packs = [env.obs_pack(j) for j in range(players)]
        feats = [env.feat(j) for j in range(players)]
        ctrl = np.zeros((args.envs, players, 3), dtype=np.float32)
        eps = max(config.EPS_END, config.EPS_START - steps / config.EPS_DECAY)
        acts = np.zeros((args.envs, players), dtype=np.int64)
        for g in range(groups):
            sl = slice(g * size, (g + 1) * size)
            if g == 0:
                for j in range(players):
                    acts[sl, j] = policy_actions(policy, packs[j], feats[j], sl, eps)
            else:
                k = pick_snapshot(pfsp)
                group_snap[g] = k
                snap = snapshots[k]
                own = (g - 1) % players
                for j in range(players):
                    if j == own:
                        acts[sl, j] = policy_actions(policy, packs[j], feats[j], sl, eps)
                    else:
                        acts[sl, j] = snap.act(packs[j][sl], feats[j][sl], config.OPP_EPS)
        for j in range(players):
            ctrl[:, j] = tank_vec.CTRL[acts[:, j]]

        before = env.t_alive.copy()
        running = ~env.over.copy()
        prev_rounds = env.rounds.copy()
        env.step(ctrl, DT)
        steps += args.envs * players
        new = (env.rounds - prev_rounds) > 0
        died = env.just_died
        rew = np.full((args.envs, players), config.STEP_PENALTY, dtype=np.float32)
        rew += config.ITEM_REWARD * env.picked
        rew[died] = -1.0
        won_seats = 0
        if new.any():
            r = np.nonzero(new)[0]
            w = env.winner[r]
            ok = w >= 0
            rew[r[ok], w[ok]] = 1.0
            won_seats = int(ok.sum())
        done = died | new[:, None]
        rounds_total += int(new.sum())
        rew_sum += float(rew.sum())
        rew_n += args.envs * players
        place_sum += float(env.place[died].sum()) + float(won_seats)
        place_n += float(died.sum()) + won_seats
        death_sum += float(died.sum())
        suicide_sum += float((died & env.suicide).sum())
        draw_sum += float((new & (env.winner < 0)).sum())
        if new.any():
            span_sum += float(env.round_frames[new].sum())
            span_n += int(new.sum())
        for g in range(1, groups):
            k = group_snap[g]
            sl = slice(g * size, (g + 1) * size)
            m = new[sl]
            if m.any():
                p = env.place[sl][m, (g - 1) % players]
                p = p[p > 0]
                snap_place[k] += float(p.sum())
                snap_rounds[k] += float(p.size)
        if tmon is not None:
            tmon.count_acts(acts)

        packs_next = [env.obs_pack(j) for j in range(players)]
        feats_next = [env.feat(j) for j in range(players)]
        valid = before & running[:, None]
        for i in range(args.envs):
            for j in range(players):
                if not valid[i, j]:
                    continue
                s = i * players + j
                for t in nstep.push(s, packs[j][i], feats[j][i], acts[i, j], rew[i, j],
                                    packs_next[j][i], feats_next[j][i], float(done[i, j])):
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
                q_sum += float(q.detach().mean().item())
                td_sum += float(td.detach().abs().mean().item())
                td_n += 1
                if tmon is not None:
                    tmon.tick_update(policy)
            if steps - last_target >= config.TARGET_EVERY:
                last_target = steps
                target.load_state_dict(policy.state_dict())

        if steps - last_snap >= config.POOL_SNAPSHOT_EVERY:
            last_snap = steps
            snapshots.append(Snapshot(policy.state_dict(), feat_dim))
            snap_at.append(steps)
            pfsp.append(1.0)
            snap_place.append(0.0)
            snap_rounds.append(0.0)
            if len(snapshots) > config.POOL_MAX:
                drop_first(len(snapshots) - config.POOL_MAX)

        now = time.time()
        if now - t_last >= 10.0:
            el = now - t0
            spm = steps / el
            place = place_sum / max(1.0, place_n)
            draw_rate = draw_sum / max(1.0, rounds_total)
            span = span_sum / max(1.0, span_n) * DT
            rpm = 60.0 * rounds_total / el
            kill = suicide_sum / max(1.0, death_sum)
            for k in range(len(pfsp)):
                if snap_rounds[k] > 0:
                    pfsp[k] = float(min(1.0, max(0.05, (snap_place[k] / snap_rounds[k] - 1.0) / 3.0)))
            stats.update({"loss": loss_sum / max(1, loss_n), "rew": rew_sum / max(1, rew_n),
                          "eps": eps, "spm": spm, "buf": replay.size / 1000.0,
                          "place": place, "suicide": kill, "draw": draw_rate, "span": span,
                          "rpm": rpm, "rounds": float(rounds_total),
                          "q_mean": q_sum / max(1.0, td_n),
                          "td_mean": td_sum / max(1.0, td_n)})
            log.info("步 %d  %.0f 步/秒  ε %.3f  回报 %+.4f  平均名次 %.2f（随机 2.5）  "
                     "自杀占死因 %.0f%%  平局 %.0f%%  回合 %.1f 秒  %.0f 局/分  回放 %.0fk  损失 %.4f",
                     steps, spm, eps, stats["rew"], place, 100.0 * kill, 100.0 * draw_rate,
                     span, rpm, stats["buf"], stats["loss"])
            loss_sum = loss_n = 0.0
            q_sum = td_sum = td_n = 0.0
            rew_sum = rew_n = 0.0
            place_sum = place_n = 0.0
            suicide_sum = death_sum = 0.0
            draw_sum = 0.0
            span_sum = 0.0
            span_n = 0
            t_last = now

        if steps - last_eval >= config.EVAL_EVERY:
            last_eval = steps
            opps = {"随机": RandomBot(7), "脚本Lv1": ScriptedBot(config.EVAL_ENVS, 1, 3),
                    "脚本Lv2": ScriptedBot(config.EVAL_ENVS, 2, 3),
                    "脚本Lv3": ScriptedBot(config.EVAL_ENVS, 3, 3),
                    "最早快照": snapshots[0]}
            res = run_eval(policy, opps, config.EVAL_ENVS, config.EVAL_FRAMES, 999, players)
            stats["eval"] = res
            log.info("评测 %s", "  ".join(
                "%s 胜 %.0f%%（不含平局）平 %.0f%% 名次 %.2f 共 %d 局 %.1f 秒" % (
                    k, 100.0 * v["win"], 100.0 * v["draws"] / max(1.0, v["rounds"]),
                    v["place"] or 0.0, v["rounds"], v["span"] * DT)
                for k, v in res.items()))
            torch.save({"model": policy.state_dict(), "steps": steps}, config.CHECKPOINT)

        if writer is not None:
            writer.push(env)
            writer.stats(stats["loss"], stats["rew"], eps, stats["spm"], stats["buf"],
                         stats["place"], stats["draw"], stats["span"], stats["rpm"], steps,
                         [(stats["eval"].get(k) or {}).get("win", 0.0) for k in EVAL_NAMES])

    torch.save({"model": policy.state_dict(), "steps": steps}, config.CHECKPOINT)
    log.info("训练结束，共 %d 步", steps)
    if tmon is not None:
        tmon.close()
    if monitor is not None:
        monitor.stop()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
