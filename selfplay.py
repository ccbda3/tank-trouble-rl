# =============================================================================
# 坦克动荡（Tank Trouble 复刻）四人混战自对弈强化学习 —— 训练主脚本
#
# 训练流程（主循环每一轮按顺序做这几件事）：
#   1) 采样：用当前策略做 ε-贪心，在 256 个并行环境里各推进一步。
#      每局 4 个座位，所以一帧产出 256 × 4 = 1024 条转移（transition）。
#   2) 存池：把奖励、done、下一帧观测写进回放池。为了不等一局打完再学，
#      用一个 n-step 队列把连续 NSTEP 步压缩成一条 n-step 回报再入池。
#   3) 抽样：从 20 万容量的回放池里按优先级（PER）抽 512 条经验。
#   4) 算目标：按 Double DQN 公式算 n-step 目标 y，再算 TD 误差。
#   5) 反传：按重要性采样权重加权做 MSE、裁梯度、Adam 更新在线网络。
#   6) 更新优先级：用刚算出的 |TD 误差| 回写这 512 条的采样概率。
#   7) 周期性动作：每 20000 步同步目标网络；每 50 万步冻结一份快照进对手池；
#      每 100 万步跑一次固定基准评测并存检查点。
#
# 为什么 4 个座位共用同一张网络：
#   四个座位的规则完全对称（没有先后手、没有座位优势），最优策略本身就是
#   同一个函数；拆成 4 张网会让它们各自朝对方的最优反应跑，在收敛期互相
#   追着震荡。共用一张网既省显存，又天然避免了这种"追尾巴"。
#
# 为什么既做自对弈又留历史对手：
#   全部用当前策略自对弈，虽然数据量最大，但对手水平永远等于自己，容易停在
#   "两个同水平选手互相啄"的平台期；把历史快照冻结成对手，能提供不会退化
#   的固定压力，逼策略学会打败"过去的自己"。
# =============================================================================
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

# 物理步长，固定 60Hz，和环境引擎内部的时间约定一致
DT = 1.0 / 60.0
log = logging.getLogger("tank")
# 专供对手池 PFSP 抽样用的固定随机源：只决定"这一次抽哪份快照"，
# 不参与环境重置和网络探索的随机性，方便单独复现对手选择过程
RNG = np.random.default_rng(20240922)
# 观战面板上展示评测胜率的顺序，必须和 run_eval 里 opponents 字典的键顺序一致
EVAL_NAMES = ("随机", "脚本Lv1", "脚本Lv2", "脚本Lv3", "最早快照")


# 把（转向, 油门, 开火）三元组压成一个整数动作 id。
# 动作空间 = 2(开火) × 3(油门: 前进/停/后退) × 3(转向: 左/停/右) = 18 个。
# 编码顺序必须和 tank.ACTION_CTRL 的枚举顺序完全一致，否则动作会被映射错。
# 其中 (1 - thrust) 把 thrust ∈ {-1,0,1} 映射到索引 {2,1,0}，
# (turn + 1) 把 turn ∈ {-1,0,1} 映射到 {0,1,2}。
def pack_action(turn, thrust, fire):
    return (fire.astype(np.int64) * 9 + (1 - thrust) * 3 + (turn + 1)).astype(np.int64)


# 脚本机器人：不学习、纯用几何规则瞄准的"电脑玩家"，分 1~3 档难度。
# 它有两个用途：
#   1) 当"鲶鱼"——训练早期纯自对弈容易被同样幼稚的对手拖住，插几个会瞄准的
#      脚本能逼策略学会走位和闪避；
#   2) 当唯一固定的评估标尺——它的水平不随训练变化，所以"打脚本的胜率"上升
#      才真正说明策略变强了（如果对手也在同步变强，胜率不变说明不了任何事）。
# 三档难度只调三个旋钮：period（决策周期，越小反应越快）、tol（瞄准容差，
# 越小越不轻易开火、越要求瞄得准）、lead（提前量，越大越会预判走位）。
class ScriptedBot:
    # n: 并行环境数；level: 难度 1/2/3；seed: 本机器人自己的随机源。
    # period / tol / lead 用同一个下标 level-1 从三个元组取值，构成难度阶梯。
    # cache: 本决策周期内沿用的动作，让机器人有"反应延迟"而不是每帧完美微调。
    def __init__(self, n, level, seed=0):
        self.n = n
        self.level = level
        self.rng = np.random.default_rng(seed)
        self.period = (8, 3, 2)[level - 1]
        self.tol = (0.34, 0.13, 0.05)[level - 1]
        self.lead = (0.0, 0.08, 0.22)[level - 1]
        self.tick = 0
        self.cache = None

    # 对外接口：只在决策时刻（self.tick 是 period 的整数倍）重新计算动作，
    # 其余帧沿用缓存动作，模拟人类的反应延迟。
    def act(self, env, side, foes):
        if self.cache is None or self.tick % self.period == 0:
            self.cache = self._decide(env, side, foes)
        self.tick += 1
        return self.cache

    # 给每个并行环境挑一个"当前最值得打的目标"：所有还活着的对手里距离最近的那个。
    # 同时返回它的坐标和速度——速度供下面的提前量使用（打移动目标要往它前面打）。
    # 已阵亡的对手距离记成 1e18（等效无穷远），自然不会被选中。
    # 这里是在 numpy 里手写向量化的 argmin：best 记当前最近距离，
    # tx/ty/vx/vy 跟着 better 掩码一起更新成"最近那个对手"的状态。
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

    # 一帧的完整决策：选目标 → 算朝向偏差 → 决定转向/油门/开火。
    def _decide(self, env, side, foes):
        # n 是在"自己 → 预测点"连线上采样的点数，用来做视线通畅检查
        n = 20
        rows, tx, ty = self._target(env, side, foes)
        # want: 指向目标（含提前量）的期望朝向；
        # diff: 期望朝向与当前朝向之差，归一化到 (-π, π]，正值表示需要逆时针转。
        want = np.arctan2(ty - env.ty[:, side], tx - env.tx[:, side])
        diff = (want - env.ta[:, side] + np.pi) % (2.0 * np.pi) - np.pi
        # 偏差小于 0.05 弧度就不转，避免在正前方反复左右抖动
        turn = np.where(np.abs(diff) < 0.05, 0, np.where(diff > 0, 1, -1))
        f = (np.arange(1, n + 1) / n)[None, :]
        sx = env.tx[:, side][:, None] + (tx - env.tx[:, side])[:, None] * f
        sy = env.ty[:, side][:, None] + (ty - env.ty[:, side])[:, None] * f
        # 沿射线采样 n 个点，全部落在空地上才算视线通畅（可以开火）。
        # 这里复用和真实子弹完全相同的碰撞检测 env.free，所以脚本认为能打中的弹道，
        # 实际子弹也会真的走过去，不会出现"看着能打其实打在墙上"的情况。
        clear = env.free(np.repeat(rows[:, None], n, axis=1), sx, sy, 1.0).all(axis=1)
        # 车头前方一个车长加半个格距处被挡住 → 即将撞墙，需要一个脱困动作
        blocked = ~env.free(rows,
                            env.tx[:, side] + np.cos(env.ta[:, side]) * (tank.TANK_H + tank.PITCH * 0.55),
                            env.ty[:, side] + np.sin(env.ta[:, side]) * (tank.TANK_H + tank.PITCH * 0.55),
                            tank.TANK_H)
        # 朝向大致对上（<1.5 弧度）就前进；前方被堵时只有小角度才允许倒车脱困，
        # 角度差大时先原地转向，免得一边乱撞一边后退。
        thrust = np.where(np.abs(diff) < 1.5, 1, 0)
        thrust = np.where(blocked, np.where(np.abs(diff) < 0.25, -1, 0), thrust)
        # 视线通畅且在瞄准容差 tol 内才开火；tol 由难度档位决定，档位越高容差越小
        fire = (clear & (np.abs(diff) < self.tol)).astype(np.int64)
        return pack_action(turn, thrust, fire)


# 随机机器人：在 18 个动作里等概率乱按。
# 只用于评测时充当最低基线（一个及格的策略应该能稳定打赢纯随机），不参与训练。
class RandomBot:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, side, foes):
        return self.rng.integers(0, network.ACTIONS, size=env.n)


# 对手池里的一份历史快照：把某个时刻的策略网络权重深拷贝下来并冻结。
# 为什么不直接引用当前网络当对手：直接引用的话对手会随训练一起变，训练目标就
# 变成了"打一个和我同步乱动的幽灵"，梯度噪声极大、还容易陷入策略互追的循环；
# 冻结成快照后对手行为固定，策略面对的是一个稳定的、可以针对性学习的靶子。
class Snapshot:
    # 深拷贝参数（clone 避免和在线网共享内存），并切到 eval 模式关掉训练态行为
    def __init__(self, state, feat_dim):
        self.net = network.make(feat_dim)
        self.net.load_state_dict({k: v.clone() for k, v in state.items()})
        self.net.eval()

    # 用这张冻结的网做推理；eps 默认 0（评估或当对手时都用贪心，不让对手"手滑"）
    def act(self, pack, feat, eps=0.0):
        return self.net.act(network.obs_tensor(pack), network.feat_tensor(feat), eps).cpu().numpy()


# 让当前策略在指定的环境切片 sl 上批量出动作（纯推理，不建计算图）。
# 采样时 eps > 0 用于探索；评估或当对手时 eps = 0 走贪心。
def policy_actions(policy, pack, feat, sl, eps):
    with torch.no_grad():
        return policy.act(network.obs_tensor(pack[sl]), network.feat_tensor(feat[sl]), eps).cpu().numpy()


# 批量版：调用方已经把多个座位的观测拼成一个大 batch，这里直接整批前向。
# 为什么要拼：如果按 (组, 座位) 逐个调用，一帧要发起 16 次 batch=64 的极小前向，
# 每次都要做一次 GPU 同步并把结果拷回主机。实测这一项占掉整轮迭代 85% 的时间
# （环境推进本身只占 9%），GPU 利用率显示 98% 但功耗只有 64W —— 典型的发射/同步受限。
# 合并成"当前策略一次 + 对手快照一次"之后，内核发射数和同步次数都降到八分之一。
def policy_actions_batch(policy, pack, feat, eps):
    with torch.no_grad():
        return policy.act(network.obs_tensor(pack), network.feat_tensor(feat), eps).cpu().numpy()


# 把若干个座位的同帧观测按座位分组、拼成一个连续 batch（顺序必须和 _scatter_seats 一致）。
def _gather_seats(src, rows, idx):
    parts = [src[j][rows[idx[j]]] for j in range(len(idx)) if idx[j].size]
    return np.concatenate(parts) if parts else None


# 把整批前向输出的动作按座位拆开，散回 acts 对应的 (环境行, 座位) 位置。
def _scatter_seats(dst, a, rows, idx):
    off = 0
    for j in range(len(idx)):
        n_j = int(idx[j].size)
        if n_j:
            dst[rows[idx[j]], j] = a[off:off + n_j]
            off += n_j


# 按 PFSP（Prioritized Fictitious Self-Play）思路从对手池里抽一份快照当本组对手。
# 核心是"越是打不过的对手，越要多打"：权重先做 PFSP_POWER=2 的幂次放大
# （拉开强弱快照之间的差距），再归一化，最后用累积分布做轮盘赌抽样。
# 权重本身由主循环定期根据"受测座位对该快照的平均名次"回写。
def pick_snapshot(pfsp):
    w = np.asarray(pfsp, dtype=np.float64) ** config.PFSP_POWER
    w = w / w.sum()
    return int(min(np.searchsorted(np.cumsum(w), RNG.random()), len(pfsp) - 1))


# 把快照池的内部状态整理成监控面板上的一张表：
# 快照名 / 诞生步数 / 当前采样权重 / 已评测的有效局数 / 受测座位对该快照的平均名次。
# 平均名次里 1 = 赢、4 = 垫底，越小代表这个快照越弱；rounds 为 0 表示还没打过。
def _snap_table(snapshots, snap_at, pfsp, place, rounds):
    rows = []
    for i in range(len(snapshots)):
        avg = place[i] / rounds[i] if rounds[i] > 0 else 0.0
        rows.append(["快照%d" % (i + 1), int(snap_at[i]), round(float(pfsp[i]), 3),
                     int(rounds[i]), round(float(avg), 3)])
    return rows


# 固定基准评测：让当前策略坐 0 号座位，其余座位由固定对手填充，各跑 EVAL_FRAMES 帧，
# 统计胜率、平局率、平均名次和平均回合时长。
# 几个关键设计：
#   * 胜率分母排除平局（wins / (wins + losses)）：一局有 30 秒超时上限（ROUND_LIMIT），
#     "打不死对手但也死不掉"的局很常见，把平局算进分母会系统性压低分数、掩盖真实进步；
#   * 用"名次"作为主指标：四人局名次有 1~4 四档区分度，比只有 0/1 的胜率对进步更敏感；
#   * 全部用 ε=0 的贪心动作 + 固定种子，保证不同训练阶段的评测结果可以横向比较。
def run_eval(policy, opponents, n_envs, frames, seed, players):
    # 单独开一份评测环境（不占用训练环境），固定种子保证每次评测的棋盘和出生点可比
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
            # 0 号座位永远是当前策略（贪心，ε=0）
            ctrl[:, 0] = tank_vec.CTRL[policy_actions(policy, env.obs_pack(0), env.feat(0),
                                                      slice(0, env.n), 0.0)]
            for j in foes:
                # 1~3 号座位交给对手：快照走批量网络推理，脚本/随机机器人按行调用。
                # 脚本机器人要显式排除自己座位，才能正确挑出最近的敌人。
                if isinstance(opp, Snapshot):
                    a = opp.act(env.obs_pack(j), env.feat(j), 0.0)
                else:
                    a = opp.act(env, j, [k for k in range(players) if k != j])
                ctrl[:, j] = tank_vec.CTRL[a]
            env.step(ctrl, DT)
            # rounds 计数增加，说明这一帧里有环境刚打完一局（有人活到最后，或超时收局）
            new = env.rounds - prev
            prev = env.rounds.copy()
            rows = np.nonzero(new > 0)[0]
            if rows.size:
                # 受测座位是赢家 → 胜；名次 > 0 但没赢（说明已阵亡）→ 负；
                # 其余情况（超时平局，名次为 0）单独记，不计入胜负分母。
                won = rows[env.winner[rows] == 0]
                died = rows[(env.winner[rows] != 0) & (env.place[rows, 0] > 0)]
                wins += float(won.size)
                losses += float(died.size)
                draws += float(rows.size - won.size - died.size)
                # 名次 0 是无效值（该座位没参与本局结算），过滤掉再求平均
                p = env.place[rows, 0]
                p = p[p > 0]
                place_sum += float(p.sum())
                place_n += int(p.size)
                span.append(float(np.mean(env.round_frames[rows])))
                # 手动重置这些环境开下一局（评测循环自己控制节奏，不走内部自动重置）
                env._new_round(rows)
        tot = wins + losses + draws
        # win 的分母刻意不含平局；place 是平均名次（四人局随机乱打的期望约 2.5，越小越强）
        out[name] = {"win": wins / max(1.0, wins + losses),
                     "rounds": int(tot),
                     "draws": int(draws),
                     "span": float(np.mean(span)) if span else 0.0,
                     "place": place_sum / place_n if place_n > 0 else 0.0}
    return out


# 训练主循环：解析参数 → 建环境/网络/优化器/回放池 → 反复「采样 + 训练」→
# 定期评测存检查点 → 收尾。下面按阶段注释。
def main():
    ap = argparse.ArgumentParser()
    # --minutes: 本次训练跑多少墙钟分钟；--envs: 并行环境数（默认 256）；
    # --monitor: 开 Web 监控面板；--view: 开观战共享内存（配合 tank_view.py）；
    # --resume: 从检查点续训。
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--envs", type=int, default=config.ENVS)
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--resume", action="store_true")
    # --ckpt/--log：覆盖默认的检查点与日志路径。跑 A/B 对比实验时用得上——
    # 两组实验的存档必须分开，否则后一组会把前一组的成果覆盖掉。
    ap.add_argument("--ckpt", default=config.CHECKPOINT)
    ap.add_argument("--log", default=config.LOG)
    # 迷宫连通度课程（课程学习）：从 --braid-from 线性退火到 --braid-to，
    # 跨 --braid-steps 步走完。不给 --braid-from 就固定用生成器默认值（0.70）。
    ap.add_argument("--braid-from", type=float, default=None)
    ap.add_argument("--braid-to", type=float, default=None)
    ap.add_argument("--braid-steps", type=float, default=4.0e7)
    args = ap.parse_args()

    # 日志同时写控制台和文件（UTF-8），这样跑在无控制台的 pythonw 下也能留档。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.log, encoding="utf-8")])
    # 启动横幅：打印设备、并行度、人数、观测张量形状和特征维度，
    # 方便一眼确认观测/特征是否和网络输入对得上。
    log.info("设备 %s  envs %d  %d 人混战  观测 %d 通道 %dx%d  特征 %d 维",
             network.DEVICE, args.envs, config.PLAYERS, tank_vec.OBS_CH,
             tank_vec.OBS_H, tank_vec.OBS_W,
             tank_vec.VecEnv(1, config.PLAYERS, 0).feat(0).shape[1])

    players = config.PLAYERS
    env = tank_vec.VecEnv(args.envs, players, seed=config.SEED)
    feat_dim = env.feat(0).shape[1]
    # policy 是在线网络：负责选动作，也是唯一被梯度更新的网。
    # target 是目标网络：只负责提供 bootstrap 的目标价值，参数定期从 policy 硬拷贝、
    # 其余时间冻结。这是 DQN 能稳定的关键——如果目标和在线是同一张网，每次更新都会
    # 同时移动"被逼近的目标"，等于在追一个自己推着跑的靶子，极易发散。
    policy = network.make(feat_dim)
    target = network.make(feat_dim)
    target.load_state_dict(policy.state_dict())
    target.eval()
    opt = torch.optim.Adam(policy.parameters(), lr=config.LR)
    # 优先经验回放池：容量 20 万条。太小则样本高度相关、容易灾难性遗忘；太大则池里
    # 大量数据来自很久以前的旧策略（离策略程度过高），反而拖慢学习。按 1024 条/帧的
    # 产出算，20 万大约相当于最近 200 帧的新鲜数据。
    replay = buffer.Replay(config.REPLAY_CAP, (tank_vec.OBS_H, tank_vec.OBS_W), feat_dim)
    # n-step 队列的"时间线"数 = 环境数 × 座位数：每个 (环境, 座位) 是一条独立的时间线，
    # 各自的 n-step 缓冲不能混在一起（不同座位之间没有时间连续性）。
    # 传观测图尺寸和特征维度是因为批量版内部要用数组整批存放队列内容。
    nstep = buffer.NStep(args.envs * players, config.NSTEP, config.GAMMA,
                         (tank_vec.OBS_H, tank_vec.OBS_W), feat_dim)

    # 把 256 个并行环境分成 4 组，用来混合"自对弈数据"和"打强敌数据"：
    #   第 0 组：4 个座位全部由当前策略驱动 —— 纯自对弈，提供大规模、水平匹配的数据；
    #   第 1~3 组：每组只有 1 个"受测座位"是当前策略，另外 3 个座位都来自同一份历史
    #             快照 —— 专门制造"我必须打赢这个老版本"的压力。
    # 每组锁定同一份快照、而不是每个座位各抽一份，是为了让名次统计干净：
    # 3 个同源对手 vs 1 个受测者，受测座位的名次才有明确含义（见下面 PFSP 权重更新）。
    groups = 4
    size = args.envs // groups
    # 预先算好"每个 (环境, 座位) 该由谁驱动"，主循环每帧直接用，避免每帧重复比较和索引。
    #   own[g]      —— 第 1~3 组的受测座位号；(g-1) % players 让受测座位在组间轮换，
    #                  使"(受测座位, 对手快照)"的所有组合都被公平覆盖
    #   seat_group  —— 每个 (环境, 座位) 属于哪一组
    #   is_policy   —— True 表示该座位本帧用当前策略出动作；False 表示用对手快照
    # rows_p/rows_s 是环境行号，idx_p/idx_s 是"按座位分组后在该批次里的位置"，
    # 供 _gather_seats / _scatter_seats 拼接与散回观测用。
    own = np.zeros(groups, dtype=np.int64)
    for g in range(1, groups):
        own[g] = (g - 1) % players
    seat_group = np.zeros((args.envs, players), dtype=np.int64)
    for g in range(groups):
        seat_group[g * size:(g + 1) * size] = g
    is_policy = (seat_group == 0) | (own[seat_group] == np.arange(players)[None, :])
    rows_p, seats_p = np.nonzero(is_policy)
    rows_s, seats_s = np.nonzero(~is_policy)
    idx_p = [np.nonzero(seats_p == j)[0] for j in range(players)]
    idx_s = [np.nonzero(seats_s == j)[0] for j in range(players)]
    # 对手池的五个平行数组（下标一一对应）：
    #   snapshots   —— 冻结的网络本体
    #   snap_at     —— 各快照诞生在第几步（用于面板展示）
    #   pfsp        —— 该快照的采样权重，越大表示越该优先拿来打
    #   snap_place  —— 受测座位对它累计拿到的名次和
    #   snap_rounds —— 累计的有效对局数（求平均名次的分母）
    # 开局先放一份"当前网络"的快照，保证池子不为空。
    snapshots = [Snapshot(policy.state_dict(), feat_dim)]
    snap_at = [0]
    pfsp = [1.0]
    snap_place = [0.0]
    snap_rounds = [0.0]

    # 快照数超过上限时从最旧的开始丢弃（列表首部是最旧的）
    def drop_first(n):
        for lst in (snapshots, snap_at, pfsp, snap_place, snap_rounds):
            del lst[0:n]

    # steps: 全局训练步数（每帧 += 环境数 × 座位数）；
    # 三个 last_* 记录各类周期性动作上次触发的步数。
    steps = 0
    last_target = 0
    last_snap = 0
    last_eval = 0
    # 断点续训：先把检查点备份成 .bak，再载入模型权重和历史步数。
    # 注意对手池会被重置成"只有当前网络这一份快照"（旧池子没有随检查点一起保存），
    # 续训后需要重新积累快照——这比用一份来历不明的旧池子更可控。
    if args.resume and os.path.exists(args.ckpt):
        shutil.copyfile(args.ckpt, args.ckpt + ".bak")
        ck = torch.load(args.ckpt, map_location=network.DEVICE)
        policy.load_state_dict(ck["model"])
        target.load_state_dict(policy.state_dict())
        snapshots = [Snapshot(policy.state_dict(), feat_dim)]
        snap_at = [0]
        pfsp = [1.0]
        snap_place = [0.0]
        snap_rounds = [0.0]
        steps = int(ck.get("steps", 0))
        # 三个周期计时器一起推到当前步数，避免刚恢复就立刻触发同步/存快照/评测
        last_target = last_snap = last_eval = steps
        log.info("断点续训：载入 %s，从第 %d 步继续", args.ckpt, steps)

    # 以下累加器都服务于"每 10 秒统计窗口"：打日志时求平均然后清零。
    it = 0
    # steps0 是本窗口起点的步数，用来算本窗口的步速
    steps0 = steps
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
    # 共享给监控面板/观战窗口的实时指标字典（对方每帧读它取最新值）。
    # place 初值给 2.5 = 四人局随机乱打的期望名次，当作观测起点。
    stats = {"loss": 0.0, "rew": 0.0, "eps": 1.0, "spm": 0.0, "buf": 0.0,
             "place": 2.5, "suicide": 0.0, "draw": 0.0, "span": 0.0, "rpm": 0.0,
             "rounds": 0.0, "q_mean": 0.0, "td_mean": 0.0, "eval": {}}

    # 观战/监控都是可选的，只影响可观测性，不影响训练逻辑本身。
    monitor = None
    tmon = None
    writer = None
    # 观战模式：把每帧画面写进共享内存，由 tank_view.py 在另一个窗口渲染。
    if args.view:
        import tank_view
        writer = tank_view.Writer()
        log.info("观战共享内存已开启：另开一个窗口运行 python tank_view.py 就能看到实时对弈")
    # 监控面板拿到的是"取值回调"（lambda）而不是当时的快照值，这样每次刷新页面都能
    # 读到最新状态——对手池权重、快照表都是在训练过程中不断变化的。
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

    # 主循环：按墙钟时间跑满 --minutes 分钟（而不是按固定步数，便于控制实验时长）。
    deadline = t0 + args.minutes * 60.0
    # 迷宫难度课程（课程学习）的步数起点：从续训点开始算，跑满 --braid-steps 步退火到位。
    braid_base = steps
    while time.time() < deadline:
        it += 1
        # 按训练进度线性调整迷宫连通度：一开始给最开阔的地图（几乎没死胡同），
        # 让策略先学会"瞄准、躲弹、别把自己打死"这些能在开阔场里获得密集反馈的技能，
        # 再逐步加回死胡同，逼它学走廊反弹这种真正的难点。
        # 只影响"之后新生成的迷宫"，已经在打的这局不会被中途换图。
        if args.braid_from is not None:
            frac = min(1.0, max(0.0, (steps - braid_base) / args.braid_steps))
            env.braid = args.braid_from + (args.braid_to - args.braid_from) * frac
        # ---- 采样阶段 ----
        # 每个座位一份观测（网络按座位分别看自己视角的画面）。
        packs = [env.obs_pack(j) for j in range(players)]
        feats = [env.feat(j) for j in range(players)]
        ctrl = np.zeros((args.envs, players, 3), dtype=np.float32)
        # ε-贪心探索率：从 EPS_START=1.0 线性衰减到 EPS_END=0.05，衰减跨 3000 万步。
        # 3000 万步是上千局的量级：足够让策略在"敢乱试"阶段把走位/瞄准/道具的基本盘踩完。
        # 衰减太快会提前锁死在坏策略上再也出不来，太慢则后期还在乱按、曲线上看不出进步。
        # 末端保留 0.05 的残余探索，是为了始终维持一点探索能力。
        eps = max(config.EPS_END, config.EPS_START - steps / config.EPS_DECAY)
        # acts 是网络输出的离散动作 id，ctrl 是喂给物理引擎的连续控制量。
        acts = np.zeros((args.envs, players), dtype=np.int64)
        # 本帧给第 1~3 组抽一份共同的陪练快照。抽签仍然是按 PFSP 权重来的，
        # 只是"每帧抽一次"而不是"每组各抽一次"：快照的采样分布不变，
        # 但这样 3 个组的 9 个对手座位可以合并成一次前向。
        snap_k = pick_snapshot(pfsp)
        # 当前策略负责的座位一次算完：第 0 组全部 4 个座位 + 第 1~3 组各 1 个受测座位。
        # 这些座位规则完全对称、共用一张网，合批不会改变任何决策结果。
        a = policy_actions_batch(policy, _gather_seats(packs, rows_p, idx_p),
                                 _gather_seats(feats, rows_p, idx_p), eps)
        _scatter_seats(acts, a, rows_p, idx_p)
        # 对手座位也一次算完。统一用 snapshots[snap_k]，并保留 OPP_EPS=0.08 的随机性：
        # 完全确定的对手会让策略学会"背板"而不是学会应对不确定性，也更容易被针对性利用。
        a = snapshots[snap_k].act(_gather_seats(packs, rows_s, idx_s),
                                  _gather_seats(feats, rows_s, idx_s), config.OPP_EPS)
        _scatter_seats(acts, a, rows_s, idx_s)
        # 把 18 个离散动作 id 翻译回连续的（转向, 油门, 开火）控制量。
        for j in range(players):
            ctrl[:, j] = tank_vec.CTRL[acts[:, j]]

        # 推进物理之前先快照三个"推进前"的状态，后面判断哪些转移算有效经验全靠它们：
        #   before      —— 该座位这一帧开始时是否还活着
        #   running     —— 该局这一帧开始时是否还没结束
        #   prev_rounds —— 与推进后的 rounds 比较，用来识别"这一帧打完了哪些局"
        before = env.t_alive.copy()
        running = ~env.over.copy()
        prev_rounds = env.rounds.copy()
        env.step(ctrl, DT)
        # 一帧 = 一个决策步；步数按 环境数 × 座位数 累加，因为每个 (环境, 座位) 都产出一条转移
        steps += args.envs * players
        new = (env.rounds - prev_rounds) > 0
        died = env.just_died
        # 奖励设计：整体是稀疏奖励，只有终局才有 ±1 的实质信号，中途加一点点塑形信号。
        #   * 基线 STEP_PENALTY = -0.001：每步都扣一点，让"磨蹭不出手"有代价。
        #     否则在纯稀疏奖励下，最优行为很容易退化成原地躲着不动、等别人互相打死
        #     （因为只要不动就不会失误）。这个小惩罚不改变"赢 = +1"的优先级，
        #     只是在同等胜率下更偏好"更快解决战斗"的策略。
        #   * 拾取道具 +0.05：给稀疏奖励补一点中途正反馈，鼓励主动跑图抢资源。
        #   * 被淘汰 -1：终局负信号；活到最后 +1（见下面）。
        # 顺序是：先全填步惩罚 → 叠加道具奖励 → 覆盖阵亡惩罚 → 再补赢家 +1。
        rew = np.full((args.envs, players), config.STEP_PENALTY, dtype=np.float32)
        rew += config.ITEM_REWARD * env.picked
        rew[died] = -1.0
        won_seats = 0
        if new.any():
            r = np.nonzero(new)[0]
            w = env.winner[r]
            # winner < 0 表示超时平局、没有赢家，这种情况不给任何终局奖励
            ok = w >= 0
            rew[r[ok], w[ok]] = 1.0
            won_seats = int(ok.sum())
        # done 表示"这条转移之后该座位的时间线断了"，有两种情况：该座位阵亡，
        # 或者这一局结束了（new[:, None] 会盖住该局的所有座位）。
        # 为什么幸存者也要 done=1：回合结束后环境立刻重置，下一帧 obs 已经是新一局的
        # 随机开局，和本局终局状态没有时间连续性；如果还按 done=0 去 bootstrap，就等于
        # 把"下一局开局的价值"错误地接到"本局终局"后面，目标 y 会被系统性污染。
        done = died | new[:, None]
        # ---- 本窗口的统计累加 ----
        rounds_total += int(new.sum())
        rew_sum += float(rew.sum())
        rew_n += args.envs * players
        # 名次只在"有人阵亡"或"有人夺冠"时才有效，所以分子分母都只按这两类事件累加
        place_sum += float(env.place[died].sum()) + float(won_seats)
        place_n += float(died.sum()) + won_seats
        death_sum += float(died.sum())
        # 自杀 = 被自己打出的子弹打死，是衡量策略"会不会自爆"的关键指标
        suicide_sum += float((died & env.suicide).sum())
        # 平局 = 回合结束但没有赢家（超时收局）
        draw_sum += float((new & (env.winner < 0)).sum())
        if new.any():
            span_sum += float(env.round_frames[new].sum())
            span_n += int(new.sum())
        # 回填快照名次统计（只统计第 1~3 组，也就是有快照当陪练的那些组）。
        # 受测座位号是 own[g]，取它在本局拿到的名次。
        # 名次 0 表示该座位没有参与这一局的结算，过滤掉。
        # 这两个累加器是后面 PFSP 权重更新的唯一数据来源。
        for g in range(1, groups):
            sl = slice(g * size, (g + 1) * size)
            m = new[sl]
            if m.any():
                p = env.place[sl][m, own[g]]
                p = p[p > 0]
                snap_place[snap_k] += float(p.sum())
                snap_rounds[snap_k] += float(p.size)
        if tmon is not None:
            # 把本帧的动作分布喂给面板，用来看策略有没有退化成一两个动作
            tmon.count_acts(acts)

        # ---- 存池阶段 ----
        # 推进后的观测就是"下一帧状态"
        packs_next = [env.obs_pack(j) for j in range(players)]
        feats_next = [env.feat(j) for j in range(players)]
        # valid = 该座位这一帧开始时还活着，且该局当时还没结束。
        # 为什么要判这个：如果座位这一帧开始时就已阵亡，或者这局已经结束，那么它这一帧的
        # obs/act/rew 都是毫无意义的"死亡停顿帧"（引擎已不再驱动它，动作和奖励都是空的），
        # 把它们当经验灌进池子等于给网络喂噪声。
        valid = before & running[:, None]
        # stream 编号 s = 环境序号 × 座位数 + 座位号：把二维的 (环境, 座位) 展平成
        # 一维时间线编号，和 NStep 初始化时的 n_streams = 环境数 × 座位数 对应，
        # 保证每条 (环境, 座位) 时间线都有自己独立的 n-step 队列。
        # np.stack(axis=1) 把 4 个座位的观测拼成 (环境, 座位, H, W)，再 reshape 成
        # (环境×座位, H, W) 就是零拷贝视图，索引恰好是 i*座位数 + j，与 stream 编号一致。
        shape = (tank_vec.OBS_H, tank_vec.OBS_W)
        pk = np.stack(packs, axis=1).reshape(-1, *shape)
        nk = np.stack(packs_next, axis=1).reshape(-1, *shape)
        ft = np.stack(feats, axis=1).reshape(-1, feat_dim)
        nf = np.stack(feats_next, axis=1).reshape(-1, feat_dim)
        # nstep.push 整批推进所有时间线，返回本帧"已经凑够 NSTEP 步 / 遇到 done"的样本；
        # 一条都没凑够时返回 None。样本里存的是 n-step 折扣回报，不是单步奖励。
        out = nstep.push(valid.reshape(-1), pk, ft, acts.reshape(-1), rew.reshape(-1),
                         nk, nf, done.reshape(-1))
        if out is not None:
            replay.push_batch(*out)
        # 打完的局在这里统一重置开新局。必须放在采样和存池之后：上面用到的终局观测、
        # 奖励、名次都是"打完那一刻"的状态，提前重置会把它们覆盖掉。
        rows = np.nonzero(new)[0]
        if rows.size:
            env._new_round(rows)

        # ---- 训练阶段 ----
        # LEARN_START：先攒够 2 万步经验再开始学。池子太小时抽出来的 512 条几乎全来自最近
        # 几帧、高度相关，早期更新基本等于在拟合最近几帧，白费算力。
        # TRAIN_EVERY=2：实测吞吐 6000~8000 步/秒，如果每一帧都做一次梯度更新，训练会变成
        # "采样等训练"的瓶颈；每 2 帧更新一次，让采样和计算大致平衡。
        if steps >= config.LEARN_START and it % config.TRAIN_EVERY == 0:
            # 按优先级抽 512 条；池子不够 512 条时返回 None，跳过本次更新
            batch = replay.sample(config.BATCH)
            if batch is not None:
                obs_p, feat_p, act_p, ret_p, nobs_p, nfeat_p, done_p, w_p, slot = batch
                # 把 numpy 批次搬上 GPU 并转成张量
                obs = network.obs_tensor(obs_p)
                feat = network.feat_tensor(feat_p)
                nobs = network.obs_tensor(nobs_p)
                nfeat = network.feat_tensor(nfeat_p)
                act_t = torch.as_tensor(act_p, dtype=torch.long, device=network.DEVICE)
                ret = torch.as_tensor(ret_p, dtype=torch.float32, device=network.DEVICE)
                done = torch.as_tensor(done_p, dtype=torch.float32, device=network.DEVICE)
                wts = torch.as_tensor(w_p, dtype=torch.float32, device=network.DEVICE)
                # q：在线网络对"实际执行过的动作"的估值，也就是要被拉向目标的那一侧
                q = policy(obs, feat).gather(1, act_t[:, None]).squeeze(1)
                with torch.no_grad():
                    # Double DQN 目标：动作由在线网 argmax 选出，价值由目标网给出。
                    # 这样做是为了消除普通 DQN 的"最大化偏差"——如果选动作和估价值都用同一个
                    # 带噪声的网络，max 操作会系统性地偏向那些被噪声高估的动作，
                    # 训练越久高估越离谱。
                    best = policy(nobs, nfeat).argmax(dim=1, keepdim=True)
                    nq = target(nobs, nfeat).gather(1, best).squeeze(1)
                    # ret 已经是 n-step 折扣回报（γ^0 ~ γ^(NSTEP-1) 累加），再乘 γ^NSTEP 把
                    # 第 NSTEP 步之后的价值打折接上，两部分共同构成完整的 n-step 目标；
                    # 乘 (1 - done) 则在终局处切断 bootstrap。
                    # 为什么用 n-step=5 而不是单步：一局要 30 秒才结束，单步回报几乎全是
                    # -0.001 的惩罚，价值信号要几十步才能往回传播一次。5 步回报（γ^5≈0.951）
                    # 让"我这步之后 5 帧会不会出事"立刻有信号，既显著加快收敛，又没有引入
                    # 太多自举误差（自举越深、方差越大）。
                    y = ret + (config.GAMMA ** config.NSTEP) * nq * (1.0 - done)
                td = y - q
                # 损失是按 PER 重要性权重加权的 MSE。
                # wts 是对"优先采样"造成分布偏差的修正（β=0.4）：优先回放会特意多抽高误差
                # 样本，样本分布和真实分布不再一致，用 (1/采样概率)^β 加权可以把偏差往回收
                # 一部分，避免训练被少数异常样本带偏。
                loss = (wts * td.pow(2)).mean()
                opt.zero_grad()
                loss.backward()
                # 梯度裁剪上限 10：偶尔出现的大 TD 误差会让梯度很猛，裁一下防止参数一步被打飞
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                opt.step()
                # 用刚算出的 |TD 误差| 更新这 512 条的采样概率（p = |TD|^α，α=0.6）。
                # 误差大说明"这块还没学会"，下次更该被抽到；用 α 做平滑是为了不让采样概率
                # 完全被少数极端样本垄断。
                replay.update_prio(slot, td.detach().cpu().numpy())
                # 累加诊断量：损失、Q 均值、|TD| 均值。
                # Q 均值和 TD 均值是判断训练是否发散的主要窗口——Q 一路涨而 TD 不降，
                # 通常意味着价值在虚高（高估）。
                loss_sum += float(loss.item())
                loss_n += 1
                q_sum += float(q.detach().mean().item())
                td_sum += float(td.detach().abs().mean().item())
                td_n += 1
                if tmon is not None:
                    tmon.tick_update(policy)
            # 目标网络定期硬同步：每 TARGET_EVERY=20000 步把在线网参数整体拷到目标网。
            # 间隔太小，目标跟着在线一起动，"稳定靶子"的作用就没了；间隔太大，目标严重
            # 滞后于当前策略，学到的价值会一直落后。
            if steps - last_target >= config.TARGET_EVERY:
                last_target = steps
                target.load_state_dict(policy.state_dict())

        # ---- 对手池维护 ----
        # 每 POOL_SNAPSHOT_EVERY=50 万步把当前策略冻结一份存进池子。50 万步这个量级意味着
        # 相邻两份快照之间策略已经有可感知的差异，快照才算"不同水平的对手"；如果间隔太短，
        # 池子里全是几乎相同的网，PFSP"优先练打不过的"就失去意义了。
        # 新快照的初始权重设为 1.0（最高档），相当于给新对手一次优先被打的机会。
        if steps - last_snap >= config.POOL_SNAPSHOT_EVERY:
            last_snap = steps
            snapshots.append(Snapshot(policy.state_dict(), feat_dim))
            snap_at.append(steps)
            pfsp.append(1.0)
            snap_place.append(0.0)
            snap_rounds.append(0.0)
            # POOL_MAX=12 是池子上限：太小对手多样性不足，太大则每个快照分到的评测局数被
            # 稀释，PFSP 权重更新会变得很慢、很迟钝。
            if len(snapshots) > config.POOL_MAX:
                drop_first(len(snapshots) - config.POOL_MAX)

        # ---- 每 10 秒打一次日志并刷新监控 ----
        now = time.time()
        if now - t_last >= 10.0:
            el = now - t0
            # 各项指标的含义：
            #   spm       —— 步/秒，训练吞吐
            #   place     —— 平均名次，四人局随机乱打的期望约 2.5，越小越强
            #   draw_rate —— 平局率（30 秒超时未分胜负的局占比）
            #   span      —— 回合平均时长（秒）
            #   rpm       —— 每分钟打完多少局
            #   kill      —— 阵亡里"自杀"的占比，用来衡量策略会不会被自己的子弹打死
            spm = (steps - steps0) / el
            place = place_sum / max(1.0, place_n)
            draw_rate = draw_sum / max(1.0, rounds_total)
            span = span_sum / max(1.0, span_n) * DT
            rpm = 60.0 * rounds_total / el
            kill = suicide_sum / max(1.0, death_sum)
            # —— PFSP 采样权重更新 ——
            # 目标是"越是打不过的对手，越要多打"。avg 是受测座位对该快照的平均名次：
            # 1 = 我总能赢，4 = 我总垫底。
            # (avg - 1) / 3 把名次 1~4 映射到 0~1：名次 1 → 权重 0（不值得再练），
            # 名次 4 → 权重 1（最该优先练）。
            # 再夹到 [0.05, 1.0]：下限 0.05 保证已经打赢的弱对手偶尔还会出现，避免完全不
            # 复习而遗忘；上限 1.0 避免某个强敌垄断采样、让其他对手饿死。
            # 没打过（rounds=0）的快照保持初始权重不变，会被优先探索到。
            for k in range(len(pfsp)):
                if snap_rounds[k] > 0:
                    pfsp[k] = float(min(1.0, max(0.05, (snap_place[k] / snap_rounds[k] - 1.0) / 3.0)))
            # 刷新给监控面板读取的指标快照
            stats.update({"loss": loss_sum / max(1, loss_n), "rew": rew_sum / max(1, rew_n),
                          "eps": eps, "spm": spm, "buf": replay.size / 1000.0,
                          "place": place, "suicide": kill, "draw": draw_rate, "span": span,
                          "rpm": rpm, "rounds": float(rounds_total),
                          "q_mean": q_sum / max(1.0, td_n),
                          "td_mean": td_sum / max(1.0, td_n)})
            # 一行日志把"训练是否健康"浓缩成一眼能看的样子：步速、探索率、平均回报、平均
            # 名次（对照随机基线 2.5）、自杀占比、平局率、回合时长、出局速度、回放池大小和损失。
            log.info("步 %d  %.0f 步/秒  ε %.3f  回报 %+.4f  平均名次 %.2f（随机 2.5）  "
                     "自杀占死因 %.0f%%  平局 %.0f%%  回合 %.1f 秒  %.0f 局/分  回放 %.0fk  损失 %.4f",
                     steps, spm, eps, stats["rew"], place, 100.0 * kill, 100.0 * draw_rate,
                     span, rpm, stats["buf"], stats["loss"])
            # 课程学习期间把当前迷宫难度也打出来，方便确认退火真的在推进
            if args.braid_from is not None:
                log.info("本轮迷宫连通度 %.3f（从 %.2f 退火到 %.2f）",
                         float(env.braid), args.braid_from, args.braid_to)
            # 清空本窗口的累加器，开始下一个 10 秒统计窗口
            loss_sum = loss_n = 0.0
            q_sum = td_sum = td_n = 0.0
            rew_sum = rew_n = 0.0
            place_sum = place_n = 0.0
            suicide_sum = death_sum = 0.0
            draw_sum = 0.0
            span_sum = 0.0
            span_n = 0
            t_last = now

        # ---- 定期评测 ----
        # 每 EVAL_EVERY=100 万步用一组固定对手评估一次当前策略。
        # 5 类对手覆盖了从"最弱基线"到"训练起点的自己"的完整谱系：
        #   随机 / 脚本Lv1~Lv3 —— 水平固定、不随训练变化，是横向比较历史曲线的标尺；
        #   最早快照 —— 池子里最老的那份网，代表训练起点；打它的胜率上升，说明策略确实
        #               超越了自己出发时的水平（这是最有信息量的一个指标）。
        # 评测用固定种子 999，保证每次评测的棋盘/出生点一致、结果可比。
        if steps - last_eval >= config.EVAL_EVERY:
            last_eval = steps
            opps = {"随机": RandomBot(7), "脚本Lv1": ScriptedBot(config.EVAL_ENVS, 1, 3),
                    "脚本Lv2": ScriptedBot(config.EVAL_ENVS, 2, 3),
                    "脚本Lv3": ScriptedBot(config.EVAL_ENVS, 3, 3),
                    "最早快照": snapshots[0]}
            res = run_eval(policy, opps, config.EVAL_ENVS, config.EVAL_FRAMES, 999, players)
            stats["eval"] = res
            # 日志里特意标注"不含平局"，避免看曲线时把平局误当成失败、或者忽略掉
            log.info("评测 %s", "  ".join(
                "%s 胜 %.0f%%（不含平局）平 %.0f%% 名次 %.2f 共 %d 局 %.1f 秒" % (
                    k, 100.0 * v["win"], 100.0 * v["draws"] / max(1.0, v["rounds"]),
                    v["place"] or 0.0, v["rounds"], v["span"] * DT)
                for k, v in res.items()))
            # 评测后存一次检查点（带上总步数，供 --resume 恢复进度）
            torch.save({"model": policy.state_dict(), "steps": steps}, args.ckpt)

        # 把当前帧画面和各项统计推给观战窗口（另一个进程的 tank_view.py 通过共享内存读）
        if writer is not None:
            writer.push(env)
            writer.stats(stats["loss"], stats["rew"], eps, stats["spm"], stats["buf"],
                         stats["place"], stats["draw"], stats["span"], stats["rpm"], steps,
                         [(stats["eval"].get(k) or {}).get("win", 0.0) for k in EVAL_NAMES])

    # 到达时长上限后收尾：最后存一次检查点，再关掉监控/观战相关资源。
    torch.save({"model": policy.state_dict(), "steps": steps}, args.ckpt)
    log.info("训练结束，共 %d 步", steps)
    if tmon is not None:
        tmon.close()
    if monitor is not None:
        monitor.stop()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
