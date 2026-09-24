# =============================================================================
# tank_pool.py —— 多进程环境池
#
# 为什么需要它：
#   训练一轮迭代里，真正占时间的是"用 numpy 推进 256 局物理 + 算观测"这一段
#   （约 15 毫秒），而它只用到 1 个 CPU 核；网络前向和梯度更新加起来才 8 毫秒。
#   机器有 24 个逻辑核却只忙 1 个 —— 也就是说瓶颈完全是"单线程算得太慢"，
#   跟 GPU 没关系（这也是 GPU 利用率一直上不去的原因）。
#   这里把环境按局切成 W 份交给 W 个子进程并行推演，主进程只负责网络与训练。
#   实测线程不行：numpy 这些小数组运算是 Python 层调用为主，全局解释器锁(GIL)
#   根本不放行，4 线程反而慢 3 倍；只有真进程才能真正并行。
#
# 数据怎么传（这是整个设计里最要紧的一点）：
#   一帧要交换 1.8 MB 的观测图，如果走队列/管道就要 pickle 一遍再传一遍，
#   省下来的时间会被序列化全吃掉。这里改成：主进程开一块共享内存，
#   里面按固定布局摆好 观测 / 特征 / 控制量 / 逐局标量 / 镜像状态，
#   子进程直接往对应的行区间里写，主进程直接读——全程零拷贝零序列化。
#   帧号只用两组信号量握手（主进程 go → 子进程算完 done），一次约几微秒。
#
# 观测用"双缓冲"（形状里最前面那个 2）：
#   子进程写入第 t 帧的同时，主进程还要拿第 t-1 帧的观测去算经验存进回放池，
#   单缓冲会被覆盖掉。两块轮换之后两边都不需要任何拷贝。
#
# 镜像状态（可选）：
#   监控面板和观战窗口要读整局的物理状态（坦克位置、子弹、墙……）。这里让子进程
#   额外把这些数组抄进共享内存，主进程那边用一个 MirrorEnv 把它包装成"和 VecEnv
#   长得一模一样、只是只读"的对象，于是监控/观战代码一行都不用改。
#
# 等价性：
#   所有计算（物理、观测、奖励、名次）仍然在 tank_vec.py 里、用的是同一份代码，
#   这里只是换了个执行位置。唯一的语义差别是"回合结束的局立刻重开"这一步挪到了
#   子进程里、并且把重开后的观测当作下一帧状态交出去——这些局的经验 done=1，
#   它的 next_obs 在训练里会被 (1-done) 直接乘成 0，所以数值上完全等价。
# =============================================================================
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

import tank_vec
from tank import NC, NR
from tank_vec import B_MAX, C_MAX, L_MAX, M_MAX, Z_MAX

# 物理步长，和 selfplay.py / tank_vec.py 内部约定一致
DT = 1.0 / 60.0

# 逐座位标量在 aux 数组最后一维里的列号
A_VALID, A_DIED, A_SUICIDE, A_PICKED, A_PLACE, A_ALIVE = range(6)
A_SEAT_COLS = 6
# 逐局标量在 aux_env 数组里的列号
E_NEW, E_WINNER, E_FRAMES = range(3)
E_ENV_COLS = 3
# 观测（OBS）/特征（FEAT）里的固定维度
OBS_H = tank_vec.OBS_H
OBS_W = tank_vec.OBS_W


# 镜像要抄的逐局状态：名字、dtype、单局形状。
# dtype 必须和 VecEnv 里同名数组完全一致（特别是那些按位运算的布尔量，
# 抄成 uint8 的话监控那边拿去做 & 运算会直接报错）。
def _mirror_spec(players):
    p = players
    return (
        ("vwall", np.bool_, (NC + 1, NR)),
        ("hwall", np.bool_, (NR + 1, NC)),
        ("tx", np.float32, (p,)), ("ty", np.float32, (p,)), ("ta", np.float32, (p,)),
        ("t_alive", np.bool_, (p,)), ("t_cool", np.float32, (p,)), ("t_wt", np.float32, (p,)),
        ("t_weapon", np.int8, (p,)), ("t_score", np.int32, (p,)),
        ("t_vx", np.float32, (p,)), ("t_vy", np.float32, (p,)),
        ("picked", np.bool_, (p,)), ("just_died", np.bool_, (p,)),
        ("place", np.int8, (p,)), ("suicide", np.bool_, (p,)),
        ("kills", np.int32, (p,)), ("ep_kills", np.int32, (p,)), ("ep_deaths", np.int32, (p,)),
        ("b_on", np.bool_, (B_MAX,)), ("b_x", np.float32, (B_MAX,)),
        ("b_y", np.float32, (B_MAX,)), ("b_vx", np.float32, (B_MAX,)),
        ("b_vy", np.float32, (B_MAX,)), ("b_life", np.float32, (B_MAX,)),
        ("b_bounce", np.int16, (B_MAX,)), ("b_owner", np.int8, (B_MAX,)),
        ("b_kind", np.int8, (B_MAX,)),
        ("m_on", np.bool_, (M_MAX,)), ("m_x", np.float32, (M_MAX,)),
        ("m_y", np.float32, (M_MAX,)), ("m_arm", np.float32, (M_MAX,)),
        ("m_owner", np.int8, (M_MAX,)),
        ("c_on", np.bool_, (C_MAX,)), ("c_x", np.float32, (C_MAX,)),
        ("c_y", np.float32, (C_MAX,)), ("c_ttl", np.float32, (C_MAX,)),
        ("c_kind", np.int8, (C_MAX,)),
        ("z_on", np.bool_, (Z_MAX,)), ("z_x", np.float32, (Z_MAX,)),
        ("z_y", np.float32, (Z_MAX,)), ("z_r", np.float32, (Z_MAX,)),
        ("z_ttl", np.float32, (Z_MAX,)),
        ("l_on", np.bool_, (L_MAX,)), ("l_xy", np.float32, (L_MAX, 4)),
        ("l_ttl", np.float32, (L_MAX,)),
        ("spawn_t", np.float32, ()), ("pause", np.float32, ()),
        ("over", np.bool_, ()), ("winner", np.int8, ()),
        ("maze_seq", np.int32, ()), ("round_frames", np.int32, ()),
        ("rounds", np.int32, ()), ("ep_steps", np.int32, ()),
    )


# 共享内存的完整布局。主进程和子进程都调用这个函数算出同样的形状与偏移，
# 所以不需要把布局本身传过去，只需要传参数。
def _layout(cfg):
    n, p, mirror, w = cfg["n"], cfg["players"], cfg["mirror"], cfg["workers"]
    items = [
        ("obs", np.uint8, (2, n, p, OBS_H, OBS_W)),
        ("feat", np.float32, (2, n, p, cfg["feat_dim"])),
        ("ctrl", np.float32, (n, p, 3)),
        ("aux", np.float32, (n, p, A_SEAT_COLS)),
        ("aux_env", np.float32, (n, E_ENV_COLS)),
        ("braid", np.float32, (w,)),
        ("stop", np.int32, (w,)),
    ]
    # 镜像的名字统一加前缀 m_，避免和上面的控制字段重名
    if mirror:
        items += [("m_" + name, dt, (n,) + shp) for name, dt, shp in _mirror_spec(p)]
    return items


# 把布局摊平成一串带字节偏移的条目。每项按 64 字节对齐起步：
# float32/float64 混排时不对齐会让 numpy 建出来的数组行为异常（而且更慢）。
def _offsets(items):
    out = []
    off = 0
    for name, dt, shape in items:
        off = (off + 63) // 64 * 64
        out.append((name, dt, shape, off))
        off += int(np.prod(shape)) * np.dtype(dt).itemsize
    return out, off


# 按偏移把共享内存切成一组 numpy 数组（都是内存的视图，不占额外内存）
def _views(buf, packed):
    return {name: np.ndarray(shape, dtype=dt, buffer=buf, offset=off)
            for name, dt, shape, off in packed}


# 每个 worker 负责的局区间：把 n 局尽量均分，且每个都是"连续的一段"。
# 连续很关键——写共享内存时就是一次大拷贝，镜像那边也能按行整片切。
def _bounds(n, w):
    return [(n * i // w, n * (i + 1) // w) for i in range(w)]


# 子进程主循环：自己建一个只负责 n_local 局的 VecEnv，然后等着主进程喊"走"。
def _worker_main(cfg, shm_name, rank, go, done):
    import numpy as np

    packed, _ = _offsets(_layout(cfg))
    shm = shared_memory.SharedMemory(name=shm_name)
    v = _views(shm.buf, packed)
    lo, hi = _bounds(cfg["n"], cfg["workers"])[rank]
    env = tank_vec.VecEnv(hi - lo, cfg["players"], seed=cfg["seed"] + rank,
                          braid=cfg["braid"])
    rows = slice(lo, hi)
    mirror = cfg["mirror"]
    use_braid = cfg["braid"] is not None
    names = [name for name, _, _ in _mirror_spec(env.p)] if mirror else []
    parity = 0

    def pub_obs():
        # 观测/特征每个座位一份，写进"局 × 座位"的对应位置
        for j in range(env.p):
            v["obs"][parity, rows, j] = env.obs_pack(j)
            v["feat"][parity, rows, j] = env.feat(j)

    def pub_mirror():
        # 整局物理状态整片抄进共享内存，给监控面板/观战窗口读
        for name in names:
            v["m_" + name][rows] = getattr(env, name)

    # 初始状态（第 0 帧）先写进 0 号缓冲，然后才告诉主进程"就绪"
    pub_obs()
    pub_mirror()
    done.release()
    while True:
        go.acquire()
        if v["stop"][rank]:
            done.release()
            break
        if use_braid:
            # 课程学习：主进程每帧把当前迷宫连通度写在共享内存里，这里读回来。
            # 只影响之后新生成的迷宫，已经在打的这局不受影响。
            env.braid = float(v["braid"][rank])
        # 记下"推进前"的存活、进行状态和回合数：
        #   before/running 用来算 valid（这条经验要不要入池）
        #   回合数差分用来判这一帧哪些局打完了
        before = env.t_alive.copy()
        running = ~env.over
        prev_rounds = env.rounds.copy()
        env.step(v["ctrl"][rows], DT)
        died = env.just_died
        aux = v["aux"][rows]
        aux[:, :, A_VALID] = before & running[:, None]
        aux[:, :, A_DIED] = died
        aux[:, :, A_SUICIDE] = died & env.suicide
        aux[:, :, A_PICKED] = env.picked
        aux[:, :, A_PLACE] = env.place
        aux[:, :, A_ALIVE] = env.t_alive
        new = env.rounds > prev_rounds
        env_aux = v["aux_env"][rows]
        env_aux[:, E_NEW] = new
        env_aux[:, E_WINNER] = env.winner
        env_aux[:, E_FRAMES] = env.round_frames
        parity ^= 1
        pub_obs()
        pub_mirror()
        # 打完的局立刻重开（和单进程版一样的顺序：先留下终局观测，再重开）。
        # 重开换的是地图和出生点，所以这几行的观测要按新局重算；
        # 这几行的经验 done=1，训练时不会再拿它的 next_obs 去 bootstrap，
        # 所以"直接交出重开后的观测"与单进程版数值完全等价。
        if new.any():
            rr = np.nonzero(new)[0]
            env._new_round(rr)
            for j in range(env.p):
                v["obs"][parity, rows, j][rr] = env.obs_pack(j, rr)
                v["feat"][parity, rows, j][rr] = env.feat(j, rr)
            pub_mirror()
        done.release()
    shm.close()


# 只读镜像：把共享内存里的整局状态包装成一个"和 VecEnv 长得一样"的对象，
# 好让监控面板和观战窗口原来的代码（它们到处读 env.tx / env.b_on / env.obs_pack）
# 一行都不用改。故意不走 VecEnv 的构造函数——那会分配一整套内存并把世界重置一遍。
class MirrorEnv(tank_vec.VecEnv):
    def __init__(self, views, n, players):
        self.n = int(n)
        self.p = int(players)
        self.rows = np.arange(self.n)
        self.rng = None
        self.braid = None
        self._base = None
        # wall_bits 不必走共享内存：它是从 vwall/hwall 推出来的，镜像这边按需重算
        self.wall_bits = np.zeros((self.n, OBS_H, OBS_W), dtype=np.uint8)
        for name, _, _ in _mirror_spec(self.p):
            setattr(self, name, views["m_" + name])

    # 镜像的状态每帧都在被覆盖，观测底图缓存（_base）绝不能留着：
    # 每次读都按当前的墙与实体重建，读完立刻失效，下次重新算。
    def _obs_base(self):
        self._refresh_wall_bits(self.rows)
        self._base = None
        out = tank_vec.VecEnv._obs_base(self)
        self._base = None
        return out


# 主进程这一侧的环境池：把 ctrl 写进共享内存、喊 worker 干活、等它们算完，
# 然后直接读共享内存里的观测/特征/标量。整轮迭代里主进程只做网络和训练。
class EnvPool:
    # n_envs 总并行局数 / players 座位数 / feat_dim 特征维度（要和网络输入一致）
    # workers 子进程数；seed 随机种子；braid 迷宫连通度（None = 用生成器默认值）
    # mirror 是否让子进程额外抄一份整局状态给监控/观战
    def __init__(self, n_envs, players, feat_dim, workers=4, seed=0, braid=None,
                 mirror=False):
        self.n = int(n_envs)
        self.p = int(players)
        self.w = max(1, min(int(workers), self.n))
        cfg = {"n": self.n, "players": self.p, "workers": self.w, "mirror": bool(mirror),
               "seed": int(seed), "braid": (None if braid is None else float(braid)),
               "feat_dim": int(feat_dim)}
        self.cfg = cfg
        packed, total = _offsets(_layout(cfg))
        ctx = mp.get_context("spawn")
        self.shm = shared_memory.SharedMemory(create=True, size=total)
        self.v = _views(self.shm.buf, packed)
        self.v["braid"][:] = 0.0 if braid is None else float(braid)
        self.v["stop"][:] = 0
        self.go = [ctx.Semaphore(0) for _ in range(self.w)]
        self.done = [ctx.Semaphore(0) for _ in range(self.w)]
        self.procs = [
            ctx.Process(target=_worker_main,
                        args=(cfg, self.shm.name, r, self.go[r], self.done[r]), daemon=True)
            for r in range(self.w)]
        for x in self.procs:
            x.start()
        for r in range(self.w):
            self.done[r].acquire()   # 等所有 worker 把第 0 帧的观测写完再开跑
        # t 是已经推进过的帧数；k = t % 2 就是"当前状态"落在哪一块双缓冲里
        self.t = 0
        self.mirror = MirrorEnv(self.v, self.n, self.p) if mirror else None

    # 当前帧（还没推进的那一帧）的观测，形状 (局, 座位, OBS_H, OBS_W)
    @property
    def obs(self):
        return self.v["obs"][self.t % 2]

    # 展平成 (局×座位, OBS_H, OBS_W)：索引恰好是 i*座位数 + j，
    # 和 NStep 的时间线编号、以及网络要吃的批形状都对得上，且是零拷贝视图。
    @property
    def obs_flat(self):
        return self.obs.reshape(-1, OBS_H, OBS_W)

    @property
    def feat(self):
        return self.v["feat"][self.t % 2]

    @property
    def feat_flat(self):
        return self.feat.reshape(-1, self.cfg["feat_dim"])

    # 推进一帧之后，上一步的观测就落在另一块缓冲里了（存经验要用它当 obs）
    @property
    def prev_obs_flat(self):
        return self.v["obs"][(self.t + 1) % 2].reshape(-1, OBS_H, OBS_W)

    @property
    def prev_feat_flat(self):
        return self.v["feat"][(self.t + 1) % 2].reshape(-1, self.cfg["feat_dim"])

    # 逐座位标量 (局, 座位, 6)；逐局标量 (局, 3)。列的含义见文件开头的 A_* / E_* 常量。
    @property
    def aux(self):
        return self.v["aux"]

    @property
    def aux_env(self):
        return self.v["aux_env"]

    # 课程学习用：设置迷宫连通度。只影响之后新生成的迷宫。
    def set_braid(self, x):
        self.v["braid"][:] = float(x)

    # 推进一帧：把控制量写进共享内存 → 喊所有 worker → 等全部算完。
    # 由于是"一次性把 ctrl 摆好再集体发令"，每个 worker 只等自己那一份算完，
    # W 个 worker 的重活是完全并行跑的。
    def step(self, ctrl):
        self.v["ctrl"][:] = ctrl
        for r in range(self.w):
            self.go[r].release()
        for r in range(self.w):
            self.done[r].acquire()
        self.t += 1

    def close(self):
        if self.shm is None:
            return
        self.v["stop"][:] = 1
        for r in range(self.w):
            self.go[r].release()
        for r in range(self.w):
            self.done[r].acquire()
        for x in self.procs:
            x.join(timeout=5)
        self.shm.close()
        self.shm = None
