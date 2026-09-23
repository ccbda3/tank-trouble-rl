# =============================================================================
# buffer.py —— 训练数据的两块「缓冲」：优先经验回放池 + n 步回报累加器
#
# 在项目中的位置：
#   selfplay.py 每走一步就把 (观测, 特征, 动作, 奖励, 下一观测, 下一特征, done)
#   交给 NStep → NStep 攒够 nstep 步后吐出一段打包样本 → 交给 Replay 存起来
#   → 训练时 Replay.sample() 按优先级抽一批 → 算完 TD 误差再 update_prio() 写回。
#
# 为什么要 n 步回报：这个任务的奖励极其稀疏，一局 30 秒只在结束时给 ±1，
# 中间只有 -0.001 的步进惩罚。单步 TD 要让「终点奖励」传到开局，得靠 bootstrap
# 反复迭代几十上百次；改成 n 步折扣累加，相当于把终点奖励按 γ^k 提前分摊到前
# 面若干步，价值信号传播快得多。
#
# 为什么要优先回放：均匀随机抽样时绝大多数样本的 TD 误差已经接近 0（学明白了），
# 训练算力被浪费；按 TD 误差加权抽样能把算力集中到还没学明白的局面上。代价是
# 改变了样本分布、引入估计偏差，所以必须配重要性权重 w 一起用。
# =============================================================================

from collections import deque

import numpy as np


# 容量固定的环形回放池。
# 没有用 SumTree，而是 cumsum + 二分查找做加权抽样：20 万条规模下每次采样
# 一次 cumsum 只要几毫秒，实现简单、够用，不必引入额外数据结构。
class Replay:
    def __init__(self, cap, obs_shape, feat_dim):
        self.cap = cap
        # 观测存的是按位打包的 uint8 图（8 个 0/1 平面压进 1 个字节）。
        # 这是整个文件里最关键的一个决定：20 万条 × 8 通道 × 35×51 的观测，
        # 展开成 float32 要 11 GB 内存，打包成 uint8 后只要 340 MB 左右。
        self.obs = np.zeros((cap,) + obs_shape, dtype=np.uint8)
        self.feat = np.zeros((cap, feat_dim), dtype=np.float32)
        self.act = np.zeros(cap, dtype=np.int64)
        # ret 存的是 n 步折扣累加回报（见 NStep），不是单步奖励
        self.ret = np.zeros(cap, dtype=np.float32)
        self.next_obs = np.zeros((cap,) + obs_shape, dtype=np.uint8)
        self.next_feat = np.zeros((cap, feat_dim), dtype=np.float32)
        self.done = np.zeros(cap, dtype=np.float32)
        # 优先级 p = (|TD| + eps)^alpha。必须用 float64：要对 20 万个数做 cumsum，
        # float32 的累加误差会让抽样分布出现肉眼可见的偏差。
        self.prio = np.zeros(cap, dtype=np.float64)
        # alpha：优先程度。0 退化成均匀采样，1 是完全按 TD 加权，0.6 是常用折中
        self.alpha = 0.6
        # beta：重要性权重的指数，越大越强地抵消「优先抽难样本」造成的偏差；
        # 固定 0.4 已能明显稳定训练，不值得为线性退火再引入额外状态
        self.beta = 0.4
        # eps：加在 |TD| 上，既让 TD=0 的样本仍有一点点被抽中的机会，
        # 也顺便避免后面幂运算除以零
        self.eps = 1e-3
        # 归一化重要性权重时用的分母，记录池子里出现过的最大优先级
        self.max_prio = 1.0
        # head 是环形写入位置，size 是当前有效样本数（池子没填满时 size < cap）
        self.head = 0
        self.size = 0

    # 写入一条样本。写满一圈后自动覆盖最老的样本：策略一直在变，太老的样本和
    # 当前策略的分布已经对不上，留着反而有害。
    def push(self, obs, feat, act, ret, next_obs, next_feat, done):
        i = self.head
        self.obs[i] = obs
        self.feat[i] = feat
        self.act[i] = act
        self.ret[i] = ret
        self.next_obs[i] = next_obs
        self.next_feat[i] = next_feat
        self.done[i] = done
        # 新样本的优先级直接给当前最大值，保证它至少被抽中一次、能算出真实 TD 误差，
        # 之后再由 update_prio() 决定它值不值得反复抽
        self.prio[i] = self.max_prio
        self.head = (self.head + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    # 按优先级抽 n 条样本。
    # 返回 (观测, 特征, 动作, 回报, 下一观测, 下一特征, done, 重要性权重, 下标)；
    # 最后那个 slot 是给 update_prio() 回写新优先级用的。
    def sample(self, n):
        if self.size < n:
            return None
        # 加权抽样的实现：把每条样本的优先级当成一段长度铺在 [0, 总和) 上，
        # 随机取 n 个点，点落在哪一段就抽哪一条。cumsum + searchsorted 就是
        # 对「取点找段」的向量化写法。
        p = self.prio[: self.size]
        cs = np.cumsum(p)
        if cs[-1] <= 0.0:
            return None
        masses = np.random.uniform(0.0, cs[-1], size=n)
        slot = np.searchsorted(cs, masses)
        # searchsorted 在浮点边界上可能返回 size，这里兜一下底
        slot = np.clip(slot, 0, self.size - 1)
        # 重要性权重 w = (p / max_prio)^(-beta)：
        # 优先抽高 TD 样本会把它们的梯度放大，用接近「1/抽样概率」的幂次把梯度
        # 拉回来；除以池内最大优先级是归一化，防止整体学习率随 max_prio 漂移。
        w = (p[slot] / (self.max_prio + self.eps)) ** (-self.beta)
        w = w / w.max()
        return (self.obs[slot], self.feat[slot], self.act[slot], self.ret[slot],
                self.next_obs[slot], self.next_feat[slot], self.done[slot], w, slot)

    # 用刚算出的新 TD 误差刷新这些样本的优先级：TD 越大说明越没学明白，
    # 下一轮更容易被抽到，形成「难样本优先」的循环。
    def update_prio(self, slot, td_error):
        p = np.abs(np.asarray(td_error, dtype=np.float64)) + self.eps
        self.prio[slot] = p ** self.alpha
        self.max_prio = max(self.max_prio, float(self.prio[slot].max()))


# n 步回报累加器。
# 「流」= 一条独立的决策序列，编号 = 环境序号 × 座位数 + 座位号（见 selfplay.py）：
# 同一局里每个座位都是独立智能体，必须各自维护自己的 n 步队列，不能混在一起。
# 每个流一个长度 nstep 的 deque：
#   攒满 nstep 步 → 打包成一段吐出去；
#   中途遇到 done  → 把队列里剩下的全部打包吐出去，否则这段经验会永远压在队列里。
class NStep:
    def __init__(self, n_envs, nstep, gamma):
        self.n_envs = n_envs
        self.nstep = nstep
        self.gamma = gamma
        self.q = [deque(maxlen=nstep) for _ in range(n_envs)]

    # 塞入一步转移，返回本次可以落库的完整样本列表。
    # 常态下返回 0 条或 1 条（攒满 nstep 时），遇到 done 时一次返回多条。
    def push(self, env_idx, obs, feat, act, rew, next_obs, next_feat, done):
        out = []
        q = self.q[env_idx]
        q.append((obs, feat, act, rew, next_obs, next_feat, done))
        if done:
            # 回合结束：把队列里从最早一步起的每一步都当成一段的起点依次吐出，吐完清空。
            # 因为只有 done 才走这里，剩下每一段的末状态都是同一个终止状态，
            # 所以 done 一律记 1.0，训练时不会再去 bootstrap 一个已经结束的局面，
            # 可以避免对终止后的状态做价值高估。
            while q:
                ret = 0.0
                for k, item in enumerate(q):
                    ret += (self.gamma ** k) * item[3]
                first = q[0]
                last = q[-1]
                out.append((first[0], first[1], first[2], ret, last[4], last[5], 1.0))
                q.popleft()
        elif len(q) == self.nstep:
            # 攒满 nstep 步：整段打包成一条 (首状态, 首动作, n 步折扣回报, 末状态, done=0)。
            # 折扣累加是 Σ γ^k r_k，k 用队列内的相对步数；
            # 末状态留给 selfplay.py 做 bootstrap：目标 = ret + γ^NSTEP · 目标网络Q(末状态)。
            ret = 0.0
            for k, item in enumerate(q):
                ret += (self.gamma ** k) * item[3]
            first = q[0]
            last = q[-1]
            out.append((first[0], first[1], first[2], ret, last[4], last[5], 0.0))
        return out
