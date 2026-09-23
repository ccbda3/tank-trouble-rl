# =============================================================================
# buffer.py —— 训练数据的两块「缓冲」：优先经验回放池 + n 步回报累加器
#
# 在项目中的位置：
#   selfplay.py 每走一步就把 (观测, 特征, 动作, 奖励, 下一观测, 下一特征, done)
#   交给 NStep → NStep 攒够 nstep 步后吐出一批打包样本 → 交给 Replay 整批存起来
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

    # 批量写入一批样本（配合批量 NStep 用）。写满一圈后自动覆盖最老的样本：
    # 策略一直在变，太老的样本和当前策略的分布已经对不上，留着反而有害。
    # 新样本的优先级一律给当前最大值，保证它至少被抽中一次、能算出真实 TD 误差，
    # 之后再由 update_prio() 决定它值不值得反复抽。
    # 每帧要落库的样本有上千条（1024 个时间线各约一条），逐条写是 7 次带 Python 开销
    # 的数组赋值 = 每帧上万次 Python 调用；改成整批一次搬运后，实际执行的就是几条大
    # memcpy——这一步从 9 毫秒降到 1 毫秒以内。
    # 环形池跨过尾部时会折成两段写；极端情况下（一批比整个池子还大）只保留最新的一批。
    def push_batch(self, obs, feat, act, ret, next_obs, next_feat, done):
        t = obs.shape[0]
        if t == 0:
            return
        if t > self.cap:
            obs, feat, act, ret, next_obs, next_feat, done = (
                x[-self.cap:] for x in (obs, feat, act, ret, next_obs, next_feat, done))
            t = self.cap
        cut = self.cap - self.head
        parts = [(self.head, cut), (0, t - cut)] if t > cut else [(self.head, t)]
        off = 0
        for a, n in parts:
            if n <= 0:
                continue
            sl = slice(a, a + n)
            self.obs[sl] = obs[off:off + n]
            self.feat[sl] = feat[off:off + n]
            self.act[sl] = act[off:off + n]
            self.ret[sl] = ret[off:off + n]
            self.next_obs[sl] = next_obs[off:off + n]
            self.next_feat[sl] = next_feat[off:off + n]
            self.done[sl] = done[off:off + n]
            self.prio[sl] = self.max_prio
            off += n
        self.head = (self.head + t) % self.cap
        self.size = min(self.size + t, self.cap)

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


# n 步回报累加器（整批数组版）。
# 「流」= 一条独立的决策序列，编号 = 环境序号 × 座位数 + 座位号（见 selfplay.py）：
# 同一局里每个座位都是独立智能体，必须各自维护自己的 n 步队列，不能混在一起。
#
# 为什么是整批数组，而不是"每条流一个 deque"：
#   训练一帧要处理 256 局 × 4 座位 = 1024 条流。用 deque 就得写 1024 次 Python 循环，
#   每帧上万个 Python 调用（deque.append、元组打包、生成器），实测独占整轮迭代 25%
#   的时间，而真正干活的时间（算 n 步折扣累加）不到其中的十分之一。
#   改成"所有流共用一组 numpy 数组 + 全局帧号取模定位"之后，一帧只做几次数组运算。
#
# 三条关键设计：
#   1) 环形缓冲按"全局帧号 % nstep"定位。每条流每帧都往同一个相对位置写，于是所有
#      流的写位置相同，写入退化成一次连续切片赋值，不需要对 1024 条流逐个 scatter
#      （scatter 会把省下来的时间又赔回去）。无效帧也照写不误，反正不会被读到。
#   2) len[] 记录每条流"当前攒了几步"，取代 deque 的长度。
#   3) 不变式：某条流无效 ⟹ 它的队列为空。这不是巧合——座位阵亡或回合结束的那一帧
#      本身是有效的（帧开始时还活着）且 done=1，队列会被一次吐空；此后无效的帧没有
#      任何残留可攒。所以无效帧直接清 len 与旧行为完全一致。
#
# 语义与旧的 deque 版本逐条对齐：
#   攒满 nstep 步 → 打包成一条 (首状态, 首动作, Σγ^k·r_k, 末状态, done=0)
#   遇到 done    → 把队列里每一步都当成一段的起点依次吐出，吐完清空，done 一律记 1.0
#   （done 处记 1.0 是为了不去 bootstrap 一个已经结束的局面，避免对终止后的状态高估价值）
class NStep:
    def __init__(self, n_streams, nstep, gamma, obs_shape, feat_dim):
        self.n = n_streams
        self.nstep = nstep
        self.gamma = gamma
        # obs_shape 是打包后的观测图尺寸 (OBS_H, OBS_W)，内存与 Replay 里存的一致
        self.obs = np.zeros((nstep, n_streams) + tuple(obs_shape), dtype=np.uint8)
        self.feat = np.zeros((nstep, n_streams, feat_dim), dtype=np.float32)
        self.act = np.zeros((nstep, n_streams), dtype=np.int64)
        self.rew = np.zeros((nstep, n_streams), dtype=np.float32)
        self.len = np.zeros(n_streams, dtype=np.int32)
        # t 是全局帧号（每调用一次 push +1），环形写位置就是 t % nstep
        self.t = 0
        # 预先算好 γ^0..γ^(nstep-1)，算 n 步折扣累加时直接查表
        self.gpick = (gamma ** np.arange(nstep)).astype(np.float32)

    # 塞入整批一步转移，返回"本次可以落库的样本"。
    #   valid/obs/feat/act/rew/next_obs/next_feat/done 都是 (n_streams, ...) 的整批数组
    #   （无效的流只是不参与吐出，但照常写进环形缓冲）
    # 返回 (obs, feat, act, ret, next_obs, next_feat, done) 或 None（本帧没有样本落库）。
    def push(self, valid, obs, feat, act, rew, next_obs, next_feat, done):
        p = self.t % self.nstep
        self.obs[p] = obs
        self.feat[p] = feat
        self.act[p] = act
        self.rew[p] = rew
        # 有效流攒一步；无效流清 0（见上面第 3) 条不变式）。攒满后不再增长，封顶在 nstep。
        self.len = np.where(valid, np.minimum(self.len + 1, self.nstep), 0)
        self.t += 1
        n = self.len   # 队列实际长度；注意 self.len 后面会被整体替换，n 保留的是替换前的值
        sidx, midx, lidx, dfin = [], [], [], []
        # ---- 情况 A：这一帧遇到 done ----
        # 队列里每一步都是一段样本的起点：样本 m 用第 m 步的状态，回报只累加到末步为止，
        # 全部标 done=1（不同样本的续接长度不同，所以要按 m 逐个算，见下面的累加循环）。
        fin = valid & done
        if fin.any():
            s = np.nonzero(fin)[0]
            l = n[s]
            cnt = np.cumsum(l)
            sidx.append(s[np.repeat(np.arange(s.size), l)])
            # 每条样本在它自己队列里的序号 0,1,2,...（用累积和拼出来，避免 Python 循环）
            midx.append(np.arange(int(cnt[-1])) - np.repeat(cnt - l, l))
            lidx.append(np.repeat(l, l))
            dfin.append(np.ones(int(cnt[-1]), dtype=np.float32))
        # ---- 情况 B：攒满 nstep 步 ----
        # 每帧每流最多吐一条：起点是队列最老的元素，末状态就是本帧的 next_obs。
        full = valid & ~done & (n >= self.nstep)
        if full.any():
            s = np.nonzero(full)[0]
            sidx.append(s)
            midx.append(np.zeros(s.size, dtype=np.int64))
            lidx.append(np.full(s.size, self.nstep, dtype=np.int64))
            dfin.append(np.zeros(s.size, dtype=np.float32))
        if not sidx:
            return None
        s = np.concatenate(sidx)
        m = np.concatenate(midx)
        l = np.concatenate(lidx)
        # pos：这条样本在环形缓冲里的位置。队列最老的元素在当前写位置往前 (l-1) 格，
        # 再往后数 m 格就是第 m 个元素。取模后落在 [0, 2*nstep)，下面统一再取一次模。
        pos = ((p - l + 1) % self.nstep) + m
        ret = np.zeros(s.size, dtype=np.float32)
        for k in range(self.nstep):
            ok = (m + k) < l   # 样本 m 只累加它后面还剩下的步数（done 的长短不一）
            if not ok.any():
                break
            ret += np.where(ok, self.rew[(pos + k) % self.nstep, s] * self.gpick[k], 0.0)
        pos %= self.nstep
        # 遇到 done 的流已经"把队列一次吐空"，长度必须归零；否则它的下一次队列
        # 会把上一段残留下来的长度算进去，吐出的样本起点就错位了。
        # 注意上面算 ret 用的是替换前的 n，所以这一步放在最后做。
        self.len = np.where(fin, 0, self.len)
        return (self.obs[pos, s], self.feat[pos, s], self.act[pos, s], ret,
                next_obs[s], next_feat[s], np.concatenate(dfin))
