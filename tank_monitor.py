# 坦克动荡（Tank Trouble 复刻）4 人混战自对弈训练的「监控面板适配层」。
#
# 位置：本模块不含任何训练逻辑，它跑在训练进程内部——selfplay.py 里用
#   tank_monitor.TankMonitor(env, policy, target, replay, stats, ...) 实例化，
#   再 build() 拿回一个 aimonitor.Monitor 并 start()。
#   因此它和训练共享同一份 policy / target 网络、同一个回放池、同一个统计字典 stats，
#   所有指标都是"就地观测"，训练侧不需要额外导出任何数据。
#
# 它暴露给谁：aimonitor 是本项目自带的通用 HTTP 监控 SDK。这里的工作只是把训练的内部状态
#   翻译成 aimonitor 能渲染的对象：
#     metric    逐点标量曲线（平均名次、损失、ε ...）
#     health    健康度指标卡（梯度/权重范数、训练/采样比 ...）
#     graph     自定义"神经网络图"（NetProbe 提供逐层节点数与层间连接矩阵）
#     surface   3D 损失曲面（LandscapeProbe 在后台线程里算）
#     image     PNG 图片（实时对局、观测通道、卷积特征图）
#     histogram / bars / table   分布、条形图、表格
#     alert / watchdog           告警与卡死检测
#
# 数据怎么流到网页：aimonitor 的 HTTP 服务每隔 interval 秒（这里 0.5s）调用上面注册的
#   回调函数，回调把 numpy / torch 张量算成 python 标量或数组，再序列化成 JSON / PNG 返回浏览器。
#   也就是说这些回调是被网页轮询驱动、而不是每帧都跑的，所以本文件里到处用带时间戳的缓存
#   （_edge_t / _gs / _gn / _gpu）来避免同一份重计算被反复触发。
#
# 阅读顺序建议：先看顶部三个矩阵辅助函数（它们决定神经网络图上连线的形状与数值），
#   再看 NetProbe（前向激活 + 特征图），然后 LandscapeProbe（损失曲面），
#   最后 TankMonitor（把所有东西注册进面板并管理生命周期）。
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

# 层间连接矩阵的缓存时长：连接矩阵只由权重决定，而权重只在梯度更新后变化，
# 5 秒重算一次足够跟上训练，又不会每个 HTTP 刷新周期都给神经网络图做一次全量权重搬运。
EDGE_TTL = 5.0
# 直方图相关的历史缓存时长。当前代码里权重/梯度的直方图缓存由 aimonitor 自己按 refresh 管理，
# 这个常量保留在此，文件内没有直接引用。
HIST_TTL = 5.0
# 梯度快照的抓取间隔（单位：梯度更新次数，不是秒）。
# 梯度是"用完就丢"的：opt.step() 之后到下一次 zero_grad 之前才短暂存在，外部没法事后回溯，
# 只能在训练循环里主动抓。但每抓一次都要把全部参数梯度拷一份到 numpy（几十 MB 量级），
# 所以不能每次更新都抓；200 次是经验折中——太少会把拷贝开销摊进训练热路径，
# 太多又会在梯度范数曲线上漏掉爆炸/消失的尖峰。第一次更新也抓一份，保证面板早期就有数据。
GRAD_EVERY = 200
# GPU 利用率的采样间隔（秒）。取利用率要起一个 nvidia-smi 子进程，代价是几十毫秒级；
# 而网页 0.5 秒就刷新一次，如果每次都调会明显拖慢面板线程。10 秒的粒度对
# "GPU 是打满还是闲置"这个判断已经完全够用。
GPU_EVERY = 10.0

# 面板上画权重/梯度直方图的 6 个主要张量：3 个卷积核 + 全连接层权重 + Dueling 的 V / A 两个线性层。
# 选它们的理由是这几层参数量最大、也最能反映"网络到底在不在学"；
# 如果把每一层都列出来，面板会被几十张直方图淹没，反而看不出重点。
WEIGHT_KEYS = ("conv.0.weight", "conv.2.weight", "conv.4.weight",
               "fc.0.weight", "value.weight", "adv.weight")


# 取出网络里所有二维卷积层（net.conv 是 Sequential，里面还夹着 ReLU 和 Flatten，需要过滤）
def _convs(net):
    return [m for m in net.conv if isinstance(m, torch.nn.Conv2d)]


# 统计一个模块的参数量，用于神经网络图上每层的"参数"标注
def _nparam(mod):
    return sum(p.numel() for p in mod.parameters())


# 把卷积核 (out, in, kH, kW) 压成 (out, in) 的连接强度矩阵：沿核的空间维度逐步取均值。
# 图上要画的是"第 i 个输出通道和第 j 个输入通道之间有多强"，而核本身还有 k×k 个权重，
# 取均值是最直观的降维方式。再取绝对值只保留强度（连线方向由节点值体现，符号在这里不可视化）。
def _chan_matrix(weight):
    w = torch.abs(weight)
    while w.dim() > 2:
        w = w.mean(-1)
    return np.round(w.detach().cpu().numpy().astype(np.float32), 6)


# 把全连接层权重中"属于卷积特征的那部分"压成 (隐藏层, 卷积通道) 的连接矩阵。
# fc.0.weight 的形状是 (HIDDEN, flat + nfeat)，后半段接的是标量特征，不属于视觉通路，先切掉；
# 再按 (隐藏神经元, 卷积通道, 空间位置) 重塑，沿空间维度取均值 —— 于是每个隐藏神经元到
# C3 每个通道都有一个平均连接强度，正好对上图上 C3 → FC 这组连线。
def _flat_matrix(weight, n_conv, n_spatial):
    w = torch.abs(weight)[:, :n_conv * n_spatial].detach().cpu().numpy().astype(np.float32)
    return np.round(w.reshape(w.shape[0], n_conv, n_spatial).mean(2), 6)


# 把 Dueling 的 Q 输出精确线性化成 (动作数, 隐藏层) 的连接矩阵。
# 网络里 q = V + A - mean_a(A)，而 V、A 都是 h 的线性层，所以：
#   q_a(h) = value.weight[0]·h + value.bias[0] + adv.weight[a]·h + adv.bias[a]
#            - mean_b(adv.weight[b]·h + adv.bias[b])
#          = (adv.weight[a] - mean_b(adv.weight[b]) + value.weight[0]) · h + 常数
# 即第 a 行对 h 的线性系数就是 adv.weight[a] - mean_b(adv.weight[b]) + value.weight[0]，
# 代码里正是 adv - adv.mean(0) 再加上 value.weight（形状 (1,H)，自动广播到 (A,H)）。
# 取绝对值只是为了画连线强度。
def _out_matrix(net):
    adv = net.adv.weight.detach()
    m = adv - adv.mean(0, keepdim=True) + net.value.weight.detach()
    return np.round(torch.abs(m).cpu().numpy().astype(np.float32), 6)


# 网络探针：把 DQN 的中间激活抽出来，供前端画"神经网络图"和"特征图"。
# 它做了两件事：
#   1) 用一次前向记录每层的节点值（观测通道均值、各卷积层逐通道均值、隐藏层激活、Q 值），
#      再配上离线算好的层间连接矩阵，让面板能把真实的网络结构和实时数据叠在一起画；
#   2) 把卷积层的特征图逐通道归一化后拼成图片，用于"肉眼看网络学到了什么"。
class NetProbe:
    def __init__(self, env, net):
        self.env = env
        self.net = net
        # 最近一次前向的中间结果，feature_image 直接复用，避免每个图片回调都重跑一次网络
        self.maps = None
        self._edge_t = 0.0
        self._edge_v = None
        cv = _convs(net)
        # 用一个全零输入跑一遍网络，目的只是"探明形状"：每层卷积输出多少通道、多大空间尺寸，
        # 以及展平后的视觉特征维度。这些数字后面要写进面板的层描述里。
        with torch.no_grad():
            obs = network.obs_tensor(np.zeros((1,) + (tank_vec.OBS_H, tank_vec.OBS_W), np.uint8))
            feat = network.feat_tensor(np.zeros((1, env.feat(0).shape[1]), np.float32))
            feats, y = [], obs
            for m in net.conv:
                y = m(y)
                if isinstance(m, torch.nn.Conv2d):
                    feats.append(y)
        # 三层的空间边长与通道数（形状探明后固定不变，图上不必再重算）
        self.shapes = [int(f.shape[-1]) for f in feats]
        self.chans = [int(f.shape[1]) for f in feats]
        # C3 展平后的视觉特征维数，以及标量特征维数（自身状态/对手状态/最近子弹）
        self.flat = int(np.prod(feats[-1].shape[1:]))
        self.nfeat = int(feat.shape[1])
        self.hidden = int(net.value.in_features)
        s1, s2, s3 = self.shapes
        k1, k2, k3 = self.chans
        # meta 是神经网络图的"逐层元信息"：层名、参数量、真实节点数、一句话说明。
        # 前端按这个顺序把 6 层节点排成 6 列，节点值则由 graph() 每次刷新时填入。
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

    # 用当前局面跑一次前向，把每层中间结果存进 self.maps。
    # 走的是 1 号位（下标 0）的观测，batch 维度为 1；全程 no_grad，探针不能影响训练。
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
            # 与 network.DQN.forward 完全一致，这里只是额外把中间量留下来
            q = v + a - a.mean(dim=1, keepdim=True)
        self.maps = (obs[0], feats, h[0], v[0], a[0], q[0])
        return self.maps

    # aimonitor 的 graph 回调：返回 {layers, edges}，前端据此画神经网络图。
    # layers 里每层带节点数值（用于节点着色/大小），edges 里是 5 个连接矩阵（层与层之间）。
    def graph(self):
        obs, feats, h, v, a, q = self.forward()
        # 观测是按位打包的整数，解包后每个通道都是 0/1，取均值即"该通道点亮的格子占比"
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
            # nodeCount 必须等于 nodes 的长度：前端用它决定这层画多少个圆点
            lay["nodeCount"] = len(nodes[i])
            lay["nodes"] = nodes[i]
            layers.append(lay)
        return {"layers": layers, "edges": self._edges()}

    # 5 个连接矩阵对应 6 层之间的 5 条边：观测→C1、C1→C2、C2→C3、C3→FC、FC→Q。
    # 但权重只有在梯度更新后才会变，所以按 EDGE_TTL 缓存，避免每次刷新都做一次权重搬运。
    def _edges(self):
        now = time.time()
        if self._edge_v is not None and now - self._edge_t < EDGE_TTL:
            return self._edge_v
        cv = _convs(self.net)
        self._edge_v = [
            _chan_matrix(cv[0].weight),
            _chan_matrix(cv[1].weight),
            _chan_matrix(cv[2].weight),
            # n_spatial 由 C3 展平维度除以 C3 通道数得到（即每个通道的空间位置数）
            _flat_matrix(self.net.fc[0].weight, self.chans[2], self.flat // self.chans[2]),
            _out_matrix(self.net),
        ]
        self._edge_t = now
        return self._edge_v

    # 生成一张特征图：which 为 "fc" / "q" 时把 512 维隐藏激活或 18 个 Q 值拉成一列显示；
    # 为 0/1/2 时取对应卷积层的特征图。
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
            # 逐通道 min-max 归一化到 0~255：不同卷积核的输出数值范围差别很大，
            # 统一归一化后才可能在同一张拼图里看清各自的空间结构（否则数值大的通道会把别人压成一片黑）。
            # 通道内取值几乎恒定时（hi-lo≈0）直接给全 0，避免除零。
            maps.append(np.zeros_like(a) if hi - lo < 1e-9 else (a - lo) / (hi - lo) * 255.0)
        return imaging.montage(maps, cols=8, pad=1, pad_val=0.0)


# 损失曲面探针：沿两个正交随机方向在参数空间里扫一个 res×res 的网格，每个网格点跑一次前向算损失，
# 把结果画成 3D 曲面，并把训练轨迹（参数相对初始点的位移在这两个方向上的投影）叠在曲面上。
# 用途：直观判断优化器是不是走在一个又陡又窄的"峡谷"里、损失面平不平、有没有跑出有效区域。
# 它必须另跑一个后台线程，因为 res×res 次前向 + 一次性的目标值计算对训练循环来说太贵了。
class LandscapeProbe:
    # size=11 网格边长（11×11=121 个点）、batch=16 冻结样本数、interval=60s 测量间隔、
    # extent=0.5 初始扫描半径（会自适应调整）。这些默认值直接决定面板上曲面的分辨率与刷新节奏。
    def __init__(self, net, target, replay, nfeat_dim, size=11, batch=16, interval=60.0, extent=0.5):
        self.net = net
        self.target = target
        self.replay = replay
        self.size = size
        self.batch = batch
        self.interval = interval
        self.extent = extent
        # 单独造一个同构网络当"探针"，参数从 policy 拷过来。
        # 因为扫曲面时必须反复把参数搬到 θ0 + a·u + b·v 上做前向，如果直接在训练用的网络上改，
        # 就会污染正在训练的模型，所以只能用一个可随意摆布的副本。
        self.probe = network.make(nfeat_dim)
        self.du = None
        self.dv = None
        self.first = None
        self.fixed = None
        self.path = []
        self.count = 0
        self.out = {"ready": 0}
        # out / path / count 会被后台测量线程写、被 HTTP 线程读，必须加锁
        self.lock = threading.Lock()

    # aimonitor 的 surface 回调：给出一份当前结果的浅拷贝，避免前端读的时候后台正好在改。
    # path 需要单独拷贝一份列表，否则锁外遍历时后台线程可能正好在 append/del。
    def snapshot(self):
        with self.lock:
            d = dict(self.out)
            d["path"] = list(self.path)
            d["probeCount"] = self.count
            return d

    # 构造两个正交的随机方向 u、v（每个参数张量各有一个，形状与该参数一致）。
    # 只在第一次调用时生成；用固定种子保证方向可复现——否则每次重启进程曲面都不一样，没法比较。
    # 做法：对每个参数先取一个归一化高斯作为 u，再取一个高斯减去它在 u 上的投影做 Gram-Schmidt
    # 正交化，最后归一化得到 v（这样 a、b 两个坐标轴互相垂直，投影才有"两个独立方向"的意义）。
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
            # 单元素参数（例如 value.bias 形状是 (1,)）只有一个自由度：a 就是 ±1，b 必然与它平行，
            # 正交化后 b 恒为 0，再用 0 去归一化就会除零得到 NaN，后面整个曲面都会变成 NaN。
            # 检测到这种退化方向就直接置零（该参数只用 a 方向扫描），这是真实修过的 bug。
            if float(nb) < 1e-6:
                b = torch.zeros_like(a)
            else:
                b = b / nb
            self.du[n] = a.to(p.device)
            self.dv[n] = b.to(p.device)

    # 冻结一批样本作为曲面的"横轴底数"：从回放池抽一个 batch，用 Double DQN 的方式算好目标值并固定住。
    # 之所以要冻结：曲面高度必须只随参数变化。如果每个网格点都用新抽的样本，损失差异里会混进
    # 数据噪声，扫出来的曲面就没法比较（看起来凹凸不平，其实是样本不同造成的）。
    # 回放池还没攒够一个 batch 时直接返回，等下一轮轮询再试。
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
            # Double DQN：动作由在线网选（argmax），价值由目标网给，done 处不加 bootstrap
            best = self.net(nobs, nfeat).argmax(dim=1, keepdim=True)
            tgt = ret + (config.GAMMA ** config.NSTEP) * (1.0 - done) * \
                self.target(nobs, nfeat).gather(1, best)
        self.fixed = (obs, feat, a, tgt)

    # 曲面高度 z 的定义：探针网络在冻结样本上的 Huber(smooth_l1) 损失。
    # 只取被采样动作对应的那个 Q 值（gather(1, a)），和训练时的 TD 目标口径一致。
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
        # 记下"初始参数"（第一次调用时的网络参数），作为训练轨迹投影的原点
        if self.first is None:
            self.first = {n: p.detach().clone() for n, p in named}
        # 把当前训练参数拷到探针上，之后所有网格点都是从这个基准点出发做偏移
        self.probe.load_state_dict(self.net.state_dict())
        params = list(self.probe.named_parameters())
        alpha = 0.0
        beta = 0.0
        base = {}
        for n, p in params:
            base[n] = p.detach().clone()
        with torch.no_grad():
            # 把"训练参数相对初始点的位移"投影到 u、v 两个方向上，得到轨迹在曲面平面上的坐标 (α, β)。
            # 注意这是投影而不是精确坐标：曲面上的原点(0,0)代表初始参数，训练越久轨迹点离原点越远。
            for n, p in named:
                dd = p.detach() - self.first[n]
                alpha += float((dd * self.du[n]).sum())
                beta += float((dd * self.dv[n]).sum())

            # 把探针参数摆到 base + a·u + b·v，返回该点的损失。用 copy_ + add_ 原地改是为了避免
            # 每个网格点都新建一遍参数（121 个点 × 全参数拷贝的开销很可观）。
            def at(au, bv):
                for n, p in params:
                    p.copy_(base[n]).add_(self.du[n], alpha=au).add_(self.dv[n], alpha=bv)
                return self._loss()

            cur = at(0.0, 0.0)
            # extent 自适应：先在 ±e 的四个角上探一下损失，看边角比中心高多少（相对涨幅 r）。
            # 目标是让角落涨幅落在 3 倍左右：r 太小（曲面太平，像块平板）就放大扫描范围，
            # r 太大（太陡，曲面会飞出画面）就缩小。因为损失随位移近似二次增长，
            # 需要的缩放因子约为 sqrt(3/r)；再把单次调整限制在 0.02~5 倍、最终 extent 夹在 1e-5~1
            # 之间，防止一次跳太狠。ref 用 max(cur, 1e-6) 是为了 cur≈0 时不做除零。
            e = self.extent
            zmax = max(at(e, e), at(e, -e), at(-e, e), at(-e, -e))
            ref = cur if cur > 1e-6 else 1e-6
            r = (zmax - cur) / ref
            if r > 1e-9:
                self.extent = max(1e-5, min(1.0, e * max(0.02, min(5.0, (3.0 / r) ** 0.5))))
            ext = self.extent
            res = self.size
            # 在 [-ext, ext] × [-ext, ext] 上均匀取 res×res 个点，逐点算损失。
            # 行优先展开成一维数组，前端按 res 还原成二维网格。
            vals = []
            for i in range(res):
                au = -ext + 2 * ext * i / (res - 1)
                for j in range(res):
                    bv = -ext + 2 * ext * j / (res - 1)
                    vals.append(at(au, bv))
            # 扫完把探针参数还原到基准点，让 probe 保持在"当前训练参数"状态（下次 load_state_dict 前
            # 也不会残留某个网格角落上的参数）
            for n, p in params:
                p.copy_(base[n])
        # 轨迹点记录成 (α, β, 该点的损失)：其中 (α,β) 由上面精确投影算出，z 用 cur。
        # 注意 cur 是 at(0,0) 的损失，不是 at(α,β) —— 轨迹叠在曲面上时会有细微偏差，属预期。
        # 只保留最近 400 个点，否则轨迹无限增长既占内存又让网页画满。
        self.path.append({"a": alpha, "b": beta, "z": cur})
        if len(self.path) > 400:
            del self.path[0]
        self.count += 1
        with self.lock:
            # ready=1 告诉前端已经有数据了；center 是训练参数当前在曲面平面上的位置，
            # zmin/zmax 是网格上的损失范围（前端用来定色标）。
            self.out = {"ready": 1, "res": res, "ext": ext, "zmin": min(vals),
                        "zmax": max(vals), "values": vals, "center": [alpha, beta], "cur": cur}

    # 后台守护线程主体：按 interval 周期测量。异常只记 warning 不向外抛——
    # 监控模块绝不能把训练搞崩；单次测量失败（例如回放池还没数据、显存瞬时不足）跳过等下一轮即可。
    # 用 stop.wait 代替 sleep：收到关闭信号时能立刻退出，不用干等满一个 interval。
    def loop(self, stop):
        while not stop.is_set():
            try:
                self.measure()
            except Exception as e:
                log.warning("损失曲面测量失败: %s", e)
            if stop.wait(self.interval):
                break


# 查询 GPU 利用率（百分比）。取不到就返回 None，面板显示为空，不影响其它指标。
def _gpu_util():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
            # CREATE_NO_WINDOW 是必需的：训练通常用 pythonw 启动（本身没有控制台），
            # 这种情况下创建子进程时 Windows 会给它新开一个控制台窗口并抢走焦点，
            # 表现为"每隔几秒弹一次黑框"。加上这个标志让子进程无窗口运行。
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


# 监控总装配：把各个探针的结果注册成 aimonitor 的指标/图表/图片/表格，并管理生命周期。
# 由 selfplay.py 在训练进程里实例化，build() 返回的 Monitor 由训练侧 start()。
class TankMonitor:
    # env / policy / target / replay / stats / steps_fn 都是训练侧的活对象（不是快照）：
    # stats 由训练循环每 10 秒更新一次，steps_fn 是读当前已训练步数的闭包，
    # 所以面板回调每次读到的都是最新值。
    def __init__(self, env, policy, target, replay, stats, steps_fn, args):
        self.env = env
        self.policy = policy
        self.target = target
        self.replay = replay
        self.stats = stats
        self.steps_fn = steps_fn
        self.args = args
        self.probe = NetProbe(env, policy)
        # 损失曲面探针复用 NetProbe 探到的标量特征维数，保证探针网络与 policy 结构一致
        self.land = LandscapeProbe(policy, target, replay, self.probe.nfeat)
        self.updates = 0
        # 动作使用率累计：act_count 记每个动作被选中的次数，act_step 记总决策次数（分母）。
        # 用累加而不是每帧重算，是为了让回调的开销与训练时长无关。
        self.act_count = np.zeros(network.ACTIONS, dtype=np.float64)
        self.act_step = np.zeros(network.ACTIONS, dtype=np.float64)
        self.grad = {}
        self.grad_at = 0.0
        # 梯度快照由训练线程写、HTTP 线程读，必须加锁
        self._lock = threading.Lock()
        # 三个带时间戳的缓存：整体梯度/权重统计、逐层梯度范数、GPU 利用率
        self._gs = {"t": 0.0, "v": (0.0, 0.0, 0.0)}
        self._gn = {"t": 0.0, "v": []}
        self._gpu = {"t": 0.0, "v": None}
        self.mon = None
        self.stop = threading.Event()

    # 训练侧每做完一次梯度更新调用一次。这里只做计数和"是否该抓梯度"的判断，
    # 真正的抓取代价被 GRAD_EVERY 摊薄了。
    def tick_update(self, net):
        self.updates += 1
        if self.updates == 1 or self.updates % GRAD_EVERY == 0:
            self.grab_grads(net)

    # 把当前全部参数梯度拷成 numpy 存下来。
    # 只收 p.grad 不为 None 的参数（某一层没参与这次反传时就没有梯度）。
    # 取绝对值：直方图关心的是梯度的量级分布（是否整体塌到 0、有没有极端长尾），符号在这里没有意义，
    # 取绝对值可以让 48 个 bin 全部用来刻画幅度分布，而不必给负半轴分掉一半。
    # 必须 .copy()：numpy 数组和 torch 张量是共享内存的，下一次反传的梯度会原地累加，
    # 不拷贝的话面板上的"历史快照"会被悄悄改掉。
    def grab_grads(self, net):
        out = {}
        for n, p in net.named_parameters():
            if p.grad is not None:
                out[n] = p.grad.detach().abs().cpu().numpy().copy()
        with self._lock:
            self.grad = out

    # 统计本帧各动作被选中的次数。bincount 的 minlength 保证即使某个动作一次没被选中也有桶，
    # [:ACTIONS] 是防御性截断（bincount 会按最大值自动扩容）。
    # act_step 累加的是 env.n × env.p（局数 × 玩家数），即这一帧一共产生了多少次决策——
    # 因为 count_acts 收到的 acts 形状是 (局, 玩家)，一帧里每个玩家都做了一次决策。
    def count_acts(self, acts):
        self.act_count += np.bincount(np.asarray(acts).ravel(),
                                      minlength=network.ACTIONS)[:network.ACTIONS]
        self.act_step += self.env.n * self.env.p

    # 取当前全部权重的绝对值（理由同梯度：直方图只看幅度分布）。每次都重新取，
    # 由 aimonitor 的 refresh 机制控制频率。
    def _weights(self):
        with torch.no_grad():
            return {n: p.detach().abs().cpu().numpy() for n, p in self.policy.named_parameters()}

    # 全局统计：(梯度 L2 范数, 权重 L2 范数, 梯度最大绝对值)。
    # 1 秒缓存：网页 0.5 秒刷一次，全参数平方和虽然不贵但没必要每次刷新都做。
    # 权重范数单独算出来是为了和梯度范数对照：能区分"梯度大是因为参数本身已经很大"还是真的在爆炸。
    # g_max 反映最坏的那一个元素，比 L2 范数更早暴露个别层炸掉的情况。
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
        # 平方和开根号得到 L2 范数
        self._gs = {"t": now, "v": (g_sq ** 0.5, w_sq ** 0.5, g_max)}
        return self._gs["v"]

    # 逐层梯度 L2 范数，用于条形图定位"是哪一层在爆炸/不动"。
    # 2 秒缓存（比全局统计慢一点）：层数多、要遍历所有梯度数组，调用频率可以更低。
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

    # 逐层权重与梯度统计表：每行是 [层类型前缀, 参数名, 形状, 参数量, 权重均值, 权重标准差, 梯度范数]。
    # 权重均值/标准差可以看出某层是否整体塌缩（标准差趋 0）；
    # 单元素参数（如 value.bias）求标准差没有意义，直接给 0；
    # 本轮没有梯度的参数（grad 里缺失）梯度范数给 0，表示"这次反传没轮到它"而不是"梯度为 0"。
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

    # 直方图数据源。key 带 "@grad" 后缀就取梯度快照（key[:-5] 去掉后缀），否则取当前权重。
    # 返回 None 时 aimonitor 会沿用上一次的直方图，所以回放池没数据或梯度还没抓到时面板不会闪断。
    def _hist_values(self, key):
        if key.endswith("@grad"):
            with self._lock:
                g = self.grad.get(key[:-5])
            return None if g is None else g.ravel()
        w = self._weights().get(key)
        return None if w is None else w.ravel()

    # GPU 利用率的方法级缓存（真正的采样间隔见 GPU_EVERY 的说明）
    def _gpu_util(self):
        now = time.time()
        if now - self._gpu["t"] < GPU_EVERY:
            return self._gpu["v"]
        self._gpu = {"t": now, "v": _gpu_util()}
        return self._gpu["v"]

    # 动作使用率条形图数据。这里返回的是累计次数（act_count），
    # 次数之间的相对大小就等价于各动作被选中的频率，条形图横向比较即可。
    # 还一次决策都没有时返回空列表，让面板留空。
    def _acts(self):
        total = float(self.act_count.sum())
        if total <= 0:
            return []
        return [[tank.ACTION_NAMES[i], float(v)] for i, v in enumerate(self.act_count)]

    # 从优先回放池里还原 |TD 误差| 的分布，画成直方图。
    # 回放池存的 prio 是 (|td| + eps)^α，所以反解 |td| = prio^(1/α) - eps。
    # p > 0 过滤掉那些还没被 update_prio 更新过的槽位（它们的优先级没有意义）。
    # 这个分布越靠左，说明价值估计整体已经比较准；长尾说明还有一批样本误差很大。
    def _buffer_td(self):
        b = self.replay
        if b.size <= 0:
            return None
        p = b.prio[:b.size]
        p = p[p > 0]
        if p.size == 0:
            return None
        return np.power(p, 1.0 / b.alpha) - b.eps

    # 告警：梯度范数异常。阈值参考训练里的梯度裁剪上限 10.0——正常情况下范数应该在 10 以下，
    # 超过 200 基本就是爆炸了；反过来小于 1e-7 说明梯度消失到几乎为 0。
    # 严格大于 0 才判消失：范数正好为 0 更可能是"这一轮没抓到梯度"而不是真的消失。
    def _alert_grad(self):
        g = self._grad_stats()[0]
        if g > 200.0:
            return "梯度范数 %.1f，疑似梯度爆炸" % g
        if 0.0 < g < 1e-7:
            return "梯度范数 %.2e，疑似梯度消失" % g
        return None

    # 告警：平均 Q 值量级发散。
    # 前 5000 次更新不报警：训练初期 Q 值本来就在乱跑，报了也是噪声。
    # 阈值 200 是经验量级判断——这个任务的回报只有 ±1 加少量步惩罚，Q 值长期处在百量级不正常。
    def _alert_q(self):
        q = self.stats.get("q_mean")
        if q is None or self.updates < 5000:
            return None
        if abs(q) > 200.0:
            return "平均 Q 值 %.1f 量级异常，价值函数可能发散" % q
        return None

    # 告警：策略退化。只拿"随机"对手当基线（脚本 Lv1~Lv3 难度分层，随机才是固定参照）。
    # 需要足够多的评测局数（≥400）才有统计意义，且训练超过 2000 万步（早期表现差不代表退化）。
    # 4 人混战里随机水平是平均名次 2.5，若明显差于随机（>3.2）就提示策略可能已崩。
    def _alert_eval(self):
        r = (self.stats.get("eval") or {}).get("随机")
        if r and r["rounds"] >= 400 and r["place"] > 3.2 and self.steps_fn() > 20000000:
            return "对随机平均名次 %.2f（随机水平 2.5），策略可能已退化" % r["place"]
        return None

    # 面板"参数"页的内容：网络结构、DQN 超参数、运行配置三张表。
    # 除了运行时才知道的值（当前更新次数、实际回放池 α 等），其余全部直接读 config / tank_vec 常量，
    # 这样面板展示的永远是真实生效的配置，不会出现"文档和代码不一致"。
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
        # interval=0.5：网页刷新节奏；history=1200：每 0.5 秒一个点，曲线窗口约 10 分钟
        mon = Monitor("坦克动荡 · 自对弈训练监控", port=config.MONITOR_PORT,
                      interval=0.5, history=1200)
        k1, k2, k3 = self.probe.chans
        s1, s2, s3 = self.probe.shapes
        # 顶部常驻信息条（设备 / 迷宫规模 / 并行局数 / 参数量）
        mon.info(lambda: {
            "设备": str(network.DEVICE).upper(),
            "迷宫": "%d×%d" % (tank_vec.NC, tank_vec.NR),
            "并行": "%d 局" % self.args.envs,
            "参数": "%.2fM" % (sum(p.numel() for p in self.policy.parameters()) / 1e6),
        })
        # 下面两个 lambda 是"回调工厂"：ev(nm) 返回一个每次调用都去 stats["eval"][nm] 里
        # 读最新胜率的闭包。stats 由训练循环定期更新，所以指标曲线天然是活的。
        ev = lambda k: (lambda: (self.stats.get("eval") or {}).get(k, {}).get("win"))
        ep = lambda k: (lambda: (self.stats.get("eval") or {}).get(k, {}).get("place"))
        # 各对手评测胜率曲线
        for nm in ("随机", "脚本Lv1", "脚本Lv2", "脚本Lv3", "最早快照"):
            mon.metric("eval_" + nm, ev(nm), label="对" + nm + "胜率", fmt="pct")
        # 4 人混战里"平均名次"是最核心的训练信号：随机水平是 2.5（4 人平均名次的理论期望），
        # 越小越好（第 1 名记 1，因此完美策略趋近 1）。label 里特意标出 2.5 作为参照基线。
        mon.metric("place", lambda: self.stats.get("place"), label="平均名次（随机 2.5）", fmt="f3",
                   color="#3ddc84")
        # 自杀占死因比：自己把自己打死（炸弹/激光/地雷）的比例。这个数字高说明策略在乱开火，
        # 是个比胜率更早暴露"行为退化"的信号。
        mon.metric("suicide", lambda: self.stats.get("suicide"), label="自杀占死因比", fmt="pct",
                   color="#e08f7f")
        # 平局率：回合到时还有多人存活。偏高说明策略学会了苟而不打（回避战斗），
        # 需要和"平均回合秒数"一起看。
        mon.metric("draw", lambda: self.stats.get("draw"), label="平局率", fmt="pct")
        mon.metric("span", lambda: self.stats.get("span"), label="平均回合秒数", fmt="f2")
        # 局/分：训练吞吐量。掉下来通常意味着环境变慢（例如局面变复杂、子弹变多），不是策略问题。
        mon.metric("rpm", lambda: self.stats.get("rpm"), label="局/分", fmt="int")
        mon.metric("rew", lambda: self.stats.get("rew"), label="平均回报", fmt="f4")
        # 训练损失（训练侧用的是带重要性权重的平方 TD 误差，记录口径见 stats["loss"]）
        mon.metric("loss", lambda: self.stats.get("loss"), label="Loss(Huber)", fmt="f4", color="#ff8f5a")
        # 平均 Q 值：Q 的绝对量级。缓慢上升是正常的（价值估计在变大），短期内暴涨/暴跌才是异常。
        mon.metric("q_mean", lambda: self.stats.get("q_mean"), label="平均 Q 值", fmt="f4", color="#c39a5c")
        # 平均 |TD| 误差：价值估计的当前误差幅度，是判断"还在不在学"最直接的标量。
        mon.metric("td_mean", lambda: self.stats.get("td_mean"), label="平均 |TD| 误差", fmt="f4", color="#ff9f7a")
        # 探索率 ε：线性衰减，和"平均名次"一起看才能解释行为变化（ε 降低通常伴随行为变果断）
        mon.metric("eps", lambda: self.stats.get("eps"), label="探索率 ε", fmt="f3", color="#8fb9ff")
        # 每秒步数：训练速度，用于判断有没有被监控/探针拖慢
        mon.metric("spm", lambda: self.stats.get("spm"), label="每秒步数", fmt="int")
        # 累计梯度更新次数 + 经验池规模（千条）：看数据量和训练量是否同步增长
        mon.metric("updates", lambda: self.updates, label="梯度更新", fmt="int")
        mon.metric("buf", lambda: self.stats.get("buf"), label="经验池(千)", fmt="f1")
        mon.metric("gpu", self._gpu_util, label="GPU 利用率", fmt="int", color="#cf7fff")

        # 健康度指标卡：梯度/权重范数与最大值（判断爆炸/消失）、学习率与 γ（确认超参生效）、
        # 参数量、训练/采样比、步数、对局数、各对手平均名次
        mon.health("grad_norm", lambda: self._grad_stats()[0], label="梯度范数", fmt="sci")
        mon.health("weight_norm", lambda: self._grad_stats()[1], label="权重范数", fmt="f2")
        mon.health("grad_max", lambda: self._grad_stats()[2], label="梯度最大绝对值", fmt="sci")
        mon.health("lr", lambda: config.LR, label="学习率", fmt="sci")
        mon.health("gamma", lambda: config.GAMMA, label="折扣因子 γ", fmt="f3")
        mon.health("params", lambda: sum(p.numel() for p in self.policy.parameters()),
                   label="可训练参数", fmt="int")
        # 训练/采样比 = 累计用于梯度更新的样本数(更新次数×批大小) / 累计环境步数。
        # 它衡量"每条经验平均被训练了几遍"：比值偏高说明同一批数据被反复回炉（更新过快，容易过拟合/发散），
        # 偏低说明采样远快于学习（数据大部分只被用过一次）。它是 Off-policy 训练里判断
        # 更新节奏是否失衡的一个综合指标。分母 max(1.0, ...) 防止训练刚开始步数为 0 时除零。
        mon.health("ratio", lambda: self.updates * config.BATCH / max(1.0, self.steps_fn()),
                   label="训练/采样比", unit="×", fmt="f3")
        mon.health("steps", lambda: self.steps_fn(), label="已训练步数", fmt="int")
        mon.health("rounds", lambda: self.stats.get("rounds"), label="累计对局数", fmt="int")
        for nm in ("随机", "脚本Lv1", "脚本Lv2", "脚本Lv3"):
            mon.health("place_" + nm, ep(nm), label="对" + nm + "平均名次", fmt="f2")

        # 神经网络图与 3D 损失曲面（分别由两个探针提供数据）
        mon.graph(self.probe.graph)
        mon.surface(self.land.snapshot)

        # 图片区。实时对局用 tank_view 的软件渲染；观测通道放大 3 倍便于看清 16px 的格子。
        # group 参数把图片在页面上分组（"战场" / "输入与特征"）。
        mon.image("battle", lambda: _render(self.env, 0), label="实时对局 1", scale=1, group="战场")
        mon.image("battle2", lambda: _render(self.env, 1), label="实时对局 2", scale=1, group="战场")
        mon.image("obs", lambda: _obs(self.env), label="观测通道（1 号位视角）", scale=3, group="输入与特征")
        # 三个卷积层各出一张"全部通道"的拼图。这里 (lambda k: (lambda: ...))(i) 的写法是
        # 为了把循环变量 i 绑进闭包——否则所有回调运行时都会用循环结束后的 i（全都变成最后一个值）。
        for i, (nm, cnt, sp) in enumerate((("C1", k1, s1), ("C2", k2, s2), ("C3", k3, s3))):
            mon.image("feat_" + nm, (lambda k: (lambda: self.probe.feature_image(k)))(i),
                      label="%s 全部 %d 通道（每通道 %d×%d，逐通道归一）" % (nm, cnt, sp, sp),
                      scale=3, group="输入与特征")

        # 每个主要权重张量画两张直方图：权重本身和它的梯度（梯度用 "@grad" 后缀区分）。
        # 两组分布对照着看：权重是"学成了什么形状"，梯度是"还在不在被推动"。
        for key in WEIGHT_KEYS:
            mon.histogram(key, (lambda k: (lambda: self._hist_values(k)))(key),
                          label=key + " · 权重", bins=48)
            mon.histogram(key + "@grad", (lambda k: (lambda: self._hist_values(k)))(key + "@grad"),
                          label=key + " · 梯度", bins=48)
        # 回放池 TD 误差分布单独设 refresh=5.0：要反解百万级 prio 数组，代价比看权重直方图高得多，
        # 放宽刷新间隔可以避免这个回调拖慢面板
        mon.histogram("buffer_td", self._buffer_td, label="经验池 |TD 误差| 分布", bins=40, refresh=5.0)

        mon.bars("layer_grad", self._layer_grad_norm, label="逐层梯度范数")
        mon.bars("acts", self._acts, label="动作使用率（累计）")
        mon.table("layer_stats", self._layer_stats, label="逐层权重与梯度统计",
                  columns=["层", "参数名", "形状", "参数量", "权重均值", "权重标准差", "梯度范数"])
        # 三类告警 + 一个看门狗：看门狗监控"训练步数"这个只会单调增长的量，
        # 30 秒不变化就说明训练主循环卡住了（某局死循环、GPU hang 等），
        # aimonitor 只在"训练中"状态下才触发看门狗告警（暂停时不会误报）。
        mon.alert("grad", self._alert_grad, label="梯度", level="warn")
        mon.alert("q", self._alert_q, label="价值量级", level="warn")
        mon.alert("eval", self._alert_eval, label="策略质量", level="warn")
        mon.watchdog("steps", lambda: self.steps_fn(), label="训练步数", timeout=30.0)
        # 把训练 logger 镜像到网页，方便对照"曲线异动"和"日志里当时发生了什么"
        mon.attach_logger(log, key="train_log", label="训练日志（实时镜像）")
        mon.params(self._params)
        self.mon = mon
        # 启动损失曲面探针的后台守护线程。daemon=True：训练进程退出时线程自动消失，
        # 不需要显式 join；close() 里置 stop 事件则让它干净地提前结束。
        threading.Thread(target=self.land.loop, args=(self.stop,), daemon=True).start()
        return mon

    # 关闭监控：通知后台测量线程退出（不阻塞等待，让训练侧可以立刻收尾）
    def close(self):
        self.stop.set()


# 实时对局渲染。tank_view 是图形相关的模块，推迟到函数内部导入，
# 这样监控模块的导入期不会把渲染依赖一起拉进来，没有观战需求时也少一点开销。
def _render(env, i):
    import tank_view
    return tank_view.render_rgb(env, i)


# 把第 i 号位的打包观测拆成 8 个二值通道，各通道乘 255 变成黑白图，
# 用 montage 拼成 4 列的一张图——这是"网络实际看到的世界"，用于排查观测编码是否符合预期。
def _obs(env, i=0, cols=4):
    ch = tank_vec.unpack(env.obs_pack(i)[i:i + 1])[0]
    maps = [(ch[c] * 255).astype(np.uint8) for c in range(tank_vec.OBS_CH)]
    return imaging.montage(maps, cols=cols, pad=1, pad_val=0)
