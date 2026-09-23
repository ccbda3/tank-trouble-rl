# =============================================================================
# network.py —— 策略网络（Dueling 结构）的定义与推理封装
#
# 在项目中的位置：
#   tank_vec.VecEnv 产出观测 → 本文件把观测变成张量、用网络算出每个动作的 Q 值
#   selfplay.py 才是训练主循环（采样、算 TD 目标、反向更新、维护目标网络）
#   buffer.py 提供训练样本
#   本文件只描述「网络长什么样」和「怎么用它选动作」，不含任何训练逻辑。
#
# 算法：Dueling Double DQN
#   Dueling：输出头拆成状态价值 V（这个局面本身有多好）和动作优势 A（每个动作
#   相对其它动作好多少），再合成 Q。动作之间差异不大的局面里，A 接近相等、由 V
#   单独表达局面好坏，两个头不会互相干扰，训练更稳。
#   Double：动作由在线网络挑、该动作的价值由目标网络给，实现在 selfplay.py，
#   本文件不涉及。
#
# 数据流：
#   obs_pack(j) 的 8 通道按位打包观测图 → obs_tensor() 解包成 float32 → 3 层卷积
#   feat(j) 的 81 维标量特征 → feat_tensor() → 与卷积输出拼接 → 512 维全连接
#   → V 头（1 维）/ A 头（18 维）→ Q = V + A - mean(A)
# =============================================================================

import torch
import torch.nn as nn

import config
import tank_vec

# 设备：训练和推理都在 GPU 上（本机为 RTX 5060）。几百个并行环境每步都要前向，
# 全靠 CPU 根本喂不上。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 动作表：tank.ACTION_CTRL 是 转向(-1/0/1) × 油门(前进/停/后退) × 开火(0/1) 的组合，
# 共 ACTIONS=18 个离散动作。张量行号就是动作编号，Q 值的列号和它一一对应。
CTRL = torch.tensor(tank_vec.CTRL, dtype=torch.float32)
ACTIONS = CTRL.shape[0]


# 网络主体：观测图走卷积提取空间特征，标量特征走全连接，两路拼接后拆成 V/A 两个头。
class DQN(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # 卷积通道数 16/32/32，来自 config，统一在一处调参
        c1, c2, c3 = config.CONV_CH
        # 输入是 8 通道（OBS_CH）的低分辨率地图，3×3 卷积、stride 2 逐层降采样三次，
        # 空间尺寸压缩到约 1/8，作用是把「墙在哪、敌人和子弹在哪」这类局部关系
        # 层层聚合成特征图。观测图是 0/1 的按位打包图，卷积只关心 3×3 邻域。
        self.conv = nn.Sequential(
            nn.Conv2d(tank_vec.OBS_CH, c1, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(c2, c3, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        # 用一次全零输入探出卷积输出的展平长度，而不是把尺寸写死成常量：
        # 以后改地图分辨率或改卷积层数，这里会自动跟着变。
        with torch.no_grad():
            probe = self.conv(torch.zeros(1, tank_vec.OBS_CH, tank_vec.OBS_H, tank_vec.OBS_W))
        flat = probe.shape[1]
        # 卷积特征与标量特征拼接后进 512 维全连接（config.HIDDEN）。
        # 拼接而不是相加，是为了让网络自己学「地图信息和标量信息怎么融合」。
        self.fc = nn.Sequential(nn.Linear(flat + feat_dim, config.HIDDEN), nn.ReLU())
        # Dueling 的两个头：V 只需要一个数，A 每个动作一个数
        self.value = nn.Linear(config.HIDDEN, 1)
        self.adv = nn.Linear(config.HIDDEN, ACTIONS)

    # 前向：卷积吃观测图、全连接吃「卷积特征 ⊕ 标量特征」，最后合成 Q。
    def forward(self, obs, feat):
        h = self.fc(torch.cat([self.conv(obs), feat], dim=1))
        v = self.value(h)
        a = self.adv(h)
        # Dueling 合成式 Q = V + A - mean(A)。
        # 减 mean(A) 是为了消除 V/A 的不可辨识性：A 整体加常数、V 减去同样的常数
        # 会得到同一组 Q，不约束的话两个头会一起漂移、训练不稳。
        return v + a - a.mean(dim=1, keepdim=True)

    # 推理专用：只出动作、不建计算图（采样阶段不需要梯度，no_grad 更省显存也更快）。
    @torch.no_grad()
    def act(self, obs, feat, eps=0.0):
        q = self(obs, feat)
        greedy = q.argmax(dim=1)
        if eps <= 0.0:
            # 贪心模式：评测和「跟随观战」的默认模式，反映策略真实水平
            return greedy
        # ε-贪心探索：每步以 eps 的概率把贪心动作替换成均匀随机动作。
        # 用 torch.where 一次性整批算完，比在 Python 里逐条判断快得多
        # （每步要处理 环境数 × 座位数 个决策）。
        noise = torch.rand_like(greedy, dtype=torch.float32) < eps
        rand = torch.randint(0, ACTIONS, greedy.shape, device=greedy.device)
        return torch.where(noise, rand, greedy)


# 建网并搬到 GPU；feat_dim 由 env.feat(0).shape[1] 决定，当前是 81 维
def make(feat_dim):
    return DQN(feat_dim).to(DEVICE)


# 观测解包 + 上卡。tank_vec 里的观测把 8 个 0/1 平面按位压进一个 uint8
# （相比展开成 float32 省 32 倍内存，回放池能装 20 万条全靠它），
# 送进卷积前必须逐位展开成 float32 并按通道排好。
def obs_tensor(packed):
    return torch.as_tensor(tank_vec.unpack(packed), dtype=torch.float32, device=DEVICE)


# 标量特征上卡：回放池里存的本就是 float32，这里只做类型统一和拷贝到 GPU
def feat_tensor(feat):
    return torch.as_tensor(feat, dtype=torch.float32, device=DEVICE)
