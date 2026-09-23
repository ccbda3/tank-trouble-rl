# ============================================================================
# tank_vec.py —— 向量化多智能体对战环境（强化学习的"世界"）
#
# 本文件在整个项目中的位置：
#   - tank.py        ：可以真正玩的游戏本体（含画面渲染、键盘输入、地图生成）。
#   - tank_vec.py    ：本文件。把 tank.py 的那套规则搬进一个纯 numpy 的批量模拟器，
#                      一次同时推进 N 局（训练里常用 256）互相独立的 4 人混战。
#   - 训练 / 回放脚本 ：调用本文件取"观测 + 奖励"，喂给神经网络。
#
# 数据流方向：
#   动作(整型, 形状 (N, P, 3)) --step()--> 按 dt 推进一帧物理，写回内部状态
#   （坦克 / 子弹 / 地雷 / 道具 / 爆炸 / 激光）--obs_pack()/feat()--> 观测
#   --> 训练脚本据此算 loss、把观测存进回放池。
#
# 本文件完全不负责渲染：它只按帧算物理、算观测，画面上看到的东西由外部
# 读取这里暴露出来的 numpy 状态数组自行绘制（所以既能在无窗口环境跑，也能观战）。
#
# 为什么要有它：
#   强化学习每秒需要产生成千上万帧经验。用 tank.py 那种带画面、带主循环的写法
#   一次只能推进一局，太慢。这里改用"面向整批数组"的写法——每次对整个 (N, ...)
#   张量做一次 numpy 运算，所有对局一起前进，CPU 利用率高得多。
#
# 与 tank.py 的关系：
#   物理常量（WALL_W / PITCH / TANK_SPEED / BOUNCE_LIMIT ...）与地图生成函数
#   ref.generate() 都直接从 tank.py 导入复用，不在这里重新定义一遍——
#   这样模拟器和可玩游戏的规则永远一致，不会两边慢慢跑偏。
# ============================================================================
import numpy as np

import tank as ref
from tank import (
    BARREL_LEN, BLAST_R, BLAST_TTL, BOUNCE_LIMIT, BULLET_R, BULLET_SPEED, CAPSULE_EVERY,
    CAPSULE_MAX, CAPSULE_R, CAPSULE_TTL, CORR_W, DEATH_PAUSE, DOUBLE_COOLDOWN, FIRE_COOLDOWN,
    FRAG_COOLDOWN, FRAG_LIFE, FRAG_SPEED, GATLING_COOLDOWN, GATLING_LIFE, GATLING_SPEED,
    HEIGHT, LASER_COOLDOWN, LASER_TTL, MAX_BULLETS, MINE_ARM, MINE_COOLDOWN, MINE_R,
    MISSILE_COOLDOWN, MISSILE_DELAY, MISSILE_LIFE, MISSILE_SPEED, MISSILE_TURN, NC, NR,
    PITCH, SHARD_BOUNCE, SHARD_COUNT, SHARD_LIFE, SHARD_SPEED, TANK_BACK, TANK_H, TANK_SPEED,
    TANK_TURN, WALL_W, WEAPON_KINDS, WEAPON_TIME, WIDTH,
)

# 上面从 tank.py 导入的常量，几个对理解本文件特别重要的：
#   地图：WALL_W=7(墙厚) / CORR_W=36(通道净宽) / PITCH=WALL_W+CORR_W(一个格子的间距)
#         / NC=19、NR=13(竖、横方向的格子数) / WIDTH、HEIGHT 是由这两者推出的整图像素尺寸。
#   坦克：TANK_H=10(半车宽，坦克按边长 2*TANK_H 的正方形做碰撞) / TANK_SPEED=170(前进)
#         / TANK_BACK=108(后退更慢) / TANK_TURN=3.0(每秒最多转 3 弧度) / BARREL_LEN=16(炮口离车心距离)。
#   子弹：BULLET_R=3(子弹是圆，半径 3) / BULLET_SPEED=450 / MAX_BULLETS=5(同屏普通弹上限)。
#   BOUNCE_LIMIT=7：子弹撞墙反弹超过 7 次就消失。取 7 是手感折中——够它在迷宫里弹几下
#         绕到刁钻角度打到人，又不至于一直乱飞、把局面拖成双方互相躲子弹。
#   DEATH_PAUSE=0.7：一方阵亡后停顿 0.7 秒再开下一局，给画面留出表现死亡的时间。

# 8 种武器的"本文件内部编号"：0 普通、1 加特林、2 追踪导弹、3 破片、4 双管炮、
# 5 激光、6 护盾、7 地雷。道具胶囊里存的是"武器种类下标"0~6，捡到时 +1 就变成这里的编号。
W_NORMAL, W_GATLING, W_MISSILE, W_FRAG, W_DOUBLE, W_LASER, W_SHIELD, W_MINE = range(8)
# 把"每种武器能维持多少秒"查表成数组，好按武器编号直接索引；
# 下标 0 是普通武器（无时限），所以补一个 0.0 占位。
WEAPON_SECONDS = np.array([0.0] + [WEAPON_TIME[k] for k in WEAPON_KINDS], dtype=np.float32)
# 飞行物的"种类"编号（注意与上面的武器编号不同，它描述子弹本身的行为）：
# K_BULLET 普通直飞弹 / K_GATLING 加特林弹 / K_SHARD 破片炸弹炸出的碎片
# / K_MISSILE 追踪导弹 / K_FRAG 破片弹本体（它自己不打人，撞墙才炸）。
K_BULLET, K_GATLING, K_SHARD, K_MISSILE, K_FRAG = range(5)
# 各类实体的槽位容量，都按"一局 4 人混战"标定：
# 子弹 96（4 人各种武器一起扫射也够用）、地雷 5、道具胶囊 4、爆炸 8、激光 4。
# 用"定长数组 + 布尔开关(on)"的对象池写法，才能全程向量化运算，
# 不必为每个实体分配 Python 对象（那样慢得多）。
B_MAX = 96
M_MAX = 5
C_MAX = 4
Z_MAX = 8
L_MAX = 4
# 把一帧的子弹位移切成 4 个子步推进。子弹 450 px/s，60fps 时一帧走 7.5 px，
# 而墙只有 7 px 厚、通道只有 36 px 宽；不分步的话高速子弹会"跳过"薄墙。
# 4 个子步（每步约 1.9 px）已足以覆盖墙厚，再细分只是徒增计算量。
SUBSTEPS = 4
# 标量特征里只取离自己最近的 8 颗子弹：既能感知威胁方向，又不至于让特征维度爆炸。
NEAR_BULLETS = 8
# 单个回合的帧数上限。游戏按 60 帧/秒跑，1800 帧刚好 30 秒：
# 双方都苟着不打时，30 秒后判平局（两边都 0 奖励），避免一个回合无限拖长。
ROUND_LIMIT = 1800
# tank.py 的合法动作表拷贝成 numpy 数组：共 18 个动作
# = 转向(-1/0/1) × 油门(前进/停/后退) × 开火(0/1)，顺序由 tank.ACTION_CTRL 决定。
CTRL = np.array(ref.ACTION_CTRL, dtype=np.float32)

# ---- 观测图的分辨率与通道定义 ----
# OBS_RES=16：物理世界 16×16 像素合并成观测图上的 1 个像素（降采样）。
# 这样一张 51×35 的小图就能覆盖整张地图，同时大幅省内存、省算力。
OBS_RES = 16
OBS_W = int(WIDTH) // OBS_RES
OBS_H = int(HEIGHT) // OBS_RES
# 观测一共 8 个通道：墙 / 自机 / 其余对手(合并) / 保留未用 / 子弹 / 地雷 / 道具 / 激光
OBS_CH = 8
# 各通道对应的比特位。观测图是"按位打包"的：1 个像素只占 1 个字节，
# 第 c 位就是第 c 个通道（1 表示该处有此物）。相比"每个通道单独存一份 float32"，
# 内存直接降到 1/8，回放池因此能装下多得多的经验。
# 注意 bit3 被跳过（保留未用），所以 BIT_ENEMY=2 与 BIT_BULLET=4 之间空了一位。
BIT_WALL, BIT_SELF, BIT_ENEMY, BIT_BULLET, BIT_MINE, BIT_CAPSULE, BIT_BEAM = 0, 1, 2, 4, 5, 6, 7


# 把按位打包的观测（每像素 1 字节）反打包成 (B, 8, H, W) 的 float32 多通道图。
# 主要给"人看/调试/可视化"用；训练时网络通常直接吃打包后的字节，自己用位移
# 拆通道，省掉这里 8 倍的内存扩张（回放池就是用打包格式存的）。
def unpack(packed):
    b = packed.shape[0]
    out = np.empty((b, OBS_CH) + packed.shape[1:], dtype=np.float32)
    for c in range(OBS_CH):
        out[:, c] = (packed >> c) & 1
    return out


# 预先算好"观测图像素坐标 -> 地图网格"的映射关系。地图是规则的墙+通道网格，
# 这层映射与对局状态无关，所以模块加载时算一次、之后全局复用，不必每帧重算。
def _maps():
    ow = np.arange(OBS_W, dtype=np.int64) * OBS_RES
    oh = np.arange(OBS_H, dtype=np.int64) * OBS_RES
    # sv/sh：每个观测列的 x（每个观测行的 y）是否落在竖直/水平墙体内。
    sv = (ow % int(PITCH)) < WALL_W
    sh = (oh % int(PITCH)) < WALL_W
    # vs/hs：该观测列/行对应"第几条竖墙线/横墙线"，夹到合法索引范围 0..NC / 0..NR。
    vs = np.clip(ow // int(PITCH), 0, NC)
    hs = np.clip(oh // int(PITCH), 0, NR)
    # ic/jc：该观测列/行对应"第几个格子"（用格子中心反推，四舍五入），夹到有效范围。
    ic = np.clip(np.round((ow - WALL_W - CORR_W * 0.5) / PITCH).astype(np.int64), 0, NC - 1)
    jc = np.clip(np.round((oh - WALL_W - CORR_W * 0.5) / PITCH).astype(np.int64), 0, NR - 1)
    return sv, sh, vs, hs, ic, jc


# 6 张查找表（见 _refresh_wall_bits 用它们把墙"投影"到观测图上），全局只算一次。
_SV, _SH, _VS, _HS, _IC, _JC = _maps()
# 3×3 九宫格的偏移 (dy, dx)，用来把点状实体在观测图上画成一个小方块（见 _blit_bit）。
_DISC = np.array([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=np.int64)


# 判断两个"旋转正方形"（坦克）是否相交，返回逐元素的布尔数组。
# 坦克是边长 2h 的正方形刚体，两辆车挤在一起时必须能查出重叠（见 _tank_free）。
# 用分离轴定理(SAT)：只要能找到一根轴，使两个正方形在轴上的投影互不重叠，
# 就说明它们没相交。候选轴取 4 条：世界坐标两主轴 + A、B 各自的边方向。
def _sq_overlap(ax, ay, aa, bx, by, ba, h):
    ca = np.cos(aa)
    sa = np.sin(aa)
    cb = np.cos(ba)
    sb = np.sin(ba)
    dx = bx - ax
    dy = by - ay
    ok = np.ones(np.shape(dx), dtype=bool)
    # 世界坐标的两条主轴 (1,0)、(0,1)：把两个正方形分别投影上去比较。
    for nx, ny in ((1.0, 0.0), (0.0, 1.0)):
        p = np.abs(dx * nx + dy * ny)
        # 旋转后的正方形在某轴上的"半宽" = h*(|cos| + |sin|)，即投影长度的一半。
        ea = h * (np.abs(ca * nx + sa * ny) + np.abs(-sa * nx + ca * ny))
        eb = h * (np.abs(cb * nx + sb * ny) + np.abs(-sb * nx + cb * ny))
        ok &= p <= ea + eb
    # 再加上 A 自身的两条边方向作为候选轴；此时 A 在该轴上的半宽恒为 h。
    for ux, uy in ((ca, sa), (-sa, ca)):
        p = np.abs(dx * ux + dy * uy)
        eb = h * (np.abs(cb * ux + sb * uy) + np.abs(-sb * ux + cb * uy))
        ok &= p <= h + eb
    return ok


# 射线与"旋转正方形"求交，返回沿射线方向的距离；未命中返回 -1。
# 用途：激光发射时从炮口沿朝向打一条射线，看多远会打到某辆坦克（见 _laser）。
# 做法：把射线与正方形都变换到正方形的局部坐标系（那里正方形是轴对齐的），
# 再用 slab 法——分别求射线与 x 方向两条边、y 方向两条边相交的参数区间，取交集。
def _ray_square(ox, oy, dx, dy, tx, ty, ta, half):
    ca = np.cos(ta)
    sa = np.sin(ta)
    # 把射线起点、方向旋转进正方形的局部坐标。
    rx = ox - tx
    ry = oy - ty
    lx = rx * ca + ry * sa
    ly = ry * ca - rx * sa
    ex = dx * ca + dy * sa
    ey = dy * ca - dx * sa
    # t0/t1 维护"射线参数的有效区间"，初值取极大/极小值表示"区间暂定全部有效"。
    t0 = np.full(np.shape(ox), -1e9, dtype=np.float32)
    t1 = np.full(np.shape(ox), 1e9, dtype=np.float32)
    for o, d in ((lx, ex), (ly, ey)):
        # 该方向分量接近 0 时射线平行于这组边，不能做除法，单独处理。
        par = np.abs(d) < 1e-9
        safe = np.where(par, 1.0, d)
        lo = np.minimum((-half - o) / safe, (half - o) / safe)
        hi = np.maximum((-half - o) / safe, (half - o) / safe)
        # 平行、且起点落在正方形外侧（与这组边错开）时永远打不中，置成无效区间。
        blocked = par & ((o < -half) | (o > half))
        t0 = np.where(par, np.where(blocked, 1e9, t0), np.maximum(t0, lo))
        t1 = np.where(par, t1, np.minimum(t1, hi))
    # 区间为空（t0>t1）或整段都在起点后方（t1<0）都算未命中。
    bad = (t0 > t1) | (t1 < 0.0)
    return np.where(bad, -1.0, np.maximum(0.0, t0))


# 向量化环境：同时模拟 self.n 局、每局 self.p 个座位的对战。
# 约定：所有状态数组的第一维都是"局"(batch)，后续维度才是"座位/实体槽位"，
# 这样一次 numpy 运算就能同时推进所有对局，Python 层几乎没有循环。
# 命名前缀含义：t_ 坦克、b_ 子弹、m_ 地雷、c_ 道具胶囊、z_ 爆炸、l_ 激光、
# ep_ 一条 episode（训练用的一整条轨迹）内的累计统计。
class VecEnv:
    # n：并行对局数（训练里常用 256）；players：每局座位数（混战传 4，默认 2）。
    def __init__(self, n, players=2, seed=None, braid=None):
        self.n = int(n)
        self.p = int(players)
        self.rng = np.random.default_rng(seed)
        # braid：迷宫连通度，None 表示用 tank.generate 的默认值（0.70）。
        # 训练时把它设成浮点数就能控制难度，也可以中途改（只影响之后新生成的迷宫），
        # 用来做"课程学习"：先给开阔地图，再逐步加回死胡同。
        self.braid = braid
        # 全体局索引数组，用于"一次操作部分局"的高级索引（比如只给死亡的那些局开新回合）。
        self.rows = np.arange(self.n)
        # ---- 迷宫（墙）----
        # vwall[i, c, r]：第 i 局第 c 条竖墙线、第 r 行格子的左边是否有竖墙；hwall 同理表示横墙。
        # 结构由 ref.generate() 生成，与可玩游戏完全一致。
        self.vwall = np.zeros((self.n, NC + 1, NR), dtype=bool)
        self.hwall = np.zeros((self.n, NR + 1, NC), dtype=bool)
        # ---- 坦克 ----
        # tx/ty/ta：车心坐标与朝向(弧度)；t_alive：是否存活；
        # t_cool：开火冷却剩余秒数；t_wt：当前特殊武器的剩余持续时间；t_weapon：当前武器编号(0~7)；
        # t_score：本局得分。
        self.tx = np.zeros((self.n, self.p), dtype=np.float32)
        self.ty = np.zeros((self.n, self.p), dtype=np.float32)
        self.ta = np.zeros((self.n, self.p), dtype=np.float32)
        self.t_alive = np.ones((self.n, self.p), dtype=bool)
        self.t_cool = np.zeros((self.n, self.p), dtype=np.float32)
        self.t_wt = np.zeros((self.n, self.p), dtype=np.float32)
        self.t_weapon = np.zeros((self.n, self.p), dtype=np.int8)
        self.t_score = np.zeros((self.n, self.p), dtype=np.int32)
        # round_frames：本回合已经跑了多少帧（用来判 30 秒上限平局）；
        # maze_seq：迷宫序号，每换一次地图 +1，供外部渲染端判断"缓存的图案该不该重建"。
        self.round_frames = np.zeros(self.n, dtype=np.int32)
        self.maze_seq = np.zeros(self.n, dtype=np.int32)
        # picked：本帧该座位是否刚吃到道具（供上层统计/奖励用）。
        self.picked = np.zeros((self.n, self.p), dtype=bool)
        # t_px/t_py：上一帧的车心坐标；两者相减除以 dt 就得到当前速度 t_vx/t_vy（供观测使用）。
        self.t_px = np.zeros((self.n, self.p), dtype=np.float32)
        self.t_py = np.zeros((self.n, self.p), dtype=np.float32)
        self.t_vx = np.zeros((self.n, self.p), dtype=np.float32)
        self.t_vy = np.zeros((self.n, self.p), dtype=np.float32)
        # ---- 子弹 ----
        # 每个槽位：on(是否在使用) + 位置 + 速度 + 剩余寿命 b_life + 已反弹次数 b_bounce
        # + 归属玩家 b_owner + 种类 b_kind。
        # b_owner：归属玩家编号，初值 -1；归因击杀时把小于 0 的值当作"无主"处理（见 _damage）。
        self.b_on = np.zeros((self.n, B_MAX), dtype=bool)
        self.b_x = np.zeros((self.n, B_MAX), dtype=np.float32)
        self.b_y = np.zeros((self.n, B_MAX), dtype=np.float32)
        self.b_vx = np.zeros((self.n, B_MAX), dtype=np.float32)
        self.b_vy = np.zeros((self.n, B_MAX), dtype=np.float32)
        self.b_life = np.zeros((self.n, B_MAX), dtype=np.float32)
        self.b_bounce = np.zeros((self.n, B_MAX), dtype=np.int16)
        self.b_owner = np.full((self.n, B_MAX), -1, dtype=np.int8)
        self.b_kind = np.zeros((self.n, B_MAX), dtype=np.int8)
        # ---- 地雷 ----
        # 每个槽位：on + 位置 + m_arm(还需多少秒才"解除保险"、可以伤人) + 归属玩家。
        self.m_on = np.zeros((self.n, M_MAX), dtype=bool)
        self.m_x = np.zeros((self.n, M_MAX), dtype=np.float32)
        self.m_y = np.zeros((self.n, M_MAX), dtype=np.float32)
        self.m_arm = np.zeros((self.n, M_MAX), dtype=np.float32)
        self.m_owner = np.full((self.n, M_MAX), -1, dtype=np.int8)
        # ---- 道具胶囊 ----
        # 每个槽位：on + 位置 + c_ttl(剩余存在时间) + c_kind(武器种类下标 0~6，捡到时 +1 得到 W_ 编号)。
        self.c_on = np.zeros((self.n, C_MAX), dtype=bool)
        self.c_x = np.zeros((self.n, C_MAX), dtype=np.float32)
        self.c_y = np.zeros((self.n, C_MAX), dtype=np.float32)
        self.c_ttl = np.zeros((self.n, C_MAX), dtype=np.float32)
        self.c_kind = np.zeros((self.n, C_MAX), dtype=np.int8)
        # ---- 爆炸（纯视觉 + 半径信息）----
        # 每个槽位：on + 圆心 + 半径 + 剩余显示时间。
        self.z_on = np.zeros((self.n, Z_MAX), dtype=bool)
        self.z_x = np.zeros((self.n, Z_MAX), dtype=np.float32)
        self.z_y = np.zeros((self.n, Z_MAX), dtype=np.float32)
        self.z_r = np.zeros((self.n, Z_MAX), dtype=np.float32)
        self.z_ttl = np.zeros((self.n, Z_MAX), dtype=np.float32)
        # ---- 激光（一帧的线段，只在画面上闪过）----
        # 每个槽位：on + 四个数 (x0, y0, x1, y1) 记录线段两端 + 剩余显示时间。
        self.l_on = np.zeros((self.n, L_MAX), dtype=bool)
        self.l_xy = np.zeros((self.n, L_MAX, 4), dtype=np.float32)
        self.l_ttl = np.zeros((self.n, L_MAX), dtype=np.float32)
        # spawn_t：距离下次刷新道具还有多少秒。初值取半个刷新周期，
        # 避免开局第 0 帧就立刻刷出胶囊。
        self.spawn_t = np.full(self.n, CAPSULE_EVERY * 0.5, dtype=np.float32)
        # pause：分胜负后的停顿倒计时；over：本局是否已结束、正在等 pause 走完。
        self.pause = np.zeros(self.n, dtype=np.float32)
        self.over = np.zeros(self.n, dtype=bool)
        # winner：赢家座位号(-1 表示还没赢家)。
        self.winner = np.full(self.n, -1, dtype=np.int8)
        # just_died：本帧刚阵亡的座位，由 step() 用"帧前存活 & 帧后死亡"差分得到；
        # place：本局名次（1 为赢家）；suicide：是否死于自己（自己的子弹/地雷）。
        self.just_died = np.zeros((self.n, self.p), dtype=bool)
        self.place = np.zeros((self.n, self.p), dtype=np.int8)
        self.suicide = np.zeros((self.n, self.p), dtype=bool)
        # kills/t_score：本局累计击杀数与积分；rounds：本 episode 已经打完的局数。
        self.kills = np.zeros((self.n, self.p), dtype=np.int32)
        self.rounds = np.zeros(self.n, dtype=np.int32)
        # ep_kills/ep_deaths/ep_steps：一条 episode 内的累计击杀、累计阵亡、总步数（用于日志与统计）。
        self.ep_kills = np.zeros((self.n, self.p), dtype=np.int32)
        self.ep_deaths = np.zeros((self.n, self.p), dtype=np.int32)
        self.ep_steps = np.zeros(self.n, dtype=np.int32)
        # wall_bits：把当前迷宫的墙预先"蒸"进观测图的一个通道（已经是打包好的字节），
        # 换地图时才重算一次，供 _obs_base 每帧复用。
        self.wall_bits = np.zeros((self.n, OBS_H, OBS_W), dtype=np.uint8)
        # _base：观测底图缓存（与视角无关的那几个通道）。None 表示已失效，下次取时重算。
        self._base = None
        self.reset()

    # 整个 episode 归零：清空积分与统计，然后给所有局开新的一回合。
    # 注意"episode"比"回合(round)"大：一条 episode 会连续打好多个回合
    # （每回合内部由 step() 自动重开），这里清的是跨越多个回合的累计量。
    def reset(self):
        self.t_score[:] = 0
        self.kills[:] = 0
        self.ep_kills[:] = 0
        self.ep_deaths[:] = 0
        self.ep_steps[:] = 0
        self.rounds[:] = 0
        self._new_round(self.rows)
        return self.obs_pack(0)   # 默认返回 0 号座位的观测，方便外层直接拿到首帧

    # 给指定这些局重开一回合：清空场上所有实体、重新生成迷宫、重掷出生点。
    # 这就是原版规则——"任意一方死亡即整局结束、全员满血重生、地图重生成新迷宫"。
    def _new_round(self, rows):
        self.b_on[rows] = False
        self.m_on[rows] = False
        self.z_on[rows] = False
        self.l_on[rows] = False
        self.c_on[rows] = False
        self.pause[rows] = 0.0
        self.over[rows] = False
        self.winner[rows] = -1
        self.just_died[rows] = False
        self.place[rows] = 0
        self.suicide[rows] = False
        self.round_frames[rows] = 0
        self.maze_seq[rows] += 1   # 迷宫序号 +1：外部渲染端据此判断"图案缓存失效了、要重建"
        self._base = None          # 地图变了，观测底图缓存作废
        for i in rows:
            # 每局用独立随机种子生成迷宫（与可玩游戏同一套生成器，保证规则一致）。
            # self.braid 为 None 时用生成器的默认连通度；设了值就按这个值生成，
            # 所以课程学习只要在每个训练迭代里改 self.braid，之后新生成的迷宫就跟着变。
            seed_i = int(self.rng.integers(1 << 30))
            if self.braid is None:
                vwall, hwall = ref.generate(seed_i)
            else:
                vwall, hwall = ref.generate(seed_i, braid=float(self.braid))
            self.vwall[i] = vwall
            self.hwall[i] = hwall
        self.picked[rows] = False
        self.spawn_t[rows] = CAPSULE_EVERY * 0.5
        # 出生点用"格子编号"表示，再反解成坐标。下面用拒绝采样反复重掷：
        # 任何两人间距不足 3 格(PITCH*3)的出生点都要重新抽，最多重试 60 轮，
        # 避免开局就把两个人叠在一起（叠一起会导致碰撞判定/朝向都退化）。
        cells = self.rng.integers(0, NC * NR, size=(rows.size, self.p))
        for _ in range(60):
            ci = cells // NR
            cj = cells % NR
            x = WALL_W + ci * PITCH + CORR_W * 0.5
            y = WALL_W + cj * PITCH + CORR_W * 0.5
            bad = np.zeros((rows.size, self.p), dtype=bool)
            for a in range(self.p):
                for b in range(a + 1, self.p):
                    close = (x[:, a] - x[:, b]) ** 2 + (y[:, a] - y[:, b]) ** 2 < (PITCH * 3.0) ** 2
                    bad[:, a] |= close
                    bad[:, b] |= close
            if not bad.any():
                break
            cells[bad] = self.rng.integers(0, NC * NR, size=int(bad.sum()))
        # 最终统一把格子编号换算成世界坐标（取格子中心点）。
        ci = cells // NR
        cj = cells % NR
        x = (WALL_W + ci * PITCH + CORR_W * 0.5).astype(np.float32)
        y = (WALL_W + cj * PITCH + CORR_W * 0.5).astype(np.float32)
        self.tx[rows] = x
        self.ty[rows] = y
        self.t_px[rows] = x   # 上一帧位置也同步，避免开局第一帧算出巨大的瞬时速度
        self.t_py[rows] = y
        self.t_alive[rows] = True
        self.t_cool[rows] = 0.0
        self.t_wt[rows] = 0.0
        self.t_weapon[rows] = W_NORMAL
        self.t_vx[rows] = 0.0
        self.t_vy[rows] = 0.0
        # 初始朝向：每个座位转向"离自己最远的那个座位"，即开局大致面向对手（而不是背对）。
        ang = np.zeros((rows.size, self.p), dtype=np.float32)
        for k in range(self.p):
            dx = x - x[:, k:k + 1]
            dy = y - y[:, k:k + 1]
            d2 = dx * dx + dy * dy
            d2[:, k] = -1.0   # 把自己排除，否则"最远/最近"会算到自己头上
            far = np.argmax(d2, axis=1)
            ang[:, k] = np.arctan2(dy[np.arange(rows.size), far], dx[np.arange(rows.size), far])
        self.ta[rows] = ang
        self._refresh_wall_bits(rows)

    # 把迷宫墙"投影"进观测图，只在换地图时调用一次（比较贵，没必要每帧做）。
    # 思路：先用 _VS/_JC（_HS/_IC）从 vwall（hwall）里取出每个观测列/行对应的那条墙线，
    # 再与 _SV/_SH（该像素是否落在墙厚范围内）做与运算，就得到墙的位图。
    def _refresh_wall_bits(self, rows):
        vw = self.vwall[np.ix_(rows, _VS, _JC)]
        hw = self.hwall[np.ix_(rows, _HS, _IC)]
        vw = np.transpose(vw, (0, 2, 1))   # 转置成与 (OBS_H, OBS_W) 对齐的排布
        wall = (_SV[None, None, :] & vw) | (_SH[None, :, None] & hw)
        self.wall_bits[rows] = wall.astype(np.uint8) << BIT_WALL   # 直接写成打包好的字节

    # 判断"以 (x,y) 为圆心、半径 r 的圆"能否放下（不与任何墙重叠）。
    # 先定位该点所在的格子 (ci,cj)，再逐面检查它到上下左右四面墙的距离是否 >= r。
    def free(self, rows, x, y, r):
        ci = np.clip(np.floor((x - WALL_W) / PITCH).astype(np.int64), 0, NC - 1)
        cj = np.clip(np.floor((y - WALL_W) / PITCH).astype(np.int64), 0, NR - 1)
        left = WALL_W + ci * PITCH
        top = WALL_W + cj * PITCH
        ok = np.ones(np.shape(x), dtype=bool)
        ok &= (~self.vwall[rows, ci, cj]) | (x >= left + r)                    # 左边那面墙：要么没墙，要么离得够远
        ok &= (~self.vwall[rows, ci + 1, cj]) | (x <= left + CORR_W - r)       # 右边那面墙
        ok &= (~self.hwall[rows, cj, ci]) | (y >= top + r)                     # 上边那面墙
        ok &= (~self.hwall[rows, cj + 1, ci]) | (y <= top + CORR_W - r)        # 下边那面墙
        return ok

    # 判断 side 号坦克能否停在 (x, y, ang) 这个位姿：不能出边界、不能压墙、
    # 不能和其它"还活着"的坦克重叠。
    def _tank_free(self, rows, x, y, ang, side):
        ok = (x >= TANK_H) & (y >= TANK_H) & (x <= WIDTH - TANK_H) & (y <= HEIGHT - TANK_H)
        ok &= self.free(rows, x, y, TANK_H)
        for j in range(self.p):
            if j == side:
                continue
            live = self.t_alive[rows, j]
            if not live.any():
                continue
            ov = _sq_overlap(x, y, ang, self.tx[rows, j], self.ty[rows, j], self.ta[rows, j], TANK_H)
            ok &= ~(live & ov)   # 只有"对方活着且重叠"才算阻挡；已阵亡的坦克不再挡路
        return ok

    # 推进一帧。ctrl 形状 (n, p, 3)，最后一维是 (转向, 油门, 开火)；
    # dt 是这一帧的时长（秒），通常取 1/60。
    # 本方法不返回观测，调用方另外用 obs_pack()/feat() 去取。
    def step(self, ctrl, dt):
        run = ~self.over          # 只有"还在打"的局才推进物理
        self._base = None         # 每帧子弹/道具/地雷都可能变，观测底图缓存一律作废
        before = self.t_alive.copy()   # 记下帧前的存活情况，帧后一对比就知道谁刚死
        if run.any():
            self.picked[:] = False
            self._drive(ctrl, dt, run)     # 1) 移动、转向、开火
            self._items(dt, run)           # 2) 道具刷新/拾取、地雷起爆、各类计时衰减
            self._bullets(dt, run)         # 3) 子弹推进、反弹、命中判定
        # just_died 用"帧前存活 & 帧后死亡"做差分。好处是无论死于子弹、地雷、破片、
        # 激光还是自己，都被自动捕捉到，不必在每个致死分支里各写一遍"标记死亡"。
        self.just_died = before & ~self.t_alive
        self.ep_steps[run] += 1
        alive_cnt = self.t_alive.sum(axis=1)   # 本帧结算后各局还活着的人数
        playing = run & ~self.over
        self.round_frames[playing] += 1
        # 回合结束条件：除自己外剩下的活人不超过 1 个（分出胜负），或已经打满帧数上限（平局）。
        ended = playing & ((alive_cnt <= 1) | (self.round_frames >= ROUND_LIMIT))
        if ended.any():
            self.over |= ended
            self.pause[ended] = DEATH_PAUSE   # 结束后先停顿 0.7 秒，给画面留出表现死亡的时间
            self.rounds[ended] += 1
            one = ended & (alive_cnt == 1)
            r = np.nonzero(one)[0]
            if r.size:
                self.winner[r] = self.t_alive[r].argmax(axis=1)   # 唯一还活着的人就是赢家
                self.place[r, self.winner[r]] = 1                  # 赢家名次固定为 1
            for j in range(self.p):
                self.ep_deaths[ended, j] += (~self.t_alive[ended, j]).astype(np.int32)
        # 名次 place 的算法：某人阵亡那一刻，场上还剩 k 个活人，那他就是第 k+1 名
        # （因为他后面还有 k 个人排着）。所以"阵亡后存活人数 + 1"就是名次。
        # np.maximum(..., 2) 专门处理"互杀导致同时全灭"：此时 k=0 会算出 1 名，
        # 但这不是真正的赢家，于是强制抬到 2 名并列；这样 place==1 只可能是真正的赢家。
        r, j = np.nonzero(self.just_died & run[:, None])
        if r.size:
            self.place[r, j] = np.maximum(alive_cnt[r] + 1, 2)
        self.pause = np.maximum(0.0, self.pause - dt)
        done = self.over & (self.pause <= 0.0)
        if done.any():
            rows = np.nonzero(done)[0]
            self._new_round(rows)   # 停顿走完，这些局开新的一回合

    # 处理所有座位的"转向 / 油门 / 开火"输入。这是塔克式(Tank Trouble)操作：
    # 只能原地左右转 + 沿车头方向前进/后退，没有侧移。
    def _drive(self, ctrl, dt, run):
        turn = ctrl[:, :, 0].astype(np.float32)
        thrust = ctrl[:, :, 1].astype(np.float32)
        fire = ctrl[:, :, 2] > 0
        alive = self.t_alive & run[:, None]
        for side in range(self.p):
            # 逐座位处理：不同座位要各自判断"撞没撞到人/墙"，难以完全并成一次运算。
            rows = np.nonzero(alive[:, side])[0]
            if rows.size == 0:
                continue
            sel = turn[rows, side] != 0
            if sel.any():
                r = rows[sel]
                na = self.ta[r, side] + turn[r, side] * TANK_TURN * dt
                # 关键：用"新角度 + 旧位置"试算。坦克绕中心旋转时四个角会扫过一片区域，
                # 这里要求整辆车转过去之后仍然 _tank_free，否则这一帧宁可不转。
                # 这样车角就不会拧进墙里、也不会卡在墙角里打转。
                ok = self._tank_free(r, self.tx[r, side], self.ty[r, side], na, side)
                self.ta[r[ok], side] = na[ok]
            sel = thrust[rows, side] != 0
            if sel.any():
                r = rows[sel]
                # 前进用 TANK_SPEED，后退用较慢的 TANK_BACK（倒车更慢，鼓励正面交战）。
                v = np.where(thrust[r, side] > 0, TANK_SPEED, -TANK_BACK) * dt
                nx = self.tx[r, side] + np.cos(self.ta[r, side]) * v
                ny = self.ty[r, side] + np.sin(self.ta[r, side]) * v
                ok = self._tank_free(r, nx, ny, self.ta[r, side], side)   # 只有能去的位置才更新
                self.tx[r[ok], side] = nx[ok]
                self.ty[r[ok], side] = ny[ok]
        # 用位置差分算出车速（供观测特征使用），并刷新"上一帧位置"。
        self.t_vx = (self.tx - self.t_px) / dt
        self.t_vy = (self.ty - self.t_py) / dt
        self.t_px = self.tx.copy()
        self.t_py = self.ty.copy()
        self.t_cool = np.maximum(0.0, self.t_cool - dt)
        # 特殊武器计时：时间用完就自动打回普通武器（护盾被打碎也走同一套复位逻辑）。
        armed = self.t_weapon > W_NORMAL
        self.t_wt[armed] -= dt
        expired = armed & (self.t_wt <= 0.0)
        self.t_weapon[expired] = W_NORMAL
        self.t_wt[expired] = 0.0
        for side in range(self.p):
            # 开火条件：活着 + 按了开火键 + 冷却已结束。
            rows = np.nonzero(alive[:, side] & fire[:, side] & (self.t_cool[:, side] <= 0.0))[0]
            if rows.size:
                self._fire(rows, side)

    # 炮口的世界坐标：从车心沿朝向推进 BARREL_LEN。
    # 子弹/激光都从炮口发出（而不是车心），否则贴脸时子弹会从自己车身里冒出来。
    def _muzzle(self, rows, side):
        return (self.tx[rows, side] + np.cos(self.ta[rows, side]) * BARREL_LEN,
                self.ty[rows, side] + np.sin(self.ta[rows, side]) * BARREL_LEN)

    # 为 side 座位的这些局各发射一个飞行物，写进它的空槽位。
    # 因为要批量操作，这里不能随便挑空位，统一取每局的"第一个空槽"（argmax 布尔数组）。
    def _emit(self, rows, side, ang, speed, kind, bounce, life):
        empty = ~self.b_on[rows]
        has = empty.any(axis=1)     # 只处理"还有空槽"的局，槽满的局这次就不发射了
        if not has.any():
            return
        r = rows[has]
        slot = np.argmax(empty[has], axis=1)
        a = ang[has] if np.ndim(ang) else ang   # ang 可能是整批角度数组，也可能是单个标量角度
        mx, my = self._muzzle(r, side)
        self.b_on[r, slot] = True
        self.b_x[r, slot] = mx
        self.b_y[r, slot] = my
        self.b_vx[r, slot] = np.cos(a) * speed
        self.b_vy[r, slot] = np.sin(a) * speed
        self.b_kind[r, slot] = kind
        self.b_bounce[r, slot] = 0
        self.b_life[r, slot] = life
        self.b_owner[r, slot] = side

    # 统计该座位当前已有多少"某种子弹"，用来做发射数量限制
    # （普通弹最多 MAX_BULLETS 发、追踪导弹和破片弹各只允许同时存在 1 颗）。
    def _count(self, rows, side, kind):
        return (self.b_on[rows] & (self.b_owner[rows] == side) & (self.b_kind[rows] == kind)).sum(axis=1)

    # 按座位当前持有的武器，决定这一枪打什么、冷却设多长。
    def _fire(self, rows, side):
        w = self.t_weapon[rows, side]
        for kind_id, mask in ((W_NORMAL, w == W_NORMAL), (W_GATLING, w == W_GATLING),
                              (W_MISSILE, w == W_MISSILE), (W_FRAG, w == W_FRAG),
                              (W_DOUBLE, w == W_DOUBLE)):
            if not mask.any():
                continue
            r = rows[mask]
            if kind_id == W_NORMAL:
                # 普通弹：受"同屏最多 MAX_BULLETS 发"的限制；寿命固定 4 秒。
                r = r[self._count(r, side, K_BULLET) < MAX_BULLETS]
                if r.size == 0:
                    continue
                self._emit(r, side, self.ta[r, side], BULLET_SPEED, K_BULLET, BOUNCE_LIMIT, 4.0)
                self.t_cool[r, side] = FIRE_COOLDOWN
            elif kind_id == W_GATLING:
                # 加特林：冷却极短、弹速更快、寿命更短，靠"泼洒"压制。
                self._emit(r, side, self.ta[r, side], GATLING_SPEED, K_GATLING, BOUNCE_LIMIT, GATLING_LIFE)
                self.t_cool[r, side] = GATLING_COOLDOWN
            elif kind_id == W_MISSILE:
                # 追踪导弹：同一时刻只允许存在 1 颗，打完才能再发。
                r = r[self._count(r, side, K_MISSILE) == 0]
                if r.size == 0:
                    continue
                self._emit(r, side, self.ta[r, side], MISSILE_SPEED, K_MISSILE, BOUNCE_LIMIT, MISSILE_LIFE)
                self.t_cool[r, side] = MISSILE_COOLDOWN
            elif kind_id == W_FRAG:
                # 破片弹：也只允许 1 颗。bounce 参数传 0，表示它不允许反弹——
                # 一碰到墙就立刻炸开成碎片（见 _bullets 里的 frag 分支）。
                r = r[self._count(r, side, K_FRAG) == 0]
                if r.size == 0:
                    continue
                self._emit(r, side, self.ta[r, side], FRAG_SPEED, K_FRAG, 0, FRAG_LIFE)
                self.t_cool[r, side] = FRAG_COOLDOWN
            else:
                # 双管炮：一次发两颗普通弹，分别偏离车头 ±0.055 弧度（约 3°），形成小散开。
                self._emit(r, side, self.ta[r, side] - 0.055, BULLET_SPEED, K_BULLET, BOUNCE_LIMIT, 4.0)
                self._emit(r, side, self.ta[r, side] + 0.055, BULLET_SPEED, K_BULLET, BOUNCE_LIMIT, 4.0)
                self.t_cool[r, side] = DOUBLE_COOLDOWN
        laser = rows[w == W_LASER]
        if laser.size:
            self._laser(laser, side)
            self.t_cool[laser, side] = LASER_COOLDOWN
        mine = rows[w == W_MINE]
        if mine.size:
            self._place_mine(mine, side)
            self.t_cool[mine, side] = MINE_COOLDOWN

    # 给 side 座位埋一颗地雷。优先复用"自己已经埋下的那颗雷"（一个人同时只留一颗），
    # 自己没有雷时才去占一个空槽。
    def _place_mine(self, rows, side):
        if rows.size == 0:
            return
        mine_slot = self.m_owner[rows] == side
        empty = ~self.m_on[rows]
        has = mine_slot.any(axis=1) | empty.any(axis=1)   # 两种槽位都找不到的局这次就不埋了
        r = rows[has]
        if r.size == 0:
            return
        mine_slot = self.m_owner[r] == side
        empty = ~self.m_on[r]
        slot = np.where(mine_slot.any(axis=1), np.argmax(mine_slot, axis=1), np.argmax(empty, axis=1))
        self.m_on[r, slot] = True
        self.m_x[r, slot] = self.tx[r, side]     # 就埋在自己当前脚下
        self.m_y[r, slot] = self.ty[r, side]
        self.m_arm[r, slot] = MINE_ARM           # 刚埋下要先"上保险"一段时间才生效，防止立刻炸到自己
        self.m_owner[r, slot] = side

    # 激光是"瞬间命中"，不做飞行。对每辆坦克都算一下"从炮口沿朝向射出去的射线
    # 多久能打到它"，取最近的那一辆——因为激光会被前面的坦克挡住。
    def _laser(self, rows, side):
        mx, my = self._muzzle(rows, side)
        ca = np.cos(self.ta[rows, side])
        sa = np.sin(self.ta[rows, side])
        best = np.full(rows.size, 1e9, dtype=np.float32)
        dists = []   # 保留对每个座位的命中距离，后面用它决定谁挨打
        for j in range(self.p):
            d = _ray_square(mx, my, ca, sa, self.tx[rows, j], self.ty[rows, j], self.ta[rows, j], TANK_H)
            # 已阵亡的坦克不算遮挡（距离置成一个大值），只有活着的人才参与"最近"的比较。
            d = np.where(self.t_alive[rows, j] & (d >= 0.0), d, 1e9)
            dists.append(d)
            best = np.minimum(best, d)
        far = best >= 1e8
        # 谁都没打到时，把线段画到 4000 长，纯粹为了让渲染出的激光不断线。
        end_x = mx + ca * np.where(far, 4000.0, best)
        end_y = my + sa * np.where(far, 4000.0, best)
        # 存进激光槽位，供渲染端画这条线段（LASER_TTL 很短，一闪即消）。
        empty = ~self.l_on[rows]
        has = empty.any(axis=1)
        r = rows[has]
        if r.size:
            m = has
            slot = np.argmax(empty[m], axis=1)
            self.l_on[r, slot] = True
            self.l_xy[r, slot, 0] = mx[m]
            self.l_xy[r, slot, 1] = my[m]
            self.l_xy[r, slot, 2] = end_x[m]
            self.l_xy[r, slot, 3] = end_y[m]
            self.l_ttl[r, slot] = LASER_TTL
        for j in range(self.p):
            hit = dists[j] < 1e8   # 距离被算出来的就是被射线打到的
            if hit.any():
                self._damage(rows[hit], j, side)   # killer 就是开激光的 side

    # 让 rows 这些局里的 side 座位受到致命伤（本函数只处理"死亡"，不处理扣血）。
    # killer 有两种传法：
    #   - 整数/标量：全场同一个来源（如爆炸、激光）；
    #   - 数组：每个局各自的来源（如子弹，每颗子弹的归属可能不同，-1 表示无主）。
    def _damage(self, rows, side, killer):
        if rows.size == 0:
            return
        shielded = self.t_weapon[rows, side] == W_SHIELD
        if shielded.any():
            # 护盾能挡下一次致死，挡完立刻碎掉（武器打回普通）。
            r = rows[shielded]
            self.t_weapon[r, side] = W_NORMAL
            self.t_wt[r, side] = 0.0
            # 重要：killer 是数组时必须按同一个掩码一起过滤。
            # 因为下面 rows 已经被筛掉了护盾生效的那些元素，如果 killer 不同步筛，
            # 它就会和 rows 错位、长度也对不上，后面的 killer==side 等判断会全错。
            if not isinstance(killer, (int, np.integer)):
                killer = np.asarray(killer)[~shielded]
            rows = rows[~shielded]
            if rows.size == 0:
                return
        self.t_alive[rows, side] = False
        self.suicide[rows, side] |= np.asarray(killer == side)
        if isinstance(killer, (int, np.integer)):
            # 标量来源：杀的不是自己就把积分记给 killer；如果是自己（比如被自己的地雷炸死），
            # 按规则算是"送分"，给其他所有座位各加一分。
            if killer != side:
                self.t_score[rows, killer] += 1
                self.kills[rows, killer] += 1
                self.ep_kills[rows, killer] += 1
            else:
                for j in range(self.p):
                    if j != side:
                        self.t_score[rows, j] += 1
                        self.ep_kills[rows, j] += 1
        else:
            # 数组来源：逐个座位归因。
            other = (killer >= 0) & (killer != side)   # 有主（非 -1）、且不是自己
            if other.any():
                r = rows[other]
                k = killer[other]
                for j in range(self.p):
                    m = k == j
                    if m.any():
                        self.t_score[r[m], j] += 1
                        self.kills[r[m], j] += 1
                        self.ep_kills[r[m], j] += 1
            mine = ~other   # 无主(killer<0) 或自杀：同样按"送分"处理，给其他所有座位加分
            if mine.any():
                r = rows[mine]
                for j in range(self.p):
                    if j != side:
                        self.t_score[r, j] += 1
                        self.ep_kills[r, j] += 1

    # 产生一次爆炸。两种用法：
    #   shards=False（地雷）：爆炸本体直接结算伤害——对半径 rad 内的存活坦克造成伤害；
    #   shards=True （破片弹撞墙炸开）：本体不结算伤害，改为放出 SHARD_COUNT 个碎片，
    #                 靠碎片弹来弹去再伤人（所以这时爆炸只是视觉表现 + 碎片的来源）。
    # owner 是这次爆炸/碎片的归属玩家，命中时用它来归因击杀。
    def _blast(self, rows, x, y, rad, owner, shards):
        empty = ~self.z_on[rows]
        has = empty.any(axis=1)
        r = rows[has]
        if r.size:
            slot = np.argmax(empty[has], axis=1)
            m = has
            self.z_on[r, slot] = True
            self.z_x[r, slot] = x[m]
            self.z_y[r, slot] = y[m]
            self.z_r[r, slot] = rad[m]
            self.z_ttl[r, slot] = BLAST_TTL
        if shards:
            for k in range(SHARD_COUNT):
                # 碎片按圆周均匀分布：第 k 个碎片的角度是 2πk/SHARD_COUNT，形成一圈散射。
                ang = np.full(rows.size, 2.0 * np.pi * k / SHARD_COUNT, dtype=np.float32)
                self._emit(rows, owner, ang, SHARD_SPEED, K_SHARD, SHARD_BOUNCE, SHARD_LIFE)
            return
        for j in range(self.p):
            # 命中判定用"圆心距 <= 爆炸半径"，即把坦克车心当成一个点来比较（够用且便宜）。
            live = self.t_alive[rows, j]
            near = live & ((x - self.tx[rows, j]) ** 2 + (y - self.ty[rows, j]) ** 2 <= rad ** 2)
            if near.any():
                self._damage(rows[near], j, owner)

    # 每帧处理：道具胶囊的刷新 / 存活时间 / 拾取，地雷的保险与起爆，
    # 以及爆炸、激光这些短暂特效的剩余时间（TTL）。
    def _items(self, dt, run):
        self.spawn_t = self.spawn_t - dt
        want = run & (self.spawn_t <= 0.0)
        self.spawn_t = np.where(want, CAPSULE_EVERY, self.spawn_t)   # 到点的局把刷新计时器重置
        try_spawn = want & (self.c_on.sum(axis=1) < CAPSULE_MAX)     # 场上胶囊数没到上限才刷
        if try_spawn.any():
            self._spawn_capsule(np.nonzero(try_spawn)[0])
        self.c_ttl = self.c_ttl - dt
        self.c_on &= self.c_ttl > 0.0   # 到期的胶囊自动消失
        for j in range(self.p):
            # 拾取判定：用"胶囊半径 + 半车宽"当阈值，相当于把圆(胶囊)与方(坦克)近似成圆-圆。
            live = self.t_alive[:, j] & run
            rows = np.nonzero(live)[0]
            if rows.size == 0:
                continue
            dx = self.c_x[rows] - self.tx[rows, j][:, None]
            dy = self.c_y[rows] - self.ty[rows, j][:, None]
            near = self.c_on[rows] & (dx * dx + dy * dy <= (CAPSULE_R + TANK_H) ** 2)
            if near.any():
                r, c = np.nonzero(near)
                # 捡到后：把胶囊种类下标 +1 得到武器编号，并按武器持续时长设 t_wt。
                self.t_weapon[rows[r], j] = self.c_kind[rows[r], c] + 1
                self.t_wt[rows[r], j] = WEAPON_SECONDS[self.c_kind[rows[r], c] + 1]
                self.c_on[rows[r], c] = False
                self.picked[rows[r], j] = True
        self.m_arm = self.m_arm - dt
        armed = self.m_on & (self.m_arm <= 0.0)   # 只有保险时间已过的地雷才会被踩爆
        blown = np.zeros_like(self.m_on)
        for j in range(self.p):
            # 踩雷判定：阈值用 MINE_R + TANK_H*1.42。1.42≈√2，即把边长 2*TANK_H 的方形坦克
            # 近似成它外接圆（半径 √2*TANK_H），这样贴边擦过也能触发，避免明明压上去却不炸。
            live = self.t_alive[:, j] & run
            rows = np.nonzero(live)[0]
            if rows.size == 0:
                continue
            dx = self.m_x[rows] - self.tx[rows, j][:, None]
            dy = self.m_y[rows] - self.ty[rows, j][:, None]
            touch = armed[rows] & (dx * dx + dy * dy <= (MINE_R + TANK_H * 1.42) ** 2)
            if touch.any():
                # 先记下"被踩到的地雷"，避免边遍历边修改 m_on 造成混乱。
                r, c = np.nonzero(touch)
                blown[rows[r], c] = True
        if blown.any():
            r, c = np.nonzero(blown)
            self.m_on[r, c] = False
            self._blast(r, self.m_x[r, c].copy(), self.m_y[r, c].copy(),
                        np.full(r.size, BLAST_R, dtype=np.float32), self.m_owner[r, c].copy(), False)
        self.z_ttl = self.z_ttl - dt
        self.z_on &= self.z_ttl > 0.0
        self.l_ttl = self.l_ttl - dt
        self.l_on &= self.l_ttl > 0.0

    # 随机挑一个格子刷新道具，要求离所有坦克、以及已有的道具都足够远(> 2 格)。
    # 同样是拒绝采样：最多试 30 次，实在找不到合适的点就放弃这一轮刷新。
    def _spawn_capsule(self, rows):
        for _ in range(30):
            cell = self.rng.integers(0, NC * NR, size=rows.size)
            x = (WALL_W + (cell // NR) * PITCH + CORR_W * 0.5).astype(np.float32)
            y = (WALL_W + (cell % NR) * PITCH + CORR_W * 0.5).astype(np.float32)
            bad = np.zeros(rows.size, dtype=bool)
            for j in range(self.p):
                bad |= (self.tx[rows, j] - x) ** 2 + (self.ty[rows, j] - y) ** 2 < (PITCH * 2.0) ** 2
            for q in range(C_MAX):
                bad |= self.c_on[rows, q] & ((self.c_x[rows, q] - x) ** 2
                                             + (self.c_y[rows, q] - y) ** 2 < (PITCH * 2.0) ** 2)
            room = ~self.c_on[rows].all(axis=1)   # 该局还有空槽位可放
            good = (~bad) & room
            if not good.any():
                return
            r = rows[good]
            slot = np.argmax(~self.c_on[r], axis=1)
            self.c_on[r, slot] = True
            self.c_x[r, slot] = x[good]
            self.c_y[r, slot] = y[good]
            self.c_ttl[r, slot] = CAPSULE_TTL
            self.c_kind[r, slot] = self.rng.integers(0, len(WEAPON_KINDS), size=r.size)   # 随机武器种类
            return

    # 追踪导弹的导引逻辑。b_life 是从满寿命往下递减的剩余秒数，所以
    # "剩余寿命 <= 总寿命 - 延迟" 就等价于"它已经飞够了 MISSILE_DELAY 秒"，
    # 从这一刻起才开始追踪；在此之前导弹沿发射方向直飞（给对手反应和躲避的机会）。
    def _home(self, dt):
        ready = self.b_on & (self.b_kind == K_MISSILE) & (self.b_life <= MISSILE_LIFE - MISSILE_DELAY)
        if not ready.any():
            return
        r, c = np.nonzero(ready)
        bx = self.b_x[r, c]
        by = self.b_y[r, c]
        best = np.full(r.size, 1e18, dtype=np.float32)
        target = np.full(r.size, -1, dtype=np.int64)
        for j in range(self.p):
            # 逐个候选目标算距离，挑最近的存活坦克当目标（已阵亡的用大值排除）。
            d = np.where(self.t_alive[r, j], (self.tx[r, j] - bx) ** 2 + (self.ty[r, j] - by) ** 2, 1e18)
            better = d < best
            best = np.where(better, d, best)
            target = np.where(better, j, target)
        has = target >= 0
        if not has.any():
            return
        r = r[has]
        c = c[has]
        tgt = target[has]
        cur = np.arctan2(self.b_vy[r, c], self.b_vx[r, c])            # 导弹当前朝向
        want = np.arctan2(self.ty[r, tgt] - self.b_y[r, c], self.tx[r, tgt] - self.b_x[r, c])   # 期望朝向
        # 取"当前朝向 -> 期望朝向"的最小夹角差（先规整到 (-π, π] 再限幅），
        # 限幅在 MISSILE_TURN*dt 之内，即每秒最多转 MISSILE_TURN 弧度：
        # 这样导弹画出的是可被躲开的追踪曲线，而不是瞬间锁定。
        diff = np.clip((want - cur + np.pi) % (2.0 * np.pi) - np.pi, -MISSILE_TURN * dt, MISSILE_TURN * dt)
        na = cur + diff
        sp = np.hypot(self.b_vx[r, c], self.b_vy[r, c])   # 只改方向、保持速度大小不变
        self.b_vx[r, c] = np.cos(na) * sp
        self.b_vy[r, c] = np.sin(na) * sp

    # 返回"本帧确实有子弹在飞"的槽位下标。子弹池固定 96 个槽，但实战平均只有
    # 个位数子弹同时在天上；所有跟子弹有关的运算都先过这一层筛选，把 96 降到 ~9，
    # 这是整个环境里省算力最狠的一刀（不改变任何物理结果，只是不再空算空槽）。
    def columns(self):
        return np.nonzero(self.b_on.any(axis=0))[0]

    # 把"卡在墙里的子弹"顶出来。理论上不该发生，但换地图、边界上的浮点误差
    # 都可能让子弹落在墙体内；卡住的子弹会一直占着槽位，还会污染观测图。
    def _unstick(self):
        rows = np.nonzero(self.b_on.any(axis=1))[0]
        if rows.size == 0:
            return
        act = self.columns()
        ix = np.ix_(rows, act)
        bx = self.b_x[ix]
        by = self.b_y[ix]
        on = self.b_on[ix]
        ang = np.arctan2(self.b_vy[ix], self.b_vx[ix])
        stuck = on & ~self.free(rows[:, None], bx, by, BULLET_R)
        for k in range(1, 11):
            if not stuck.any():
                break
            back = 1.5 * k   # 沿进入方向的反方向逐步后退搜索，最远退到 15 px
            nx = bx - np.cos(ang) * back
            ny = by - np.sin(ang) * back
            ok = self.free(rows[:, None], nx, ny, BULLET_R)
            fix = stuck & ok
            bx = np.where(fix, nx, bx)
            by = np.where(fix, ny, by)
            stuck = stuck & ~ok   # 已经找到合法位置的，从待处理集合里去掉
        self.b_x[ix] = bx
        self.b_y[ix] = by
        if stuck.any():
            self.b_on[ix] = on & ~stuck   # 实在救不回来的就直接删掉这颗子弹

    # 子弹推进的主体。把一帧拆成 SUBSTEPS 个子步，每个子步里分别处理
    # x、y 两个方向的移动与撞墙反弹——这是标准做法：先沿 x 走，撞了就翻 vx 并留在原地；
    # 再沿 y 走，撞了就翻 vy。这样斜射进墙角时两个方向都能各自正确反弹。
    def _bullets(self, dt, run):
        if not self.b_on.any():
            return
        self._unstick()   # 先把卡墙的子弹救出来，再谈推进
        self._home(dt)    # 追踪导弹要先转向再前进
        h = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            pick = self.b_on & run[:, None]     # 本子步真正要推进的 (局, 槽) 位置
            if not pick.any():
                return
            # 只取"有子弹在飞的槽位"（act）和"有子弹在飞的局"（rows），
            # 两者交叉出的那个小矩形块才是真正需要算的，其余 96 槽的空位一概跳过。
            act = np.nonzero(pick.any(axis=0))[0]
            rows = np.nonzero(pick[:, act].any(axis=1))[0]
            ix = np.ix_(rows, act)
            bx = self.b_x[ix]
            by = self.b_y[ix]
            bvx = self.b_vx[ix]
            bvy = self.b_vy[ix]
            live = pick[ix]
            nx = bx + bvx * h
            # 撞墙时位置不更新、速度取反，效果就是反弹。
            hit_x = ~self.free(rows[:, None], nx, by, BULLET_R)
            bx = np.where(hit_x, bx, nx)
            bvx = np.where(hit_x, -bvx, bvx)
            ny = by + bvy * h
            hit_y = ~self.free(rows[:, None], bx, ny, BULLET_R)
            by = np.where(hit_y, by, ny)
            bvy = np.where(hit_y, -bvy, bvy)
            self.b_x[ix] = bx
            self.b_y[ix] = by
            self.b_vx[ix] = bvx
            self.b_vy[ix] = bvy
            bounce = self.b_bounce[ix] + hit_x.astype(np.int16) + hit_y.astype(np.int16)   # 累计反弹次数
            self.b_bounce[ix] = bounce
            # 破片弹(K_FRAG)一碰墙就炸开：本体消失，原地放出一圈碎片（shards=True）。
            # 注意 rows 只是"有子弹的局"的子集，命中位置的全局局号必须写成 rows[fr]，
            # 不能直接拿块内下标 fr 去索引环境（那样会在错误的局里炸出爆炸）。
            frag = live & (self.b_kind[ix] == K_FRAG) & (hit_x | hit_y)
            if frag.any():
                fr, fc = np.nonzero(frag)
                gr = rows[fr]
                gc = act[fc]
                self.b_on[gr, gc] = False
                self._blast(gr, self.b_x[gr, gc].copy(), self.b_y[gr, gc].copy(),
                            np.full(fr.size, BLAST_R * 0.7, dtype=np.float32),
                            self.b_owner[gr, gc].copy(), True)
            # 寿命按子步扣减，保证一帧总共扣掉 dt（而不是每个子步都扣一整个 dt）。
            life = np.where(live, self.b_life[ix] - h, self.b_life[ix])
            self.b_life[ix] = life
            # 反弹次数超过 BOUNCE_LIMIT 的子弹直接消失（为什么是 7 见文件顶部说明）。
            dead = live & (bounce > BOUNCE_LIMIT)
            if dead.any():
                dr, dc = np.nonzero(dead)
                self.b_on[rows[dr], act[dc]] = False
            gone = live & (life <= 0.0)
            if gone.any():
                gr, gc = np.nonzero(gone)
                self.b_on[rows[gr], act[gc]] = False
        self._bullet_hits(run)   # 所有子步走完，再统一判定"有没有打中坦克"

    # 判定子弹是否打中坦克。子弹是圆、坦克是旋转正方形，精确的圆-方相交比较麻烦，
    # 这里改用保守近似：把子弹也当成边长 2*BULLET_R 的小方块，先把它的相对位置
    # 旋转到坦克的局部坐标系里，然后直接做轴对齐的 AABB 判定（两个方向的门限都是
    # TANK_H + BULLET_R）。判定范围略大于真实形状，但计算最省、手感上可以接受。
    def _bullet_hits(self, run):
        live = self.b_on & run[:, None]
        if not live.any():
            return
        act = np.nonzero(live.any(axis=0))[0]   # 同样只在"有子弹在飞的槽位"里找命中
        for j in range(self.p):
            rows = np.nonzero(self.t_alive[:, j] & run)[0]
            if rows.size == 0:
                continue
            ix = np.ix_(rows, act)
            ca = np.cos(self.ta[rows, j])[:, None]
            sa = np.sin(self.ta[rows, j])[:, None]
            dx = self.b_x[ix] - self.tx[rows, j][:, None]
            dy = self.b_y[ix] - self.ty[rows, j][:, None]
            # 排除破片弹本体：它只在撞墙时炸开，不会直接把坦克撞死（伤人的是它的碎片）。
            hit = live[ix] & (self.b_kind[ix] != K_FRAG)
            hit &= np.abs(dx * ca + dy * sa) <= TANK_H + BULLET_R    # 沿车头方向的偏移
            hit &= np.abs(dy * ca - dx * sa) <= TANK_H + BULLET_R    # 垂直车头方向的偏移
            if not hit.any():
                continue
            hr, hc = np.nonzero(hit)
            gr = rows[hr]
            gc = act[hc]
            self.b_on[gr, gc] = False   # 命中后子弹消失
            self._damage(gr, j, self.b_owner[gr, gc].astype(np.int64))   # 用子弹归属做击杀归因

    # 计算"与观察者视角无关"的那几个观测通道，并缓存起来：
    # 墙 + 子弹 + 地雷 + 道具 + 激光。这几样对 4 个座位来说完全相同，
    # 所以每帧只算一次，obs_pack() 里 4 个座位直接复用同一份底图，
    # 避免同一份工作重复做 4 遍。
    # 缓存（self._base）由 step() 和 _new_round() 置空失效——因为任何一帧
    # 都可能改变子弹/道具/地雷的位置，换地图更是全变了。
    def _obs_base(self):
        if self._base is not None:
            return self._base
        out = self.wall_bits.copy()
        # _dot_bit 把"点状实体"写到它所在的那个观测像素上（按位或进对应通道）。
        # 子弹只遍历"在飞的槽位"（avg ~9 个），而不是 96 个空槽——否则每帧要空转
        # 87 次 _dot_bit，白扔几百次 numpy 调用。
        for q in self.columns():
            self._dot_bit(out, BIT_BULLET, self.b_x[:, q], self.b_y[:, q], self.b_on[:, q])
        for q in range(M_MAX):
            self._dot_bit(out, BIT_MINE, self.m_x[:, q], self.m_y[:, q], self.m_on[:, q])
        for q in range(C_MAX):
            self._dot_bit(out, BIT_CAPSULE, self.c_x[:, q], self.c_y[:, q], self.c_on[:, q])
        for q in range(L_MAX):
            # 激光是一条线段，不能只点一个像素：要沿着它均匀采样一串点逐个打上去。
            # 采样步长取约一个观测像素(OBS_RES)那么长，画出来就是一条连续的线。
            on = self.l_on[:, q]
            if not on.any():
                continue
            x0 = self.l_xy[:, q, 0]
            y0 = self.l_xy[:, q, 1]
            x1 = self.l_xy[:, q, 2]
            y1 = self.l_xy[:, q, 3]
            steps = np.maximum(2, (np.hypot(x1 - x0, y1 - y0) / OBS_RES).astype(np.int64))
            for t in range(int(steps.max()) + 1):
                sel = on & (t <= steps)          # 各局线段长度不同，用 t<=steps 把多余采样点裁掉
                f = t / np.maximum(steps, 1)     # 归一化参数：0 在起点、1 在终点
                self._dot_bit(out, BIT_BEAM, x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, sel)
        self._base = out
        return out

    # 产出 side 号座位的观测，形状 (n, OBS_H, OBS_W) 的 uint8 打包位图。
    # 拿底图复制一份，再把"自机/对手"这两个随视角变化的通道画上去。
    # 关键设计：除自己外的所有对手都合并画进同一个 BIT_ENEMY 通道，
    # 所以无论 2 人还是 4 人，图像格式都固定是 8 通道——网络结构不必随
    # 对手人数变化（每个对手的个体信息改在标量特征 feat() 里给出）。
    def obs_pack(self, side):
        out = self._obs_base().copy()
        for j in range(self.p):
            self._blit_bit(out, BIT_SELF if j == side else BIT_ENEMY,
                           self.tx[:, j], self.ty[:, j], self.t_alive[:, j])
        return out

    # 把一个实体画成以它所在像素为中心的 3×3 小方块（而不是单个像素）。
    # 因为观测分辨率是 16:1 的降采样，坦克在图上只有一两个像素；
    # 只点一个像素的话它会随位置抖动而时隐时现，网络很难稳定识别，
    # 膨胀成 3×3 后"看得见"且位置稳定。
    def _blit_bit(self, out, bit, x, y, mask):
        rows = np.nonzero(mask)[0]
        if rows.size == 0:
            return
        cx = np.clip((x[rows] / OBS_RES).astype(np.int64), 0, OBS_W - 1)
        cy = np.clip((y[rows] / OBS_RES).astype(np.int64), 0, OBS_H - 1)
        for dy, dx in _DISC:
            oy = np.clip(cy + dy, 0, OBS_H - 1)   # 贴边时夹回图内，避免越界
            ox = np.clip(cx + dx, 0, OBS_W - 1)
            out[rows, oy, ox] |= np.uint8(1 << bit)   # 按位或：同像素多个物体互不影响

    # 只写单个像素的点。用于本身很小或数量很多的实体（子弹、地雷、道具、激光采样点），
    # 它们没必要占 3×3 那么大；而且点比块更能表达精确位置。
    def _dot_bit(self, out, bit, x, y, mask):
        rows = np.nonzero(mask)[0]
        if rows.size == 0:
            return
        cx = np.clip((x[rows] / OBS_RES).astype(np.int64), 0, OBS_W - 1)
        cy = np.clip((y[rows] / OBS_RES).astype(np.int64), 0, OBS_H - 1)
        out[rows, cy, cx] |= np.uint8(1 << bit)

    # 标量特征向量（p=4 时共 81 维）。打包后的图像擅长表达"空间布局"，
    # 但一些连续量（速度、剩余时间、名次）用图不好表达，就把它们单独列成向量。
    # 布局：自身 8 维 + 每个对手 7 维（4 人局是 3 个对手） + 最近 8 颗子弹各 6 维
    #       + 末尾 4 维（含回合进度）。
    def feat(self, side):
        others = [j for j in range(self.p) if j != side]
        cols = 8 + 7 * len(others) + 6 * NEAR_BULLETS + 4
        out = np.zeros((self.n, cols), dtype=np.float32)
        # ---- 自身 8 维 ----
        # 位置归一化到 0~1；朝向用 cos/sin 两个分量而不是角度值，
        # 因为角度在 ±π 处会跳变（359° 和 1° 其实很近，但数值差 358），
        # 用 cos/sin 表达方向对神经网络更友好。
        out[:, 0] = self.tx[:, side] / WIDTH
        out[:, 1] = self.ty[:, side] / HEIGHT
        out[:, 2] = np.cos(self.ta[:, side])
        out[:, 3] = np.sin(self.ta[:, side])
        out[:, 4] = self.t_alive[:, side]
        out[:, 5] = self.t_weapon[:, side] / 7.0                 # 武器编号最大为 7，除以 7 归一
        out[:, 6] = np.clip(self.t_wt[:, side] / 10.0, 0.0, 1.0)  # 武器剩余时间，按最长 10 秒归一
        out[:, 7] = self.t_cool[:, side]                          # 开火冷却剩余秒数
        b = 8
        for j in others:
            # ---- 每个对手 7 维：位置(2) + 朝向 cos/sin(2) + 存活(1) + 武器(1) + 武器剩余时间(1) ----
            out[:, b + 0] = self.tx[:, j] / WIDTH
            out[:, b + 1] = self.ty[:, j] / HEIGHT
            out[:, b + 2] = np.cos(self.ta[:, j])
            out[:, b + 3] = np.sin(self.ta[:, j])
            out[:, b + 4] = self.t_alive[:, j]
            out[:, b + 5] = self.t_weapon[:, j] / 7.0
            out[:, b + 6] = np.clip(self.t_wt[:, j] / 10.0, 0.0, 1.0)
            b += 7
        # ---- 最近 NEAR_BULLETS 颗子弹，每颗 6 维 ----
        # 先把子弹位置/速度变换到"以自机为原点、自机朝向为 x 轴"的相对坐标系，
        # 这样网络看到的是"我正前方的威胁"，与自机在地图上的绝对朝向无关，
        # 学出来的策略自然就具备旋转不变性，泛化更好。
        # 只在"在飞的槽位"（act）里排序：96 个槽里绝大多数是空的，对整个 96 列做
        # argsort 等于给 87 个 ±1e9 的假数据排序，纯属浪费。空槽对排序的贡献永远是
        # 排在最后并被 sel 过滤掉，所以只排真实候选、结果与排全表完全一致。
        act = self.columns()
        na = int(act.size)
        if na:
            bx = self.b_x[:, act]
            by = self.b_y[:, act]
            on = self.b_on[:, act]
            bkx = self.b_kind[:, act]
            bvx = self.b_vx[:, act]
            bvy = self.b_vy[:, act]
            dx = bx - self.tx[:, side][:, None]
            dy = by - self.ty[:, side][:, None]
            ref_ang = self.ta[:, side][:, None]
            ca = np.cos(ref_ang)
            sa = np.sin(ref_ang)
            rx = dx * ca + dy * sa
            ry = dy * ca - dx * sa
            dist = np.where(on, np.hypot(dx, dy), 1e9)   # 未启用的槽位距离设为极大，排序时自然落到最后
            rows = np.arange(self.n)
            pick_n = min(NEAR_BULLETS, na)   # 子弹不足 8 颗时只填前 pick_n 组，其余保持 0
            order = np.argsort(dist, axis=1)[:, :pick_n]   # 每局取最近的那些槽位
            cs = np.cos(self.ta[:, side])
            sn = np.sin(self.ta[:, side])
            for k in range(pick_n):
                c = order[:, k]
                sel = dist[rows, c] < 1e8   # 该局不足 pick_n 颗子弹时，剩下的槽位整组填 0
                out[:, b + 6 * k + 0] = np.where(sel, rx[rows, c] / 300.0, 0.0)   # 相对位置 x
                out[:, b + 6 * k + 1] = np.where(sel, ry[rows, c] / 300.0, 0.0)   # 相对位置 y
                # 相对速度同样旋到自机坐标系；除以 540（最快弹速 GATLING_SPEED）做归一。
                out[:, b + 6 * k + 2] = np.where(sel, (bvx[rows, c] * cs + bvy[rows, c] * sn) / 540.0, 0.0)
                out[:, b + 6 * k + 3] = np.where(sel, (bvy[rows, c] * cs - bvx[rows, c] * sn) / 540.0, 0.0)
                out[:, b + 6 * k + 4] = np.where(sel, bkx[rows, c] / 4.0, 0.0)               # 子弹种类，最大 4
                out[:, b + 6 * k + 5] = np.where(sel, np.minimum(dist[rows, c], 300.0) / 300.0, 0.0) # 距离，截断在 300
        b += 6 * NEAR_BULLETS
        # ---- 末尾 4 维：本局得分、死亡停顿进度、本局是否已结束、本回合时间进度 ----
        out[:, b + 0] = self.t_score[:, side] / 10.0
        out[:, b + 1] = self.pause / DEATH_PAUSE
        out[:, b + 2] = self.over
        out[:, b + 3] = np.clip(self.round_frames / ROUND_LIMIT, 0.0, 1.0)   # 0=刚开局，1=即将判平局
        return out
