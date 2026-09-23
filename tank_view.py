# =============================================================================
# tank_view.py —— 观战窗口（pygame），提供两种互不干扰的模式
#
# 模式一 Viewer / viewer()：共享内存观战
#   训练进程（selfplay.py 带 --view）通过 multiprocessing.shared_memory 把
#   前 VIEW_ENVS=4 局的状态写进一块固定结构（DTYPE）的 numpy 数组，本脚本只把
#   它读出来、按 60 帧渲染。缺陷是刷新率被训练迭代率绑死：几百个并行环境时
#   训练进程每秒只推几次状态，画面看起来一直在跳帧。
#
# 模式二 follow()：跟随观战（本文件的主要用法）
#   本地起 4 局模拟，每 2 秒按修改时间检查一次检查点文件，拉到新权重就用真实
#   60 帧/秒推演，能看到当前策略到底在做什么。
#   按 1-4 切台，按 M 把 0 号座位换成脚本 Lv3 当对手。
#   推演永远用训练当时的 ε（实时从训练进程的统计量里读），所以画面就是训练现场：
#   探索噪声照常掺进去，能看到"训练时它到底在做什么、为什么会自杀"。
#
# 数据流：
#   tank_vec.VecEnv 的字段 → _fill() 抄进 DTYPE 结构化数组
#   → draw_scene() / draw_pause_overlay() 用 pygame 画出来
#   共享内存模式下这条链的前半段跑在训练进程里，即本文件的 Writer 类。
# =============================================================================

import argparse
import sys

import numpy as np

import tank
import tank_vec
from tank_vec import B_MAX, C_MAX, L_MAX, M_MAX, NC, NR, ROUND_LIMIT, WEAPON_KINDS, Z_MAX

# 两块共享内存的名字：一块放局面状态，一块放训练统计量（HUD 用）
SHM_NAME = "tank_view_state"
STATS_NAME = "tank_view_stats"
# 共享内存里固定放 4 局（训练进程有几百局，但只挑前 4 局写出来，观战只看这几局）
VIEW_ENVS = 4
PLAYERS_MAX = 4
# 4 个座位 4 种颜色，索引就是座位号，和 DTYPE 里 score/tank 的座位顺序一一对应
COLORS = [(214, 64, 64), (64, 122, 224), (72, 190, 100), (232, 186, 72)]

# 共享内存里一局状态的布局（结构化 dtype）。
# 用结构化 dtype 而不是 JSON/pickle 的原因：训练进程和观战进程直接共享同一块
# 内存，零拷贝、零序列化，观战不会拖慢训练。
# 字段含义：
#   round  回合号（回合结束就自增）
#   maze   迷宫序号（真正换迷宫时才自增，见 follow() 里的墙壁缓存注释）
#   winner 胜者座位号，-1 表示超时平局
#   pause  死亡停顿剩余秒数，> 0 时该局处于「已结束但还没重开」的停帧状态
#   step   本次训练迭代累计的步数，rframe 本回合已走的帧数
#   score  各座位比分
#   vwall/hwall 竖墙、横墙的存在标志，按 tank 里的格点编号索引，用来重建墙图
#   tank   各座位 (x, y, 朝向, 是否存活, 武器编号, 武器剩余时间)
#   bullet/mine/capsule/beam/blast 场上物件，各列含义对应 _fill() 的写入顺序
DTYPE = np.dtype([
    ("round", np.int32),
    ("maze", np.int32),
    ("winner", np.int8),
    ("pause", np.float32),
    ("step", np.int32),
    ("rframe", np.int32),
    ("score", np.int32, (PLAYERS_MAX,)),
    ("vwall", np.uint8, (NC + 1, NR)),
    ("hwall", np.uint8, (NR + 1, NC)),
    ("tank", np.float32, (PLAYERS_MAX, 6)),
    ("bullet", np.float32, (B_MAX, 5)),
    ("mine", np.float32, (M_MAX, 4)),
    ("capsule", np.float32, (C_MAX, 4)),
    ("beam", np.float32, (L_MAX, 5)),
    ("blast", np.float32, (Z_MAX, 5)),
])

# 训练进程推给观战窗口的统计量。
# loss/rew 训练损失与平均奖励，eps 当前探索率，spm 步/秒，buf 回放池千条数，
# place 平均名次，draw 平局率，span 平均回合秒数，rpm 局/分，steps 累计训练步数，
# eval 是各评测基准（随机、脚本1/2/3、快照）上的胜率。
STATS = np.dtype([
    ("loss", np.float32),
    ("rew", np.float32),
    ("eps", np.float32),
    ("spm", np.float32),
    ("buf", np.float32),
    ("place", np.float32),
    ("draw", np.float32),
    ("span", np.float32),
    ("rpm", np.float32),
    ("steps", np.float32),
    ("eval", np.float32, (5,)),
])

# 道具配色表，键就是 tank.WEAPON_KINDS 里的名字，观战里按名字取色
ITEM_COLOR = {
    "gatling": (120, 226, 255), "missile": (176, 132, 255), "frag": (255, 156, 74),
    "double": (255, 226, 106), "laser": (110, 226, 255), "shield": (140, 172, 255),
    "mine": (214, 96, 160),
}


# 把 VecEnv 里第 i 局的状态抄进一行结构化数据。
# 这是「训练侧 / 观战侧」唯一的桥：env 内部字段名（tx、b_on 之类）只在这里出现一次，
# 观战代码之后只认 DTYPE 的字段名，两边改动互不影响。
def _fill(s, env, i):
    # p 是本局的座位数，用变量而不是写死 4：座位数变了这里不用改
    p = env.p
    s["round"] = env.rounds[i]
    s["maze"] = env.maze_seq[i]
    s["winner"] = env.winner[i]
    s["pause"] = env.pause[i]
    s["step"] = env.ep_steps[i]
    s["rframe"] = env.round_frames[i]
    # 各座位数组只填前 p 个，避免把预留的槽位写脏
    s["score"][:p] = env.t_score[i]
    s["vwall"] = env.vwall[i]
    s["hwall"] = env.hwall[i]
    s["tank"][:p, 0] = env.tx[i]
    s["tank"][:p, 1] = env.ty[i]
    s["tank"][:p, 2] = env.ta[i]
    s["tank"][:p, 3] = env.t_alive[i]
    s["tank"][:p, 4] = env.t_weapon[i]
    s["tank"][:p, 5] = env.t_wt[i]
    # 子弹/地雷/道具/激光/爆炸这些物件是定长数组，不在场的槽位靠 b_on/m_on 之类的
    # 标志位判断，所以全部照抄、由绘制端过滤
    s["bullet"][:, 0] = env.b_x[i]
    s["bullet"][:, 1] = env.b_y[i]
    s["bullet"][:, 2] = env.b_kind[i]
    s["bullet"][:, 3] = env.b_life[i]
    s["bullet"][:, 4] = env.b_on[i]
    s["mine"][:, 0] = env.m_x[i]
    s["mine"][:, 1] = env.m_y[i]
    s["mine"][:, 2] = env.m_on[i]
    s["mine"][:, 3] = env.m_arm[i]
    s["capsule"][:, 0] = env.c_x[i]
    s["capsule"][:, 1] = env.c_y[i]
    s["capsule"][:, 2] = env.c_kind[i]
    s["capsule"][:, 3] = env.c_on[i]
    # 激光的 l_xy 是 (x1, y1, x2, y2) 四个数，直接用切片整体拷
    s["beam"][:, 0:4] = env.l_xy[i]
    s["beam"][:, 4] = env.l_ttl[i]
    s["blast"][:, 0] = env.z_x[i]
    s["blast"][:, 1] = env.z_y[i]
    s["blast"][:, 2] = env.z_r[i]
    s["blast"][:, 3] = env.z_ttl[i]
    s["blast"][:, 4] = env.z_on[i]


# 把白墙画到一张离屏 Surface 上。
# 墙的数量是几十条，每帧重画要几十次 draw.rect，所以画一次缓存起来复用，
# 只在换迷宫时重建（见 follow()/viewer() 里的 wall_key 判断）。
# 格点索引换算成像素用的是和物理一致的 PITCH / WALL_W，保证画面和碰撞判定对齐。
def build_walls(s):
    import pygame
    W, H = int(tank.WIDTH), int(tank.HEIGHT)
    surf = pygame.Surface((W, H))
    surf.fill((0, 0, 0))
    P, WW = tank.PITCH, tank.WALL_W
    for i in range(NC + 1):
        for j in range(NR):
            if s["vwall"][i, j]:
                # 竖墙位于格点 i 的左侧：x = i * P - WW
                pygame.draw.rect(surf, (255, 255, 255), (i * P - WW, j * P, WW, P))
    for j in range(NR + 1):
        for i in range(NC):
            if s["hwall"][j, i]:
                # 横墙位于格点 j 的上方：y = j * P - WW
                pygame.draw.rect(surf, (255, 255, 255), (i * P, j * P - WW, P, WW))
    return surf


# 每帧重画整局：先铺墙图，再按「视觉遮挡顺序」画各类物件。
# 绘制顺序（爆炸 → 激光 → 道具 → 地雷 → 子弹 → 坦克）本身没有物理含义，
# 只是让坦克压在最上层、看得清位置。
# 坐标用的是世界像素坐标，tank 里 y 轴向下、pygame 也是，所以不需要翻转。
def draw_scene(screen, s, wall_surf=None, p=PLAYERS_MAX):
    import pygame
    screen.fill((0, 0, 0))
    # 没传墙图就现画一张（墙图在调用方缓存）
    if wall_surf is None:
        wall_surf = build_walls(s)
    screen.blit(wall_surf, (0, 0))
    colors = COLORS
    for q in range(Z_MAX):
        if s["blast"][q, 4] > 0:
            # 爆炸圈半径随剩余存活时间线性收缩（tank.BLAST_TTL = 0.5 秒为满半径），
            # 这是纯视觉表现，物理判定用的始终是 z_r
            k = float(s["blast"][q, 3]) / 0.5
            r = int(float(s["blast"][q, 2]) * max(0.0, k))
            if r > 1:
                pygame.draw.circle(screen, (255, 240, 200),
                                   (int(s["blast"][q, 0]), int(s["blast"][q, 1])), r, 2)
    # 激光画成一条细线：(x1, y1) 到 (x2, y2)
    for q in range(L_MAX):
        if s["beam"][q, 4] > 0:
            pygame.draw.line(screen, (255, 255, 255),
                             (int(s["beam"][q, 0]), int(s["beam"][q, 1])),
                             (int(s["beam"][q, 2]), int(s["beam"][q, 3])), 3)
    # 道具：先铺深色底色再描一圈彩色边，让空心圈在深色背景上更醒目
    for q in range(C_MAX):
        if s["capsule"][q, 3] > 0:
            kind = WEAPON_KINDS[int(s["capsule"][q, 2])]
            pygame.draw.circle(screen, (18, 18, 18),
                               (int(s["capsule"][q, 0]), int(s["capsule"][q, 1])),
                               int(tank.CAPSULE_R))
            pygame.draw.circle(screen, ITEM_COLOR[kind],
                               (int(s["capsule"][q, 0]), int(s["capsule"][q, 1])),
                               int(tank.CAPSULE_R), 2)
    for q in range(M_MAX):
        if s["mine"][q, 2] > 0:
            pygame.draw.circle(screen, (150, 60, 110),
                               (int(s["mine"][q, 0]), int(s["mine"][q, 1])), int(tank.MINE_R))
    for q in range(B_MAX):
        if s["bullet"][q, 4] > 0:
            # 普通直飞弹和加特林弹（kind 0、1）画 2 像素，其余弹种画 3 像素，便于分辨
            r = 2 if s["bullet"][q, 2] in (0.0, 1.0) else 3
            pygame.draw.circle(screen, (255, 255, 255),
                               (int(s["bullet"][q, 0]), int(s["bullet"][q, 1])), r)
    for i in range(p):
        # 死掉的坦克直接跳过（tank[:, 3] 是存活标记）
        if s["tank"][i, 3] <= 0:
            continue
        x, y, a = float(s["tank"][i, 0]), float(s["tank"][i, 1]), float(s["tank"][i, 2])
        # 机体画成以 h = TANK_H 为半宽、绕朝向 a 旋转的正方形：
        # 四个顶点 = 中心 + 旋转后的 (±h, ±h)，这样不用贴图也能看出朝向
        ca, sa = np.cos(a), np.sin(a)
        h = tank.TANK_H
        pts = [(x + ca * h - sa * h, y + sa * h + ca * h), (x + ca * h + sa * h, y + sa * h - ca * h),
               (x - ca * h + sa * h, y - sa * h - ca * h), (x - ca * h - sa * h, y - sa * h + ca * h)]
        pygame.draw.polygon(screen, colors[i], pts)
        pygame.draw.polygon(screen, (255, 255, 255), pts, 1)
        # 炮管从中心沿朝向画到机体外 BARREL_LEN 处，长度和物理上的炮口位置一致
        bx = x + ca * (h + tank.BARREL_LEN)
        by = y + sa * (h + tank.BARREL_LEN)
        pygame.draw.line(screen, (255, 255, 255), (x, y), (bx, by), 5)
        # 武器标记：6 号武器是护盾，单独用淡蓝圈；其它武器用各自的颜色描圈；
        # 0 表示只有普通弹，不画标记
        w = int(s["tank"][i, 4])
        if w == 6:
            pygame.draw.circle(screen, (200, 220, 255), (int(x), int(y)), int(h + 4), 1)
        elif w > 0:
            pygame.draw.circle(screen, ITEM_COLOR[WEAPON_KINDS[w - 1]], (int(x), int(y)), int(h + 3), 1)
    return screen


# 死亡停顿（tank.DEATH_PAUSE，约 0.7 秒）期间的提示层。
# 这段时间物理不再推进，如果什么都不画，画面完全静止、看起来像程序卡死；
# 压一层半透明黑幕再写清楚「谁赢了、还有多久重开」，观战体验完全不同。
def draw_pause_overlay(screen, s, font):
    import pygame
    left = float(s["pause"])
    if left <= 0.0:
        return
    W, H = int(tank.WIDTH), int(tank.HEIGHT)
    veil = pygame.Surface((W, H))
    veil.set_alpha(150)
    veil.fill((0, 0, 0))
    screen.blit(veil, (0, 0))
    w = int(s["winner"])
    if w >= 0:
        # winner 存的是座位下标，显示时 +1 变成 P1~P4
        head = "P%d 存活到最后" % (w + 1)
        col = COLORS[w % len(COLORS)]
    else:
        head = "本回合平局（超时）"
        col = (200, 200, 200)
    # pause 同时也是倒计时剩余秒数，直接拿来显示
    lines = [head, "%.1f 秒后换新迷宫、全员重生" % left]
    y = H // 2 - 34
    for k, txt in enumerate(lines):
        surf = font.render(txt, True, col if k == 0 else (170, 170, 180))
        # 居中放置：以文字宽度算左边界
        screen.blit(surf, ((W - surf.get_width()) // 2, y))
        y += 30


# 把一局渲染成 RGB 数组（不弹窗口），供需要图像的调用方使用（比如训练侧抓图）。
# pygame 的 surfarray 是 (W, H, 3)，转成常规的 (H, W, 3) 后 copy 出来，
# copy 是必须的：不复制的话返回的是 Surface 的内存视图，Surface 一释放就成野指针。
def render_rgb(env, i=0):
    import pygame
    s = np.zeros((1,), dtype=DTYPE)
    _fill(s[0], env, i)
    surf = pygame.Surface((int(tank.WIDTH), int(tank.HEIGHT)))
    draw_scene(surf, s[0])
    return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)).copy()


# 挂上已有的共享内存块，没有就创建一块。
# 需要这个函数是因为训练进程和观战窗口谁先启动都有可能：
# 先连上；连到的旧块尺寸不够（改了 DTYPE 或局数）就销毁重建；
# create 时撞上 FileExistsError 是并发的另一个进程刚建好，回头挂上即可。
def _attach_or_create(name, size):
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name=name)
        if shm.size >= size:
            return shm
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass
    try:
        return shared_memory.SharedMemory(create=True, size=size, name=name)
    except FileExistsError:
        return shared_memory.SharedMemory(name=name)


# 训练进程侧的共享内存写端（selfplay.py 带 --view 时才创建）。
# 两块内存分别对应状态 DTYPE 和统计量 STATS，各用一个 np.ndarray 视图包住缓冲区，
# 写入即直接落到共享内存，没有序列化开销。
class Writer:
    def __init__(self, envs=VIEW_ENVS):
        self.envs = envs
        self.state_shm = _attach_or_create(SHM_NAME, DTYPE.itemsize * envs)
        self.stats_shm = _attach_or_create(STATS_NAME, STATS.itemsize)
        self.state = np.ndarray((envs,), dtype=DTYPE, buffer=self.state_shm.buf)
        self.stat = np.ndarray((1,), dtype=STATS, buffer=self.stats_shm.buf)
        # 回合号置 -1 作为「还没写过数据」的哨兵值，观战侧据此能区分空数据
        self.state["round"] = -1

    # 把前 upto 局抄进共享内存（默认只写 VIEW_ENVS=4 局）。
    # 只写 4 局是刻意的：观战只看这几局，写几百局纯属浪费训练时间。
    def push(self, env, upto=None):
        upto = self.envs if upto is None else upto
        for i in range(upto):
            _fill(self.state[i], env, i)

    def stats(self, loss, rew, eps, spm, buf, place, draw, span, rpm, steps, evals):
        s = self.stat[0]
        s["loss"] = loss
        s["rew"] = rew
        s["eps"] = eps
        s["spm"] = spm
        s["buf"] = buf
        s["place"] = place
        s["draw"] = draw
        s["span"] = span
        s["rpm"] = rpm
        s["steps"] = steps
        s["eval"][:len(evals)] = evals

    # 只解除本进程的映射，不 unlink：内存块要留给还在运行的观战进程用，
    # 真正负责回收的是创建方（这里是训练进程，退出时由操作系统/后续启动逻辑处理）。
    def close(self):
        self.state_shm.close()
        self.stats_shm.close()


# 跟随观战：本地推演 4 局，周期性拉取最新检查点权重。
# 和 viewer() 的区别在于「权重的来源」——viewer 看的是训练进程推过来的状态，
# 受训练速度限制；follow 自己用显卡以 60 帧/秒跑，看到的是当前策略真实的行为。
def follow(envs=VIEW_ENVS, seed=20240923):
    import os
    import time

    import pygame
    import torch

    import config
    import network
    import selfplay

    # 本地环境：envs 局 × config.PLAYERS 个座位；固定 seed 便于复现某个观察到的行为
    env = tank_vec.VecEnv(envs, config.PLAYERS, seed=seed)
    envs_p = env.p
    # 特征维度从环境自己问出来（当前 81），不写死，改特征时不至于崩
    feat_dim = env.feat(0).shape[1]
    net = network.make(feat_dim)
    # 推理模式：这里只前向，没有 BatchNorm/Dropout，但保持和训练侧一致的调用习惯
    net.eval()
    # 动作表：(局数, 座位数)，每个座位独立决策
    acts = np.zeros((envs, envs_p), dtype=np.int64)
    bot = None
    # mode 0 = 四个座位都用自己（对等自对弈），1 = 0 号座位换成脚本 Lv3
    mode = 0
    # snap 缓存本帧各局状态，避免同一帧里重复调用 _fill
    snap = np.zeros((envs,), dtype=DTYPE)
    ck_steps = 0
    ck_mtime = None
    last_ck = 0.0

    # 尝试挂上训练进程的统计量共享内存；单独跑 follow（没有训练进程）时是 None
    stat = None
    stat_shm = None
    try:
        from multiprocessing import shared_memory
        stat_shm = shared_memory.SharedMemory(name=STATS_NAME)
        stat = np.ndarray((1,), dtype=STATS, buffer=stat_shm.buf)
    except (FileNotFoundError, ValueError):
        stat = None

    pygame.init()
    W, H = int(tank.WIDTH), int(tank.HEIGHT)
    # 画面下方留 74 像素的 HUD 条显示文字
    HUD = 74
    screen = pygame.display.set_mode((W, H + HUD))
    pygame.display.set_caption("坦克动荡 · 跟随训练观战")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Microsoft YaHei", 17)
    # 每局一张墙图缓存 + 对应的缓存键（见下面换迷宫的判断）
    wall_surf = [None] * envs
    wall_key = [None] * envs
    # 固定步长 1/60：与 clock.tick(60) 对齐，保证推演速度和真实游戏一致
    # （用真实耗时当 dt 的话，掉帧时物理会变慢、行为看着就不像游戏了）
    dt = 1.0 / 60.0
    env_index = 0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            # 1-4 切台
            if e.type == pygame.KEYDOWN and pygame.K_1 <= e.key <= pygame.K_4:
                env_index = min(envs - 1, e.key - pygame.K_1)
            # M 切换 0 号座位的对手；脚本机器人只在第一次按下时才创建（创建有开销）
            if e.type == pygame.KEYDOWN and e.key == pygame.K_m:
                mode = 1 - mode
                if mode == 1 and bot is None:
                    bot = selfplay.ScriptedBot(envs, 3, 11)

        # 每 2 秒最多检查一次检查点：torch.load 是磁盘 IO + 反序列化，
        # 每帧都查会明显掉帧，所以用 last_ck 限流；再用 mtime 判断文件是否真的变过，
        # 没变过就不做无谓的加载。
        now = time.time()
        if now - last_ck >= 2.0:
            last_ck = now
            try:
                m = os.path.getmtime(config.CHECKPOINT)
            except OSError:
                m = None
            if m is not None and m != ck_mtime:
                ck_mtime = m
                try:
                    # 训练进程边跑边写这个文件，可能正好读到半截文件而抛异常，
                    # 所以包一层 try：这次读失败就把 mtime 置回 None，下轮重试。
                    ck = torch.load(config.CHECKPOINT, map_location=network.DEVICE)
                    net.load_state_dict(ck["model"])
                    net.eval()
                    ck_steps = int(ck.get("steps", 0))
                except Exception:
                    ck_mtime = None

        # 永远按训练时的 ε 推演，看到的就是训练现场的真实画面
        # （ε 实时从训练进程的统计量里读；没有训练进程时退化成最终值 0.05）
        eps_show = float(stat[0]["eps"]) if stat is not None else 0.05
        with torch.no_grad():
            # 四个座位合成一次前向：同一张网、同一个 ε，合批不改变任何决策，
            # 但把每帧 4 次小前向压成 1 次，少和训练进程抢 GPU。
            # obs_pack(j)/feat(j) 是第 j 个座位自己视角的信息（视线遮挡等都在这里体现）
            pk = np.stack([env.obs_pack(j) for j in range(envs_p)])
            ft = np.stack([env.feat(j) for j in range(envs_p)])
            q = net.act(network.obs_tensor(pk.reshape(-1, pk.shape[2], pk.shape[3])),
                        network.feat_tensor(ft.reshape(-1, ft.shape[2])), eps_show)
            acts[:] = q.cpu().numpy().reshape(envs_p, envs).T
            # 用脚本机器人覆盖 0 号座位：其余座位仍然是网络，形成「网络 vs 脚本」的对比
            if mode == 1 and bot is not None:
                acts[:, 0] = bot.act(env, 0, [k for k in range(1, envs_p)])
        # 动作编号查表还原成 (转向, 油门, 开火) 三通道控制量，再交给环境步进
        ctrl = np.zeros((envs, envs_p, 3), dtype=np.float32)
        for j in range(envs_p):
            ctrl[:, j] = tank_vec.CTRL[acts[:, j]]
        env.step(ctrl, dt)

        for i in range(envs):
            _fill(snap[i], env, i)
        for i in range(envs):
            # 墙图缓存键用 (迷宫序号, 局号)，不能用回合号：
            # 回合号在「回合结束」时就自增了（那会儿迷宫还没换），真正换迷宫时它反而
            # 短暂不变；拿回合号当键会导致换了迷宫之后一直画着上一张墙图。
            key = (int(snap[i]["maze"]), i)
            if key != wall_key[i]:
                wall_key[i] = key
                wall_surf[i] = build_walls(snap[i])
        s = snap[env_index]
        draw_scene(screen, s, wall_surf[env_index], envs_p)
        draw_pause_overlay(screen, s, font)
        st = None if stat is None else stat[0]
        # HUD 三行：本局概况 / 对手与探索率 / 训练统计
        pygame.draw.rect(screen, (14, 14, 18), (0, H, W, HUD))
        # rframe / ROUND_LIMIT 就是本回合的时间进度（ROUND_LIMIT=1800 帧 ≈ 30 秒）
        line1 = "跟随检查点 %d 步   对局 %d/%d   比分 %s   回合 %d   回合进度 %.0f%%" % (
            ck_steps, env_index + 1, envs, "/".join(str(int(v)) for v in s["score"]),
            int(s["round"]), 100.0 * min(1.0, int(s["rframe"]) / float(ROUND_LIMIT)))
        screen.blit(font.render(line1, True, (230, 230, 235)), (10, H + 4))
        screen.blit(font.render("1-4 切台   M 切对手   ESC 退出", True, (140, 140, 150)), (520, H + 4))
        line2 = "对手 %s   按训练时的 ε %.2f 推演（探索噪声照常掺入）   本地 60 帧/秒，2 秒拉一次最新检查点" % (
            "脚本Lv3" if mode == 1 else "自己（对等）", eps_show)
        screen.blit(font.render(line2, True, (150, 190, 235)), (10, H + 26))
        if st is None:
            line3 = "训练进程未开启，仅看当前检查点的对局"
        else:
            # eval 是 5 个基准上的胜率，和上面写 STATS 时的顺序一一对应
            names = ("随机", "脚本1", "脚本2", "脚本3", "快照")
            line3 = "训练 ε %.2f   名次 %.2f（随机 2.5）   平局 %.0f%%   回合 %.1f 秒   评测胜率: %s" % (
                float(st["eps"]), float(st["place"]), 100.0 * float(st["draw"]),
                float(st["span"]),
                " ".join("%s%.0f%%" % (n, 100.0 * float(v)) for n, v in zip(names, st["eval"])))
        screen.blit(font.render(line3, True, (150, 150, 160)), (10, H + 48))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    return 0


# 共享内存观战窗口（读端）。只读训练进程写好的状态，自己不做任何模拟，
# 所以窗口本身几乎不耗 GPU，可以随时开关。
def viewer(env_index=0):
    import pygame
    from multiprocessing import shared_memory
    try:
        state_shm = shared_memory.SharedMemory(name=SHM_NAME)
        stats_shm = shared_memory.SharedMemory(name=STATS_NAME)
    except FileNotFoundError:
        # 训练进程没带 --view 时不会有这两块内存，这里给出明确提示而不是报栈
        print("没有找到正在运行的自对弈进程，先启动训练再加 --view")
        return 1
    state = np.ndarray((VIEW_ENVS,), dtype=DTYPE, buffer=state_shm.buf)
    stat = np.ndarray((1,), dtype=STATS, buffer=stats_shm.buf)

    pygame.init()
    W, H = int(tank.WIDTH), int(tank.HEIGHT)
    HUD = 74
    screen = pygame.display.set_mode((W, H + HUD))
    pygame.display.set_caption("坦克动荡 · 自对弈观战")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Microsoft YaHei", 17)
    # 只看一局，所以墙图缓存只有一份
    wall_surf = None
    wall_key = None
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            if e.type == pygame.KEYDOWN and pygame.K_1 <= e.key <= pygame.K_4:
                env_index = e.key - pygame.K_1
        # 直接取共享内存视图，训练进程写什么这里就读到什么
        s = state[env_index]
        # 缓存键同上：用 (迷宫序号, 局号) 而不是回合号，才能正确反映「换迷宫」
        key = (int(s["maze"]), env_index)
        if key != wall_key:
            wall_key = key
            wall_surf = build_walls(s)

        draw_scene(screen, s, wall_surf)
        draw_pause_overlay(screen, s, font)
        st = stat[0]
        pygame.draw.rect(screen, (14, 14, 18), (0, H, W, HUD))
        # 0 号座位的武器编号决定 HUD 里显示哪把武器（0 = 普通弹）
        w = int(s["tank"][0, 4])
        line1 = "对局 %d   比分 %s   回合 %d   武器 %s   回合进度 %.0f%%" % (
            env_index + 1, "/".join(str(int(v)) for v in s["score"]), int(s["round"]),
            WEAPON_KINDS[w - 1] if w > 0 else "普通弹",
            100.0 * min(1.0, int(s["rframe"]) / float(ROUND_LIMIT)))
        screen.blit(font.render(line1, True, (230, 230, 235)), (10, H + 4))
        screen.blit(font.render("1-4 切台   ESC 退出", True, (140, 140, 150)), (560, H + 4))
        names = ("随机", "脚本1", "脚本2", "脚本3", "快照")
        line2 = "ε %.2f   名次 %.2f（随机 2.5）   平局 %.0f%%   回合 %.1f 秒   评测胜率: %s" % (
            float(st["eps"]), float(st["place"]), 100.0 * float(st["draw"]),
            float(st["span"]),
            " ".join("%s%.0f%%" % (n, 100.0 * float(v)) for n, v in zip(names, st["eval"])))
        screen.blit(font.render(line2, True, (150, 190, 235)), (10, H + 26))
        line3 = "已训 %d 步   %.0f 步/秒   %.0f 局/分   回放池 %.0fk   损失 %.4f" % (
            int(st["steps"]), float(st["spm"]), float(st["rpm"]), float(st["buf"]),
            float(st["loss"]))
        screen.blit(font.render(line3, True, (150, 150, 160)), (10, H + 48))
        pygame.display.flip()
        # 训练推送的速度决定实际刷新率，这里限 60 帧只是上限
        clock.tick(60)
    pygame.quit()
    state_shm.close()
    stats_shm.close()
    return 0


# 两个入口：默认起共享内存观战窗口，--follow 起本地推演的跟随观战
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--envs", type=int, default=VIEW_ENVS)
    a = ap.parse_args()
    if a.follow:
        sys.exit(follow(a.envs))
    sys.exit(viewer(a.env))
