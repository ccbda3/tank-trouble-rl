import argparse
import sys

import numpy as np

import tank
from tank_vec import B_MAX, C_MAX, L_MAX, M_MAX, NC, NR, WEAPON_KINDS, Z_MAX

SHM_NAME = "tank_view_state"
STATS_NAME = "tank_view_stats"
VIEW_ENVS = 4

DTYPE = np.dtype([
    ("round", np.int32),
    ("step", np.int32),
    ("score", np.int32, (3,)),
    ("vwall", np.uint8, (NC + 1, NR)),
    ("hwall", np.uint8, (NR + 1, NC)),
    ("tank", np.float32, (3, 6)),
    ("bullet", np.float32, (B_MAX, 5)),
    ("mine", np.float32, (M_MAX, 4)),
    ("capsule", np.float32, (C_MAX, 4)),
    ("beam", np.float32, (L_MAX, 5)),
    ("blast", np.float32, (Z_MAX, 5)),
])

STATS = np.dtype([
    ("loss", np.float32),
    ("rew", np.float32),
    ("eps", np.float32),
    ("spm", np.float32),
    ("buf", np.float32),
    ("win", np.float32),
    ("steps", np.float32),
    ("eval", np.float32, (5,)),
])


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


class Writer:
    def __init__(self, envs=VIEW_ENVS):
        self.envs = envs
        self.state_shm = _attach_or_create(SHM_NAME, DTYPE.itemsize * envs)
        self.stats_shm = _attach_or_create(STATS_NAME, STATS.itemsize)
        self.state = np.ndarray((envs,), dtype=DTYPE, buffer=self.state_shm.buf)
        self.stat = np.ndarray((1,), dtype=STATS, buffer=self.stats_shm.buf)
        self.state["round"] = -1

    def push(self, env, upto=None):
        upto = self.envs if upto is None else upto
        for i in range(upto):
            p = env.p
            s = self.state[i]
            s["round"] = env.rounds[i]
            s["step"] = env.ep_steps[i]
            s["score"][:p] = env.t_score[i]
            s["vwall"] = env.vwall[i]
            s["hwall"] = env.hwall[i]
            s["tank"][:p, 0] = env.tx[i]
            s["tank"][:p, 1] = env.ty[i]
            s["tank"][:p, 2] = env.ta[i]
            s["tank"][:p, 3] = env.t_alive[i]
            s["tank"][:p, 4] = env.t_weapon[i]
            s["tank"][:p, 5] = env.t_wt[i]
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
            s["beam"][:, 0:4] = env.l_xy[i]
            s["beam"][:, 4] = env.l_ttl[i]
            s["blast"][:, 0] = env.z_x[i]
            s["blast"][:, 1] = env.z_y[i]
            s["blast"][:, 2] = env.z_r[i]
            s["blast"][:, 3] = env.z_ttl[i]
            s["blast"][:, 4] = env.z_on[i]

    def stats(self, loss, rew, eps, spm, buf, win, steps, evals):
        s = self.stat[0]
        s["loss"] = loss
        s["rew"] = rew
        s["eps"] = eps
        s["spm"] = spm
        s["buf"] = buf
        s["win"] = win
        s["steps"] = steps
        s["eval"][:len(evals)] = evals

    def close(self):
        self.state_shm.close()
        self.stats_shm.close()


def viewer(env_index=0):
    import pygame
    from multiprocessing import shared_memory
    try:
        state_shm = shared_memory.SharedMemory(name=SHM_NAME)
        stats_shm = shared_memory.SharedMemory(name=STATS_NAME)
    except FileNotFoundError:
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
    colors = [(214, 64, 64), (64, 122, 224), (72, 190, 100)]
    item_color = {
        "gatling": (120, 226, 255), "missile": (176, 132, 255), "frag": (255, 156, 74),
        "double": (255, 226, 106), "laser": (110, 226, 255), "shield": (140, 172, 255),
        "mine": (214, 96, 160),
    }
    wall_surf = None
    wall_key = None
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            if e.type == pygame.KEYDOWN and pygame.K_1 <= e.key <= pygame.K_4:
                env_index = e.key - pygame.K_1
        s = state[env_index]
        key = (int(s["round"]), env_index)
        if key != wall_key:
            wall_key = key
            wall_surf = pygame.Surface((W, H))
            wall_surf.fill((0, 0, 0))
            P, WW = tank.PITCH, tank.WALL_W
            for i in range(NC + 1):
                for j in range(NR):
                    if s["vwall"][i, j]:
                        pygame.draw.rect(wall_surf, (255, 255, 255),
                                         (i * P - WW, j * P, WW, P))
            for j in range(NR + 1):
                for i in range(NC):
                    if s["hwall"][j, i]:
                        pygame.draw.rect(wall_surf, (255, 255, 255),
                                         (i * P, j * P - WW, P, WW))

        screen.fill((0, 0, 0))
        screen.blit(wall_surf, (0, 0))
        for q in range(Z_MAX):
            if s["blast"][q, 4] > 0:
                k = float(s["blast"][q, 3]) / 0.5
                r = int(float(s["blast"][q, 2]) * max(0.0, k))
                if r > 1:
                    pygame.draw.circle(screen, (255, 240, 200),
                                       (int(s["blast"][q, 0]), int(s["blast"][q, 1])), r, 2)
        for q in range(L_MAX):
            if s["beam"][q, 4] > 0:
                pygame.draw.line(screen, (255, 255, 255),
                                 (int(s["beam"][q, 0]), int(s["beam"][q, 1])),
                                 (int(s["beam"][q, 2]), int(s["beam"][q, 3])), 3)
        for q in range(C_MAX):
            if s["capsule"][q, 3] > 0:
                kind = WEAPON_KINDS[int(s["capsule"][q, 2])]
                pygame.draw.circle(screen, (18, 18, 18),
                                   (int(s["capsule"][q, 0]), int(s["capsule"][q, 1])), int(tank.CAPSULE_R))
                pygame.draw.circle(screen, item_color[kind],
                                   (int(s["capsule"][q, 0]), int(s["capsule"][q, 1])), int(tank.CAPSULE_R), 2)
        for q in range(M_MAX):
            if s["mine"][q, 2] > 0:
                pygame.draw.circle(screen, (150, 60, 110),
                                   (int(s["mine"][q, 0]), int(s["mine"][q, 1])), int(tank.MINE_R))
        for q in range(B_MAX):
            if s["bullet"][q, 4] > 0:
                r = 2 if s["bullet"][q, 2] in (0.0, 1.0) else 3
                pygame.draw.circle(screen, (255, 255, 255),
                                   (int(s["bullet"][q, 0]), int(s["bullet"][q, 1])), r)
        for i in range(3):
            if s["tank"][i, 3] <= 0:
                continue
            x, y, a = float(s["tank"][i, 0]), float(s["tank"][i, 1]), float(s["tank"][i, 2])
            ca, sa = np.cos(a), np.sin(a)
            h = tank.TANK_H
            pts = [(x + ca * h - sa * h, y + sa * h + ca * h), (x + ca * h + sa * h, y + sa * h - ca * h),
                   (x - ca * h + sa * h, y - sa * h - ca * h), (x - ca * h - sa * h, y - sa * h + ca * h)]
            pygame.draw.polygon(screen, colors[i], pts)
            pygame.draw.polygon(screen, (255, 255, 255), pts, 1)
            bx = x + ca * (h + tank.BARREL_LEN)
            by = y + sa * (h + tank.BARREL_LEN)
            pygame.draw.line(screen, (255, 255, 255), (x, y), (bx, by), 5)
            w = int(s["tank"][i, 4])
            if w == 6:
                pygame.draw.circle(screen, (200, 220, 255), (int(x), int(y)), int(h + 4), 1)
            elif w > 0:
                pygame.draw.circle(screen, item_color[WEAPON_KINDS[w - 1]], (int(x), int(y)), int(h + 3), 1)
        st = stat[0]
        pygame.draw.rect(screen, (14, 14, 18), (0, H, W, HUD))
        w = int(s["tank"][0, 4])
        line1 = "对局 %d   比分 %d : %d   回合 %d   武器 %s" % (
            env_index + 1, int(s["score"][0]), int(s["score"][1]), int(s["round"]),
            WEAPON_KINDS[w - 1] if w > 0 else "普通弹")
        screen.blit(font.render(line1, True, (230, 230, 235)), (10, H + 4))
        screen.blit(font.render("1-4 切台   ESC 退出", True, (140, 140, 150)), (560, H + 4))
        names = ("随机", "脚本1", "脚本2", "脚本3", "快照")
        line2 = "ε %.2f   当前胜率 %.0f%%   评测: %s" % (
            float(st["eps"]), 100.0 * float(st["win"]),
            " ".join("%s%.0f%%" % (n, 100.0 * float(v)) for n, v in zip(names, st["eval"])))
        screen.blit(font.render(line2, True, (150, 190, 235)), (10, H + 26))
        line3 = "已训 %d 步   %.0f 步/秒   回放池 %.0fk   损失 %.4f" % (
            int(st["steps"]), float(st["spm"]), float(st["buf"]), float(st["loss"]))
        screen.blit(font.render(line3, True, (150, 150, 160)), (10, H + 48))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    state_shm.close()
    stats_shm.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=int, default=0)
    a = ap.parse_args()
    sys.exit(viewer(a.env))
