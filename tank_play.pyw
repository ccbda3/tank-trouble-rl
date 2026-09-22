import math
import sys

import pygame

import tank
from tank import HEIGHT, TANK_COLORS, TANK_H, WIDTH, World, read_ctrl

HUD = 122
DT = 1.0 / 60.0
BAR = (0, 0, 0)
FLOOR = (0, 0, 0)
WALL = (255, 255, 255)
SHOT = (255, 255, 255)
TEXT = (238, 238, 238)
DIM = (150, 150, 150)
LINE = (70, 70, 70)


def make_font(size):
    for name in ("microsoftyahei", "msyh", "simhei", "simsun"):
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


def read_drive(keys, keyset):
    return read_ctrl(keys.__getitem__, keyset)


def shade(col, k):
    return tuple(min(255, int(c * k)) for c in col)


def rot_pts(cx, cy, ang, pts):
    ca = math.cos(ang)
    sa = math.sin(ang)
    return [
        (int(round(cx + px * ca - py * sa)), int(round(cy + px * sa + py * ca)))
        for px, py in pts
    ]


def draw_world():
    screen.fill(FLOOR)
    for x, y, w, h in world.walls:
        pygame.draw.rect(screen, WALL, (x, y, w, h))


def draw_capsules():
    for c in world.capsules:
        if c.ttl < 4.0 and int(c.ttl * 6) % 2 == 0:
            continue
        cx = int(c.x)
        cy = int(c.y)
        pygame.draw.circle(screen, (10, 10, 10), (cx, cy), 12)
        pygame.draw.circle(screen, SHOT, (cx, cy), 12, 2)
        k = c.kind
        if k == "gatling":
            for dx in (-4, 0, 4):
                pygame.draw.circle(screen, SHOT, (cx + dx, cy), 2)
        elif k == "missile":
            pygame.draw.polygon(screen, SHOT, [(cx, cy - 7), (cx + 4, cy + 2), (cx, cy + 8), (cx - 4, cy + 2)])
        elif k == "frag":
            pygame.draw.circle(screen, SHOT, (cx, cy), 3)
            for a in range(0, 8):
                ang = math.pi * a / 4.0
                pygame.draw.line(
                    screen,
                    SHOT,
                    (int(cx + math.cos(ang) * 5), int(cy + math.sin(ang) * 5)),
                    (int(cx + math.cos(ang) * 9), int(cy + math.sin(ang) * 9)),
                    2,
                )
        elif k == "double":
            pygame.draw.rect(screen, SHOT, (cx - 6, cy - 7, 5, 14))
            pygame.draw.rect(screen, SHOT, (cx + 1, cy - 7, 5, 14))
        elif k == "laser":
            pygame.draw.polygon(
                screen, SHOT,
                [(cx + 4, cy - 8), (cx - 5, cy + 1), (cx - 1, cy + 1),
                 (cx - 4, cy + 8), (cx + 5, cy - 1), (cx + 1, cy - 1)],
            )
        elif k == "shield":
            pygame.draw.polygon(
                screen, SHOT,
                [(cx, cy - 8), (cx + 7, cy - 4), (cx + 7, cy + 2),
                 (cx, cy + 8), (cx - 7, cy + 2), (cx - 7, cy - 4)],
            )
        else:
            pygame.draw.circle(screen, SHOT, (cx, cy), 3)
            for a in range(0, 8, 2):
                ang = math.pi * a / 4.0
                pygame.draw.line(
                    screen,
                    SHOT,
                    (int(cx + math.cos(ang) * 5), int(cy + math.sin(ang) * 5)),
                    (int(cx + math.cos(ang) * 9), int(cy + math.sin(ang) * 9)),
                    2,
                )
        tag = font_tag.render(tank.WEAPON_NAME[k], True, SHOT)
        tw = tag.get_width() + 10
        th = tag.get_height() + 4
        ty = cy + 15
        if ty + th > HEIGHT - 2:
            ty = cy - 15 - th
        tx = min(max(cx - tw // 2, 2), WIDTH - tw - 2)
        pygame.draw.rect(screen, (10, 10, 10), (tx, ty, tw, th))
        pygame.draw.rect(screen, LINE, (tx, ty, tw, th), 1)
        screen.blit(tag, (tx + 5, ty + 2))


def draw_mines():
    pulse = (pygame.time.get_ticks() % 1200) / 1200.0
    alpha = 26 + int(22 * abs(math.sin(pulse * math.pi)))
    for m in world.mines:
        r = int(tank.MINE_R)
        surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (255, 255, 255, alpha), (r + 1, r + 1), r)
        pygame.draw.circle(surf, (255, 255, 255, alpha + 40), (r + 1, r + 1), 2)
        screen.blit(surf, (int(m.x) - r - 1, int(m.y) - r - 1))


def draw_beams():
    for x0, y0, x1, y1, ttl in world.beams:
        a = max(0.0, ttl / tank.LASER_TTL)
        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.line(surf, (255, 255, 255, int(90 * a)), (x0, y0), (x1, y1), 11)
        pygame.draw.line(surf, (255, 255, 255, int(255 * a)), (x0, y0), (x1, y1), 4)
        screen.blit(surf, (0, 0))


def draw_blasts():
    for x, y, rad, ttl in world.blasts:
        k = max(0.0, ttl / tank.BLAST_TTL)
        rr = max(3, int(rad * (1.45 - 0.6 * k)))
        surf = pygame.Surface((rr * 2, rr * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (255, 255, 255, int(70 * k)), (rr, rr), rr)
        pygame.draw.circle(surf, (255, 255, 255, int(255 * k)), (rr, rr), rr, max(2, rr // 6))
        screen.blit(surf, (int(x) - rr, int(y) - rr))


def draw_bullets():
    for b in world.bullets:
        r = 2 if b.kind in ("gatling", "shard") else 3
        pygame.draw.circle(screen, SHOT, (int(b.x), int(b.y)), r)


def draw_tank(t):
    if not t.alive:
        return
    col = TANK_COLORS[t.side]
    edge = shade(col, 0.45)
    pygame.draw.polygon(
        screen, edge,
        rot_pts(t.x, t.y, t.angle,
                ((-4.0, -4.0), (tank.BARREL_LEN, -4.0), (tank.BARREL_LEN, 4.0), (-4.0, 4.0))),
    )
    body = ((TANK_H, TANK_H), (TANK_H, -TANK_H), (-TANK_H, -TANK_H), (-TANK_H, TANK_H))
    pygame.draw.polygon(screen, col, rot_pts(t.x, t.y, t.angle, body))
    pygame.draw.polygon(screen, edge, rot_pts(t.x, t.y, t.angle, body), 2)
    pygame.draw.circle(screen, edge, (int(t.x), int(t.y)), 3)
    if t.shield:
        r = TANK_H + 7
        surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(surf, (255, 255, 255, 60), (r + 2, r + 2), r)
        pygame.draw.circle(surf, (255, 255, 255, 210), (r + 2, r + 2), r, 2)
        screen.blit(surf, (int(t.x) - r - 2, int(t.y) - r - 2))


def weapon_state(t):
    if t.weapon == "normal":
        return ""
    return "%s %.1fs" % (tank.WEAPON_NAME[t.weapon], t.weapon_t)


def draw_hud():
    pygame.draw.rect(screen, BAR, (0, HEIGHT, WIDTH, HUD))
    pygame.draw.line(screen, LINE, (0, HEIGHT), (WIDTH, HEIGHT), 2)
    x = 12
    for t in world.tanks:
        col = TANK_COLORS[t.side]
        surf = font_big.render("P%d" % (t.side + 1), True, col)
        screen.blit(surf, (x, HEIGHT + 4))
        x += surf.get_width() + 6
        surf = font_big.render("%d" % t.score, True, TEXT)
        screen.blit(surf, (x, HEIGHT + 4))
        x += surf.get_width() + 16
    x += 8
    for t in world.tanks:
        txt = weapon_state(t)
        if not txt:
            continue
        surf = font_small.render(txt, True, TANK_COLORS[t.side])
        screen.blit(surf, (x, HEIGHT + 9))
        x += surf.get_width() + 16
    if world.over and world.dead is not None:
        surf = font_big.render("P%d 阵亡" % (world.dead + 1), True, TANK_COLORS[world.dead])
        screen.blit(surf, (WIDTH - surf.get_width() - 14, HEIGHT + 4))
    screen.blit(font_tip.render(TIP1, True, TEXT), (12, HEIGHT + 36))
    screen.blit(font_tip.render(TIP2, True, DIM), (12, HEIGHT + 36 + tip_h + 3))
    screen.blit(font_tip.render(TIP3, True, DIM), (12, HEIGHT + 36 + 2 * (tip_h + 3)))


def draw_menu():
    screen.fill((0, 0, 0))
    surf = font_menu.render("坦 克 动 荡", True, TEXT)
    screen.blit(surf, ((WIDTH - surf.get_width()) // 2, 120))
    if stage == 1:
        rows = (
            "1   单人（对战 Laika）",
            "2   双人同屏",
            "3   三人同屏（P3 用鼠标）",
            "",
            "ESC  退出",
        )
    else:
        rows = (
            "难度",
            "",
            "1   简单（Laika 慢、乱打、常自杀）",
            "2   普通（会捡道具、会瞄你、会用反弹）",
            "3   疯狂（快、预判走位、多段反弹预瞄）",
        )
    y = 240
    for row in rows:
        if row:
            surf = font_menu.render(row, True, TEXT if row[0].isdigit() else DIM)
            screen.blit(surf, ((WIDTH - surf.get_width()) // 2, y))
        y += 52
    surf = font_small.render(
        "任意坦克阵亡即本小局结束：击杀方 +1 分，0.7 秒后重生成迷宫、全员在新位置重生、道具清空",
        True, DIM,
    )
    screen.blit(surf, ((WIDTH - surf.get_width()) // 2, HEIGHT - 90))


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + HUD))
pygame.display.set_caption("坦克动荡")
clock = pygame.time.Clock()
font_big = make_font(21)
font_small = make_font(15)
font_tag = make_font(14)
font_menu = make_font(26)

TIP1 = "P1：↑ ↓ ← → 移动   M 开火        P2：W A S D 移动   Q 开火        P3：鼠标转向 + 左键开火"
TIP2 = "道具限时，捡到新的直接覆盖并重置倒计时；道具到时间自动变回普通弹。R 重开对局   ESC 回菜单"
TIP3 = "加特林 按住连发 | 追踪导弹 延迟后追最近敌车 | 破片炸弹 撞墙炸出弹片 | 双管炮 一次两发 | 激光 穿墙瞬发秒杀 | 护盾 挡一次 | 地雷 原地埋雷谁踩谁炸"
tip_size = 15
while tip_size > 9 and max(make_font(tip_size).size(t)[0] for t in (TIP1, TIP2, TIP3)) > WIDTH - 24:
    tip_size -= 1
font_tip = make_font(tip_size)
tip_h = font_tip.get_height()

KEYCODE = {
    "w": pygame.K_w, "a": pygame.K_a, "s": pygame.K_s, "d": pygame.K_d, "q": pygame.K_q,
    "up": pygame.K_UP, "down": pygame.K_DOWN, "left": pygame.K_LEFT, "right": pygame.K_RIGHT,
    "m": pygame.K_m,
}


def bind_keys(keyset):
    return {name: tuple(KEYCODE[k] for k in keyset[name]) for name in keyset}


KEYS_P1 = bind_keys(tank.P1_KEYMAP)
KEYS_P2 = bind_keys(tank.P2_KEYMAP)

stage = 1
mode = 0
level = 2
laika = None
world = None
running = True


def start(m, lv):
    global world, laika, mode, level, stage
    mode = m
    level = lv
    world = World(None, tanks=m)
    laika = tank.Laika(lv, seed=m * 31 + lv) if m == 1 else None
    stage = 0


def p3_ctrl():
    t = world.tanks[2]
    mx, my = pygame.mouse.get_pos()
    ang = math.atan2(my - t.y, mx - t.x)
    diff = (ang - t.angle + math.pi) % (2.0 * math.pi) - math.pi
    turn = 0 if abs(diff) < 0.05 else (1 if diff > 0 else -1)
    thrust = 1 if math.hypot(mx - t.x, my - t.y) > 26.0 else 0
    return (turn, thrust, 1 if pygame.mouse.get_pressed()[0] else 0)


while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if stage:
                    running = False
                else:
                    stage = 1
            elif stage == 1 and event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                if event.key == pygame.K_1:
                    stage = 2
                else:
                    start(event.key - pygame.K_0 - 1, level)
            elif stage == 2 and event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                start(1, event.key - pygame.K_0)
            elif not stage and event.key == pygame.K_r:
                world.reset()
    if stage:
        draw_menu()
        pygame.display.flip()
        continue
    keys = pygame.key.get_pressed()
    ctrls = []
    for i in range(len(world.tanks)):
        if mode == 1 and i == 1:
            ctrls.append(laika.act(world, world.tanks[i], world.tanks[0], DT))
        elif mode == 3 and i == 2:
            ctrls.append(p3_ctrl())
        else:
            ctrls.append(read_drive(keys, KEYS_P1 if i == 0 else KEYS_P2))
    world.step(DT, ctrls)
    draw_world()
    draw_mines()
    draw_capsules()
    draw_blasts()
    draw_beams()
    draw_bullets()
    for t in world.tanks:
        draw_tank(t)
    draw_hud()
    pygame.display.flip()

pygame.quit()
sys.exit(0)
