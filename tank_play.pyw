import math
import sys

import pygame

import tank
from tank import CELL, COLS, HEIGHT, ROWS, TANK_R, WIDTH, Bot, World, read_ctrl

HUD = 74
DT = 1.0 / 60.0
BAR = (22, 24, 31)
FLOOR = (34, 37, 45)
WALL = (86, 93, 110)
WALL_TOP = (122, 131, 152)
WALL_BOT = (56, 62, 76)
P1_COLOR = (92, 186, 255)
P2_COLOR = (255, 108, 96)
SHOT = (255, 216, 104)
TEXT = (212, 218, 230)
DIM = (134, 142, 160)
POWER_COLOR = {
    "laser": (110, 226, 255),
    "bomb": (255, 156, 74),
    "speed": (255, 226, 106),
    "shield": (140, 172, 255),
}
BULLET_COLOR = {"normal": SHOT, "laser": (110, 226, 255), "bomb": (255, 156, 74)}

def make_font(size):
    for name in ("microsoftyahei", "msyh", "simhei", "simsun"):
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


def read_drive(keys, keyset):
    return read_ctrl(keys.__getitem__, keyset)


def draw_world():
    screen.fill(FLOOR)
    grid = world.grid
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] != "#":
                continue
            x = c * CELL
            y = r * CELL
            pygame.draw.rect(screen, WALL, (x, y, CELL, CELL))
            pygame.draw.rect(screen, WALL_TOP, (x, y, CELL, 3))
            pygame.draw.rect(screen, WALL_BOT, (x, y + CELL - 3, CELL, 3))


def draw_tank(t, color):
    if not t.alive:
        return
    cx = int(t.x)
    cy = int(t.y)
    ex = int(t.x + math.cos(t.angle) * (TANK_R + 8))
    ey = int(t.y + math.sin(t.angle) * (TANK_R + 8))
    pygame.draw.line(screen, (228, 233, 242), (cx, cy), (ex, ey), 7)
    pygame.draw.circle(screen, color, (cx, cy), int(TANK_R))
    pygame.draw.circle(screen, (14, 16, 21), (cx, cy), int(TANK_R), 2)
    pygame.draw.circle(screen, (228, 233, 242), (cx, cy), 4)
    pygame.draw.circle(screen, (14, 16, 21), (cx, cy), 4, 1)
    if t.shield:
        pygame.draw.circle(screen, POWER_COLOR["shield"], (cx, cy), int(TANK_R) + 5, 2)
    if t.boost > 0.0:
        pygame.draw.circle(screen, POWER_COLOR["speed"], (cx, cy), int(TANK_R) + 8, 2)
    if t.ammo_left > 0:
        acol = POWER_COLOR[t.ammo_kind]
        pygame.draw.circle(
            screen,
            acol,
            (int(cx - math.cos(t.angle) * 5), int(cy - math.sin(t.angle) * 5)),
            3,
        )


def draw_bullets():
    for b in world.bullets:
        col = BULLET_COLOR[b.kind]
        cx = int(b.x)
        cy = int(b.y)
        if b.kind == "laser":
            pygame.draw.line(
                screen, col, (int(b.x - b.vx * 0.018), int(b.y - b.vy * 0.018)), (cx, cy), 3
            )
        r = 5 if b.kind == "bomb" else 4
        pygame.draw.circle(screen, col, (cx, cy), r)
        pygame.draw.circle(screen, (255, 252, 232), (cx, cy), r, 1)


def draw_powerups():
    for p in world.powerups:
        if p.ttl < 3.5 and int(p.ttl * 6) % 2 == 0:
            continue
        col = POWER_COLOR[p.kind]
        cx = int(p.x)
        cy = int(p.y)
        pygame.draw.circle(screen, (18, 20, 26), (cx, cy), 10)
        pygame.draw.circle(screen, col, (cx, cy), 10, 2)
        if p.kind == "laser":
            pygame.draw.line(screen, col, (cx - 6, cy), (cx + 6, cy), 3)
        elif p.kind == "bomb":
            pygame.draw.circle(screen, col, (cx, cy), 4)
        elif p.kind == "speed":
            pygame.draw.polygon(screen, col, [(cx - 5, cy - 4), (cx + 5, cy), (cx - 5, cy + 4)])
        else:
            pygame.draw.circle(screen, col, (cx, cy), 6, 2)


def draw_blasts():
    for x, y, rad, ttl in world.blasts:
        k = max(0.0, ttl / tank.BLAST_TTL)
        rr = max(2, int(rad * (1.35 - 0.55 * k)))
        surf = pygame.Surface((rr * 2, rr * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (255, 176, 92, int(150 * k)), (rr, rr), rr)
        pygame.draw.circle(surf, (255, 240, 200, int(235 * k)), (rr, rr), rr, 3)
        screen.blit(surf, (int(x) - rr, int(y) - rr))


def draw_hud():
    pygame.draw.rect(screen, BAR, (0, HEIGHT, WIDTH, HUD))
    pygame.draw.line(screen, (48, 52, 64), (0, HEIGHT), (WIDTH, HEIGHT))
    screen.blit(font_big.render("P1", True, P1_COLOR), (12, HEIGHT + 4))
    screen.blit(font_big.render("%d" % tally[0], True, TEXT), (52, HEIGHT + 4))
    screen.blit(font_big.render(":", True, DIM), (76, HEIGHT + 4))
    screen.blit(font_big.render("%d" % tally[1], True, TEXT), (90, HEIGHT + 4))
    screen.blit(font_big.render("P2", True, P2_COLOR), (122, HEIGHT + 4))
    roles = "机器人 Lv%d" % bot_level if bot_on else "双人同屏"
    screen.blit(font_small.render(roles, True, DIM), (176, HEIGHT + 9))
    if world.over:
        msg = "平局" if world.winner is None else "P%d 胜" % (world.winner + 1)
        color = DIM if world.winner is None else (P1_COLOR if world.winner == 0 else P2_COLOR)
        screen.blit(font_big.render(msg, True, color), (WIDTH - 96, HEIGHT + 4))
    screen.blit(font_tip.render(TIP1, True, TEXT), (12, HEIGHT + 31))
    screen.blit(font_tip.render(TIP2, True, DIM), (12, HEIGHT + 31 + tip_h + 3))


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + HUD))
pygame.display.set_caption("坦克动荡 · 手动试玩")
clock = pygame.time.Clock()
font_big = make_font(21)
font_small = make_font(15)

TIP1 = "你(P2)：← → 原地转向     ↑ ↓ 前进后退     空格 / M / 回车 开火"
TIP2 = "P1：A D 转向   W S 前进后退   Q 开火          B 换人    R 重开    N 换迷宫    ESC 退出"
tip_size = 15
while tip_size > 9 and max(make_font(tip_size).size(TIP1)[0], make_font(tip_size).size(TIP2)[0]) > WIDTH - 24:
    tip_size -= 1
font_tip = make_font(tip_size)
tip_h = font_tip.get_height()

world = World()
bot_on = True
bot_level = 2
bot = Bot(0.05 + 0.25 * bot_level, seed=7)
tally = [0, 0]
restart_at = None
running = True

KEYCODE = {
    "w": pygame.K_w,
    "a": pygame.K_a,
    "s": pygame.K_s,
    "d": pygame.K_d,
    "q": pygame.K_q,
    "up": pygame.K_UP,
    "down": pygame.K_DOWN,
    "left": pygame.K_LEFT,
    "right": pygame.K_RIGHT,
    "m": pygame.K_m,
    "return": pygame.K_RETURN,
    "space": pygame.K_SPACE,
}


def bind_keys(keyset):
    return {name: tuple(KEYCODE[k] for k in keyset[name]) for name in keyset}


KEYS_P1 = bind_keys(tank.P1_KEYMAP)
KEYS_P2 = bind_keys(tank.P2_KEYMAP)

while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                world.reset(world.seed)
                restart_at = None
            elif event.key == pygame.K_n:
                world.reset()
                restart_at = None
            elif event.key == pygame.K_b:
                bot_on = not bot_on
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                bot_level = event.key - pygame.K_0
                bot = Bot(0.05 + 0.25 * bot_level, seed=7)
                bot_on = True
    keys = pygame.key.get_pressed()
    c0 = bot.act(world, world.tanks[0], world.tanks[1], DT) if bot_on else read_drive(keys, KEYS_P1)
    c1 = read_drive(keys, KEYS_P2)
    was_over = world.over
    world.step(DT, c0, c1)
    if world.over and not was_over:
        if world.winner is not None:
            tally[world.winner] += 1
        restart_at = pygame.time.get_ticks() + 1800
    if world.over and restart_at is not None and pygame.time.get_ticks() >= restart_at:
        world.reset()
        restart_at = None
    draw_world()
    draw_powerups()
    draw_blasts()
    draw_bullets()
    draw_tank(world.tanks[0], P1_COLOR)
    draw_tank(world.tanks[1], P2_COLOR)
    draw_hud()
    pygame.display.flip()

pygame.quit()
sys.exit(0)
