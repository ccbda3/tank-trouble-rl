import math
import sys

import pygame

import tank
from tank import HEIGHT, TANK_R, WIDTH, World, read_ctrl

HUD = 74
DT = 1.0 / 60.0
BAR = (219, 219, 214)
FLOOR = (233, 233, 229)
WALL = (74, 74, 74)
WALL_EDGE = (52, 52, 52)
P1_COLOR = (38, 38, 44)
P2_COLOR = (205, 45, 41)
TEXT = (38, 38, 44)
DIM = (118, 118, 122)
POWER_COLOR = {
    "laser": (28, 138, 218),
    "bomb": (222, 92, 26),
    "speed": (231, 183, 17),
    "shield": (26, 158, 108),
}
BULLET_COLOR = {"normal": (32, 32, 38), "laser": (24, 120, 214), "bomb": (196, 54, 20)}


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
        pygame.draw.rect(screen, WALL_EDGE, (x, y, w, h), 1)


def draw_tank(t, body):
    if not t.alive:
        return
    edge = shade(body, 0.45)
    pygame.draw.polygon(
        screen,
        edge,
        rot_pts(
            t.x,
            t.y,
            t.angle,
            ((-4.0, -4.0), (TANK_R + 11.0, -4.0), (TANK_R + 11.0, 4.0), (-4.0, 4.0)),
        ),
    )
    pygame.draw.polygon(
        screen,
        body,
        rot_pts(
            t.x,
            t.y,
            t.angle,
            ((-TANK_R, -TANK_R), (TANK_R, -TANK_R), (TANK_R, TANK_R), (-TANK_R, TANK_R)),
        ),
    )
    pygame.draw.polygon(
        screen,
        edge,
        rot_pts(
            t.x,
            t.y,
            t.angle,
            ((-TANK_R, -TANK_R), (TANK_R, -TANK_R), (TANK_R, TANK_R), (-TANK_R, TANK_R)),
        ),
        2,
    )
    pygame.draw.circle(screen, edge, (int(t.x), int(t.y)), 3)
    if t.shield:
        pygame.draw.circle(screen, POWER_COLOR["shield"], (int(t.x), int(t.y)), TANK_R + 6, 2)
    if t.boost > 0.0:
        pygame.draw.circle(screen, POWER_COLOR["speed"], (int(t.x), int(t.y)), TANK_R + 10, 2)
    if t.ammo_left > 0:
        pygame.draw.circle(
            screen,
            edge,
            (
                int(t.x - math.cos(t.angle) * 6.0),
                int(t.y - math.sin(t.angle) * 6.0),
            ),
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
        pygame.draw.circle(screen, (250, 250, 246), (cx, cy), r, 1)


def draw_powerups():
    for p in world.powerups:
        if p.ttl < 3.5 and int(p.ttl * 6) % 2 == 0:
            continue
        col = POWER_COLOR[p.kind]
        cx = int(p.x)
        cy = int(p.y)
        pygame.draw.circle(screen, (250, 250, 246), (cx, cy), 11)
        pygame.draw.circle(screen, col, (cx, cy), 11, 3)
        if p.kind == "laser":
            pygame.draw.polygon(
                screen,
                col,
                [
                    (cx + 3, cy - 7),
                    (cx - 4, cy + 1),
                    (cx - 1, cy + 1),
                    (cx - 3, cy + 7),
                    (cx + 4, cy - 1),
                    (cx + 1, cy - 1),
                ],
            )
        elif p.kind == "bomb":
            pygame.draw.circle(screen, col, (cx, cy + 2), 5)
            pygame.draw.line(screen, col, (cx + 2, cy - 2), (cx + 5, cy - 8), 2)
        elif p.kind == "speed":
            for dx in (-5, 2):
                pygame.draw.polygon(
                    screen,
                    col,
                    [(cx + dx - 2, cy - 6), (cx + dx + 4, cy), (cx + dx - 2, cy + 6)],
                )
        else:
            pygame.draw.polygon(
                screen,
                col,
                [
                    (cx, cy - 7),
                    (cx + 6, cy - 3),
                    (cx + 6, cy + 2),
                    (cx, cy + 7),
                    (cx - 6, cy + 2),
                    (cx - 6, cy - 3),
                ],
            )


def draw_blasts():
    for x, y, rad, ttl in world.blasts:
        k = max(0.0, ttl / tank.BLAST_TTL)
        rr = max(3, int(rad * (1.35 - 0.55 * k)))
        surf = pygame.Surface((rr * 2, rr * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (232, 96, 16, int(215 * k)), (rr, rr), rr)
        pygame.draw.circle(surf, (255, 196, 74, int(255 * k)), (rr, rr), max(2, rr // 3))
        pygame.draw.circle(surf, (140, 42, 6, int(235 * k)), (rr, rr), rr, 3)
        screen.blit(surf, (int(x) - rr, int(y) - rr))


def draw_hud():
    pygame.draw.rect(screen, BAR, (0, HEIGHT, WIDTH, HUD))
    pygame.draw.line(screen, WALL, (0, HEIGHT), (WIDTH, HEIGHT), 2)
    screen.blit(font_big.render("P1", True, P1_COLOR), (12, HEIGHT + 4))
    screen.blit(font_big.render("%d" % tally[0], True, TEXT), (52, HEIGHT + 4))
    screen.blit(font_big.render(":", True, DIM), (76, HEIGHT + 4))
    screen.blit(font_big.render("%d" % tally[1], True, TEXT), (90, HEIGHT + 4))
    screen.blit(font_big.render("P2", True, P2_COLOR), (122, HEIGHT + 4))
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

TIP1 = "P1：A D 原地转向   W S 前进后退   Q 开火"
TIP2 = "P2：← → 原地转向   ↑ ↓ 前进后退   空格 / M / 回车 开火      R 重开   N 换迷宫   ESC 退出"
tip_size = 15
while tip_size > 9 and max(make_font(tip_size).size(TIP1)[0], make_font(tip_size).size(TIP2)[0]) > WIDTH - 24:
    tip_size -= 1
font_tip = make_font(tip_size)
tip_h = font_tip.get_height()

world = World()
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
    keys = pygame.key.get_pressed()
    was_over = world.over
    world.step(DT, read_drive(keys, KEYS_P1), read_drive(keys, KEYS_P2))
    if world.over and not was_over:
        if world.winner is not None:
            tally[world.winner] += 1
        restart_at = pygame.time.get_ticks() + 1800
    if world.over and restart_at is not None and pygame.time.get_ticks() >= restart_at:
        world.reset()
        restart_at = None
    draw_world()
    draw_blasts()
    draw_powerups()
    draw_bullets()
    draw_tank(world.tanks[0], P1_COLOR)
    draw_tank(world.tanks[1], P2_COLOR)
    draw_hud()
    pygame.display.flip()

pygame.quit()
sys.exit(0)
