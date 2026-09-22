import math
import random

import numpy as np

W = 720
H = 920
PADDLE_W = 110
PADDLE_H = 12
BALL_R = 9
ROWS = 50
COLS = 20
BRICK_W = 35
BRICK_H = 15
BRICK_GAP = 1
BRICK_TOP = 54
BRICK_LEFT = (W - COLS * (BRICK_W + BRICK_GAP)) // 2
PADDLE_Y = H - 46
BALL_SPEED = 7.5
COLORS = [(230, 80, 80), (230, 150, 70), (230, 210, 80), (110, 200, 100), (90, 160, 230), (170, 110, 230)]


class Breakout:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.paddle_x = W / 2
        self.bricks = [[True] * COLS for _ in range(ROWS)]
        self.score = 0
        self.spawn_ball()
        return self.state()

    def spawn_ball(self):
        angle = self.rng.uniform(-40, 40)
        rad = math.radians(angle)
        self.vx = BALL_SPEED * math.sin(rad)
        self.vy = -BALL_SPEED * math.cos(rad)
        self.ball_x = self.paddle_x
        self.ball_y = PADDLE_Y - BALL_R - 2

    def state(self):
        row_frac = [sum(row) / COLS for row in self.bricks]
        total = sum(row_frac) / ROWS
        return [
            self.paddle_x / W,
            self.ball_x / W,
            self.ball_y / H,
            self.vx / BALL_SPEED,
            self.vy / BALL_SPEED,
            *row_frac,
            total,
        ]

    def step(self, action):
        if action == 0:
            self.paddle_x = max(PADDLE_W / 2, self.paddle_x - 16)
        elif action == 1:
            self.paddle_x = min(W - PADDLE_W / 2, self.paddle_x + 16)

        bx = self.ball_x + self.vx
        by = self.ball_y + self.vy
        reward = 0.0
        done = False

        if bx < BALL_R:
            bx = BALL_R
            self.vx = abs(self.vx)
        elif bx > W - BALL_R:
            bx = W - BALL_R
            self.vx = -abs(self.vx)
        if by < BALL_R:
            by = BALL_R
            self.vy = abs(self.vy)

        hit = False
        for r in range(ROWS):
            for c in range(COLS):
                if not self.bricks[r][c]:
                    continue
                x0 = BRICK_LEFT + c * (BRICK_W + BRICK_GAP)
                y0 = BRICK_TOP + r * (BRICK_H + BRICK_GAP)
                if x0 - BALL_R < bx < x0 + BRICK_W + BALL_R and y0 - BALL_R < by < y0 + BRICK_H + BALL_R:
                    self.bricks[r][c] = False
                    self.score += 1
                    reward += 1.0
                    hit = True
                    ol = bx - (x0 - BALL_R)
                    orr = (x0 + BRICK_W + BALL_R) - bx
                    ot = by - (y0 - BALL_R)
                    ob = (y0 + BRICK_H + BALL_R) - by
                    if min(ol, orr, ot, ob) in (ol, orr):
                        self.vx = -self.vx
                    else:
                        self.vy = -self.vy
                    break
            if hit:
                break

        if self.vy > 0 and by >= PADDLE_Y - BALL_R and abs(bx - self.paddle_x) < PADDLE_W / 2 + BALL_R:
            by = PADDLE_Y - BALL_R
            rel = max(-1.0, min(1.0, (bx - self.paddle_x) / (PADDLE_W / 2)))
            self.vx = BALL_SPEED * rel * 0.7
            if abs(self.vx) < 1.0:
                self.vx = 1.0 if rel >= 0 else -1.0
            self.vy = -math.sqrt(max(1.0, BALL_SPEED * BALL_SPEED - self.vx * self.vx))

        if by > H + BALL_R:
            done = True
            reward -= 3.0
        elif not any(any(row) for row in self.bricks):
            done = True
            reward += 40.0
        else:
            reward += 0.02

        self.ball_x = bx
        self.ball_y = by
        return reward, done


def render_bw(g, size=96):
    sx = size / W
    sy = size / H
    img = np.zeros((size, size), np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            if not g.bricks[r][c]:
                continue
            x0 = BRICK_LEFT + c * (BRICK_W + BRICK_GAP)
            y0 = BRICK_TOP + r * (BRICK_H + BRICK_GAP)
            img[int(y0 * sy):int((y0 + BRICK_H) * sy) + 1, int(x0 * sx):int((x0 + BRICK_W) * sx) + 1] = 200
    px = int((g.paddle_x - PADDLE_W / 2) * sx)
    px2 = int((g.paddle_x + PADDLE_W / 2) * sx)
    py = int(PADDLE_Y * sy)
    py2 = int((PADDLE_Y + PADDLE_H) * sy)
    img[py:py2 + 1, px:px2 + 1] = 150
    bx = int(g.ball_x * sx)
    by = int(g.ball_y * sy)
    r0 = max(0, by - 1)
    r1 = min(size, by + 2)
    c0 = max(0, bx - 1)
    c1 = min(size, bx + 2)
    img[r0:r1, c0:c1] = 255
    return img
