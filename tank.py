import math
import random

CELL = 32
COLS = 21
ROWS = 15
WIDTH = COLS * CELL
HEIGHT = ROWS * CELL

TANK_R = 10.0
TANK_SPEED = 126.0
BULLET_R = 3.5
BULLET_SPEED = 340.0
BULLET_LIFE = 4.0
BOUNCE_LIMIT = 6
FIRE_COOLDOWN = 0.34
MAX_BULLETS = 5

DIR8 = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
MOVES = ((0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
MOVE_NAMES = ("停", "右", "右下", "下", "左下", "左", "左上", "上", "右上")
ACTIONS = 18
ACTION_NAMES = tuple(MOVE_NAMES[i % 9] + ("+开火" if i >= 9 else "") for i in range(ACTIONS))
ACTION_CTRL = tuple((MOVES[i % 9][0], MOVES[i % 9][1], 1 if i >= 9 else 0) for i in range(ACTIONS))


def generate(seed=None, braid=0.35):
    rng = random.Random(seed)
    g = [["#"] * COLS for _ in range(ROWS)]
    g[1][1] = "."
    stack = [(1, 1)]
    while stack:
        r, c = stack[-1]
        nbrs = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr = r + dr
            nc = c + dc
            if 1 <= nr <= ROWS - 2 and 1 <= nc <= COLS - 2 and g[nr][nc] == "#":
                nbrs.append((nr, nc, r + dr // 2, c + dc // 2))
        if not nbrs:
            stack.pop()
            continue
        nr, nc, mr, mc = rng.choice(nbrs)
        g[mr][mc] = "."
        g[nr][nc] = "."
        stack.append((nr, nc))
    for r in range(1, ROWS - 1):
        for c in range(1, COLS - 1):
            if g[r][c] == "#" and ((r % 2 == 1) != (c % 2 == 1)) and rng.random() < braid:
                g[r][c] = "."
    return tuple("".join(row) for row in g)


class Tank:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.alive = True
        self.cool = 0.0
        self.score = 0
        self.shots = 0
        self.death = None


class Bullet:
    def __init__(self, x, y, vx, vy, owner):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.owner = owner
        self.bounce = 0
        self.life = BULLET_LIFE


class World:
    def __init__(self, seed=None):
        self.grid = generate(seed)
        self.tanks = []
        self.bullets = []
        self.time = 0.0
        self.over = False
        self.winner = None
        self.seed = seed
        self.reset(seed)

    def reset(self, seed=None):
        if seed is None:
            seed = random.randrange(1 << 30)
        self.seed = seed
        self.grid = generate(seed)
        rng = random.Random(seed)
        cells = [
            (r, c)
            for r in range(1, ROWS - 1)
            for c in range(1, COLS - 1)
            if self.grid[r][c] == "."
        ]
        best = None
        for _ in range(80):
            a = rng.choice(cells)
            b = rng.choice(cells)
            d = abs(a[0] - b[0]) + abs(a[1] - b[1])
            if best is None or d > best[0]:
                best = (d, a, b)
        _, a, b = best
        self.tanks = [
            Tank((a[1] + 0.5) * CELL, (a[0] + 0.5) * CELL, 0.0),
            Tank((b[1] + 0.5) * CELL, (b[0] + 0.5) * CELL, 0.0),
        ]
        for i, t in enumerate(self.tanks):
            o = self.tanks[1 - i]
            t.angle = math.atan2(o.y - t.y, o.x - t.x)
        self.bullets = []
        self.time = 0.0
        self.over = False
        self.winner = None
        return self

    def _solid(self, r, c):
        if r < 0 or r >= ROWS or c < 0 or c >= COLS:
            return True
        return self.grid[r][c] == "#"

    def _circle_hit(self, x, y, rad):
        c0 = int((x - rad) // CELL)
        c1 = int((x + rad) // CELL)
        r0 = int((y - rad) // CELL)
        r1 = int((y + rad) // CELL)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if not self._solid(r, c):
                    continue
                nx = min(max(x, c * CELL), c * CELL + CELL)
                ny = min(max(y, r * CELL), r * CELL + CELL)
                if (x - nx) ** 2 + (y - ny) ** 2 < rad * rad:
                    return True
        return False

    def _push_out(self, x, y, rad):
        for _ in range(4):
            moved = False
            c0 = int((x - rad) // CELL)
            c1 = int((x + rad) // CELL)
            r0 = int((y - rad) // CELL)
            r1 = int((y + rad) // CELL)
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if not self._solid(r, c):
                        continue
                    rx = c * CELL
                    ry = r * CELL
                    nx = min(max(x, rx), rx + CELL)
                    ny = min(max(y, ry), ry + CELL)
                    dx = x - nx
                    dy = y - ny
                    d2 = dx * dx + dy * dy
                    if d2 >= rad * rad:
                        continue
                    moved = True
                    if d2 > 1e-8:
                        d = math.sqrt(d2)
                        k = (rad - d) / d
                        x += dx * k
                        y += dy * k
                    else:
                        left = x - rx
                        right = rx + CELL - x
                        top = y - ry
                        bottom = ry + CELL - y
                        m = min(left, right, top, bottom)
                        if m == left:
                            x = rx - rad
                        elif m == right:
                            x = rx + CELL + rad
                        elif m == top:
                            y = ry - rad
                        else:
                            y = ry + CELL + rad
            if not moved:
                break
        return x, y

    def _fire(self, t):
        if t.cool > 0.0:
            return False
        n = 0
        for b in self.bullets:
            if b.owner is t:
                n += 1
        if n >= MAX_BULLETS:
            return False
        bx = t.x + math.cos(t.angle) * (TANK_R + BULLET_R + 1.0)
        by = t.y + math.sin(t.angle) * (TANK_R + BULLET_R + 1.0)
        if self._circle_hit(bx, by, 1.5):
            return False
        t.cool = FIRE_COOLDOWN
        t.shots += 1
        self.bullets.append(
            Bullet(bx, by, math.cos(t.angle) * BULLET_SPEED, math.sin(t.angle) * BULLET_SPEED, t)
        )
        return True

    def _drive(self, t, ctrl, dt):
        dx, dy, fire = ctrl
        if dx or dy:
            n = math.hypot(dx, dy)
            t.angle = math.atan2(dy, dx)
            nx = t.x + dx / n * TANK_SPEED * dt
            ny = t.y + dy / n * TANK_SPEED * dt
            t.x, t.y = self._push_out(nx, ny, TANK_R)
        t.cool = max(0.0, t.cool - dt)
        if fire:
            self._fire(t)

    def _fly(self, b, dt):
        steps = int(BULLET_SPEED * dt / 4.0) + 1
        h = dt / steps
        for _ in range(steps):
            nx = b.x + b.vx * h
            if self._circle_hit(nx, b.y, BULLET_R):
                b.vx = -b.vx
                b.bounce += 1
                b.x, b.y = self._push_out(b.x, b.y, BULLET_R)
            else:
                b.x = nx
            ny = b.y + b.vy * h
            if self._circle_hit(b.x, ny, BULLET_R):
                b.vy = -b.vy
                b.bounce += 1
                b.x, b.y = self._push_out(b.x, b.y, BULLET_R)
            else:
                b.y = ny
            if b.bounce > BOUNCE_LIMIT:
                return False
        b.life -= dt
        return b.life > 0.0

    def clear_ahead(self, x, y, angle, limit):
        d = TANK_R + BULLET_R + 1.0
        while d < limit:
            if self._circle_hit(x + math.cos(angle) * d, y + math.sin(angle) * d, BULLET_R):
                return d
            d += 4.0
        return limit

    def step(self, dt, ctrl0, ctrl1):
        if self.over:
            return
        self.time += dt
        for t, ctrl in ((self.tanks[0], ctrl0), (self.tanks[1], ctrl1)):
            if t.alive:
                self._drive(t, ctrl, dt)
        keep = []
        for b in self.bullets:
            if not self._fly(b, dt):
                continue
            hit = False
            for t in self.tanks:
                if not t.alive:
                    continue
                if (b.x - t.x) ** 2 + (b.y - t.y) ** 2 <= (BULLET_R + TANK_R) ** 2:
                    t.alive = False
                    other = self.tanks[1] if t is self.tanks[0] else self.tanks[0]
                    if b.owner is t:
                        t.death = "被自己的子弹弹死"
                        other.score += 1
                    else:
                        t.death = "被对手打中"
                        b.owner.score += 1
                    hit = True
                    break
            if not hit:
                keep.append(b)
        self.bullets = keep
        a0 = self.tanks[0].alive
        a1 = self.tanks[1].alive
        if not a0 or not a1:
            self.over = True
            if a0 and not a1:
                self.winner = 0
            elif a1 and not a0:
                self.winner = 1
            else:
                self.winner = None

    def step_actions(self, dt, a0, a1):
        self.step(dt, ACTION_CTRL[a0], ACTION_CTRL[a1])


class Bot:
    def __init__(self, skill=0.6, seed=None):
        self.rng = random.Random(seed)
        self.skill = float(skill)
        self.think = 0.0
        self.way = None

    def _los(self, world, me, foe):
        d = math.hypot(foe.x - me.x, foe.y - me.y)
        n = int(d // 6) + 1
        for i in range(1, n):
            x = me.x + (foe.x - me.x) * i / n
            y = me.y + (foe.y - me.y) * i / n
            if world._solid(int(y // CELL), int(x // CELL)):
                return False
        return True

    def _next_cell(self, world, me, foe):
        start = (int(me.y // CELL), int(me.x // CELL))
        goal = (int(foe.y // CELL), int(foe.x // CELL))
        if world._solid(*start) or world._solid(*goal):
            return None
        prev = {start: None}
        queue = [start]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            if cur == goal:
                break
            r, c = cur
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if n in prev or world._solid(*n):
                    continue
                prev[n] = cur
                queue.append(n)
        if goal not in prev:
            return None
        node = goal
        while prev[node] is not None and prev[prev[node]] is not None:
            node = prev[node]
        return ((node[1] + 0.5) * CELL, (node[0] + 0.5) * CELL)

    def _dir_index(self, dx, dy):
        return int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8

    def _aim_hit(self, world, me, foe, angle):
        cx = math.cos(angle)
        cy = math.sin(angle)
        d = TANK_R + BULLET_R + 1.0
        while d < 420.0:
            px = me.x + cx * d
            py = me.y + cy * d
            if world._circle_hit(px, py, BULLET_R):
                return False
            if (px - foe.x) ** 2 + (py - foe.y) ** 2 <= (TANK_R + BULLET_R) ** 2:
                return True
            d += 6.0
        return False

    def act(self, world, me, foe, dt):
        if not me.alive or not foe.alive:
            return (0, 0, 0)
        self.think -= dt
        if self.think <= 0.0:
            self.think = self.rng.uniform(0.10, 0.22)
            self.way = self._next_cell(world, me, foe)
        seen = self._los(world, me, foe)
        dist = math.hypot(foe.x - me.x, foe.y - me.y)
        dx = 0
        dy = 0
        if seen:
            idx = self._dir_index(foe.x - me.x, foe.y - me.y)
            dx, dy = DIR8[idx]
        elif self.way is not None:
            tx, ty = self.way
            if math.hypot(tx - me.x, ty - me.y) < 7.0:
                self.think = 0.0
            if abs(tx - me.x) > 5.0:
                dx = 1 if tx > me.x else -1
            if abs(ty - me.y) > 5.0:
                dy = 1 if ty > me.y else -1
        if dx == 0 and dy == 0:
            dx, dy = DIR8[self.rng.randrange(8)]
        fire = 0
        if seen:
            mine = math.atan2(dy, dx)
            safe = world.clear_ahead(me.x, me.y, mine, 54.0) >= 54.0
            if safe and self._aim_hit(world, me, foe, mine):
                if self.rng.random() < 0.30 + 0.6 * self.skill:
                    fire = 1
            elif safe:
                aim = math.atan2(foe.y - me.y, foe.x - me.x)
                d2 = abs((aim - mine + math.pi) % (2 * math.pi) - math.pi)
                tol = math.atan2(14.0 + 90.0 * (1.0 - self.skill), max(40.0, dist))
                if d2 < tol and self.rng.random() < 0.04 + 0.10 * self.skill:
                    fire = 1
        return (dx, dy, fire)
