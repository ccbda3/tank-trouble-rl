import math
import random

CELL = 32
COLS = 21
ROWS = 15
WIDTH = COLS * CELL
HEIGHT = ROWS * CELL

TANK_R = 10.0
TANK_SPEED = 128.0
TANK_BACK = 82.0
TANK_TURN = 3.1
BULLET_R = 3.5
BULLET_SPEED = 340.0
BULLET_LIFE = 4.0
BOUNCE_LIMIT = 6
FIRE_COOLDOWN = 0.34
MAX_BULLETS = 5

POWERUP_KINDS = ("laser", "bomb", "speed", "shield")
POWERUP_R = 10.0
POWERUP_TTL = 14.0
POWERUP_EVERY = 6.5
POWERUP_MAX = 2
BOOST_TIME = 9.0
BOOST_MUL = 1.75
AMMO_SHOTS = 3
BLAST_R = 46.0
BLAST_TTL = 0.45

KIND_LIFE = {"normal": BULLET_LIFE, "laser": 1.1, "bomb": 1.4}
KIND_BOUNCE = {"normal": BOUNCE_LIMIT, "laser": 0, "bomb": 2}
KIND_PIERCE = {"normal": False, "laser": True, "bomb": False}
KIND_BLAST = {"normal": 0.0, "laser": 0.0, "bomb": BLAST_R}


def _ctrl_name(turn, thrust, fire):
    parts = []
    if turn < 0:
        parts.append("左转")
    elif turn > 0:
        parts.append("右转")
    if thrust > 0:
        parts.append("前进")
    elif thrust < 0:
        parts.append("后退")
    if fire:
        parts.append("开火")
    return "+".join(parts) if parts else "停"


ACTION_CTRL = tuple((t, th, f) for f in (0, 1) for th in (1, 0, -1) for t in (-1, 0, 1))
ACTION_NAMES = tuple(_ctrl_name(*c) for c in ACTION_CTRL)
ACTIONS = len(ACTION_CTRL)

P1_KEYMAP = {
    "up": ("w",),
    "down": ("s",),
    "left": ("a",),
    "right": ("d",),
    "fire": ("q",),
}
P2_KEYMAP = {
    "up": ("up",),
    "down": ("down",),
    "left": ("left",),
    "right": ("right",),
    "fire": ("m", "return", "space"),
}


def read_ctrl(down, keys):
    def held(name):
        return any(down(k) for k in keys[name])

    turn = (1 if held("right") else 0) - (1 if held("left") else 0)
    thrust = (1 if held("up") else 0) - (1 if held("down") else 0)
    return (turn, thrust, 1 if held("fire") else 0)


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
        self.shield = False
        self.boost = 0.0
        self.ammo_kind = None
        self.ammo_left = 0


class PowerUp:
    def __init__(self, x, y, kind):
        self.x = x
        self.y = y
        self.kind = kind
        self.ttl = POWERUP_TTL


class Bullet:
    def __init__(self, x, y, vx, vy, owner, kind="normal"):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.owner = owner
        self.kind = kind
        self.bounce = 0
        self.life = KIND_LIFE[kind]


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
        if rng.random() < 0.5:
            corners = ((1, 1), (ROWS - 2, COLS - 2))
        else:
            corners = ((1, COLS - 2), (ROWS - 2, 1))
        picks = []
        for cr, cc in corners:
            picks.append(min(cells, key=lambda rc: (rc[0] - cr) ** 2 + (rc[1] - cc) ** 2))
        a, b = picks
        self.rng = rng
        self.tanks = [
            Tank((a[1] + 0.5) * CELL, (a[0] + 0.5) * CELL, 0.0),
            Tank((b[1] + 0.5) * CELL, (b[0] + 0.5) * CELL, 0.0),
        ]
        for i, t in enumerate(self.tanks):
            o = self.tanks[1 - i]
            t.angle = math.atan2(o.y - t.y, o.x - t.x)
        self.bullets = []
        self.powerups = []
        self.blasts = []
        self.spawn_t = POWERUP_EVERY * 0.5
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
        kind = t.ammo_kind if t.ammo_left > 0 and t.ammo_kind else "normal"
        t.cool = FIRE_COOLDOWN
        t.shots += 1
        if kind != "normal":
            t.ammo_left -= 1
            if t.ammo_left <= 0:
                t.ammo_kind = None
        self.bullets.append(
            Bullet(
                bx,
                by,
                math.cos(t.angle) * BULLET_SPEED,
                math.sin(t.angle) * BULLET_SPEED,
                t,
                kind,
            )
        )
        return True

    def _drive(self, t, ctrl, dt):
        turn, thrust, fire = ctrl
        if turn:
            t.angle += turn * TANK_TURN * dt
        if thrust:
            v = TANK_SPEED if thrust > 0 else -TANK_BACK
            if t.boost > 0.0:
                v *= BOOST_MUL
            nx = t.x + math.cos(t.angle) * v * dt
            ny = t.y + math.sin(t.angle) * v * dt
            t.x, t.y = self._push_out(nx, ny, TANK_R)
        if t.boost > 0.0:
            t.boost = max(0.0, t.boost - dt)
        t.cool = max(0.0, t.cool - dt)
        if fire:
            self._fire(t)

    def _fly(self, b, dt):
        steps = int(BULLET_SPEED * dt / 4.0) + 1
        h = dt / steps
        pierce = KIND_PIERCE[b.kind]
        limit = KIND_BOUNCE[b.kind]
        for _ in range(steps):
            nx = b.x + b.vx * h
            if (not pierce) and self._circle_hit(nx, b.y, BULLET_R):
                b.vx = -b.vx
                b.bounce += 1
                b.x, b.y = self._push_out(b.x, b.y, BULLET_R)
            else:
                b.x = nx
            ny = b.y + b.vy * h
            if (not pierce) and self._circle_hit(b.x, ny, BULLET_R):
                b.vy = -b.vy
                b.bounce += 1
                b.x, b.y = self._push_out(b.x, b.y, BULLET_R)
            else:
                b.y = ny
            if b.bounce > limit:
                return False
            if not (0.0 <= b.x <= WIDTH and 0.0 <= b.y <= HEIGHT):
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

    def _kill(self, t, killer):
        t.alive = False
        other = self.tanks[1] if t is self.tanks[0] else self.tanks[0]
        if killer is t:
            t.death = "被自己的子弹弹死"
            other.score += 1
        else:
            t.death = "被对手打中"
            killer.score += 1

    def _damage(self, t, killer):
        if t.shield:
            t.shield = False
            return False
        self._kill(t, killer)
        return True

    def _blast(self, x, y, rad, owner):
        self.blasts.append([x, y, rad, BLAST_TTL])
        for t in self.tanks:
            if not t.alive:
                continue
            if (x - t.x) ** 2 + (y - t.y) ** 2 <= rad * rad:
                self._damage(t, owner)

    def _apply(self, t, kind):
        if kind == "shield":
            t.shield = True
        elif kind == "speed":
            t.boost = BOOST_TIME
        else:
            t.ammo_kind = kind
            t.ammo_left = AMMO_SHOTS

    def _spawn_powerup(self):
        cells = [
            (r, c)
            for r in range(1, ROWS - 1)
            for c in range(1, COLS - 1)
            if self.grid[r][c] == "."
        ]
        for _ in range(40):
            r, c = self.rng.choice(cells)
            x = (c + 0.5) * CELL
            y = (r + 0.5) * CELL
            near = False
            for t in self.tanks:
                if (t.x - x) ** 2 + (t.y - y) ** 2 < (CELL * 3.0) ** 2:
                    near = True
                    break
            if not near:
                self.powerups.append(PowerUp(x, y, self.rng.choice(POWERUP_KINDS)))
                return

    def _pickups(self, dt):
        self.spawn_t -= dt
        if self.spawn_t <= 0.0:
            self.spawn_t = POWERUP_EVERY
            if len(self.powerups) < POWERUP_MAX:
                self._spawn_powerup()
        keep = []
        for p in self.powerups:
            p.ttl -= dt
            if p.ttl <= 0.0:
                continue
            taken = False
            for t in self.tanks:
                if not t.alive:
                    continue
                if (p.x - t.x) ** 2 + (p.y - t.y) ** 2 <= (POWERUP_R + TANK_R) ** 2:
                    self._apply(t, p.kind)
                    taken = True
                    break
            if not taken:
                keep.append(p)
        self.powerups = keep
        live = []
        for b in self.blasts:
            b[3] -= dt
            if b[3] > 0.0:
                live.append(b)
        self.blasts = live

    def step(self, dt, ctrl0, ctrl1):
        if self.over:
            return
        self.time += dt
        for t, ctrl in ((self.tanks[0], ctrl0), (self.tanks[1], ctrl1)):
            if t.alive:
                self._drive(t, ctrl, dt)
        self._pickups(dt)
        keep = []
        for b in self.bullets:
            alive = self._fly(b, dt)
            hit = False
            if alive:
                for t in self.tanks:
                    if not t.alive:
                        continue
                    if (b.x - t.x) ** 2 + (b.y - t.y) ** 2 <= (BULLET_R + TANK_R) ** 2:
                        self._damage(t, b.owner)
                        hit = True
                        break
            blast = KIND_BLAST[b.kind]
            if blast > 0.0 and (hit or not alive):
                self._blast(b.x, b.y, blast, b.owner)
            if alive and not hit:
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
        self.ready = 0.0
        self.detour = 0.0
        self.detour_dir = 1

    def _los(self, world, me, foe):
        d = math.hypot(foe.x - me.x, foe.y - me.y)
        n = int(d // 6) + 1
        for i in range(1, n):
            x = me.x + (foe.x - me.x) * i / n
            y = me.y + (foe.y - me.y) * i / n
            if world._solid(int(y // CELL), int(x // CELL)):
                return False
        return True

    def _next_cell(self, world, me, tx, ty):
        start = (int(me.y // CELL), int(me.x // CELL))
        goal = (int(ty // CELL), int(tx // CELL))
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

    def act(self, world, me, foe, dt):
        if not me.alive or not foe.alive:
            return (0, 0, 0)
        seen = self._los(world, me, foe)
        dist = math.hypot(foe.x - me.x, foe.y - me.y)
        self.think -= dt
        if self.think <= 0.0:
            self.think = self.rng.uniform(0.10, 0.22)
            gx = foe.x
            gy = foe.y
            if not seen and world.powerups:
                best = None
                for p in world.powerups:
                    d = (p.x - me.x) ** 2 + (p.y - me.y) ** 2
                    if best is None or d < best[0]:
                        best = (d, p)
                if best is not None and best[0] < (CELL * 9.0) ** 2:
                    gx = best[1].x
                    gy = best[1].y
            self.way = self._next_cell(world, me, gx, gy)
        if seen:
            tx = foe.x
            ty = foe.y
        elif self.way is not None:
            tx, ty = self.way
            if math.hypot(tx - me.x, ty - me.y) < 8.0:
                self.think = 0.0
        else:
            tx = me.x + math.cos(me.angle) * 64.0
            ty = me.y + math.sin(me.angle) * 64.0
        diff = (math.atan2(ty - me.y, tx - me.x) - me.angle + math.pi) % (2 * math.pi) - math.pi
        probe_x = me.x + math.cos(me.angle) * 26.0
        probe_y = me.y + math.sin(me.angle) * 26.0
        blocked = world._circle_hit(probe_x, probe_y, TANK_R)
        if self.detour > 0.0:
            self.detour -= dt
            turn = self.detour_dir
            thrust = 1
        elif blocked and abs(diff) < 0.6:
            self.detour = self.rng.uniform(0.35, 0.9)
            self.detour_dir = self.rng.choice((-1, 1))
            turn = self.detour_dir
            thrust = 1
        else:
            turn = 0 if abs(diff) < 0.05 else (1 if diff > 0 else -1)
            thrust = 1 if (not blocked and abs(diff) < 1.5) else 0
        fire = 0
        self.ready -= dt
        if seen:
            tol = math.atan2(TANK_R * (1.0 + 3.0 * (1.0 - self.skill)), max(1.0, dist))
            safe = world.clear_ahead(me.x, me.y, me.angle, 54.0) >= 54.0
            if abs(diff) < tol and safe and self.ready <= 0.0:
                if self.rng.random() < 0.45 + 0.5 * self.skill:
                    fire = 1
                    self.ready = self.rng.uniform(0.0, 0.7 * (1.0 - self.skill))
        else:
            self.ready = 0.0
        return (turn, thrust, fire)
