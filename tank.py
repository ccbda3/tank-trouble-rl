import math
import random

WALL_W = 7.0
CORR_W = 31.0
PITCH = WALL_W + CORR_W
NC = 19
NR = 13
WIDTH = WALL_W + NC * PITCH
HEIGHT = WALL_W + NR * PITCH

TANK_R = 10.0
TANK_SPEED = 152.0
TANK_BACK = 97.0
TANK_TURN = 3.1
BULLET_R = 3.5
BULLET_SPEED = 404.0
BULLET_LIFE = 4.0
BOUNCE_LIMIT = 6
FIRE_COOLDOWN = 0.34
MAX_BULLETS = 5

POWERUP_KINDS = ("laser", "bomb", "speed", "shield")
POWERUP_R = 12.0
POWERUP_TTL = 14.0
POWERUP_EVERY = 6.5
POWERUP_MAX = 2
BOOST_TIME = 9.0
BOOST_MUL = 1.75
AMMO_SHOTS = 3
BLAST_R = 55.0
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
    vwall = [[True] * NR for _ in range(NC + 1)]
    hwall = [[True] * NC for _ in range(NR + 1)]
    seen = [[False] * NC for _ in range(NR)]
    seen[0][0] = True
    stack = [(0, 0)]
    while stack:
        i, j = stack[-1]
        nbrs = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni = i + di
            nj = j + dj
            if 0 <= ni < NC and 0 <= nj < NR and not seen[nj][ni]:
                nbrs.append((ni, nj, di, dj))
        if not nbrs:
            stack.pop()
            continue
        ni, nj, di, dj = rng.choice(nbrs)
        if di:
            vwall[max(i, ni)][j] = False
        else:
            hwall[max(j, nj)][i] = False
        seen[nj][ni] = True
        stack.append((ni, nj))
    for j in range(NR):
        for i in range(NC):
            if i + 1 < NC and vwall[i + 1][j] and rng.random() < braid:
                vwall[i + 1][j] = False
            if j + 1 < NR and hwall[j + 1][i] and rng.random() < braid:
                hwall[j + 1][i] = False
    return vwall, hwall


SPAN = CORR_W + 2.0 * WALL_W


def wall_rects(vwall, hwall):
    out = []
    for i in range(NC + 1):
        for j in range(NR):
            if vwall[i][j]:
                out.append((i * PITCH, j * PITCH, WALL_W, SPAN))
    for j in range(NR + 1):
        for i in range(NC):
            if hwall[j][i]:
                out.append((i * PITCH, j * PITCH, SPAN, WALL_W))
    return out


def cell_center(i, j):
    return (WALL_W + i * PITCH + CORR_W * 0.5, WALL_W + j * PITCH + CORR_W * 0.5)


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
        self.vwall, self.hwall = generate(seed)
        self.walls = wall_rects(self.vwall, self.hwall)
        self.index = {}
        for rect in self.walls:
            x, y, w, h = rect
            for bc in range(int(x // PITCH), int((x + w) // PITCH) + 1):
                for br in range(int(y // PITCH), int((y + h) // PITCH) + 1):
                    self.index.setdefault((bc, br), []).append(rect)
        rng = random.Random(seed)
        if rng.random() < 0.5:
            corners = ((0, 0), (NC - 1, NR - 1))
        else:
            corners = ((NC - 1, 0), (0, NR - 1))
        self.rng = rng
        self.tanks = []
        for i, j in corners:
            x, y = cell_center(i, j)
            self.tanks.append(Tank(x, y, 0.0))
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

    def _near(self, x, y, rad):
        out = []
        for bc in range(int((x - rad) // PITCH), int((x + rad) // PITCH) + 1):
            for br in range(int((y - rad) // PITCH), int((y + rad) // PITCH) + 1):
                b = self.index.get((bc, br))
                if b:
                    out.extend(b)
        return out

    def _circle_hit(self, x, y, rad):
        r2 = rad * rad
        for x0, y0, w, h in self._near(x, y, rad):
            nx = min(max(x, x0), x0 + w)
            ny = min(max(y, y0), y0 + h)
            if (x - nx) ** 2 + (y - ny) ** 2 < r2:
                return True
        return False

    def _push_out(self, x, y, rad):
        r2 = rad * rad
        for _ in range(8):
            moved = False
            for x0, y0, w, h in self._near(x, y, rad):
                nx = min(max(x, x0), x0 + w)
                ny = min(max(y, y0), y0 + h)
                dx = x - nx
                dy = y - ny
                d2 = dx * dx + dy * dy
                if d2 >= r2:
                    continue
                moved = True
                if d2 > 1e-9:
                    d = math.sqrt(d2)
                    k = (rad - d) / d
                    x += dx * k
                    y += dy * k
                else:
                    left = x - x0
                    right = x0 + w - x
                    top = y - y0
                    bottom = y0 + h - y
                    m = min(left, right, top, bottom)
                    if m == left:
                        x = x0 - rad
                    elif m == right:
                        x = x0 + w + rad
                    elif m == top:
                        y = y0 - rad
                    else:
                        y = y0 + h + rad
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
        for _ in range(40):
            i = self.rng.randrange(NC)
            j = self.rng.randrange(NR)
            x, y = cell_center(i, j)
            near = False
            for t in self.tanks:
                if (t.x - x) ** 2 + (t.y - y) ** 2 < (PITCH * 3.0) ** 2:
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

