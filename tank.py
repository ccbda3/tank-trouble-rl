import math
import random

WALL_W = 7.0
CORR_W = 36.0
PITCH = WALL_W + CORR_W
NC = 19
NR = 13
SPAN = CORR_W + 2.0 * WALL_W
WIDTH = WALL_W + NC * PITCH
HEIGHT = WALL_W + NR * PITCH

TANK_H = 10.0
TANK_SPEED = 170.0
TANK_BACK = 108.0
TANK_TURN = 3.0
BARREL_LEN = 16.0

BULLET_R = 3.0
BULLET_SPEED = 450.0
BULLET_LIFE = 4.0
BOUNCE_LIMIT = 7
MAX_BULLETS = 5
FIRE_COOLDOWN = 0.30
DOUBLE_COOLDOWN = 0.55
GATLING_SPEED = 540.0
GATLING_COOLDOWN = 0.08
GATLING_LIFE = 1.5
MISSILE_SPEED = 255.0
MISSILE_LIFE = 9.0
MISSILE_DELAY = 0.9
MISSILE_TURN = 4.2
MISSILE_COOLDOWN = 0.9
FRAG_SPEED = 205.0
FRAG_LIFE = 6.0
FRAG_COOLDOWN = 0.7
SHARD_SPEED = 300.0
SHARD_LIFE = 1.7
SHARD_BOUNCE = 3
SHARD_COUNT = 10
LASER_COOLDOWN = 0.65
LASER_TTL = 0.15
MINE_R = 8.0
MINE_ARM = 1.0
MINE_COOLDOWN = 0.5
BLAST_R = 62.0
BLAST_TTL = 0.5
DEATH_PAUSE = 0.7

WEAPON_KINDS = ("gatling", "missile", "frag", "double", "laser", "shield", "mine")
WEAPON_TIME = {
    "gatling": 6.0,
    "missile": 8.0,
    "frag": 8.0,
    "double": 8.0,
    "laser": 5.0,
    "shield": 6.0,
    "mine": 10.0,
}
WEAPON_NAME = {
    "normal": "普通弹",
    "gatling": "加特林",
    "missile": "追踪导弹",
    "frag": "破片炸弹",
    "double": "双管炮",
    "laser": "激光",
    "shield": "护盾",
    "mine": "地雷",
}

CAPSULE_R = 11.0
CAPSULE_TTL = 22.0
CAPSULE_EVERY = 5.5
CAPSULE_MAX = 3

TANK_COLORS = ((214, 48, 44), (48, 96, 214), (34, 158, 70))

P1_KEYMAP = {
    "up": ("up",),
    "down": ("down",),
    "left": ("left",),
    "right": ("right",),
    "fire": ("m",),
}
P2_KEYMAP = {
    "up": ("w",),
    "down": ("s",),
    "left": ("a",),
    "right": ("d",),
    "fire": ("q",),
}


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


def read_ctrl(down, keys):
    def held(name):
        return any(down(k) for k in keys[name])

    turn = (1 if held("right") else 0) - (1 if held("left") else 0)
    thrust = (1 if held("up") else 0) - (1 if held("down") else 0)
    return (turn, thrust, 1 if held("fire") else 0)


def generate(seed=None, braid=0.18):
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


def sq_pts(cx, cy, ang, h):
    ca = math.cos(ang)
    sa = math.sin(ang)
    return (
        (cx - h * ca + h * sa, cy - h * sa - h * ca),
        (cx + h * ca + h * sa, cy + h * sa - h * ca),
        (cx + h * ca - h * sa, cy + h * sa + h * ca),
        (cx - h * ca - h * sa, cy - h * sa + h * ca),
    )


def rect_pts(x, y, w, h):
    return ((x, y), (x + w, y), (x + w, y + h), (x, y + h))


def overlap(pts_a, pts_b):
    axes = (
        (1.0, 0.0),
        (0.0, 1.0),
        (pts_a[1][0] - pts_a[0][0], pts_a[1][1] - pts_a[0][1]),
        (pts_a[3][0] - pts_a[0][0], pts_a[3][1] - pts_a[0][1]),
    )
    for ax, ay in axes:
        n = math.hypot(ax, ay)
        if n < 1e-9:
            continue
        ax /= n
        ay /= n
        d = pts_a[0][0] * ax + pts_a[0][1] * ay
        a0 = a1 = d
        for px, py in pts_a[1:]:
            d = px * ax + py * ay
            if d < a0:
                a0 = d
            elif d > a1:
                a1 = d
        d = pts_b[0][0] * ax + pts_b[0][1] * ay
        b0 = b1 = d
        for px, py in pts_b[1:]:
            d = px * ax + py * ay
            if d < b0:
                b0 = d
            elif d > b1:
                b1 = d
        if a1 <= b0 or b1 <= a0:
            return False
    return True


def ray_box(ox, oy, dx, dy, half):
    t0 = -1e9
    t1 = 1e9
    for o, d in ((ox, dx), (oy, dy)):
        if abs(d) < 1e-9:
            if o < -half or o > half:
                return None
            continue
        a = (-half - o) / d
        b = (half - o) / d
        if a > b:
            a, b = b, a
        if a > t0:
            t0 = a
        if b < t1:
            t1 = b
        if t0 > t1:
            return None
    if t1 < 0.0:
        return None
    return max(0.0, t0)


class Tank:
    def __init__(self, x, y, angle, side):
        self.x = x
        self.y = y
        self.angle = angle
        self.side = side
        self.alive = True
        self.cool = 0.0
        self.score = 0
        self.weapon = "normal"
        self.weapon_t = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.speed_mul = 1.0
        self.turn_mul = 1.0
        self.killed_by = None

    @property
    def shield(self):
        return self.weapon == "shield" and self.weapon_t > 0.0

    def muzzle(self):
        return (
            self.x + math.cos(self.angle) * BARREL_LEN,
            self.y + math.sin(self.angle) * BARREL_LEN,
        )

    def hits(self, x, y, rad):
        ca = math.cos(self.angle)
        sa = math.sin(self.angle)
        dx = x - self.x
        dy = y - self.y
        return abs(dx * ca + dy * sa) <= TANK_H + rad and abs(dy * ca - dx * sa) <= TANK_H + rad

    def ray_dist(self, ox, oy, dx, dy):
        ca = math.cos(self.angle)
        sa = math.sin(self.angle)
        rx = ox - self.x
        ry = oy - self.y
        return ray_box(rx * ca + ry * sa, ry * ca - rx * sa, dx * ca + dy * sa, dy * ca - dx * sa, TANK_H)


class Capsule:
    def __init__(self, x, y, kind):
        self.x = x
        self.y = y
        self.kind = kind
        self.ttl = CAPSULE_TTL


class Bullet:
    def __init__(self, x, y, angle, speed, owner, kind, bounce, life):
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.owner = owner
        self.kind = kind
        self.bounce = 0
        self.limit = bounce
        self.life = life


class Mine:
    def __init__(self, x, y, owner):
        self.x = x
        self.y = y
        self.owner = owner
        self.arm = MINE_ARM


class World:
    def __init__(self, seed=None, tanks=2):
        self.tanks = [Tank(0.0, 0.0, 0.0, i) for i in range(tanks)]
        self.reset(seed)

    def reset(self, seed=None):
        self.rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
        self.seed = self.rng.randrange(1 << 30)
        for t in self.tanks:
            t.score = 0
        return self.new_round()

    def new_round(self):
        self.seed = self.rng.randrange(1 << 30)
        self.vwall, self.hwall = generate(self.seed)
        self.walls = wall_rects(self.vwall, self.hwall)
        self.index = {}
        for rect in self.walls:
            x, y, w, h = rect
            for bc in range(int(x // PITCH), int((x + w) // PITCH) + 1):
                for br in range(int(y // PITCH), int((y + h) // PITCH) + 1):
                    self.index.setdefault((bc, br), []).append(rect)
        self.bullets = []
        self.capsules = []
        self.mines = []
        self.blasts = []
        self.beams = []
        self.pause = 0.0
        self.over = False
        self.dead = None
        self.spawn_t = CAPSULE_EVERY * 0.5
        self._spawn_tanks()
        return self

    def _spawn_tanks(self):
        cells = [(i, j) for i in range(NC) for j in range(NR)]
        self.rng.shuffle(cells)
        picks = []
        for i, j in cells:
            x, y = cell_center(i, j)
            if any((px - x) ** 2 + (py - y) ** 2 < (PITCH * 3.0) ** 2 for px, py in picks):
                continue
            picks.append((x, y))
            if len(picks) == len(self.tanks):
                break
        if len(picks) < len(self.tanks):
            for i, j in cells:
                x, y = cell_center(i, j)
                if any(abs(px - x) < 1.0 and abs(py - y) < 1.0 for px, py in picks):
                    continue
                picks.append((x, y))
                if len(picks) == len(self.tanks):
                    break
        for t, (x, y) in zip(self.tanks, picks):
            t.x = x
            t.y = y
            t.angle = 0.0
            t.alive = True
            t.cool = 0.0
            t.weapon = "normal"
            t.weapon_t = 0.0
            t.vx = 0.0
            t.vy = 0.0
            t.speed_mul = 1.0
            t.turn_mul = 1.0
            t.killed_by = None
        for t in self.tanks:
            far = None
            for o in self.tanks:
                if o is t:
                    continue
                d = (o.x - t.x) ** 2 + (o.y - t.y) ** 2
                if far is None or d > far[0]:
                    far = (d, o)
            if far is not None:
                t.angle = math.atan2(far[1].y - t.y, far[1].x - t.x)

    def _near(self, x0, y0, x1, y1):
        out = []
        for bc in range(int(x0 // PITCH), int(x1 // PITCH) + 1):
            for br in range(int(y0 // PITCH), int(y1 // PITCH) + 1):
                b = self.index.get((bc, br))
                if b:
                    out.extend(b)
        return out

    def point_solid(self, x, y):
        if x < 0.0 or y < 0.0 or x > WIDTH or y > HEIGHT:
            return True
        for x0, y0, w, h in self._near(x, y, x, y):
            if x0 <= x <= x0 + w and y0 <= y <= y0 + h:
                return True
        return False

    def _blocked(self, x, y, ang, skip, tanks=True):
        if x < TANK_H or y < TANK_H or x > WIDTH - TANK_H or y > HEIGHT - TANK_H:
            return True
        if self._circle_hit(x, y, TANK_H):
            return True
        if not tanks:
            return False
        pts = sq_pts(x, y, ang, TANK_H)
        for o in self.tanks:
            if o is skip or not o.alive:
                continue
            if overlap(pts, sq_pts(o.x, o.y, o.angle, TANK_H)):
                return True
        return False

    def _circle_hit(self, x, y, rad):
        for x0, y0, w, h in self._near(x - rad, y - rad, x + rad, y + rad):
            nx = min(max(x, x0), x0 + w)
            ny = min(max(y, y0), y0 + h)
            if (x - nx) ** 2 + (y - ny) ** 2 < rad * rad:
                return True
        return False

    def _push_out(self, x, y, rad):
        r2 = rad * rad
        for _ in range(6):
            moved = False
            for x0, y0, w, h in self._near(x - rad, y - rad, x + rad, y + rad):
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
                elif x < x0:
                    x = x0 - rad
                elif x > x0 + w:
                    x = x0 + w + rad
                elif y < y0:
                    y = y0 - rad
                else:
                    y = y0 + h + rad
            if not moved:
                break
        return x, y

    def cell_at(self, x, y):
        return (
            round((x - WALL_W - CORR_W * 0.5) / PITCH),
            round((y - WALL_W - CORR_W * 0.5) / PITCH),
        )

    def can_go(self, i, j, di, dj):
        if not (0 <= i < NC and 0 <= j < NR):
            return False
        ni = i + di
        nj = j + dj
        if not (0 <= ni < NC and 0 <= nj < NR):
            return False
        return (not self.vwall[max(i, ni)][j]) if di else (not self.hwall[max(j, nj)][i])

    def route(self, i, j, gi, gj):
        if not (0 <= gi < NC and 0 <= gj < NR):
            return None
        if (i, j) == (gi, gj):
            return None
        prev = {(i, j): None}
        queue = [(i, j)]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            if cur == (gi, gj):
                break
            ci, cj = cur
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (ci + di, cj + dj)
                if n in prev or not self.can_go(ci, cj, di, dj):
                    continue
                prev[n] = cur
                queue.append(n)
        if (gi, gj) not in prev:
            return None
        node = (gi, gj)
        while prev[node] is not None and prev[prev[node]] is not None:
            node = prev[node]
        return node

    def trace(self, x, y, ang, bounces=2, ignore=None, step=None, limit=110):
        if self.point_solid(x, y):
            return [(x, y)], None
        if step is None:
            step = BULLET_SPEED * (1.0 / 60.0) / 3.0
        dx = math.cos(ang) * step
        dy = math.sin(ang) * step
        pts = [(x, y)]
        n = 0
        for _ in range(limit):
            nx = x + dx
            if self._circle_hit(nx, y, BULLET_R):
                dx = -dx
                n += 1
                pts.append((x, y))
                x, y = self._push_out(x, y, BULLET_R)
            else:
                x = nx
            ny = y + dy
            if self._circle_hit(x, ny, BULLET_R):
                dy = -dy
                n += 1
                pts.append((x, y))
                x, y = self._push_out(x, y, BULLET_R)
            else:
                y = ny
            if n > bounces:
                break
            if not (0.0 <= x <= WIDTH and 0.0 <= y <= HEIGHT):
                break
            for t in self.tanks:
                if t is ignore or not t.alive:
                    continue
                if t.hits(x, y, BULLET_R):
                    pts.append((x, y))
                    return pts, t
        pts.append((x, y))
        return pts, None

    def _drive(self, t, ctrl, dt):
        turn, thrust, fire = ctrl
        ox, oy = t.x, t.y
        if turn:
            na = t.angle + turn * TANK_TURN * t.turn_mul * dt
            if not self._blocked(t.x, t.y, na, t, False):
                t.angle = na
        if thrust:
            v = (TANK_SPEED if thrust > 0 else -TANK_BACK) * t.speed_mul
            nx = t.x + math.cos(t.angle) * v * dt
            ny = t.y + math.sin(t.angle) * v * dt
            if not self._blocked(nx, ny, t.angle, t):
                t.x = nx
                t.y = ny
        t.vx = (t.x - ox) / dt
        t.vy = (t.y - oy) / dt
        t.cool = max(0.0, t.cool - dt)
        if t.weapon != "normal":
            t.weapon_t -= dt
            if t.weapon_t <= 0.0:
                t.weapon = "normal"
                t.weapon_t = 0.0
        if fire:
            self._fire(t)

    def _count(self, t, kind):
        n = 0
        for b in self.bullets:
            if b.owner is t and b.kind == kind:
                n += 1
        return n

    def _fire(self, t):
        if t.cool > 0.0:
            return False
        mx, my = t.muzzle()
        w = t.weapon
        if w == "shield":
            return False
        if w == "mine":
            for m in list(self.mines):
                if m.owner is t:
                    self.mines.remove(m)
            self.mines.append(Mine(t.x, t.y, t))
            t.cool = MINE_COOLDOWN
            return True
        if w == "laser":
            t.cool = LASER_COOLDOWN
            self._laser(t, mx, my)
            return True
        if w == "gatling":
            self.bullets.append(
                Bullet(mx, my, t.angle, GATLING_SPEED, t, "gatling", BOUNCE_LIMIT, GATLING_LIFE)
            )
            t.cool = GATLING_COOLDOWN
            return True
        if w == "missile":
            if self._count(t, "missile") >= 1:
                return False
            self.bullets.append(
                Bullet(mx, my, t.angle, MISSILE_SPEED, t, "missile", BOUNCE_LIMIT, MISSILE_LIFE)
            )
            t.cool = MISSILE_COOLDOWN
            return True
        if w == "frag":
            if self._count(t, "frag") >= 1:
                return False
            self.bullets.append(Bullet(mx, my, t.angle, FRAG_SPEED, t, "frag", 0, FRAG_LIFE))
            t.cool = FRAG_COOLDOWN
            return True
        if w == "double":
            for off in (-0.055, 0.055):
                self.bullets.append(
                    Bullet(mx, my, t.angle + off, BULLET_SPEED, t, "bullet", BOUNCE_LIMIT, BULLET_LIFE)
                )
            t.cool = DOUBLE_COOLDOWN
            return True
        if self._count(t, "bullet") >= MAX_BULLETS:
            return False
        self.bullets.append(
            Bullet(mx, my, t.angle, BULLET_SPEED, t, "bullet", BOUNCE_LIMIT, BULLET_LIFE)
        )
        t.cool = FIRE_COOLDOWN
        return True

    def _laser(self, t, mx, my):
        ca = math.cos(t.angle)
        sa = math.sin(t.angle)
        best = None
        for o in self.tanks:
            if not o.alive:
                continue
            d = o.ray_dist(mx, my, ca, sa)
            if d is not None and (best is None or d < best[0]):
                best = (d, o)
        if best is None:
            ex = mx + ca * 4000.0
            ey = my + sa * 4000.0
        else:
            ex = mx + ca * best[0]
            ey = my + sa * best[0]
        self.beams.append([mx, my, ex, ey, LASER_TTL])
        for o in self.tanks:
            if o.alive and o.ray_dist(mx, my, ca, sa) is not None:
                self._damage(o, t)

    def _blast(self, x, y, rad, owner, shards):
        self.blasts.append([x, y, rad, BLAST_TTL])
        if shards:
            for k in range(SHARD_COUNT):
                self.bullets.append(
                    Bullet(x, y, 2.0 * math.pi * k / SHARD_COUNT, SHARD_SPEED, owner,
                           "shard", SHARD_BOUNCE, SHARD_LIFE)
                )
            return
        for t in self.tanks:
            if t.alive and (x - t.x) ** 2 + (y - t.y) ** 2 <= rad * rad:
                self._damage(t, owner)

    def _damage(self, t, killer):
        if not t.alive:
            return False
        if t.shield:
            t.weapon = "normal"
            t.weapon_t = 0.0
            return False
        t.alive = False
        t.killed_by = killer
        self.dead = t.side
        if killer is not None and killer is not t:
            killer.score += 1
        else:
            for o in self.tanks:
                if o is not t:
                    o.score += 1
        return True

    def _spawn_capsule(self):
        for _ in range(60):
            x, y = cell_center(self.rng.randrange(NC), self.rng.randrange(NR))
            if any((t.x - x) ** 2 + (t.y - y) ** 2 < (PITCH * 2.0) ** 2 for t in self.tanks):
                continue
            if any((c.x - x) ** 2 + (c.y - y) ** 2 < (PITCH * 2.0) ** 2 for c in self.capsules):
                continue
            self.capsules.append(Capsule(x, y, self.rng.choice(WEAPON_KINDS)))
            return

    def _items(self, dt):
        self.spawn_t -= dt
        if self.spawn_t <= 0.0:
            self.spawn_t = CAPSULE_EVERY
            if len(self.capsules) < CAPSULE_MAX:
                self._spawn_capsule()
        keep = []
        for c in self.capsules:
            c.ttl -= dt
            if c.ttl <= 0.0:
                continue
            taken = False
            for t in self.tanks:
                if not t.alive:
                    continue
                if (c.x - t.x) ** 2 + (c.y - t.y) ** 2 <= (CAPSULE_R + TANK_H) ** 2:
                    t.weapon = c.kind
                    t.weapon_t = WEAPON_TIME[c.kind]
                    taken = True
                    break
            if not taken:
                keep.append(c)
        self.capsules = keep
        live = []
        for m in self.mines:
            m.arm -= dt
            blown = False
            if m.arm <= 0.0:
                for t in self.tanks:
                    if t.alive and (m.x - t.x) ** 2 + (m.y - t.y) ** 2 <= (MINE_R + TANK_H * 1.42) ** 2:
                        blown = True
                        break
            if blown:
                self._blast(m.x, m.y, BLAST_R, m.owner, False)
            else:
                live.append(m)
        self.mines = live
        blast = []
        for b in self.blasts:
            b[3] -= dt
            if b[3] > 0.0:
                blast.append(b)
        self.blasts = blast
        beams = []
        for b in self.beams:
            b[4] -= dt
            if b[4] > 0.0:
                beams.append(b)
        self.beams = beams

    def _fly(self, b, dt):
        if b.kind == "missile":
            self._home(b, dt)
        steps = int(math.hypot(b.vx, b.vy) * dt / 3.0) + 1
        h = dt / steps
        for _ in range(steps):
            nx = b.x + b.vx * h
            if self._circle_hit(nx, b.y, BULLET_R):
                if b.kind == "frag":
                    self._blast(b.x, b.y, BLAST_R * 0.7, b.owner, True)
                    return False
                b.vx = -b.vx
                b.bounce += 1
                b.x, b.y = self._push_out(b.x, b.y, BULLET_R)
            else:
                b.x = nx
            ny = b.y + b.vy * h
            if self._circle_hit(b.x, ny, BULLET_R):
                if b.kind == "frag":
                    self._blast(b.x, b.y, BLAST_R * 0.7, b.owner, True)
                    return False
                b.vy = -b.vy
                b.bounce += 1
                b.x, b.y = self._push_out(b.x, b.y, BULLET_R)
            else:
                b.y = ny
            if b.bounce > b.limit:
                return False
            if not (0.0 <= b.x <= WIDTH and 0.0 <= b.y <= HEIGHT):
                return False
        b.life -= dt
        return b.life > 0.0

    def _home(self, b, dt):
        if b.life > MISSILE_LIFE - MISSILE_DELAY:
            return
        target = None
        best = None
        for t in self.tanks:
            if not t.alive:
                continue
            d = (t.x - b.x) ** 2 + (t.y - b.y) ** 2
            if best is None or d < best:
                best = d
                target = t
        if target is None:
            return
        cur = math.atan2(b.vy, b.vx)
        want = math.atan2(target.y - b.y, target.x - b.x)
        diff = (want - cur + math.pi) % (2.0 * math.pi) - math.pi
        step = MISSILE_TURN * dt
        diff = step if diff > step else (-step if diff < -step else diff)
        na = cur + diff
        sp = math.hypot(b.vx, b.vy)
        b.vx = math.cos(na) * sp
        b.vy = math.sin(na) * sp

    def step(self, dt, ctrls):
        if self.over:
            self.pause -= dt
            if self.pause <= 0.0:
                self.new_round()
            return
        for t, ctrl in zip(self.tanks, ctrls):
            if t.alive:
                self._drive(t, ctrl, dt)
        self._items(dt)
        keep = []
        for b in self.bullets:
            alive = self._fly(b, dt)
            hit = False
            if alive and b.kind != "frag":
                for t in self.tanks:
                    if t.alive and t.hits(b.x, b.y, BULLET_R):
                        self._damage(t, b.owner)
                        hit = True
                        break
            if alive and not hit:
                keep.append(b)
        self.bullets = keep
        if any(not t.alive for t in self.tanks):
            self.over = True
            self.pause = DEATH_PAUSE

    def step_actions(self, dt, actions):
        self.step(dt, [ACTION_CTRL[a] for a in actions])


def ang_diff(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


class Laika:
    def __init__(self, level, seed=None):
        self.level = max(1, min(3, level))
        self.rng = random.Random(seed)
        self.think = 0.0
        self.sol = 0.0
        self.err = 0.0
        self.armed = False
        self.dodge_ang = 0.0
        self.dodge_t = 0.0
        self.speed_mul = (0.66, 1.0, 1.15)[self.level - 1]
        self.turn_mul = (0.62, 1.0, 1.25)[self.level - 1]
        self.scan = (0.45, 0.28, 0.18)[self.level - 1]
        self.aim_err = (0.22, 0.075, 0.012)[self.level - 1]
        self.dodge_skill = (0.08, 0.75, 1.0)[self.level - 1]
        self.grab = (0.15, 0.75, 1.0)[self.level - 1]
        self.lead = (0.0, 0.10, 0.24)[self.level - 1]
        self.nav = None
        self.px = 0.0
        self.py = 0.0
        self.pa = 0.0
        self.stuck = 0.0
        self.unstick = 0.0
        self.unstick_dir = 1
        self.search = (
            ((3, 0.05, 0, 300),),
            ((3, 0.05, 0, 640), (13, math.pi, 2, 380)),
            ((5, 0.05, 0, 640), (17, math.pi, 3, 460), (29, math.pi, 5, 560)),
        )[self.level - 1]

    def _solve(self, world, me, foe):
        mx, my = me.muzzle()
        px = foe.x + foe.vx * self.lead
        py = foe.y + foe.vy * self.lead
        base = math.atan2(py - my, px - mx)
        for count, half, bounces, limit in self.search:
            best = None
            for k in range(count):
                ang = base if count == 1 else base + half * (2.0 * k / (count - 1) - 1.0)
                pts, hit = world.trace(mx, my, ang, bounces, me, None, limit)
                if hit is foe:
                    run = 0.0
                    for i in range(len(pts) - 1):
                        run += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                    if best is None or run < best[0]:
                        best = (run, ang)
            if best is not None:
                return best[1]
        return None

    def _threat(self, world, me):
        best = None
        for b in world.bullets:
            dx = b.x - me.x
            dy = b.y - me.y
            d = math.hypot(dx, dy)
            if d < 1.0 or d > 190.0:
                continue
            sp = math.hypot(b.vx, b.vy)
            if sp < 1e-6:
                continue
            ux = b.vx / sp
            uy = b.vy / sp
            if -dx * ux - dy * uy <= 0.0:
                continue
            s = -dx * uy + dy * ux
            if abs(s) > TANK_H + BULLET_R + 3.0:
                continue
            if best is None or d < best[2]:
                nx, ny = (-uy, ux) if s >= 0.0 else (uy, -ux)
                best = (nx * 0.85 - ux * 0.55, ny * 0.85 - uy * 0.55, d)
        return best

    def _clear_dir(self, world, me, tx, ty):
        want = math.atan2(ty - me.y, tx - me.x)
        near = TANK_H + 7.0
        far = TANK_H + PITCH * 1.5
        best = None
        for off in (0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4,
                    1.8, -1.8, 2.2, -2.2, 2.7, -2.7, math.pi):
            a = me.angle + off
            if world._blocked(me.x + math.cos(a) * near, me.y + math.sin(a) * near, a, me):
                continue
            score = abs(ang_diff(a, want))
            if not world._blocked(me.x + math.cos(a) * far, me.y + math.sin(a) * far, a, me):
                score -= 0.35
            if best is None or score < best[0]:
                best = (score, a)
        return None if best is None else best[1]

    def act(self, world, me, foe, dt):
        if not me.alive or world.over:
            return (0, 0, 0)
        me.speed_mul = self.speed_mul
        me.turn_mul = self.turn_mul
        self.think -= dt
        self.dodge_t -= dt
        threat = self._threat(world, me) if self.rng.random() < self.dodge_skill else None
        if threat is not None:
            d = ang_diff(math.atan2(threat[1], threat[0]), me.angle)
            me.turn_mul = self.turn_mul if abs(d) > 0.14 else self.turn_mul * 0.3
            turn = 0 if abs(d) < 0.03 else (1 if d > 0 else -1)
            if threat[2] < 56.0 and abs(d) > 2.2:
                return (turn, -1, 0)
            return (turn, 1 if abs(d) < 1.3 else 0, 0)
        if self.think <= 0.0:
            self.think = self.scan
            self.err = self.rng.uniform(-self.aim_err, self.aim_err)
            sol = self._solve(world, me, foe)
            self.armed = sol is not None
            if sol is not None:
                self.sol = sol
            gx, gy = foe.x, foe.y
            if world.capsules and self.rng.random() < self.grab:
                near = min(world.capsules, key=lambda c: (c.x - me.x) ** 2 + (c.y - me.y) ** 2)
                if (near.x - me.x) ** 2 + (near.y - me.y) ** 2 < (PITCH * 7.0) ** 2:
                    gx, gy = near.x, near.y
            nxt = world.route(*world.cell_at(me.x, me.y), *world.cell_at(gx, gy))
            self.nav = cell_center(*nxt) if nxt is not None else (gx, gy)
        aim = None
        if self.armed:
            aim = self.sol + self.err
        else:
            aim = math.atan2(self.nav[1] - me.y, self.nav[0] - me.x)
        diff = ang_diff(aim, me.angle)
        turn = 0
        thrust = 0
        fire = 0
        if self.unstick > 0.0:
            self.unstick -= dt
            return (self.unstick_dir, -1, 0)
        if self.armed and abs(diff) < 0.05:
            me.turn_mul = self.turn_mul * 0.25
            turn = 0 if abs(diff) < 0.012 else (1 if diff > 0 else -1)
            fire = 1
        elif self.level == 1 and self.rng.random() < 0.012:
            fire = 1
            me.turn_mul = self.turn_mul
            turn = self.rng.choice((-1, 0, 1))
            thrust = 1
        else:
            me.turn_mul = self.turn_mul if abs(diff) > 0.14 else self.turn_mul * 0.3
            turn = 0 if abs(diff) < 0.03 else (1 if diff > 0 else -1)
            if world._blocked(
                me.x + math.cos(me.angle) * (TANK_H + PITCH * 0.6),
                me.y + math.sin(me.angle) * (TANK_H + PITCH * 0.6),
                me.angle, me,
            ):
                fd = self._clear_dir(world, me, self.nav[0], self.nav[1])
                if fd is None:
                    turn, thrust = 0, -1
                else:
                    d = ang_diff(fd, me.angle)
                    fwd_ok = not world._blocked(
                        me.x + math.cos(me.angle) * (TANK_H + 9.0),
                        me.y + math.sin(me.angle) * (TANK_H + 9.0),
                        me.angle, me,
                    )
                    rot_ok = not world._blocked(me.x, me.y, me.angle + (0.1 if d > 0 else -0.1), me, False)
                    if not rot_ok:
                        turn, thrust = 0, (1 if fwd_ok else -1)
                    elif abs(d) < 0.04:
                        turn, thrust = 0, (1 if fwd_ok else 0)
                    else:
                        turn, thrust = (1 if d > 0 else -1), 0
            else:
                thrust = 1 if abs(diff) < 1.9 else 0
        moved = (me.x - self.px) ** 2 + (me.y - self.py) ** 2
        turned = abs(ang_diff(me.angle, self.pa))
        self.px = me.x
        self.py = me.y
        self.pa = me.angle
        if (thrust != 0 or turn != 0) and moved < 0.04 and turned < 0.008:
            self.stuck += dt
            if self.stuck > 0.3:
                self.stuck = 0.0
                self.unstick = self.rng.uniform(0.5, 1.0)
                self.unstick_dir = self.rng.choice((-1, 1))
        else:
            self.stuck = 0.0
        return (turn, thrust, fire)

