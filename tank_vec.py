import numpy as np

import tank as ref
from tank import (
    BARREL_LEN, BLAST_R, BLAST_TTL, BOUNCE_LIMIT, BULLET_R, BULLET_SPEED, CAPSULE_EVERY,
    CAPSULE_MAX, CAPSULE_R, CAPSULE_TTL, CORR_W, DEATH_PAUSE, DOUBLE_COOLDOWN, FIRE_COOLDOWN,
    FRAG_COOLDOWN, FRAG_LIFE, FRAG_SPEED, GATLING_COOLDOWN, GATLING_LIFE, GATLING_SPEED,
    HEIGHT, LASER_COOLDOWN, LASER_TTL, MAX_BULLETS, MINE_ARM, MINE_COOLDOWN, MINE_R,
    MISSILE_COOLDOWN, MISSILE_DELAY, MISSILE_LIFE, MISSILE_SPEED, MISSILE_TURN, NC, NR,
    PITCH, SHARD_BOUNCE, SHARD_COUNT, SHARD_LIFE, SHARD_SPEED, TANK_BACK, TANK_H, TANK_SPEED,
    TANK_TURN, WALL_W, WEAPON_KINDS, WEAPON_TIME, WIDTH,
)

W_NORMAL, W_GATLING, W_MISSILE, W_FRAG, W_DOUBLE, W_LASER, W_SHIELD, W_MINE = range(8)
WEAPON_SECONDS = np.array([0.0] + [WEAPON_TIME[k] for k in WEAPON_KINDS], dtype=np.float32)
K_BULLET, K_GATLING, K_SHARD, K_MISSILE, K_FRAG = range(5)
B_MAX = 48
M_MAX = 3
C_MAX = 3
Z_MAX = 6
L_MAX = 3
SUBSTEPS = 4

OBS_RES = 8
OBS_W = int(WIDTH) // OBS_RES
OBS_H = int(HEIGHT) // OBS_RES
OBS_CH = 7


def _maps():
    ow = np.arange(OBS_W, dtype=np.int64) * OBS_RES
    oh = np.arange(OBS_H, dtype=np.int64) * OBS_RES
    sv = (ow % int(PITCH)) < WALL_W
    sh = (oh % int(PITCH)) < WALL_W
    vs = np.clip(ow // int(PITCH), 0, NC)
    hs = np.clip(oh // int(PITCH), 0, NR)
    ic = np.clip(np.round((ow - WALL_W - CORR_W * 0.5) / PITCH).astype(np.int64), 0, NC - 1)
    jc = np.clip(np.round((oh - WALL_W - CORR_W * 0.5) / PITCH).astype(np.int64), 0, NR - 1)
    return sv, sh, vs, hs, ic, jc


_SV, _SH, _VS, _HS, _IC, _JC = _maps()
_DISC = np.array([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=np.int64)


def _sq_overlap(ax, ay, aa, bx, by, ba, h):
    ca = np.cos(aa)
    sa = np.sin(aa)
    cb = np.cos(ba)
    sb = np.sin(ba)
    dx = bx - ax
    dy = by - ay
    ok = np.ones(np.shape(dx), dtype=bool)
    for nx, ny in ((1.0, 0.0), (0.0, 1.0)):
        p = np.abs(dx * nx + dy * ny)
        ea = h * (np.abs(ca * nx + sa * ny) + np.abs(-sa * nx + ca * ny))
        eb = h * (np.abs(cb * nx + sb * ny) + np.abs(-sb * nx + cb * ny))
        ok &= p <= ea + eb
    for ux, uy in ((ca, sa), (-sa, ca)):
        p = np.abs(dx * ux + dy * uy)
        eb = h * (np.abs(cb * ux + sb * uy) + np.abs(-sb * ux + cb * uy))
        ok &= p <= h + eb
    return ok


def _ray_square(ox, oy, dx, dy, tx, ty, ta, half):
    ca = np.cos(ta)
    sa = np.sin(ta)
    rx = ox - tx
    ry = oy - ty
    lx = rx * ca + ry * sa
    ly = ry * ca - rx * sa
    ex = dx * ca + dy * sa
    ey = dy * ca - dx * sa
    t0 = np.full(np.shape(ox), -1e9, dtype=np.float32)
    t1 = np.full(np.shape(ox), 1e9, dtype=np.float32)
    for o, d in ((lx, ex), (ly, ey)):
        par = np.abs(d) < 1e-9
        safe = np.where(par, 1.0, d)
        lo = np.minimum((-half - o) / safe, (half - o) / safe)
        hi = np.maximum((-half - o) / safe, (half - o) / safe)
        blocked = par & ((o < -half) | (o > half))
        t0 = np.where(par, np.where(blocked, 1e9, t0), np.maximum(t0, lo))
        t1 = np.where(par, t1, np.minimum(t1, hi))
    bad = (t0 > t1) | (t1 < 0.0)
    return np.where(bad, -1.0, np.maximum(0.0, t0))


class VecEnv:
    def __init__(self, n, players=2, seed=None):
        self.n = int(n)
        self.p = int(players)
        self.rng = np.random.default_rng(seed)
        self.rows = np.arange(self.n)
        self.vwall = np.zeros((self.n, NC + 1, NR), dtype=bool)
        self.hwall = np.zeros((self.n, NR + 1, NC), dtype=bool)
        self.tx = np.zeros((self.n, self.p), dtype=np.float32)
        self.ty = np.zeros((self.n, self.p), dtype=np.float32)
        self.ta = np.zeros((self.n, self.p), dtype=np.float32)
        self.t_alive = np.ones((self.n, self.p), dtype=bool)
        self.t_cool = np.zeros((self.n, self.p), dtype=np.float32)
        self.t_wt = np.zeros((self.n, self.p), dtype=np.float32)
        self.t_weapon = np.zeros((self.n, self.p), dtype=np.int8)
        self.t_score = np.zeros((self.n, self.p), dtype=np.int32)
        self.t_px = np.zeros((self.n, self.p), dtype=np.float32)
        self.t_py = np.zeros((self.n, self.p), dtype=np.float32)
        self.t_vx = np.zeros((self.n, self.p), dtype=np.float32)
        self.t_vy = np.zeros((self.n, self.p), dtype=np.float32)
        self.b_on = np.zeros((self.n, B_MAX), dtype=bool)
        self.b_x = np.zeros((self.n, B_MAX), dtype=np.float32)
        self.b_y = np.zeros((self.n, B_MAX), dtype=np.float32)
        self.b_vx = np.zeros((self.n, B_MAX), dtype=np.float32)
        self.b_vy = np.zeros((self.n, B_MAX), dtype=np.float32)
        self.b_life = np.zeros((self.n, B_MAX), dtype=np.float32)
        self.b_bounce = np.zeros((self.n, B_MAX), dtype=np.int16)
        self.b_owner = np.full((self.n, B_MAX), -1, dtype=np.int8)
        self.b_kind = np.zeros((self.n, B_MAX), dtype=np.int8)
        self.m_on = np.zeros((self.n, M_MAX), dtype=bool)
        self.m_x = np.zeros((self.n, M_MAX), dtype=np.float32)
        self.m_y = np.zeros((self.n, M_MAX), dtype=np.float32)
        self.m_arm = np.zeros((self.n, M_MAX), dtype=np.float32)
        self.m_owner = np.full((self.n, M_MAX), -1, dtype=np.int8)
        self.c_on = np.zeros((self.n, C_MAX), dtype=bool)
        self.c_x = np.zeros((self.n, C_MAX), dtype=np.float32)
        self.c_y = np.zeros((self.n, C_MAX), dtype=np.float32)
        self.c_ttl = np.zeros((self.n, C_MAX), dtype=np.float32)
        self.c_kind = np.zeros((self.n, C_MAX), dtype=np.int8)
        self.z_on = np.zeros((self.n, Z_MAX), dtype=bool)
        self.z_x = np.zeros((self.n, Z_MAX), dtype=np.float32)
        self.z_y = np.zeros((self.n, Z_MAX), dtype=np.float32)
        self.z_r = np.zeros((self.n, Z_MAX), dtype=np.float32)
        self.z_ttl = np.zeros((self.n, Z_MAX), dtype=np.float32)
        self.l_on = np.zeros((self.n, L_MAX), dtype=bool)
        self.l_xy = np.zeros((self.n, L_MAX, 4), dtype=np.float32)
        self.l_ttl = np.zeros((self.n, L_MAX), dtype=np.float32)
        self.spawn_t = np.full(self.n, CAPSULE_EVERY * 0.5, dtype=np.float32)
        self.pause = np.zeros(self.n, dtype=np.float32)
        self.over = np.zeros(self.n, dtype=bool)
        self.dead = np.full(self.n, -1, dtype=np.int8)
        self.kills = np.zeros((self.n, self.p), dtype=np.int32)
        self.rounds = np.zeros(self.n, dtype=np.int32)
        self.ep_kills = np.zeros((self.n, self.p), dtype=np.int32)
        self.ep_deaths = np.zeros((self.n, self.p), dtype=np.int32)
        self.ep_steps = np.zeros(self.n, dtype=np.int32)
        self.obs_wall = np.zeros((self.n, OBS_H, OBS_W), dtype=np.uint8)
        self.reset()

    def reset(self):
        for i in range(self.n):
            seed = int(self.rng.integers(1 << 30))
            vwall, hwall = ref.generate(seed)
            self.vwall[i] = vwall
            self.hwall[i] = hwall
        self.t_score[:] = 0
        self.kills[:] = 0
        self.ep_kills[:] = 0
        self.ep_deaths[:] = 0
        self.ep_steps[:] = 0
        self.rounds[:] = 0
        self._new_round(self.rows)
        return self.observation(0)

    def _new_round(self, rows):
        self.b_on[rows] = False
        self.m_on[rows] = False
        self.z_on[rows] = False
        self.l_on[rows] = False
        self.c_on[rows] = False
        self.pause[rows] = 0.0
        self.over[rows] = False
        self.dead[rows] = -1
        self.spawn_t[rows] = CAPSULE_EVERY * 0.5
        cells = self.rng.integers(0, NC * NR, size=(rows.size, self.p))
        for _ in range(40):
            ci = cells // NR
            cj = cells % NR
            x = WALL_W + ci * PITCH + CORR_W * 0.5
            y = WALL_W + cj * PITCH + CORR_W * 0.5
            bad = np.zeros((rows.size, self.p), dtype=bool)
            for a in range(self.p):
                for b in range(a + 1, self.p):
                    close = (x[:, a] - x[:, b]) ** 2 + (y[:, a] - y[:, b]) ** 2 < (PITCH * 3.0) ** 2
                    bad[:, a] |= close
                    bad[:, b] |= close
            if not bad.any():
                break
            cells[bad] = self.rng.integers(0, NC * NR, size=int(bad.sum()))
        ci = cells // NR
        cj = cells % NR
        x = (WALL_W + ci * PITCH + CORR_W * 0.5).astype(np.float32)
        y = (WALL_W + cj * PITCH + CORR_W * 0.5).astype(np.float32)
        self.tx[rows] = x
        self.ty[rows] = y
        self.t_px[rows] = x
        self.t_py[rows] = y
        self.t_alive[rows] = True
        self.t_cool[rows] = 0.0
        self.t_wt[rows] = 0.0
        self.t_weapon[rows] = W_NORMAL
        self.t_vx[rows] = 0.0
        self.t_vy[rows] = 0.0
        ang = np.zeros((rows.size, self.p), dtype=np.float32)
        for k in range(self.p):
            dx = x - x[:, k:k + 1]
            dy = y - y[:, k:k + 1]
            d2 = dx * dx + dy * dy
            d2[:, k] = -1.0
            far = np.argmax(d2, axis=1)
            ang[:, k] = np.arctan2(dy[np.arange(rows.size), far], dx[np.arange(rows.size), far])
        self.ta[rows] = ang
        self._refresh_wall_obs(rows)

    def _refresh_wall_obs(self, rows):
        vw = self.vwall[np.ix_(rows, _VS, _JC)]
        hw = self.hwall[np.ix_(rows, _HS, _IC)]
        vw = np.transpose(vw, (0, 2, 1))
        wall = (_SV[None, None, :] & vw) | (_SH[None, :, None] & hw)
        self.obs_wall[rows] = np.where(wall, 255, 0).astype(np.uint8)

    def free(self, rows, x, y, r):
        ci = np.clip(np.floor((x - WALL_W) / PITCH).astype(np.int64), 0, NC - 1)
        cj = np.clip(np.floor((y - WALL_W) / PITCH).astype(np.int64), 0, NR - 1)
        left = WALL_W + ci * PITCH
        top = WALL_W + cj * PITCH
        ok = np.ones(np.shape(x), dtype=bool)
        ok &= (~self.vwall[rows, ci, cj]) | (x >= left + r)
        ok &= (~self.vwall[rows, ci + 1, cj]) | (x <= left + CORR_W - r)
        ok &= (~self.hwall[rows, cj, ci]) | (y >= top + r)
        ok &= (~self.hwall[rows, cj + 1, ci]) | (y <= top + CORR_W - r)
        return ok

    def _tank_free(self, rows, x, y, ang, side):
        ok = (x >= TANK_H) & (y >= TANK_H) & (x <= WIDTH - TANK_H) & (y <= HEIGHT - TANK_H)
        ok &= self.free(rows, x, y, TANK_H)
        for j in range(self.p):
            if j == side:
                continue
            live = self.t_alive[rows, j]
            if not live.any():
                continue
            ov = _sq_overlap(x, y, ang, self.tx[rows, j], self.ty[rows, j], self.ta[rows, j], TANK_H)
            ok &= ~(live & ov)
        return ok

    def step(self, ctrl, dt):
        run = ~self.over
        if run.any():
            self._drive(ctrl, dt, run)
            self._items(dt, run)
            self._bullets(dt, run)
            newly = run & (self.t_alive.sum(axis=1) < self.p)
            if newly.any():
                self.over |= newly
                self.pause[newly] = DEATH_PAUSE
                self.rounds[newly] += 1
                for j in range(self.p):
                    self.ep_deaths[newly, j] += (~self.t_alive[newly, j]).astype(np.int32)
        self.ep_steps[run] += 1
        self.pause = np.maximum(0.0, self.pause - dt)
        done = self.over & (self.pause <= 0.0)
        if done.any():
            rows = np.nonzero(done)[0]
            self._new_round(rows)

    def _drive(self, ctrl, dt, run):
        turn = ctrl[:, :, 0].astype(np.float32)
        thrust = ctrl[:, :, 1].astype(np.float32)
        fire = ctrl[:, :, 2] > 0
        alive = self.t_alive & run[:, None]
        for side in range(self.p):
            rows = np.nonzero(alive[:, side])[0]
            if rows.size == 0:
                continue
            sel = turn[rows, side] != 0
            if sel.any():
                r = rows[sel]
                na = self.ta[r, side] + turn[r, side] * TANK_TURN * dt
                ok = self._tank_free(r, self.tx[r, side], self.ty[r, side], na, side)
                self.ta[r[ok], side] = na[ok]
            sel = thrust[rows, side] != 0
            if sel.any():
                r = rows[sel]
                v = np.where(thrust[r, side] > 0, TANK_SPEED, -TANK_BACK) * dt
                nx = self.tx[r, side] + np.cos(self.ta[r, side]) * v
                ny = self.ty[r, side] + np.sin(self.ta[r, side]) * v
                ok = self._tank_free(r, nx, ny, self.ta[r, side], side)
                self.tx[r[ok], side] = nx[ok]
                self.ty[r[ok], side] = ny[ok]
        self.t_vx = (self.tx - self.t_px) / dt
        self.t_vy = (self.ty - self.t_py) / dt
        self.t_px = self.tx.copy()
        self.t_py = self.ty.copy()
        self.t_cool = np.maximum(0.0, self.t_cool - dt)
        armed = self.t_weapon > W_NORMAL
        self.t_wt[armed] -= dt
        expired = armed & (self.t_wt <= 0.0)
        self.t_weapon[expired] = W_NORMAL
        self.t_wt[expired] = 0.0
        for side in range(self.p):
            rows = np.nonzero(alive[:, side] & fire[:, side] & (self.t_cool[:, side] <= 0.0))[0]
            if rows.size:
                self._fire(rows, side)

    def _muzzle(self, rows, side):
        return (self.tx[rows, side] + np.cos(self.ta[rows, side]) * BARREL_LEN,
                self.ty[rows, side] + np.sin(self.ta[rows, side]) * BARREL_LEN)

    def _emit(self, rows, side, ang, speed, kind, bounce, life):
        empty = ~self.b_on[rows]
        has = empty.any(axis=1)
        if not has.any():
            return
        r = rows[has]
        slot = np.argmax(empty[has], axis=1)
        a = ang[has] if np.ndim(ang) else ang
        mx, my = self._muzzle(r, side)
        self.b_on[r, slot] = True
        self.b_x[r, slot] = mx
        self.b_y[r, slot] = my
        self.b_vx[r, slot] = np.cos(a) * speed
        self.b_vy[r, slot] = np.sin(a) * speed
        self.b_kind[r, slot] = kind
        self.b_bounce[r, slot] = 0
        self.b_life[r, slot] = life
        self.b_owner[r, slot] = side

    def _count(self, rows, side, kind):
        return (self.b_on[rows] & (self.b_owner[rows] == side) & (self.b_kind[rows] == kind)).sum(axis=1)

    def _fire(self, rows, side):
        w = self.t_weapon[rows, side]
        for kind_id, mask in ((W_NORMAL, w == W_NORMAL), (W_GATLING, w == W_GATLING),
                              (W_MISSILE, w == W_MISSILE), (W_FRAG, w == W_FRAG),
                              (W_DOUBLE, w == W_DOUBLE)):
            if not mask.any():
                continue
            r = rows[mask]
            if kind_id == W_NORMAL:
                r = r[self._count(r, side, K_BULLET) < MAX_BULLETS]
                if r.size == 0:
                    continue
                self._emit(r, side, self.ta[r, side], BULLET_SPEED, K_BULLET, BOUNCE_LIMIT, 4.0)
                self.t_cool[r, side] = FIRE_COOLDOWN
            elif kind_id == W_GATLING:
                self._emit(r, side, self.ta[r, side], GATLING_SPEED, K_GATLING, BOUNCE_LIMIT, GATLING_LIFE)
                self.t_cool[r, side] = GATLING_COOLDOWN
            elif kind_id == W_MISSILE:
                r = r[self._count(r, side, K_MISSILE) == 0]
                if r.size == 0:
                    continue
                self._emit(r, side, self.ta[r, side], MISSILE_SPEED, K_MISSILE, BOUNCE_LIMIT, MISSILE_LIFE)
                self.t_cool[r, side] = MISSILE_COOLDOWN
            elif kind_id == W_FRAG:
                r = r[self._count(r, side, K_FRAG) == 0]
                if r.size == 0:
                    continue
                self._emit(r, side, self.ta[r, side], FRAG_SPEED, K_FRAG, 0, FRAG_LIFE)
                self.t_cool[r, side] = FRAG_COOLDOWN
            else:
                self._emit(r, side, self.ta[r, side] - 0.055, BULLET_SPEED, K_BULLET, BOUNCE_LIMIT, 4.0)
                self._emit(r, side, self.ta[r, side] + 0.055, BULLET_SPEED, K_BULLET, BOUNCE_LIMIT, 4.0)
                self.t_cool[r, side] = DOUBLE_COOLDOWN
        laser = rows[w == W_LASER]
        if laser.size:
            self._laser(laser, side)
            self.t_cool[laser, side] = LASER_COOLDOWN
        mine = rows[w == W_MINE]
        if mine.size:
            self._place_mine(mine, side)
            self.t_cool[mine, side] = MINE_COOLDOWN

    def _place_mine(self, rows, side):
        if rows.size == 0:
            return
        mine_slot = self.m_owner[rows] == side
        empty = ~self.m_on[rows]
        has = mine_slot.any(axis=1) | empty.any(axis=1)
        r = rows[has]
        if r.size == 0:
            return
        mine_slot = self.m_owner[r] == side
        empty = ~self.m_on[r]
        slot = np.where(mine_slot.any(axis=1), np.argmax(mine_slot, axis=1), np.argmax(empty, axis=1))
        self.m_on[r, slot] = True
        self.m_x[r, slot] = self.tx[r, side]
        self.m_y[r, slot] = self.ty[r, side]
        self.m_arm[r, slot] = MINE_ARM
        self.m_owner[r, slot] = side

    def _laser(self, rows, side):
        mx, my = self._muzzle(rows, side)
        ca = np.cos(self.ta[rows, side])
        sa = np.sin(self.ta[rows, side])
        best = np.full(rows.size, 1e9, dtype=np.float32)
        dists = []
        for j in range(self.p):
            d = _ray_square(mx, my, ca, sa, self.tx[rows, j], self.ty[rows, j], self.ta[rows, j], TANK_H)
            d = np.where(self.t_alive[rows, j] & (d >= 0.0), d, 1e9)
            dists.append(d)
            best = np.minimum(best, d)
        far = best >= 1e8
        end_x = mx + ca * np.where(far, 4000.0, best)
        end_y = my + sa * np.where(far, 4000.0, best)
        empty = ~self.l_on[rows]
        has = empty.any(axis=1)
        r = rows[has]
        if r.size:
            m = has
            slot = np.argmax(empty[m], axis=1)
            self.l_on[r, slot] = True
            self.l_xy[r, slot, 0] = mx[m]
            self.l_xy[r, slot, 1] = my[m]
            self.l_xy[r, slot, 2] = end_x[m]
            self.l_xy[r, slot, 3] = end_y[m]
            self.l_ttl[r, slot] = LASER_TTL
        for j in range(self.p):
            hit = dists[j] < 1e8
            if hit.any():
                self._damage(rows[hit], j, side)

    def _damage(self, rows, side, killer):
        if rows.size == 0:
            return
        shielded = self.t_weapon[rows, side] == W_SHIELD
        if shielded.any():
            r = rows[shielded]
            self.t_weapon[r, side] = W_NORMAL
            self.t_wt[r, side] = 0.0
            rows = rows[~shielded]
            if rows.size == 0:
                return
        self.t_alive[rows, side] = False
        self.dead[rows] = side
        if isinstance(killer, (int, np.integer)):
            if killer != side:
                self.t_score[rows, killer] += 1
                self.kills[rows, killer] += 1
                self.ep_kills[rows, killer] += 1
            else:
                for j in range(self.p):
                    if j != side:
                        self.t_score[rows, j] += 1
                        self.ep_kills[rows, j] += 1
        else:
            other = (killer >= 0) & (killer != side)
            if other.any():
                r = rows[other]
                k = killer[other]
                for j in range(self.p):
                    m = k == j
                    if m.any():
                        self.t_score[r[m], j] += 1
                        self.kills[r[m], j] += 1
                        self.ep_kills[r[m], j] += 1
            mine = ~other
            if mine.any():
                r = rows[mine]
                for j in range(self.p):
                    if j != side:
                        self.t_score[r, j] += 1
                        self.ep_kills[r, j] += 1

    def _blast(self, rows, x, y, rad, owner, shards):
        empty = ~self.z_on[rows]
        has = empty.any(axis=1)
        r = rows[has]
        if r.size:
            slot = np.argmax(empty[has], axis=1)
            m = has
            self.z_on[r, slot] = True
            self.z_x[r, slot] = x[m]
            self.z_y[r, slot] = y[m]
            self.z_r[r, slot] = rad[m]
            self.z_ttl[r, slot] = BLAST_TTL
        if shards:
            for k in range(SHARD_COUNT):
                ang = np.full(rows.size, 2.0 * np.pi * k / SHARD_COUNT, dtype=np.float32)
                self._emit(rows, owner, ang, SHARD_SPEED, K_SHARD, SHARD_BOUNCE, SHARD_LIFE)
            return
        for j in range(self.p):
            live = self.t_alive[rows, j]
            near = live & ((x - self.tx[rows, j]) ** 2 + (y - self.ty[rows, j]) ** 2 <= rad ** 2)
            if near.any():
                self._damage(rows[near], j, owner)

    def _items(self, dt, run):
        self.spawn_t = self.spawn_t - dt
        want = run & (self.spawn_t <= 0.0)
        self.spawn_t = np.where(want, CAPSULE_EVERY, self.spawn_t)
        try_spawn = want & (self.c_on.sum(axis=1) < CAPSULE_MAX)
        if try_spawn.any():
            self._spawn_capsule(np.nonzero(try_spawn)[0])
        self.c_ttl = self.c_ttl - dt
        self.c_on &= self.c_ttl > 0.0
        for j in range(self.p):
            live = self.t_alive[:, j] & run
            rows = np.nonzero(live)[0]
            if rows.size == 0:
                continue
            dx = self.c_x[rows] - self.tx[rows, j][:, None]
            dy = self.c_y[rows] - self.ty[rows, j][:, None]
            near = self.c_on[rows] & (dx * dx + dy * dy <= (CAPSULE_R + TANK_H) ** 2)
            if near.any():
                r, c = np.nonzero(near)
                self.t_weapon[rows[r], j] = self.c_kind[rows[r], c] + 1
                self.t_wt[rows[r], j] = WEAPON_SECONDS[self.c_kind[rows[r], c] + 1]
                self.c_on[rows[r], c] = False
        self.m_arm = self.m_arm - dt
        armed = self.m_on & (self.m_arm <= 0.0)
        blown = np.zeros_like(self.m_on)
        for j in range(self.p):
            live = self.t_alive[:, j] & run
            rows = np.nonzero(live)[0]
            if rows.size == 0:
                continue
            dx = self.m_x[rows] - self.tx[rows, j][:, None]
            dy = self.m_y[rows] - self.ty[rows, j][:, None]
            touch = armed[rows] & (dx * dx + dy * dy <= (MINE_R + TANK_H * 1.42) ** 2)
            if touch.any():
                r, c = np.nonzero(touch)
                blown[rows[r], c] = True
        if blown.any():
            r, c = np.nonzero(blown)
            self.m_on[r, c] = False
            self._blast(r, self.m_x[r, c].copy(), self.m_y[r, c].copy(),
                        np.full(r.size, BLAST_R, dtype=np.float32), self.m_owner[r, c].copy(), False)
        self.z_ttl = self.z_ttl - dt
        self.z_on &= self.z_ttl > 0.0
        self.l_ttl = self.l_ttl - dt
        self.l_on &= self.l_ttl > 0.0

    def _spawn_capsule(self, rows):
        for _ in range(30):
            cell = self.rng.integers(0, NC * NR, size=rows.size)
            x = (WALL_W + (cell // NR) * PITCH + CORR_W * 0.5).astype(np.float32)
            y = (WALL_W + (cell % NR) * PITCH + CORR_W * 0.5).astype(np.float32)
            bad = np.zeros(rows.size, dtype=bool)
            for j in range(self.p):
                bad |= (self.tx[rows, j] - x) ** 2 + (self.ty[rows, j] - y) ** 2 < (PITCH * 2.0) ** 2
            for q in range(C_MAX):
                bad |= self.c_on[rows, q] & ((self.c_x[rows, q] - x) ** 2
                                             + (self.c_y[rows, q] - y) ** 2 < (PITCH * 2.0) ** 2)
            room = ~self.c_on[rows].all(axis=1)
            good = (~bad) & room
            if not good.any():
                return
            r = rows[good]
            slot = np.argmax(~self.c_on[r], axis=1)
            self.c_on[r, slot] = True
            self.c_x[r, slot] = x[good]
            self.c_y[r, slot] = y[good]
            self.c_ttl[r, slot] = CAPSULE_TTL
            self.c_kind[r, slot] = self.rng.integers(0, len(WEAPON_KINDS), size=r.size)
            return

    def _home(self, dt):
        ready = self.b_on & (self.b_kind == K_MISSILE) & (self.b_life <= MISSILE_LIFE - MISSILE_DELAY)
        if not ready.any():
            return
        r, c = np.nonzero(ready)
        bx = self.b_x[r, c]
        by = self.b_y[r, c]
        best = np.full(r.size, 1e18, dtype=np.float32)
        target = np.full(r.size, -1, dtype=np.int64)
        for j in range(self.p):
            d = np.where(self.t_alive[r, j], (self.tx[r, j] - bx) ** 2 + (self.ty[r, j] - by) ** 2, 1e18)
            better = d < best
            best = np.where(better, d, best)
            target = np.where(better, j, target)
        has = target >= 0
        if not has.any():
            return
        r = r[has]
        c = c[has]
        tgt = target[has]
        cur = np.arctan2(self.b_vy[r, c], self.b_vx[r, c])
        want = np.arctan2(self.ty[r, tgt] - self.b_y[r, c], self.tx[r, tgt] - self.b_x[r, c])
        diff = np.clip((want - cur + np.pi) % (2.0 * np.pi) - np.pi, -MISSILE_TURN * dt, MISSILE_TURN * dt)
        na = cur + diff
        sp = np.hypot(self.b_vx[r, c], self.b_vy[r, c])
        self.b_vx[r, c] = np.cos(na) * sp
        self.b_vy[r, c] = np.sin(na) * sp

    def _unstick(self):
        rows = np.nonzero(self.b_on.any(axis=1))[0]
        if rows.size == 0:
            return
        bx = self.b_x[rows].copy()
        by = self.b_y[rows].copy()
        ang = np.arctan2(self.b_vy[rows], self.b_vx[rows])
        stuck = self.b_on[rows] & ~self.free(rows[:, None], bx, by, BULLET_R)
        for k in range(1, 11):
            if not stuck.any():
                break
            back = 1.5 * k
            nx = bx - np.cos(ang) * back
            ny = by - np.sin(ang) * back
            ok = self.free(rows[:, None], nx, ny, BULLET_R)
            fix = stuck & ok
            bx = np.where(fix, nx, bx)
            by = np.where(fix, ny, by)
            stuck = stuck & ~ok
        self.b_x[rows] = bx
        self.b_y[rows] = by
        if stuck.any():
            self.b_on[rows] = self.b_on[rows] & ~stuck

    def _bullets(self, dt, run):
        if not self.b_on.any():
            return
        self._unstick()
        self._home(dt)
        h = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            live = self.b_on & run[:, None]
            if not live.any():
                return
            rows = np.nonzero(live.any(axis=1))[0]
            live_sub = live[rows]
            nx = self.b_x[rows] + self.b_vx[rows] * h
            hit_x = ~self.free(rows[:, None], nx, self.b_y[rows], BULLET_R)
            self.b_x[rows] = np.where(hit_x, self.b_x[rows], nx)
            self.b_vx[rows] = np.where(hit_x, -self.b_vx[rows], self.b_vx[rows])
            ny = self.b_y[rows] + self.b_vy[rows] * h
            hit_y = ~self.free(rows[:, None], self.b_x[rows], ny, BULLET_R)
            self.b_y[rows] = np.where(hit_y, self.b_y[rows], ny)
            self.b_vy[rows] = np.where(hit_y, -self.b_vy[rows], self.b_vy[rows])
            self.b_bounce[rows] += (hit_x.astype(np.int16) + hit_y.astype(np.int16))
            frag = live_sub & (self.b_kind[rows] == K_FRAG) & (hit_x | hit_y)
            if frag.any():
                r, c = np.nonzero(frag)
                self.b_on[r, c] = False
                self._blast(r, self.b_x[r, c].copy(), self.b_y[r, c].copy(),
                            np.full(r.size, BLAST_R * 0.7, dtype=np.float32),
                            self.b_owner[r, c].copy(), True)
            dead = self.b_on & run[:, None] & (self.b_bounce > BOUNCE_LIMIT)
            if dead.any():
                self.b_on[dead] = False
            self.b_life = np.where(self.b_on & run[:, None], self.b_life - h, self.b_life)
            gone = self.b_on & run[:, None] & (self.b_life <= 0.0)
            if gone.any():
                self.b_on[gone] = False
        self._bullet_hits(run)

    def _bullet_hits(self, run):
        live = self.b_on & run[:, None]
        if not live.any():
            return
        for j in range(self.p):
            rows = np.nonzero(self.t_alive[:, j] & run)[0]
            if rows.size == 0:
                continue
            ca = np.cos(self.ta[rows, j])[:, None]
            sa = np.sin(self.ta[rows, j])[:, None]
            dx = self.b_x[rows] - self.tx[rows, j][:, None]
            dy = self.b_y[rows] - self.ty[rows, j][:, None]
            hit = live[rows] & (self.b_kind[rows] != K_FRAG)
            hit &= np.abs(dx * ca + dy * sa) <= TANK_H + BULLET_R
            hit &= np.abs(dy * ca - dx * sa) <= TANK_H + BULLET_R
            if not hit.any():
                continue
            r, c = np.nonzero(hit)
            self.b_on[rows[r], c] = False
            self._damage(rows[r], j, self.b_owner[rows[r], c].astype(np.int64))

    def observation(self, side):
        obs = np.zeros((self.n, OBS_CH, OBS_H, OBS_W), dtype=np.uint8)
        obs[:, 0] = self.obs_wall
        for j in range(self.p):
            ch = 1 if j == side else 2
            val = 255 if j == side else 200
            self._blit(obs, ch, self.tx[:, j], self.ty[:, j], self.t_alive[:, j], val)
        for q in range(B_MAX):
            self._dot(obs, 3, self.b_x[:, q], self.b_y[:, q], self.b_on[:, q], 255)
        for q in range(M_MAX):
            self._dot(obs, 4, self.m_x[:, q], self.m_y[:, q], self.m_on[:, q], 90)
        for q in range(C_MAX):
            self._dot(obs, 5, self.c_x[:, q], self.c_y[:, q], self.c_on[:, q], 160)
        for q in range(L_MAX):
            on = self.l_on[:, q]
            if not on.any():
                continue
            x0 = self.l_xy[:, q, 0]
            y0 = self.l_xy[:, q, 1]
            x1 = self.l_xy[:, q, 2]
            y1 = self.l_xy[:, q, 3]
            steps = np.maximum(2, (np.hypot(x1 - x0, y1 - y0) / OBS_RES).astype(np.int64))
            for t in range(int(steps.max()) + 1):
                sel = on & (t <= steps)
                f = t / np.maximum(steps, 1)
                self._dot(obs, 6, x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, sel, 255)
        return obs

    def _blit(self, obs, ch, x, y, mask, value):
        rows = np.nonzero(mask)[0]
        if rows.size == 0:
            return
        cx = np.clip((x[rows] / OBS_RES).astype(np.int64), 0, OBS_W - 1)
        cy = np.clip((y[rows] / OBS_RES).astype(np.int64), 0, OBS_H - 1)
        for dy, dx in _DISC:
            obs[rows, ch, np.clip(cy + dy, 0, OBS_H - 1), np.clip(cx + dx, 0, OBS_W - 1)] = value

    def _dot(self, obs, ch, x, y, mask, value):
        rows = np.nonzero(mask)[0]
        if rows.size == 0:
            return
        cx = np.clip((x[rows] / OBS_RES).astype(np.int64), 0, OBS_W - 1)
        cy = np.clip((y[rows] / OBS_RES).astype(np.int64), 0, OBS_H - 1)
        obs[rows, ch, cy, cx] = value

    def scalars(self, side):
        out = np.zeros((self.n, self.p * 7 + 6), dtype=np.float32)
        order = [side] + [j for j in range(self.p) if j != side]
        for k, j in enumerate(order):
            b = k * 7
            out[:, b + 0] = self.tx[:, j] / WIDTH
            out[:, b + 1] = self.ty[:, j] / HEIGHT
            out[:, b + 2] = np.cos(self.ta[:, j])
            out[:, b + 3] = np.sin(self.ta[:, j])
            out[:, b + 4] = self.t_alive[:, j]
            out[:, b + 5] = self.t_weapon[:, j] / 7.0
            out[:, b + 6] = np.clip(self.t_wt[:, j] / 10.0, 0.0, 1.0)
        base = self.p * 7
        out[:, base + 0] = self.t_score[:, side] / 10.0
        for j in range(self.p):
            if j != side:
                out[:, base + 1] = self.t_score[:, j] / 10.0
            break
        out[:, base + 2] = self.t_cool[:, side]
        out[:, base + 3] = self.pause / DEATH_PAUSE
        out[:, base + 4] = self.over
        out[:, base + 5] = self.ep_steps / 1800.0
        return out

    def self_play_batch(self, policy, dt):
        obs = np.concatenate([self.observation(j) for j in range(self.p)], axis=0)
        scal = np.concatenate([self.scalars(j) for j in range(self.p)], axis=0)
        acts = policy(obs, scal)
        acts = np.asarray(acts).reshape(self.p, self.n, 3).transpose(1, 0, 2)
        self.step(acts.astype(np.float32), dt)
        return obs, scal, acts
