"""Offline simulator with the same interface as hal.RoboMasterHAL.

Virtual clock (runs much faster than real time), random maze, noisy odometry
with scale error, noisy range sensors and a pinhole camera model.  Used to
test the mission logic without the robot:  python main.py --sim --round 1
"""
import math
import random
import time

import config as C
from maze import DIRS, DV, OPEN, WALL, Maze, step
from vision import Detection, label_parts, plate_size

T = C.TILE
ROBOT_R = 0.17
IR_REACH = 0.08         # corner IR modules set to 8 cm (as on the robot)
YAW_OF_DIR = {"N": 0.0, "E": 90.0, "S": 180.0, "W": -90.0}


def random_maze(seed, loops=4):
    rnd = random.Random(seed)
    m = Maze(C.GRID_W, C.GRID_H, True)
    for c in range(m.w):
        for r in range(m.h):
            for d in "NE":
                m.set((c, r), d, WALL)
    stack, seen = [tuple(C.START_CELL)], {tuple(C.START_CELL)}
    while stack:
        c = stack[-1]
        nb = [(d, step(c, d)) for d in DIRS if m.inside(step(c, d)) and step(c, d) not in seen]
        if not nb:
            stack.pop()
            continue
        d, n = rnd.choice(nb)
        m.set(c, d, OPEN)
        seen.add(n)
        stack.append(n)
    walls = [k for k, v in m.edges.items() if v == WALL]
    for k in rnd.sample(walls, min(loops, len(walls))):
        m.edges[k] = OPEN
    return m


class SimHAL:
    def __init__(self, seed=1, speed=None, distractors=("red_circle",), extra=1):
        self.speed = speed          # None = as fast as possible, else x real time
        self.frame = None
        rnd = random.Random(seed)
        self.rnd = random.Random(seed + 1000)
        self.maze = random_maze(seed)
        # targets stand against walls (any side), several per cell allowed:
        # one of each class, `extra` more sheet cards, and the distractors.
        # The first two share a cell so every maze tests that case.
        faces = [((c, r), d) for c in range(C.GRID_W) for r in range(C.GRID_H) for d in DIRS
                 if self.maze.get((c, r), d) == WALL and (c, r) != tuple(C.START_CELL)]
        rnd.shuffle(faces)
        labels = list(C.TARGET_CLASSES) + [rnd.choice(list(C.TARGET_CLASSES))
                                           for _ in range(extra)] + list(distractors)
        first = faces[0]
        same = [f for f in faces[1:] if f[0] == first[0]]
        chosen = [first] + same[:1]
        chosen += [f for f in faces if f not in chosen][:len(labels) - len(chosen)]
        g = T / 2 - C.TARGET_WALL_GAP
        self.targets = []
        for l, (cell, d) in zip(labels, chosen):
            dx, dy = DV[d]
            self.targets.append(dict(label=l, cell=cell, side=d, id="%s@%s%s" % (l, cell, d),
                                     x=cell[0] * T + dx * g, y=cell[1] * T + dy * g))
        self.t = 0.0
        self.x, self.y = C.START_CELL[0] * T, C.START_CELL[1] * T
        self.yaw = 0.0
        self.cmd = (0.0, 0.0, 0.0)
        self.vel = (0.0, 0.0, 0.0)
        self.odo = [0.0, 0.0]
        self.odo_scale = 1.0 + self.rnd.uniform(-0.04, 0.04)
        self.gimbal_yaw, self.gimbal_pitch = 0.0, C.GIMBAL_PITCH
        self._gimbal_ready = 0.0
        self.sharp_err = {}                     # simulated calibration error per side (m)
        self.turn_slip = 0.0                    # odometry error per degree turned (m)
        self.sharp_glitch = 0.0                 # chance a Sharp reading comes out 9 cm long
        self.sharp_bias = {s: 0.0 for s in C.SHARP}
        self.hits = []
        self.collisions = set()
        self.fire_enabled = True

    # ---------------------------------------------------------- clock ----
    def now(self):
        return self.t

    def sleep(self, dt):
        end = self.t + max(dt, 0.0)
        while self.t < end - 1e-9:
            h = min(0.01, end - self.t)
            self._physics(h)
            self.t += h
        if self.speed:
            time.sleep(max(dt, 0.0) / self.speed)

    def tick(self, dt):
        """Advance physics without real sleeping (panel calls this when idle)."""
        end = self.t + dt
        while self.t < end - 1e-9:
            h = min(0.01, end - self.t)
            self._physics(h)
            self.t += h

    def rezero(self):
        """Put the robot back on the start tile (what a person does for real)."""
        self.x, self.y = C.START_CELL[0] * T, C.START_CELL[1] * T
        self.yaw, self.odo = 0.0, [0.0, 0.0]
        self.cmd = self.vel = (0.0, 0.0, 0.0)
        self.gimbal_yaw, self.gimbal_pitch = 0.0, C.GIMBAL_PITCH

    def _physics(self, h):
        a = min(1.0, h / 0.06)                        # motor lag
        self.vel = tuple(v + (c - v) * a for v, c in zip(self.vel, self.cmd))
        vx, vy, wz = self.vel
        self.yaw += (wz + self.rnd.gauss(0, 0.6) * (abs(vx) + abs(vy))) * h
        r = math.radians(self.yaw)
        vn = vx * math.cos(r) - vy * math.sin(r)
        ve = vx * math.sin(r) + vy * math.cos(r)
        nx, ny = self.x + ve * h, self.y + vn * h
        nx, ny = self._collide(nx, ny)
        self.odo[0] += vn * h * self.odo_scale
        self.odo[1] += ve * h * self.odo_scale
        if self.turn_slip and abs(wz) > 5:      # wheels slip while turning in place
            self.odo[0] += self.rnd.gauss(0, self.turn_slip * abs(wz) * h)
            self.odo[1] += self.rnd.gauss(0, self.turn_slip * abs(wz) * h)
        self.x, self.y = nx, ny
        for tg in self.targets:                   # chassis 0.32 x 0.24 m, never rotates
            ns = abs(math.cos(math.radians(self.yaw))) > 0.7        # chassis along N-S
            hy, hx = (0.16, 0.12) if ns else (0.12, 0.16)
            if abs(tg["y"] - ny) < hy + 0.01 and abs(tg["x"] - nx) < hx + 0.01:
                self.collisions.add(tg["id"])

    def _collide(self, x, y):
        c = (int(round(x / T)), int(round(y / T)))
        cx, cy = c[0] * T, c[1] * T
        lim = T / 2 - ROBOT_R
        if self.maze.get(c, "N") == WALL and y > cy + lim:
            y = cy + lim
        if self.maze.get(c, "S") == WALL and y < cy - lim:
            y = cy - lim
        if self.maze.get(c, "E") == WALL and x > cx + lim:
            x = cx + lim
        if self.maze.get(c, "W") == WALL and x < cx - lim:
            x = cx - lim
        return x, y

    # ---------------------------------------------------------- reads ----
    def odom(self):
        return self.odo[0], self.odo[1], (self.yaw + 180) % 360 - 180

    def _raw_range(self, d):
        c = (int(round(self.x / T)), int(round(self.y / T)))
        pos = self.y if d in "NS" else self.x
        s = 1 if d in "NE" else -1
        k = 0
        while True:
            if self.maze.get(c, d) == WALL:
                b = (c[1] if d in "NS" else c[0]) * T + s * T / 2
                r = abs(b - pos)
                for tg in self.targets:            # a plate on that wall, in the beam
                    lat = (self.x - tg["x"]) if d in "NS" else (self.y - tg["y"])
                    if tg["cell"] == c and tg["side"] == d and \
                            abs(lat) < plate_size(tg["label"])[0] / 2:
                        r -= C.TARGET_WALL_GAP
                return r
            c = step(c, d)
            k += 1
            if k > 20:
                return 99.0

    def ranges_ex(self):
        """Same contract as the real HAL: only directions measurable right now.
        Sharps on the chassis sides; the ToF only where the gimbal points (and
        only once it has finished turning)."""
        from hal import side_dir, tof_dir
        out = {}
        h = self.heading()
        if h is not None:
            for side in C.SHARP:
                d = side_dir(h, side)
                r = self._raw_range(d) + self.rnd.gauss(0, 0.006) + self.sharp_err.get(side, 0.0)
                if self.sharp_glitch and self.rnd.random() < self.sharp_glitch:
                    r += 0.09                       # a long reading (panel gap / fold-back)
                face = r - C.SHARP_OFFSET[side]
                if face < C.SHARP_MIN:              # GP2Y0A41 fold-back: too close reads LONG
                    r = C.SHARP_OFFSET[side] + C.SHARP_MIN + 4.0 * (C.SHARP_MIN - face) + 0.10
                ok = C.SHARP_MIN <= r - C.SHARP_OFFSET[side] <= C.SHARP_MAX
                out[d] = (r - self.sharp_bias.get(side, 0.0) if ok else None, "sharp")
        if self.t >= self._gimbal_ready:
            d = tof_dir(self.yaw + self.gimbal_yaw, self.gimbal_pitch)
            if d is not None:
                r = self._raw_range(d) + self.rnd.gauss(0, 0.004)
                out[d] = (r if r - C.TOF_OFFSET < C.TOF_MAX else None, "tof")
        return out

    def side_bias_check(self):
        from hal import side_dir
        out = {}
        h = self.heading() or "N"
        for side in C.SHARP:
            r = self._raw_range(side_dir(h, side)) + self.sharp_err.get(side, 0.0)
            if r - C.SHARP_OFFSET[side] > C.SHARP_MAX or r > C.SIDE_WALL_MAX:
                continue
            bias = r - C.SIDE_NOMINAL
            if abs(bias) <= C.SHARP_BIAS_MAX:
                self.sharp_bias[side] = bias
                out[side] = bias
        return out

    def heading(self):
        from hal import chassis_dir
        return chassis_dir(self.yaw)

    def set_mode(self, mode):
        pass                                    # the sim gimbal always turns with the chassis

    def gimbal_front(self):
        self.gimbal_to(0.0, C.GIMBAL_PITCH, wait=True)

    def gimbal_wait(self):
        pass

    def ranges(self):
        return {d: r for d, (r, _) in self.ranges_ex().items()}

    def sensor_ok(self, d):
        return d in self.ranges_ex()

    def _segments(self):
        """Wall and plate segments ((x0, y0), (x1, y1)) in metres, cached."""
        if getattr(self, "_segs", None) is None:
            segs, h = [], T / 2
            for (c, r, d), v in self.maze.edges.items():
                if v != WALL:
                    continue
                cx, cy = c * T, r * T
                segs.append(((cx - h, cy + h), (cx + h, cy + h)) if d == "N"
                            else ((cx + h, cy - h), (cx + h, cy + h)))
            W, Hh = C.GRID_W * T - h, C.GRID_H * T - h
            segs += [((-h, -h), (W, -h)), ((-h, Hh), (W, Hh)), ((-h, -h), (-h, Hh)), ((W, -h), (W, Hh))]
            for tg in self.targets:
                pw = plate_size(tg["label"])[0] / 2
                if tg["side"] in "NS":
                    segs.append(((tg["x"] - pw, tg["y"]), (tg["x"] + pw, tg["y"])))
                else:
                    segs.append(((tg["x"], tg["y"] - pw), (tg["x"], tg["y"] + pw)))
            self._segs = segs
        return self._segs

    def _ray(self, x, y, ang, maxd):
        """Distance along a ray (deg, 0 = N, + clockwise) to the first segment."""
        dx, dy = math.sin(math.radians(ang)), math.cos(math.radians(ang))
        best = maxd
        for (x0, y0), (x1, y1) in self._segments():
            ex, ey = x1 - x0, y1 - y0
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            t = ((x0 - x) * ey - (y0 - y) * ex) / den
            u = ((x0 - x) * dy - (y0 - y) * dx) / den
            if 0 <= u <= 1 and 0 < t < best:
                best = t
        return best

    def ir_state(self):
        """Each module is a ray from its mount point; it sees up to IR_REACH."""
        out = {}
        r = math.radians(self.yaw)
        for n, spec in C.IR.items():
            px, py = spec["pos"]                    # east, north in the chassis frame
            x = self.x + px * math.cos(r) + py * math.sin(r)
            y = self.y - px * math.sin(r) + py * math.cos(r)
            out[n] = self._ray(x, y, spec["ang"] + self.yaw, IR_REACH + 0.01) <= IR_REACH
        return out

    def bump(self, d, strict=False):
        from hal import side_dir
        h = self.heading()
        if h is None:
            return False
        for n, on in self.ir_state().items():
            spec = C.IR[n]
            if on and side_dir(h, spec["guards"]) == d and not (strict and spec["ang"] % 90):
                r = self.ranges().get(d)
                if r is None or r <= C.IR_TRUST_MAX:
                    return True
        return False

    def _visible(self, x0, y0, x1, y1):
        n = int(math.hypot(x1 - x0, y1 - y0) / 0.03) + 1
        prev = (int(round(x0 / T)), int(round(y0 / T)))
        for i in range(1, n + 1):
            f = i / float(n)
            c = (int(round((x0 + (x1 - x0) * f) / T)), int(round((y0 + (y1 - y0) * f) / T)))
            if c != prev:
                if abs(c[0] - prev[0]) + abs(c[1] - prev[1]) != 1:
                    return False
                d = "E" if c[0] > prev[0] else "W" if c[0] < prev[0] else "N" if c[1] > prev[1] else "S"
                if self.maze.get(prev, d) == WALL:
                    return False
                prev = c
        return True

    def detections(self):
        fw, fh = 640, 360
        f = (fw / 2.0) / math.tan(math.radians(C.CAMERA_HFOV) / 2)
        heading = self.yaw + self.gimbal_yaw
        out = []
        for tg in self.targets:
            dx, dy = tg["x"] - self.x, tg["y"] - self.y
            dist = math.hypot(dx, dy)
            if dist < 0.2:
                continue
            b = (math.degrees(math.atan2(dx, dy)) - heading + 180) % 360 - 180
            if abs(b) > C.CAMERA_HFOV / 2 - 3 or not self._visible(self.x, self.y, tg["x"], tg["y"]):
                continue
            face = self._facing(tg, dx, dy, dist)
            if face < 0.35:                               # seen edge-on / from behind
                continue
            pw, ph = plate_size(tg["label"])
            pw *= face
            color, shape = label_parts(tg["label"])
            z = dist * math.cos(math.radians(b)) * self.rnd.uniform(0.94, 1.06)
            cx = fw / 2 + f * math.tan(math.radians(b))
            tp = math.degrees(math.atan(-0.105 / z))    # plate ~10 cm below the lens
            cy = fh / 2 - f * math.tan(math.radians(tp - self.gimbal_pitch))
            out.append(Detection(tg["label"], color, shape, cx, cy,
                                 f * pw / z, f * ph / z, 0, fw, fh, False))
        return self.t - 0.03, out

    @staticmethod
    def _facing(tg, dx, dy, dist):
        """cos(angle) between the plate's normal (away from its wall) and the
        direction to the camera."""
        nx, ny = DV[tg["side"]]
        return (-nx * -dx + -ny * -dy) / max(dist, 1e-6)

    # -------------------------------------------------------- actions ----
    def drive(self, x, y, z):
        self.cmd = (x, y, z)

    def stop(self):
        self.cmd = (0.0, 0.0, 0.0)

    def gimbal_to(self, yaw, pitch=None, wait=True):
        pitch = self.gimbal_pitch if pitch is None else pitch
        dt = max(abs(yaw - self.gimbal_yaw) / C.GIMBAL_YAW_SPEED,
                 abs(pitch - self.gimbal_pitch) / C.GIMBAL_PITCH_SPEED) + 0.03
        self.gimbal_yaw, self.gimbal_pitch = yaw, pitch
        self._gimbal_ready = self.t + dt            # the ToF reads again once it stops
        if wait:
            self.sleep(dt)

    def gimbal_by(self, dyaw, dpitch):
        self.gimbal_to(self.gimbal_yaw + dyaw, self.gimbal_pitch + dpitch, True)

    def fire(self, times):
        heading = self.yaw + self.gimbal_yaw
        best = None
        for tg in self.targets:
            dx, dy = tg["x"] - self.x, tg["y"] - self.y
            dist = math.hypot(dx, dy)
            b = (math.degrees(math.atan2(dx, dy)) - heading + 180) % 360 - 180
            half = math.degrees(math.atan(plate_size(tg["label"])[0] / 2 / dist))
            if (abs(b) <= half and self._facing(tg, dx, dy, dist) > 0.2
                    and self._visible(self.x, self.y, tg["x"], tg["y"])):
                if best is None or dist < best[1]:
                    best = (tg, dist)
        hit = best[0]["label"] if best else None
        self.hits.append(dict(t=round(self.t, 2), hit=hit, id=best[0]["id"] if best else None,
                              dist_tiles=round(best[1] / T, 2) if best else None))
        self.sleep(0.05 + C.FIRE_GAP * (times - 1))

    def close(self):
        pass

    def truth(self):
        return sorted(t["id"] for t in self.targets)
