"""Round 1 (explore + map + shoot) and Round 2 (plan optimal route + shoot)."""
import collections
import math
import threading
import time

import config as C
from hal import YAW_OF_DIR, side_dir, wrap
from maze import DIRS, OPEN, OPP, UNKNOWN, WALL, Maze, compress, step
from motion import Motion
from vision import bearing, distance

T = C.TILE


class Target(dict):
    """label, cell, side, dist, shot, attempts (a dict, so it serialises).
    side = the wall of `cell` the target stands against (it faces away from it);
    it is shot by looking towards `side` from `cell` or up to 2 cells back."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Mission:
    def __init__(self, hal, round_no, shoot, maze=None, targets=None):
        self.hal = hal
        self.round = round_no
        self.shoot = set(shoot)
        self.maze = maze or Maze(C.GRID_W, C.GRID_H, C.ASSUME_BOUNDARY_WALLS)
        self.abort = threading.Event()      # set() to stop the round safely
        self.pause = threading.Event()      # set() to hold still (clock keeps running)
        self.motion = Motion(hal, self.maze, abort=self.abort, pause=self.pause)
        # aiming telemetry for the panel: phase, label, error (deg), history
        self.aim = {"phase": "IDLE", "label": None, "yaw": None, "pitch": None,
                    "t": 0.0, "wall": 0.0, "hist": collections.deque(maxlen=120)}
        self.targets = [Target(t, cell=tuple(t["cell"]), side=t.get("side"))
                        for t in (targets or [])]
        self.faces = self.maze.faces        # (cell, dir) wall faces the camera checked
        self.unreachable = set()            # goals given up after repeated obstacle stops
        self._tof_seen = {}                 # d -> (ToF wall distance, odom n, e) this visit
        self._suspects = set()              # (cell, dir) walls to re-measure (map closed in)
        self._remeasured = set()            # edge keys re-measured with the ToF
        self._visits = collections.Counter()    # scans per cell
        self._fails = collections.Counter()
        self.motion.ignore.update((t.cell, t.side) for t in self.targets if t.side)
        self.path = [tuple(C.START_CELL)]
        self.shots = []
        self.cell = tuple(C.START_CELL)
        self.last_dir = None
        self.t_ready = hal.now()           # frames older than this are stale
        self.t0 = hal.now()
        self.t_end = None                  # set when the round stops: the clock freezes
        self.deadline = self.t0 + C.ROUND_LIMIT[round_no] - C.TIME_MARGIN

    # ----------------------------------------------------------- util ----
    def elapsed(self):
        end = self.t_end if self.t_end is not None else self.hal.now()
        return end - self.t0

    def finish(self):
        """Stop the round clock (first call wins)."""
        if self.t_end is None:
            self.t_end = self.hal.now()

    def log(self, msg):
        print("[%6.1fs] %s" % (self.elapsed(), msg))

    def fresh(self, after, timeout=1.0):
        end = self.hal.now() + timeout
        while True:
            t, dets = self.hal.detections()
            if t >= after:
                return dets
            if self.hal.now() > end or self.abort.is_set():
                return []
            self.hal.sleep(0.01)

    def gimbal_for(self, d, current=None):
        """Gimbal yaw (vs the chassis) that looks along map direction d; for the
        direction behind, the closer of +180 / -180."""
        current = self.hal.gimbal_yaw if current is None else current
        rel = wrap(YAW_OF_DIR[d] - YAW_OF_DIR[self.motion.heading])
        if abs(rel) == 180.0 and current < 0:
            rel = -180.0
        return rel

    def confirmed(self, after):
        """Detections that show in CONFIRM_FRAMES frames in a row (same label,
        bearing within 3 deg) - a flicker is not a card."""
        t, dets = self.hal.detections()
        end = self.hal.now() + 1.0
        while t < after and self.hal.now() < end and not self.abort.is_set():
            self.hal.sleep(0.01)
            t, dets = self.hal.detections()
        for _ in range(C.CONFIRM_FRAMES - 1):
            t0 = t
            while t <= t0 and self.hal.now() < end and not self.abort.is_set():
                self.hal.sleep(0.01)
                t, more = self.hal.detections()
            if t <= t0:
                return []
            dets = [x for x in dets if x.label and any(
                y.label == x.label and abs(bearing(y)[0] - bearing(x)[0]) < 3.0 for y in more)]
        return dets

    def face(self, d, wait=True):
        yaw = self.gimbal_for(d)
        err = getattr(self.hal, "gimbal_error", lambda: None)()
        if (abs(yaw - self.hal.gimbal_yaw) > 0.5
                or abs(self.hal.gimbal_pitch - C.GIMBAL_PITCH) > 0.5
                or (err is not None and abs(err) > C.GIMBAL_TRIM_MIN)):
            self.hal.gimbal_to(yaw, C.GIMBAL_PITCH, wait=wait)
            if wait:
                self.t_ready = self.hal.now() + C.VIDEO_LATENCY

    def sensor_ok(self, d):
        return getattr(self.hal, "sensor_ok", lambda _d: True)(d)

    def room_behind(self, d):
        """Safe to back off a few cm towards d."""
        if self.hal.bump(d):
            return False
        r = self.hal.ranges().get(d)
        if r is not None:
            return r > T / 2 - 0.01             # at least a normal wall distance
        cell = self.motion.cell()
        e = self.maze.get(cell, d)
        if e == OPEN:
            return True
        if e == WALL and (cell, d) in self.faces and (cell, d) not in self.motion.ignore:
            # the camera checked that wall and found no plate on it
            x, y, _ = self.motion.pose()
            off = {"N": y - cell[1] * T, "S": cell[1] * T - y,
                   "E": x - cell[0] * T, "W": cell[0] * T - x}[d]
            return off < 0.02                   # centred: ~14 cm to that wall
        return False

    def wait_range(self, d, timeout=0.5):
        """Wait for a gimbal-ToF reading along d (the gimbal has just turned)."""
        end = self.hal.now() + timeout
        while self.hal.now() < end and not self.abort.is_set():
            if self.ranges_ex().get(d, (None, ""))[1] == "tof":
                return True
            self.hal.sleep(0.02)
        return False

    def hold(self):
        """Wait here while paused."""
        while self.pause.is_set() and not self.abort.is_set():
            self.hal.stop()
            self.hal.sleep(0.05)

    def _aim_state(self, phase, label=None, ey=None, ep=None):
        a = self.aim
        a["phase"], a["t"], a["wall"] = phase, self.hal.now(), time.time()
        if label is not None:
            a["label"] = label
        if ey is not None:
            a["yaw"], a["pitch"] = ey, ep
            a["hist"].append((a["t"], ey, ep))

    def pending(self):
        return [t for t in self.targets
                if t.label in self.shoot and not t.shot and t.attempts < 2]

    # --------------------------------------------------------- motion ----
    def split_segments(self, path):
        """A long run that ends at an unchecked wall stops 2 cells short, so the
        camera checks that wall (a plate on it) before the robot gets close."""
        out, c = [], self.cell
        for d, n in compress(path):
            dest = step(c, d, n)
            if n > 2 and self.maze.get(dest, d) == WALL and (dest, d) not in self.faces:
                out += [(d, n - 2), (d, 2)]
            else:
                out.append((d, n))
            c = dest
        return out

    def follow(self, path):
        for d, n in self.split_segments(path):
            start = self.cell
            if d in self.unseen_faces(start):
                # check the wall we are driving at from here (>= 0.8 m away): a plate
                # on it is then known before we arrive, so the pose snaps onto it
                self.look(start, d)
            reached = self.motion.run_segment(d, n)
            dx, dy = reached[0] - start[0], reached[1] - start[1]
            moved = abs(dx) + abs(dy) if step(start, d, abs(dx) + abs(dy)) == reached else 0
            self.path.extend(step(start, d, k) for k in range(1, moved + 1))
            for k in range(moved):
                self.maze.drove(step(start, d, k), d)
            if reached != self.path[-1]:
                self.path.append(reached)
            self.cell = reached
            self.last_dir = d
            self.t_ready = self.hal.now() + C.VIDEO_LATENCY
            self.spot()                                 # every card in view goes on the map
            if reached != step(start, d, n) or self.abort.is_set():
                return False
        return True

    # -------------------------------------------------------- mapping ----
    def ranges_ex(self):
        f = getattr(self.hal, "ranges_ex", None)
        return f() if f else {d: (r, "tof") for d, r in self.hal.ranges().items()}

    def sense_walls(self, c):
        """Map the walls of c from whatever is measurable right now: the side
        Sharps, and the gimbal ToF in the direction it points."""
        m = self.maze
        for d, (r, kind) in self.ranges_ex().items():
            if kind == "tof" and r is not None:
                n, e, _ = self.hal.odom()
                self._tof_seen[d] = (r, n, e)           # for recenter_tof (robot still here?)
            if r is None and kind == "sharp":
                # beyond its range OR closer than its minimum - ambiguous: a straight
                # IR module decides, otherwise the gimbal ToF measures it
                if self.hal.bump(d, strict=True):
                    m.set(c, d, WALL)
                continue
            wall = r is not None and r <= T / 2 + C.WALL_TOL
            m.set(c, d, WALL if wall else OPEN)
            if wall or kind != "tof":
                continue
            # long-range ToF: pre-map the corridor ahead (never overrides)
            r = C.TOF_MAX if r is None else r
            cc, k = step(c, d), 1
            while m.inside(cc):
                b = (k + 0.5) * T
                if abs(r - b) <= C.WALL_TOL:
                    m.set(cc, d, WALL, force=False)
                    break
                if r < b:
                    break
                m.set(cc, d, OPEN, force=False)
                cc, k = step(cc, d), k + 1
        m.visited.add(c)

    # --------------------------------------------------------- vision ----
    @staticmethod
    def face_dist(k):
        """Camera -> plate distance for a target on the wall k cells ahead."""
        return (k + 0.5) * T - C.TARGET_WALL_GAP

    def look(self, c, d):
        """Look towards d from c: map every card on the first wall ahead."""
        self.face(d)
        dets = self.confirmed(self.t_ready)
        if not any(x.label for x in dets) and self.hal.bump(d) and self.room_behind(OPP[d]):
            # nothing seen, yet something is within ~11 cm: we drifted onto that
            # wall (a plate there is too close to see) - back off and look again
            self.motion.nudge(OPP[d], 0.08)
            self.t_ready = self.hal.now() + C.VIDEO_LATENCY
            dets = self.confirmed(self.t_ready)
        known = self.maze.ray(c, d, C.VIEW_TILES, peek=False)
        self.maze.observed.update(known)
        end = known[-1] if known else c
        if (self.maze.get(end, d) == WALL
                and self.face_dist(len(known)) <= C.VIEW_RANGE_M):
            self.faces.add((end, d))
        line = [c] + self.maze.ray(c, d, C.VIEW_TILES)
        end_known = self.maze.get(line[-1], d) == WALL
        for det in dets:
            if det.label is None:
                continue
            if self.map_seen(det):                      # the wall right behind it (ray)
                continue
            z = distance(det)                           # depth along the view axis
            yaw_b, _ = bearing(det)
            if abs(z * math.tan(math.radians(yaw_b))) > 0.6 * T:
                continue                                # not in this corridor
            if end_known:                               # it is on the first wall
                k = len(line) - 1
                if abs(z - self.face_dist(k)) > 0.6 * T:
                    continue                            # inconsistent: not that wall
            else:
                k = max(0, int(round((z + C.TARGET_WALL_GAP) / T - 0.5)))
            cell = step(c, d, k)
            if not self.maze.inside(cell):
                cell = line[-1]
            self.add_target(det.label, cell, d, z)
        if (known and self.maze.get(end, d) == WALL
                and self.face_dist(len(known)) <= C.VIEW_RANGE_M):
            seen = {x.label.split("_")[0] for x in dets if x.label}   # colours
            missed = []
            for t in self.faced_targets(end, d):        # marked there, not in the picture
                if t.label.split("_")[0] not in seen:   # (the shape of one card can flip)
                    missed.append(t)
                    self.not_seen(t, "looked from %s" % (c,))
        else:
            missed = []
        for t in self.pending():                        # in range and facing us: shoot
            if (t.side == d and t.cell in line and t not in missed
                    and self.face_dist(line.index(t.cell)) <= C.FIRE_RANGE_M):
                self.engage_here(t, d)

    # ------------------------------------------------- every card seen ----
    def place(self, det):
        """(cell, side) of a card seen in the picture: walk the camera ray through
        the map - the card stands just in front of the first wall it meets (an
        unmapped edge counts when the card's range says so).  None if unsure."""
        z = distance(det)
        if z is None or z > C.SPOT_MAX_M:
            return None
        x, y, yaw = self.motion.pose()
        g = self.hal.gimbal_yaw
        err = getattr(self.hal, "gimbal_error", lambda: None)()
        if err is not None:
            g -= err                                    # where the gimbal really points
        b, _ = bearing(det)
        r = z / max(0.3, math.cos(math.radians(b)))     # range to the card
        th = math.radians(yaw + g + b)
        ux, uy = math.sin(th), math.cos(th)
        c = (int(round(x / T)), int(round(y / T)))
        sx, sy = (1 if ux > 0 else -1), (1 if uy > 0 else -1)
        inf = float("inf")
        while True:                                     # cell by cell along the ray
            tx = ((c[0] + 0.5 * sx) * T - x) / ux if abs(ux) > 1e-9 else inf
            ty = ((c[1] + 0.5 * sy) * T - y) / uy if abs(uy) > 1e-9 else inf
            s = min(tx, ty)
            if s > r + C.SPOT_SLACK:
                break
            d = ("E" if sx > 0 else "W") if tx < ty else ("N" if sy > 0 else "S")
            e = self.maze.get(c, d)
            if e == WALL and s < r - C.SPOT_THROUGH and self.maze._key(c, d) not in self._remeasured:
                # a card seen well beyond this wall: the wall is wrong - re-measure it
                self._suspects.add((c, d))
                self.log("card seen %.2f m away behind wall %s %s (%.2f m) - re-measure that "
                         "wall" % (r, c, d, s))
                return None
            if e == WALL or (e == UNKNOWN and abs(s - r - C.TARGET_WALL_GAP) < C.SPOT_SLACK):
                return c, d
            c = step(c, d)
            if not self.maze.inside(c):
                break
        return None                                     # no wall behind it: not placed

    def map_seen(self, det):
        """Put a card seen anywhere in the picture on the map, in its own block,
        on its wall (several per block / per wall).  Not shot from here."""
        if not det.label:
            return None
        spot = self.place(det)
        if spot is None:
            return None
        cell, side = spot
        return self.add_target(det.label, cell, side, distance(det))

    def spot(self):
        """Robot standing still: map every card confirmed in the picture now."""
        for det in self.confirmed(self.hal.now()):
            self.map_seen(det)

    def add_target(self, label, cell, side, dist):
        axis = 0 if side in "NS" else 1                 # coordinate across the line
        for t in self.targets:
            if t.label != label or t.side != side or t.cell[axis] != cell[axis]:
                continue
            gap = abs(t.cell[1 - axis] - cell[1 - axis])
            far = max(dist, t.dist) > 2.5 * T           # far estimates are coarse
            if gap <= (3 if far else 0):
                if dist < t.dist:
                    t.cell, t.dist = cell, dist
                    self.motion.ignore.add((cell, side))
                t["seen"] = t.get("seen", 1) + 1
                return t
        t = Target(label=label, cell=cell, side=side, dist=dist, shot=False, attempts=0,
                   designated=label in self.shoot)
        self.targets.append(t)
        self.motion.ignore.add((cell, side))            # its plate is not the wall
        self.log("TARGET %-12s at cell %s %s wall  (%.2f m)%s" % (
            label, cell, side, dist, "  <- designated" if t.designated else ""))
        return t

    # ------------------------------------------- wrong detections out ----
    def not_seen(self, t, why):
        """t's wall was looked at reliably (from 1+ cells away, or looking down up
        close) and t was not there.  MISS_REMOVE such misses, and at least as many
        as the times it was seen: a wrong detection - off the map (seen again
        later, it comes back).  A target that was shot is never removed."""
        t["misses"] = t.get("misses", 0) + 1
        if t.get("shot") or t["misses"] < C.MISS_REMOVE or t["misses"] < t.get("seen", 1):
            return
        self.targets.remove(t)
        if not any(tuple(o.cell) == tuple(t.cell) and o.side == t.side for o in self.targets):
            self.motion.ignore.discard((tuple(t.cell), t.side))
        self.log("REMOVED %-12s at cell %s %s wall - not there when looked at again "
                 "(%s; seen %d, missed %d)" % (t.label, tuple(t.cell), t.side, why,
                                               t.get("seen", 1), t["misses"]))

    def faced_targets(self, cell, side):
        return [t for t in self.targets if tuple(t.cell) == tuple(cell) and t.side == side]

    def blocked(self):
        if not C.AVOID_TARGET_CELLS:
            return set()
        return {t.cell for t in self.targets}

    def engage(self, t, d):
        self.face(d)
        t.attempts += 1
        expect = None                       # where the plate must be, seen from here
        if t.get("side") and (self.cell[0] == t.cell[0] or self.cell[1] == t.cell[1]):
            k = abs(self.cell[0] - t.cell[0]) + abs(self.cell[1] - t.cell[1])
            expect = self.face_dist(k)
        self._expect = expect
        if expect is not None and expect < C.MIN_SHOOT_M and self.room_behind(OPP[d]):
            back = min(0.10, C.MIN_SHOOT_M - expect + 0.01)   # too close to hit: step back
            self.motion.nudge(OPP[d], back)
            self._expect = expect + back
            self.t_ready = self.hal.now() + C.VIDEO_LATENCY
        ok = self.aim_and_fire(t.label)
        if not ok and self.hal.bump(d) and self.room_behind(OPP[d]):   # too close: back off
            self.motion.nudge(OPP[d], 0.08)
            self.t_ready = self.hal.now() + C.VIDEO_LATENCY
            ok = self.aim_and_fire(t.label)
        if not ok:                         # small sweep, then give up
            for dy in (-12, 12):
                self.hal.gimbal_by(dy, 0)
                self.t_ready = self.hal.now() + C.VIDEO_LATENCY
                ok = self.aim_and_fire(t.label)
                self.face(d)
                if ok:
                    break
        self.face(d)
        if ok:
            t.shot = True
            c = self.cell                     # a hit proves the line is clear
            while c != t.cell and self.maze.inside(c) and (c[0] == t.cell[0] or c[1] == t.cell[1]):
                self.maze.set(c, d, OPEN, force=False)
                c = step(c, d)
            self.shots.append({"label": t.label, "cell": list(t.cell), "side": t.side,
                               "from": list(self.cell), "t": round(self.elapsed(), 2)})
            self.log("SHOT   %-12s at cell %s %s wall from %s" % (
                t.label, t.cell, t.side, self.cell))
        else:
            self.log("miss   %-12s not re-acquired from %s" % (t.label, self.cell))
        return ok

    # ------------------------------------------------------ centring ----
    def tof_wall(self, d):
        """Gimbal ToF distance (m from the robot centre) to the wall on side d:
        a reading taken this visit if the robot has not moved since, else measured."""
        n, e, _ = self.hal.odom()
        got = self._tof_seen.get(d)
        if got is not None and math.hypot(got[1] - n, got[2] - e) < 0.01:
            return got[0]
        self.face(d)
        if not self.wait_range(d):
            return None
        vals = []
        for _ in range(4):
            r, kind = self.ranges_ex().get(d, (None, ""))
            if kind == "tof" and r is not None:
                vals.append(r)
            self.hal.sleep(0.05)
        return sorted(vals)[len(vals) // 2] if len(vals) >= 3 else None

    def recenter_tof(self, c):
        """Final_Robomaster's recenter: where the robot really is in its 60 x 60 cm
        cell, from the gimbal ToF on the cell's walls (accurate to ~1-2 cm standing
        still), so odometry drift never adds up cell after cell.  Per axis:
          two walls - offset = half the difference; used only when the two add up to
                      the 60 cm cell (a card plate on one reads 8 cm short);
          one wall  - used only when the camera checked it and found no plate.
        Then the pose is set there (center_in_cell drives to the true centre)."""
        m = self.maze
        for a, b in (("N", "S"), ("E", "W")):
            walls = [d for d in (a, b) if m.get(c, d) == WALL]
            if not walls:
                continue                                # open both ways: nothing to measure
            clean = [d for d in walls if (c, d) in self.faces and (c, d) not in self.motion.ignore]
            if len(walls) == 2:
                ra, rb = self.tof_wall(a), self.tof_wall(b)
                if ra is None or rb is None or abs(ra + rb - T) > C.RECENTER_SUM_TOL:
                    continue                            # a plate on one of them: unsure
                off = (rb - ra) / 2.0                   # m towards a, from the centre
            elif clean:
                r = self.tof_wall(clean[0])
                if r is None or abs(r - T / 2) > C.RECENTER_MAX:
                    continue
                off = (T / 2 - r) if clean[0] == a else (r - T / 2)
            else:
                continue
            want = (c[1] if a == "N" else c[0]) * T + off
            x, y, _ = self.motion.pose()
            have = y if a == "N" else x
            if abs(want - have) >= C.RECENTER_LOG:
                print("[centre] %s: off the centre %+.1f cm %s by ToF (pose said %+.1f)" % (
                    c, 100 * off, a, 100 * (have - (c[1] if a == "N" else c[0]) * T)))
            if a == "N":
                self.motion.off_n += want - have
            else:
                self.motion.off_e += want - have

    # ------------------------------------------------------ dead ends ----
    def dead_end(self, c):
        """A block with 3 walls."""
        return sum(self.maze.get(c, d) == WALL for d in DIRS) == 3

    def look_down(self, c, d, only=None):
        """Dead end (Dhai_8 "stamp"): gimbal to wall d, tilted down to
        DEAD_END_PITCH in the same move, sweep DEAD_END_SWEEP.  Each confirmed card is a target on
        (c, d); a designated one is shot right there, the gimbal still tilted down.
        only = shoot just this target (round 2).  True if something was shot."""
        base = self.gimbal_for(d)
        # straight to the wall AND down in one move (turning level first looked like
        # it never aimed down)
        print("[dead end] %s %s wall: looking down %.0f deg, sweep %s" % (
            c, d, C.DEAD_END_PITCH, "/".join("%+.0f" % o for o in C.DEAD_END_SWEEP)))
        hit = False
        seen = set()
        swept = True
        for off in C.DEAD_END_SWEEP:
            if self.abort.is_set():
                swept = False
                break
            self.hal.gimbal_to(base + off, C.DEAD_END_PITCH, wait=True)
            self.t_ready = self.hal.now() + C.VIDEO_LATENCY
            for det in self.confirmed(self.t_ready):
                if not det.label:
                    continue
                seen.add(det.label.split("_")[0])          # colour (shapes flip)
                if distance(det) > C.DEAD_END_NEAR:     # further off: not on this wall
                    self.map_seen(det)
                    continue
                cell, side = self.place(det) or (c, d)  # swept off the wall: maybe the next
                t = self.add_target(det.label, cell, side, distance(det))
                if (t.designated and not t.shot and t.attempts < 2
                        and (only is None or t is only)):
                    hit |= self.fire_down(t, d)
            if only is not None and only.shot:
                swept = False
                break
        # (a card NOT seen up close is no proof it is not there - ~0.12 m from the
        # camera it may not fit the picture: last run removed a real card here)
        self.faces.add((c, d))
        return hit

    def fire_down(self, t, d):
        """Aim (the usual aim_and_fire, from the tilted-down gimbal) and fire."""
        t.attempts += 1
        self._expect = None                   # no distance gate this close
        ok = self.aim_and_fire(t.label)
        if ok:
            t.shot = True
            self.shots.append({"label": t.label, "cell": list(t.cell), "side": t.side,
                               "from": list(self.cell), "t": round(self.elapsed(), 2)})
            self.log("SHOT   %-12s at cell %s %s wall from %s (dead end, looking down)" % (
                t.label, t.cell, t.side, self.cell))
        else:
            self.log("miss   %-12s not re-acquired looking down in %s" % (t.label, self.cell))
        return ok

    def engage_here(self, t, d):
        """Shoot t from the robot's cell: inside a dead end by looking down."""
        if self.cell == t.cell and self.dead_end(t.cell):
            return self.look_down(self.cell, d, only=t)
        ok = self.engage(t, d)
        k = abs(self.cell[0] - t.cell[0]) + abs(self.cell[1] - t.cell[1])
        if not ok and k >= 1 and self.aim.get("phase") == "LOST" and t in self.targets:
            self.not_seen(t, "lost from %s" % (self.cell,))   # not in the picture at all
        return ok

    def aim_and_fire(self, label):
        after = max(self.t_ready, self.hal.now() + C.VIDEO_LATENCY)
        self._aim_state("AIMING", label)
        for i in range(C.AIM_ITERS + 1):
            dets = [x for x in self.fresh(after) if x.label == label]
            if not dets:
                self._aim_state("LOST")
                return False
            expect = getattr(self, "_expect", None)
            if expect is not None:              # a card at the wrong distance is another card
                dets = [x for x in dets if abs(distance(x) - expect) < 0.35]
                if not dets:
                    self._aim_state("LOST")
                    return False
            det = min(dets, key=lambda x: abs(bearing(x)[0]))
            if distance(det) > C.FIRE_RANGE_M + 0.15:
                self._aim_state("TOO FAR")
                return False                    # rule: only fire from <= 2 tiles
            ey, ep = bearing(det)
            ey += C.AIM_YAW_OFFSET
            ep += C.AIM_PITCH_OFFSET + math.degrees(       # barrel sits below the camera
                math.atan(C.BARREL_BELOW_CAMERA_M / max(0.1, distance(det))))
            locked = abs(ey) < C.AIM_TOL and abs(ep) < C.AIM_TOL
            self._aim_state("LOCKED" if locked else "AIMING", None, ey, ep)
            if locked:
                break
            if i == C.AIM_ITERS:
                if abs(ey) > 3 * C.AIM_TOL:
                    self._aim_state("LOST")
                    return False
                break
            self.hal.gimbal_by(ey, ep)
            after = self.hal.now() + C.VIDEO_LATENCY
        self._aim_state("FIRE" if getattr(self.hal, "fire_enabled", True) else "DRY RUN")
        self.hal.fire(C.FIRE_TIMES)
        return True

    def firing_spots(self, t):
        """[(cell, dir)] to shoot t from: in front of its wall, <= 2 cells back,
        clear line, looking towards the wall."""
        out, blocked = [], self.blocked()
        for d in ([t.side] if t.get("side") else DIRS):
            for k in range(0 if t.get("side") else 1, C.FIRE_RANGE_TILES + 1):
                if t.get("side") and self.face_dist(k) > C.FIRE_RANGE_M:
                    break
                s = step(t.cell, OPP[d], k)
                if (not self.maze.inside(s) or (s in blocked and s != t.cell)
                        or not self.maze.clear_line(s, t.cell)):
                    break
                out.append((s, d))
        return out

    def needs_visit(self, c):
        """A cell still has an unknown edge or an unchecked wall face."""
        m = self.maze
        return any(m.get(c, d) == UNKNOWN or (m.get(c, d) == WALL and (c, d) not in self.faces)
                   for d in DIRS)

    def unseen_faces(self, c):
        """Directions from c whose first wall face is unchecked and close enough."""
        out = []
        for d in DIRS:
            known = self.maze.ray(c, d, C.VIEW_TILES, peek=False)
            end = known[-1] if known else c
            if (self.maze.get(end, d) == WALL and (end, d) not in self.faces
                    and self.face_dist(len(known)) <= C.VIEW_RANGE_M):
                out.append(d)
        return out

    def visit(self, c):
        """Arrived at c: map its walls (Sharps now, the gimbal ToF as the gimbal
        turns), look at every unchecked wall face, shoot what is in range."""
        self._visits[c] += 1
        self._tof_seen = {}
        self.sense_walls(c)
        self.remeasure(c)
        if not getattr(self, "_bias_done", True):
            # no wall beside the start tile: calibrate the Sharp on the first one
            self._bias_done = self.side_bias_check(deferred=True)
        for d in DIRS:                            # keep chassis + barrel off plates
            if self.hal.bump(d) and self.room_behind(OPP[d]):
                self.motion.nudge(OPP[d], 0.06)
                break
        if self.dead_end(c):                      # Dhai_8 stamp: look down at all 3 walls
            self.log("DEAD END %s: looking down at its 3 walls" % (c,))
            for d in DIRS:
                if self.maze.get(c, d) == WALL:
                    self.look_down(c, d)
        tried = set()
        while not self.abort.is_set():
            unseen = self.unseen_faces(c)
            need = [d for d in DIRS if d not in tried
                    and (self.maze.get(c, d) == UNKNOWN or d in unseen)]
            if not need:
                break
            g = self.hal.gimbal_yaw               # nearest gimbal angle first
            d = min(need, key=lambda d: abs(self.gimbal_for(d, g) - g))
            tried.add(d)
            self.face(d)
            if self.maze.get(c, d) == UNKNOWN:
                self.wait_range(d)                # the ToF now points along d
                self.sense_walls(c)
            if d in self.unseen_faces(c):
                self.look(c, d)
        if not self.dead_end(c):                  # cards marked on its own walls: check them
            for d in DIRS:                        # up close, looking down (they sit low)
                if (self.maze.get(c, d) == WALL
                        and any(not t.shot for t in self.faced_targets(c, d))):
                    self.look_down(c, d)
        self.maze.visited.add(c)
        self.spot()
        self.recenter_tof(c)                      # where it really is, from the walls
        self.motion.center_in_cell()              # walls checked: centre on the cell
        for t in self.pending():
            for s, d in self.firing_spots(t):
                if s == c and not t.shot:
                    self.engage_here(t, d)

    # --------------------------------------------------------- round 1 ----
    def done_round1(self):
        if not C.EXPECTED_TARGETS:
            return False
        return (len(self.targets) >= C.EXPECTED_TARGETS
                and not any(t.label in self.shoot and not t.shot for t in self.targets))

    def side_bias_check(self, deferred=False):
        """Start of a round: learn each side Sharp's offset against the gimbal ToF
        turned to that wall.  Assuming the robot stands exactly on the tile centre
        turned a placement error into a Sharp "calibration" that the robot then
        held all run (placed 5 cm left -> ran 5 cm left).  The ToF also puts the
        pose's sideways axis where the robot really is."""
        f = getattr(self.hal, "side_bias_check", None)
        if not f:
            return True
        ref = {}
        ex = self.ranges_ex()
        for side in C.SHARP:
            d = side_dir(self.motion.heading, side)
            r, kind = ex.get(d, (None, ""))
            if kind != "sharp" or r is None or r > C.SIDE_WALL_MAX:
                continue                            # no wall seen that side
            self.face(d)
            if not self.wait_range(d):
                continue
            vals = []
            for _ in range(5):                      # a few fresh ToF samples
                t, kind = self.ranges_ex().get(d, (None, ""))
                if kind == "tof" and t is not None:
                    vals.append(t)
                self.hal.sleep(0.05)
            if len(vals) >= 3:
                t = sorted(vals)[len(vals) // 2]
                if abs(t - C.SIDE_NOMINAL) <= C.SHARP_BIAS_MAX:   # not a target plate
                    ref[side] = t
                    self.motion.place_axis(d, t)
        if deferred and not ref:
            return False
        got = f(ref, strict=deferred)
        if got:
            self.log("Sharp bias %s: %s (removed from later readings)" % (
                "at %s" % (self.cell,) if deferred else "at start", ", ".join(
                "%s %+.1f cm%s" % (k, 100 * v, " vs ToF %.2f m" % ref[k] if k in ref else
                                   " (no ToF: assumes the tile centre)")
                for k, v in sorted(got.items()))))
        return bool(ref)

    # ------------------------------------------------- unknown map: goals ----
    def shoot_only(self):
        """Round 1 is past EXPLORE_FRACTION of its time: only shoot what is found."""
        return self.elapsed() > C.EXPLORE_FRACTION * C.ROUND_LIMIT[1]

    def goal_value(self, s, shots):
        """What standing (and scanning) at cell s gains, the map being unknown:
        its unknown edges (the map), its unchecked wall faces (the targets),
        designated targets that can be shot from there, a dead end not yet
        searched (looking down)."""
        m = self.maze
        sus = sum((s, d) in self._suspects for d in DIRS)
        if self._visits[s] >= 2:                # scanned twice: what is left there stays
            return C.VALUE_SHOT * shots.get(s, 0) + C.VALUE_EDGE * sus
        unknown = sum(m.get(s, d) == UNKNOWN for d in DIRS)
        # its own walls not yet checked (faces further off get checked on the way:
        # counting those too made extra stops - 7% slower in the simulator)
        faces = sum(m.get(s, d) == WALL and (s, d) not in self.faces for d in DIRS)
        dead = 1 if self.dead_end(s) and s not in m.visited else 0
        return (C.VALUE_EDGE * (unknown + sus) + C.VALUE_FACE * faces
                + C.VALUE_SHOT * shots.get(s, 0) + C.VALUE_DEAD_END * dead)

    def next_goal(self, blocked):
        """Round 1 on an unknown map: the next cell to go to and scan.
          * GOAL_RULE "near": the nearest cell that gains anything, a cell that
            gains more counting as nearer (VALUE_PULL tiles per point);
          * GOAL_RULE "rate": the most gain per second, value / (travel + SCAN_COST)
            (Final_Robomaster's information_rate).
        Late in the round (shoot_only) cells to shoot found designated targets
        from come first; with none of those, it keeps exploring."""
        shots = collections.Counter()
        for t in self.pending():
            for s, _ in self.firing_spots(t):
                shots[s] += 1
        cost = self.maze.costs_from(self.cell, C.COST_TILE, C.COST_SEGMENT, blocked,
                                    self.last_dir)     # the first turn costs too
        left = self.deadline - self.hal.now()
        late = self.shoot_only()
        best, best_key = None, None
        for s, c in cost.items():
            if s == self.cell or s in blocked or s in self.unreachable:
                continue
            if c * C.SEC_PER_COST > left:           # no time to get there
                continue
            v = self.goal_value(s, shots)
            if v <= 0:
                continue
            if C.GOAL_RULE == "rate":
                key = -v / (c + C.SCAN_COST)
            else:
                key = c - C.VALUE_PULL * v
            if late:                                # found targets first
                key = (0 if shots.get(s) else 1, c if shots.get(s) else key)
            else:
                key = (0, key)
            if best_key is None or key < best_key:
                best, best_key = s, key
        return best

    def find_suspects(self, blocked):
        """Nothing left to explore, yet some cells cannot be reached on the map -
        the maze is connected, so a wall on the border of the reached area is
        wrong (sensed off-centre / at a wall end: last run closed 9 cells in).
        Mark the border walls not yet re-measured as suspect.  True if any."""
        reach = set(self.maze.costs_from(self.cell, C.COST_TILE, C.COST_SEGMENT, blocked))
        m, new = self.maze, set()
        for c in reach:
            for d in DIRS:
                n = step(c, d)
                k = m._key(c, d)
                if (m.inside(n) and n not in reach and m.get(c, d) == WALL
                        and k not in self._remeasured):
                    new.add((c, d))
        if new:
            self.log("map closes the robot in (%d of %d cells reachable): re-measuring %d "
                     "border walls with the ToF" % (len(reach), m.w * m.h, len(new)))
        self._suspects |= new
        return bool(new)

    def remeasure(self, c):
        """Suspect walls of c: the gimbal ToF straight at each decides."""
        m = self.maze
        for d in DIRS:
            if (c, d) not in self._suspects:
                continue
            self._suspects.discard((c, d))
            self._remeasured.add(m._key(c, d))
            r = self.tof_wall(d)
            if r is not None and r > T / 2 + C.WALL_TOL:
                m.edges[m._key(c, d)] = OPEN
                m.driven.add(m._key(c, d))          # the ToF saw through: never a wall again
                self.log("wall %s %s was wrong: ToF %.2f m - open" % (c, d, r))

    def run_round1(self):
        self.log("ROUND 1: explore, map and shoot %s" % sorted(self.shoot))
        self._bias_done = self.side_bias_check()
        self.visit(self.cell)
        while not self.done_round1():
            self.hold()
            if self.abort.is_set():
                self.log("aborted")
                break
            if self.hal.now() > self.deadline:
                self.log("time limit reached")
                break
            blocked = self.blocked()
            goal = self.next_goal(blocked)
            if goal is None and self.find_suspects(blocked):
                goal = self.next_goal(blocked)
            if goal is None:
                self.log("every wall checked")
                break
            path, _ = self.maze.plan(self.cell, {goal}, C.COST_TILE, C.COST_SEGMENT,
                                     self.last_dir, blocked)
            if path is None:
                if len(self.maze.visited) <= 1:
                    self.log("WARNING: no way out of the start cell %s - every side reads as a "
                             "wall or has no sensor data. Check the sensor report at connect "
                             "and the Sensors card." % (self.cell,))
                else:
                    self.log("nothing left to explore")
                break
            if not self.follow(path) and not self.abort.is_set():
                self._fails[path[-1]] += 1              # stopped early (obstacle)
                if self._fails[path[-1]] >= 2:
                    self.unreachable.add(path[-1])
                    self.log("giving up on %s after repeated obstacle stops" % (path[-1],))
            self.visit(self.cell)
        self.hal.stop()
        self.finish()
        self.log("ROUND 1 finished: %d cells visited, %d wall faces checked, %d targets, %d shot" % (
            len(self.maze.visited), len(self.faces), len(self.targets),
            sum(1 for t in self.targets if t.shot)))

    # --------------------------------------------------------- round 2 ----
    def plan_round2(self):
        todo = [t for t in self.targets if t.label in self.shoot]
        spots = {id(t): self.firing_spots(t) for t in todo}
        todo = [t for t in todo if spots[id(t)]]
        start = self.cell
        cells = {start} | {s for v in spots.values() for s, _ in v}
        blocked = self.blocked()
        cost = {a: self.maze.costs_from(a, C.COST_TILE, C.COST_SEGMENT, blocked)
                for a in cells}
        n = len(todo)
        inf = float("inf")

        def hop(p, i, s):
            close = C.COST_CLOSE_SPOT if s == todo[i].cell else 0.0
            return cost[p].get(s, inf) + close

        if n <= C.ROUTE_EXACT_MAX:
            # Held-Karp: best cost to have shot the set `mask`, standing at cell s
            # (exact; grows 2^n instead of n! - 10 targets: ~1 s, not hours)
            dp = {(0, start): (0.0, None)}
            by_mask = {0: [start]}
            for mask in range(1 << n):
                for cur in by_mask.get(mask, []):
                    g = dp[(mask, cur)][0]
                    for i in range(n):
                        if mask >> i & 1:
                            continue
                        nm = mask | (1 << i)
                        for sp, d in spots[id(todo[i])]:
                            g2 = g + hop(cur, i, sp)
                            key = (nm, sp)
                            if g2 < dp.get(key, (inf,))[0]:
                                if key not in dp:
                                    by_mask.setdefault(nm, []).append(sp)
                                dp[key] = (g2, (mask, cur, i, sp, d))
            full = (1 << n) - 1
            ends = [(dp[(full, c)][0], c) for c in by_mask.get(full, [])]
            if not ends:
                return []
            _, cur = min(ends)
            plan, mask = [], full
            while mask:
                _, (pm, pc, i, sp, d) = dp[(mask, cur)]
                plan.append((todo[i], sp, d))
                mask, cur = pm, pc
            best_plan = plan[::-1]
        else:
            # many targets: nearest next shot first
            left, cur, best_plan = list(range(n)), start, []
            while left:
                g, i, sp, d = min((hop(cur, i, sp), i, sp, d)
                                  for i in left for sp, d in spots[id(todo[i])])
                if g == inf:
                    break
                best_plan.append((todo[i], sp, d))
                left.remove(i)
                cur = sp
        return best_plan

    def run_round2(self):
        self._bias_done = self.side_bias_check()
        for t in self.targets:
            t.shot, t.attempts = False, 0
        plan = self.plan_round2()
        self.log("ROUND 2 plan: " + ", ".join(
            "%s from %s" % (t.label, s) for t, s, _ in plan))
        for t, s, d in plan:
            self.hold()
            if self.abort.is_set():
                self.log("aborted")
                break
            if self.hal.now() > self.deadline:
                self.log("time limit reached")
                break
            if t.shot:
                continue
            for _ in range(C.ROUND2_TRIES):         # a move that gave up: plan again, retry
                if s == self.cell or self.abort.is_set():
                    break
                path, _ = self.maze.plan(self.cell, {s}, C.COST_TILE, C.COST_SEGMENT,
                                         self.last_dir, self.blocked())
                if path is None:
                    self.log("no path to %s" % (s,))
                    break
                self.follow(path)
            if s != self.cell:                      # blocked on the way: never shoot from elsewhere
                self.log("did not reach %s (at %s) - skipping %s" % (s, self.cell, t.label))
                continue
            self.engage_here(t, d)
        self.hal.stop()
        self.finish()
        self.log("ROUND 2 finished: %d/%d designated targets shot" % (
            sum(1 for t in self.targets if t.shot),
            sum(1 for t in self.targets if t.label in self.shoot)))

    # ------------------------------------------------------------ I/O ----
    def to_dict(self):
        return {
            "round": self.round,
            "elapsed": round(self.elapsed(), 2),
            "tile": T,
            "start": list(C.START_CELL),
            "shoot": sorted(self.shoot),
            "maze": self.maze.to_dict(),
            "targets": [dict(t, cell=list(t.cell)) for t in self.targets],
            "path": [list(c) for c in self.path],
            "trace": self.motion.trace,
            "shots": self.shots,
        }
