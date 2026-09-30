"""Round 1 (explore + map + shoot) and Round 2 (plan optimal route + shoot)."""
import collections
import math
import threading
import time

import config as C
from hal import YAW_OF_DIR, wrap
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
            if reached != self.path[-1]:
                self.path.append(reached)
            self.cell = reached
            self.last_dir = d
            self.t_ready = self.hal.now() + C.VIDEO_LATENCY
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
        for t in self.pending():                        # in range and facing us: shoot
            if (t.side == d and t.cell in line
                    and self.face_dist(line.index(t.cell)) <= C.FIRE_RANGE_M):
                self.engage(t, d)

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
                return t
        t = Target(label=label, cell=cell, side=side, dist=dist, shot=False, attempts=0,
                   designated=label in self.shoot)
        self.targets.append(t)
        self.motion.ignore.add((cell, side))            # its plate is not the wall
        self.log("TARGET %-12s at cell %s %s wall  (%.2f m)%s" % (
            label, cell, side, dist, "  <- designated" if t.designated else ""))
        return t

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
        self.sense_walls(c)
        for d in DIRS:                            # keep chassis + barrel off plates
            if self.hal.bump(d) and self.room_behind(OPP[d]):
                self.motion.nudge(OPP[d], 0.06)
                break
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
        self.maze.visited.add(c)
        self.motion.center_in_cell()              # walls checked: centre on the cell
        for t in self.pending():
            for s, d in self.firing_spots(t):
                if s == c and not t.shot:
                    self.engage(t, d)

    # --------------------------------------------------------- round 1 ----
    def done_round1(self):
        if not C.EXPECTED_TARGETS:
            return False
        return (len(self.targets) >= C.EXPECTED_TARGETS
                and not any(t.label in self.shoot and not t.shot for t in self.targets))

    def side_bias_check(self):
        """Start of a round: learn each side Sharp's offset against the start tile."""
        f = getattr(self.hal, "side_bias_check", None)
        if f:
            got = f()
            if got:
                self.log("Sharp bias at start: %s (removed from later readings)" % ", ".join(
                    "%s %+.1f cm" % (k, 100 * v) for k, v in sorted(got.items())))

    def run_round1(self):
        self.log("ROUND 1: explore, map and shoot %s" % sorted(self.shoot))
        self.side_bias_check()
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
            goals = {(c, r) for c in range(self.maze.w) for r in range(self.maze.h)
                     if self.needs_visit((c, r))}
            for t in self.pending():
                goals.update(s for s, _ in self.firing_spots(t))
            goals -= blocked | {self.cell} | self.unreachable
            if not goals:
                self.log("every wall checked")
                break
            path, _ = self.maze.plan(self.cell, goals, C.COST_TILE, C.COST_SEGMENT,
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
        self.side_bias_check()
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
            if s != self.cell:
                path, _ = self.maze.plan(self.cell, {s}, C.COST_TILE, C.COST_SEGMENT,
                                         self.last_dir, self.blocked())
                if path is None:
                    self.log("no path to %s" % (s,))
                    continue
                self.follow(path)
                if self.cell != s:              # blocked on the way: never shoot from elsewhere
                    self.log("did not reach %s (at %s) - skipping %s" % (s, self.cell, t.label))
                    continue
            self.engage(t, d)
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
