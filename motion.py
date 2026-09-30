"""Grid-snapped localisation + grid-by-grid motion (Dhai_8 style).

Motion: the chassis turns in place to face the next cell, drives FORWARD one
cell (heading held, centred between the walls by the side Sharps, stopped on
the cell centre by odometry / the front ToF) and stands still before the next
step.  While turning and driving the gimbal faces forward in chassis-lead mode;
scans run in free mode.

Pose = wheel odometry (world frame of the start) + an offset.  Walls only exist
on tile boundaries, so any range reading that lands near a boundary tells us
exactly where we are on that axis; the offset is nudged towards it (SNAP_GAIN).
"""
import math
import threading

import config as C
from maze import OPEN, step
from hal import YAW_OF_DIR, chassis_dir, wrap

T = C.TILE
SIGN = {"N": 1, "E": 1, "S": -1, "W": -1}


def clamp(v, lim):
    return max(-lim, min(lim, v))


def to_chassis(vn, ve, yaw):
    """Map-frame velocity (north, east) -> chassis frame (forward, right)."""
    r = math.radians(yaw)
    return vn * math.cos(r) + ve * math.sin(r), -vn * math.sin(r) + ve * math.cos(r)


class Motion:
    def __init__(self, hal, maze, start_cell=None, abort=None, pause=None):
        self.hal, self.maze = hal, maze
        self.abort = abort or threading.Event()
        self.pause = pause or threading.Event()
        self.ignore = set()        # (cell, dir) wall faces with a target on them
        start_cell = start_cell or C.START_CELL
        n, e, _ = hal.odom()
        self.off_n = start_cell[1] * T - n
        self.off_e = start_cell[0] * T - e
        self.trace = []
        self._last_trace = -1.0
        self.heading = chassis_dir(hal.odom()[2]) or "N"     # way the chassis faces

    # ------------------------------------------------------------ pose ----
    def pose(self):
        """(X east m, Y north m, yaw deg)."""
        n, e, yaw = self.hal.odom()
        return e + self.off_e, n + self.off_n, yaw

    def cell(self):
        x, y, _ = self.pose()
        return (int(round(x / T)), int(round(y / T)))

    def corner_ir(self):
        """(left, right): a front-corner IR on that side of the chassis is on."""
        irs = self.hal.ir_state() if hasattr(self.hal, "ir_state") else {}
        return (any(on for k, on in irs.items() if on and C.IR[k]["pos"][0] < 0),
                any(on for k, on in irs.items() if on and C.IR[k]["pos"][0] > 0))

    def localize(self, moving_axis=None, speed=0.0):
        from hal import side_dir
        x, y, _ = self.pose()
        rc = (int(round(x / T)), int(round(y / T)))
        ex = self.hal.ranges_ex() if hasattr(self.hal, "ranges_ex") else {
            k: (r, "tof") for k, r in self.hal.ranges().items()}
        # a side whose front-corner IR is on is scraping a wall: its Sharp may fold
        # back (closer than 4 cm it reads LONG) - do not trust it now
        scraping = set()
        if self.heading:
            lo, ro = self.corner_ir()
            if lo:
                scraping.add(side_dir(self.heading, "L"))
            if ro:
                scraping.add(side_dir(self.heading, "R"))
        for d, (r, kind) in ex.items():
            if r is None or r > C.SNAP_MAX_RANGE:
                continue
            if kind == "sharp" and d in scraping:
                continue
            ns = d in "NS"
            if ns and moving_axis == "Y" and speed > C.SNAP_MAX_SPEED:
                continue
            if not ns and moving_axis == "X" and speed > C.SNAP_MAX_SPEED:
                continue
            pos = y if ns else x
            s = SIGN[d]
            p = pos + s * r
            k = math.floor(p / T)                      # nearest boundary (k+0.5)*T
            b = (k + 0.5) * T
            # boundary edge on the robot's side, in the robot's row/column
            along_idx = k if s > 0 else k + 1
            edge_cell = (rc[0], along_idx) if ns else (along_idx, rc[1])
            if self.maze.get(edge_cell, d) == OPEN:
                continue                                # no wall there
            # a known target's plate stands TARGET_WALL_GAP in front of its wall:
            # the beam hits either the plate or the wall beside it
            surf = [b, b - s * C.TARGET_WALL_GAP] if (edge_cell, d) in self.ignore else [b]
            b = min(surf, key=lambda v: abs(p - v))
            if s * (p - b) > 0 and kind == "tof":
                tol = C.SNAP_TOL_FAR            # ToF: wall farther than expected, never a plate
            elif (edge_cell, d) in self.maze.faces:
                tol = C.SNAP_TOL_CHECKED        # plate or not is known here
            else:
                tol = C.SNAP_TOL
            if abs(p - b) > tol:
                continue
            err = (b - s * r) - pos
            corr = clamp(C.SNAP_GAIN * err, C.SNAP_STEP_MAX)
            if ns:
                self.off_n += corr
            else:
                self.off_e += corr

    def _trace(self):
        t = self.hal.now()
        if t - self._last_trace >= 0.1:
            x, y, _ = self.pose()
            self.trace.append((round(x, 3), round(y, 3)))
            self._last_trace = t

    # ---------------------------------------------------------- motion ----
    def settle(self, t=C.SETTLE_TIME):
        self.hal.stop()
        end = self.hal.now() + t
        while self.hal.now() < end:
            self.localize()
            self.hal.sleep(0.02)
        for _ in range(3):
            self.localize()

    def hold_wz(self, yaw):
        """Heading hold on the current grid heading."""
        return clamp(-C.YAW_KP * wrap(yaw - YAW_OF_DIR[self.heading]), C.YAW_WMAX)

    def nudge(self, d, dist, speed=0.15):
        """Small straight move towards map direction d (e.g. back off a wall), heading held."""
        hal = self.hal
        vn, ve = {"N": (speed, 0), "S": (-speed, 0), "E": (0, speed), "W": (0, -speed)}[d]
        end = hal.now() + dist / speed
        while hal.now() < end and not self.abort.is_set():
            r = hal.ranges().get(d)
            if hal.bump(d) or (r is not None and r < 0.22):
                break                                  # something close that way: stop
            yaw = self.pose()[2]
            cx, cy = to_chassis(vn, ve, yaw)
            hal.drive(cx, cy, self.hold_wz(yaw))
            hal.sleep(1.0 / C.CTRL_HZ)
        self.settle()

    def turn_to(self, d):
        """Turn the chassis in place to face map direction d (gimbal forward,
        chassis-lead mode, closed loop on yaw; the command sign is probed like
        Dhai_8 and flipped if the error grows)."""
        hal = self.hal
        target = YAW_OF_DIR[d]
        yaw = self.pose()[2]
        if d == self.heading and abs(wrap(yaw - target)) <= C.TURN_TOL:
            return True
        hal.gimbal_front()
        hal.set_mode("chassis_lead")
        sign, ok_ticks = 1.0, 0
        probe = (yaw, wrap(target - yaw), hal.now())
        end = hal.now() + C.TURN_TIMEOUT
        ok = False
        while hal.now() < end and not self.abort.is_set():
            yaw = self.pose()[2]
            err = wrap(target - yaw)
            if abs(err) <= C.TURN_TOL:
                hal.stop()
                ok_ticks += 1
                if ok_ticks >= 3:
                    ok = True
                    break
            else:
                ok_ticks = 0
                if hal.now() - probe[2] >= 0.3:        # is the error shrinking?
                    # compare |error| (a 180 deg turn flips the sign of the error
                    # between +180 and -180 - comparing signs misread that)
                    if abs(err) > abs(probe[1]) + 3.0 and sign > 0:
                        sign = -1.0                    # command sign opposite: flip once
                        print("[motion] turn command sign flipped")
                    probe = (yaw, err, hal.now())
                wz = math.copysign(max(8.0, min(C.TURN_SPEED, C.TURN_KP * abs(err))), err)
                hal.drive(0.0, 0.0, sign * wz)
            hal.sleep(1.0 / C.CTRL_HZ)
        hal.stop()
        if not ok:
            print("[motion] turn to %s not finished (yaw %.1f)" % (d, self.pose()[2]))
        self.heading = d
        self.settle()
        return ok

    def run_segment(self, d, n):
        """Face d, then drive n cells forward - one cell at a time when GRID_STEP.
        Returns the cell reached (early if blocked)."""
        self.turn_to(d)
        self.hal.gimbal_front()
        self.hal.set_mode("chassis_lead")          # the gimbal (ToF) stays facing forward
        start = self.cell()
        steps = [1] * n if C.GRID_STEP else [n]
        reached = start
        for k in steps:
            want = step(reached, d, k)
            reached = self._drive(d, k)
            if reached != want or self.abort.is_set():
                break
            if C.GRID_STEP and C.CELL_PAUSE:
                self.hal.sleep(C.CELL_PAUSE)
        self.hal.set_mode("free")
        return reached

    def _drive(self, d, n):
        """Drive n tiles forward (towards d) and stop on the cell centre."""
        hal = self.hal
        start = self.cell()
        goal = step(start, d, n)
        gx, gy = goal[0] * T, goal[1] * T
        ns, s = d in "NS", SIGN[d]
        axis = "Y" if ns else "X"
        dt = 1.0 / C.CTRL_HZ
        v = 0.0
        t_end = hal.now() + C.SEG_TIMEOUT * n + 1.0
        nxt = hal.now()
        while True:
            self.localize(axis, abs(v))
            x, y, yaw = self.pose()
            along = s * ((gy - y) if ns else (gx - x))
            lat = (gx - x) if ns else (gy - y)
            if abs(along) < C.POS_TOL and abs(v) < 0.2:
                break
            left_on, right_on = self.corner_ir()
            if along > 0.05 and left_on and right_on:
                print("[motion] both front IR on: obstacle ahead (%s) - stop" % d)
                break
            if self.abort.is_set():
                break
            if self.pause.is_set():                     # hold still; the clock runs on
                hal.stop()
                v = 0.0
                while self.pause.is_set() and not self.abort.is_set():
                    hal.sleep(0.05)
                t_end = hal.now() + C.SEG_TIMEOUT * n + 1.0
                nxt = hal.now()
                continue
            if hal.now() > t_end:
                print("[motion] segment timeout")
                break
            brake = max(0.0, abs(along) - abs(v) * C.MOTION_LAG)    # allow for the lag
            vdes = math.copysign(min(C.V_MAX, math.sqrt(2 * C.A_MAX * brake)), along)
            if abs(vdes) < C.V_MIN:
                vdes = math.copysign(C.V_MIN, along)
            v += clamp(vdes - v, C.A_MAX * dt)
            vl = clamp(C.LAT_KP * lat, C.LAT_VMAX)
            vn, ve = (s * v, vl) if ns else (vl, s * v)
            cx, cy = to_chassis(vn, ve, yaw)            # forward + sideways centring
            cy += self.too_close_push(d)                # something really close: away
            if left_on or right_on:                     # scraping a corner: steer away
                cx *= C.IR_SLOW
                cy += C.IR_STEER if left_on else -C.IR_STEER
            hal.drive(cx, cy, self.hold_wz(yaw))
            self._trace()
            nxt += dt
            hal.sleep(max(0.0, nxt - hal.now()))
        self.settle()
        return self.cell()

    def side_readings(self, d):
        """(left, right) wall distances (m from the centre) for a robot facing d,
        None where there is no reading / no wall within SIDE_WALL_MAX."""
        from hal import side_dir
        ex = self.hal.ranges_ex() if hasattr(self.hal, "ranges_ex") else {
            k: (r, "sharp") for k, r in self.hal.ranges().items()}
        out = []
        for side in ("L", "R"):
            r = ex.get(side_dir(d, side), (None, ""))[0]
            out.append(r if r is not None and r < C.SIDE_WALL_MAX else None)
        return out

    def too_close_push(self, d):
        """Sideways speed (chassis, + = right) away from a side that is REALLY close
        (< TOO_CLOSE from the centre = ~8 cm from the chassis).  A target plate on
        the wall of a centred robot reads ~0.22 m and never triggers it."""
        left, right = self.side_readings(d)
        push = 0.0
        if left is not None and left < C.TOO_CLOSE:
            push += C.WALL_VMAX
        if right is not None and right < C.TOO_CLOSE:
            push -= C.WALL_VMAX
        return push

    def center_in_cell(self):
        """Drive to the centre of the current cell by the pose estimate (Dhai_8's
        align_at_cell_center).  Called after the scan: by then the camera has checked
        the cell's walls, so the localiser knows which readings are target plates and
        its corrections can be trusted over a wider range."""
        hal = self.hal
        end = hal.now() + C.CENTER_TIME
        ok_ticks = 0
        c = self.cell()
        while hal.now() < end and not self.abort.is_set():
            self.localize()
            x, y, yaw = self.pose()
            en, ee = c[1] * T - y, c[0] * T - x         # map error to the cell centre
            if abs(en) <= C.CENTER_TOL and abs(ee) <= C.CENTER_TOL:
                hal.stop()
                ok_ticks += 1
                if ok_ticks >= 3:
                    break
            else:
                ok_ticks = 0
                vn = clamp(C.CENTER_KP * en, C.CENTER_VMAX) if abs(en) > C.CENTER_TOL else 0.0
                ve = clamp(C.CENTER_KP * ee, C.CENTER_VMAX) if abs(ee) > C.CENTER_TOL else 0.0
                cx, cy = to_chassis(vn, ve, yaw)
                hal.drive(cx, cy, self.hold_wz(yaw))
            hal.sleep(1.0 / C.CTRL_HZ)
        hal.stop()
        self.settle()
