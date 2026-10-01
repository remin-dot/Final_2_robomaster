"""Hardware layer for the DJI RoboMaster EP (robomaster SDK).

Sensors on this robot (config.py):
  * 1 ToF on the gimbal - it measures wherever the camera looks.  Each reading
    is tagged with the map direction the gimbal ACTUALLY points (sub_angle),
    and used only while the gimbal is within TOF_ALIGN_DEG of N / E / S / W.
  * 2 Sharp on the chassis sides (E = right, W = left).
  * 2 IR obstacle modules on the front corners at 45 deg (collision guards).

Everything is subscription based: callbacks keep the latest values in memory,
so reads cost nothing and the control loop never blocks on I/O.  The camera
runs in its own thread and always holds the newest detections.
"""
import math
import statistics
import threading
import time
from collections import deque

import config as C
from vision import Detector

YAW_OF_DIR = {"N": 0.0, "E": 90.0, "S": 180.0, "W": -90.0}


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def gimbal_yaw_for(d, current):
    """Gimbal yaw (deg) for map direction d, choosing the closer of S=+/-180."""
    y = YAW_OF_DIR[d]
    if d == "S" and abs(-180.0 - current) < abs(180.0 - current):
        y = -180.0
    return y


def sharp_cm(raw, cal):
    """Final_Robomaster's conversion: log-log interpolation between the calibration
    points (cm, raw), the power law cm = A * raw ** B outside them."""
    pts = sorted(cal["points"], key=lambda p: p[1])
    if len(pts) >= 2 and pts[0][1] <= raw <= pts[-1][1]:
        for (c1, r1), (c2, r2) in zip(pts, pts[1:]):
            if r1 <= raw <= r2 and r2 > r1:
                t = (math.log(raw) - math.log(r1)) / (math.log(r2) - math.log(r1))
                return math.exp(math.log(c1) + t * (math.log(c2) - math.log(c1)))
    return cal["A"] * max(raw, 1.0) ** cal["B"]


DIRS_CW = "NESW"


def chassis_dir(yaw, tol=20.0):
    """Map direction the chassis faces (yaw relative to the start), or None while
    it is turning / between directions."""
    for d, a in YAW_OF_DIR.items():
        if abs(wrap(yaw - a)) <= tol:
            return d
    return None


def side_dir(heading, side):
    """Map direction of a chassis side ("F", "R", "B", "L") for chassis heading."""
    k = DIRS_CW.index(heading)
    return DIRS_CW[(k + {"F": 0, "R": 1, "B": 2, "L": 3}[side]) % 4]


def tof_dir(yaw, pitch):
    """Map direction the gimbal ToF points along (world yaw = chassis yaw + gimbal
    yaw), or None between directions / while tilted."""
    if abs(pitch - C.GIMBAL_PITCH) > 5.0:
        return None
    for d, a in YAW_OF_DIR.items():
        if abs(wrap(yaw - a)) <= C.TOF_ALIGN_DEG:
            return d
    return None


def tof_to_centre(rel_yaw, r):
    """ToF reading (m from its face) -> distance from the chassis centre; the
    gimbal axis sits GIMBAL_AXIS_N ahead of the centre (rel_yaw = gimbal vs chassis)."""
    return r + C.TOF_OFFSET + C.GIMBAL_AXIS_N * math.cos(math.radians(rel_yaw))


class RoboMasterHAL:
    def __init__(self, fire_enabled=True, progress=None):
        """progress(text) is told each connection step (the panel shows it)."""
        self._progress = progress or (lambda _t: None)
        import logging
        import os
        import robomaster
        from robomaster import robot, camera, blaster
        os.makedirs(C.OUT_DIR, exist_ok=True)       # the SDK's own log, for diagnosis
        fh = logging.FileHandler(os.path.join(C.OUT_DIR, "sdk_log.txt"), mode="w")   # per connect
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        robomaster.logger.setLevel(logging.INFO)
        robomaster.logger.addHandler(fh)
        self._blaster_mod = blaster
        self.fire_enabled = fire_enabled
        self.ep = robot.Robot()
        if C.ROBOT_IP and C.CONN_TYPE == "sta":
            from robomaster import config as rm_config
            rm_config.ROBOT_IP_STR = C.ROBOT_IP   # skip the discovery broadcast

        def init():
            if C.CONN_TYPE == "ap":         # Wi-Fi direct: UDP is the reliable choice
                return self.ep.initialize(conn_type="ap", proto_type="udp")
            return self.ep.initialize(conn_type=C.CONN_TYPE)
        ok = self._step("robot link (%s)" % C.CONN_TYPE, init, C.CONNECT_TIMEOUT, required=True)
        if ok is False:
            raise RuntimeError("the SDK could not open a link to the robot (see out/sdk_log.txt)")
        self._step("robot mode FREE", lambda: self.ep.set_robot_mode(mode=robot.FREE), 5)

        self._odom = (0.0, 0.0)
        self._yaw = 0.0
        self._odom_yaw = None       # attitude yaw when the position frame was fixed
        self.odom_rot = 0.0         # extra frame correction (motion's self-check), deg
        self._gimbal_actual = (C.GIMBAL_PITCH, 0.0)          # (pitch, yaw) vs chassis
        self._tof = {d: deque(maxlen=5) for d in "NESW"}      # (t, metres | None)
        self._tof_seen = False                                # any non-zero ToF value yet
        self._sharp = {d: deque(maxlen=5) for d in C.SHARP}
        self.sharp_bias = {d: 0.0 for d in C.SHARP}          # set by side_bias_check()
        self._sharp_nodata = {d: True for d in C.SHARP}
        self._io = {}
        self._ir_ok = {n: False for n in C.IR}                # True once it has read "clear"

        ch = self.ep.chassis

        def subs():
            ch.sub_position(cs=0, freq=50, callback=self._on_pos)
            ch.sub_attitude(freq=50, callback=self._on_att)
            self.ep.gimbal.sub_angle(freq=50, callback=self._on_gimbal)
            self.ep.sensor.sub_distance(freq=C.SENSOR_HZ, callback=self._on_tof)
        self._step("data subscriptions", subs, 8, required=True)
        self._adaptor_polling = False
        self._adaptor_sub = False
        if (C.SHARP or C.IR) and C.ADAPTOR_POLL:
            self._start_adaptor_polling()
        elif C.SHARP or C.IR:
            try:
                ok = self.ep.sensor_adaptor.sub_adapter(
                    freq=C.SENSOR_HZ, callback=self._on_adapter)
                if ok is False:
                    raise RuntimeError("sub_adapter returned False")
                self._adaptor_sub = True
            except Exception as e:     # fall back to polling each port
                print("[hal] adaptor subscription failed (%s) -> polling" % e)
                self._start_adaptor_polling()

        self.gimbal_yaw, self.gimbal_pitch = 0.0, C.GIMBAL_PITCH   # commanded, vs chassis
        self._gimbal_action = None
        self._feed_t = 0.0                                    # time of the last angle feed
        self._sign = {"yaw": 1.0, "pitch": 1.0}               # feed sign vs commands
        self._gimbal_lock = threading.Lock()
        self._gimbal_thread = None
        self._mode = "free"
        # a gimbal left asleep (suspend) by another program ignores every command
        self._step("gimbal wake (resume)", self._gimbal_resume, 4)
        self._step("gimbal recenter", lambda: self.ep.gimbal.recenter(
            pitch_speed=C.GIMBAL_PITCH_SPEED, yaw_speed=C.GIMBAL_YAW_SPEED).wait_for_completed(
            timeout=C.GIMBAL_TIMEOUT), C.GIMBAL_TIMEOUT + 2)
        self._step("gimbal check", self._gimbal_check, 8)

        self._det = (0.0, [])
        self.frame = None
        self._running = True
        self._detector = Detector()
        self._step("camera stream %s" % C.CAMERA_RES, self._start_camera, 10)
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()
        self._progress("reading sensors")
        time.sleep(0.8)                      # let every stream deliver data
        self.yaw0 = self._yaw
        print("[hal] sensor check:")
        for line in self.sensor_report():
            print("[hal]   " + line)

    def _step(self, what, fn, timeout, required=False):
        """Run one connection step with a time limit, logging how long it took.
        A step that fails or hangs is fatal only when required."""
        self._progress(what)
        print("[hal] %s ..." % what)
        box, t0 = {}, time.monotonic()

        def run():
            try:
                box["v"] = fn()
            except BaseException as e:     # noqa: B902 - report whatever the SDK raises
                box["e"] = e
        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout)
        dt = time.monotonic() - t0
        if t.is_alive():
            msg = "%s: no answer in %.0f s" % (what, timeout)
        elif "e" in box:
            msg = "%s: %s" % (what, box["e"])
        else:
            print("[hal] %s ok (%.1f s)" % (what, dt))
            return box.get("v")
        print("[hal] %s %s" % ("FAILED" if required else "WARNING", msg))
        if required:
            raise RuntimeError(msg)
        return None

    # ----------------------------------------------------------- clock ----
    now = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)

    # ------------------------------------------------------- callbacks ----
    def _on_pos(self, info):
        # sub_position(cs=0): x forward / y right of the robot AS IT STOOD WHEN
        # SUBSCRIBED (at connect) - odom() turns it into the start frame
        self._odom = (info[0], info[1])

    def _on_att(self, info):
        self._yaw = info[0]                    # deg, + = clockwise
        if self._odom_yaw is None:             # the heading the position frame is fixed to
            self._odom_yaw = info[0]

    def _on_gimbal(self, info):
        # pitch, yaw relative to the chassis, in the same sign as our commands
        self._gimbal_actual = (info[0] * self._sign["pitch"], info[1] * self._sign["yaw"])
        self._feed_t = time.monotonic()

    # ----------------------------------------------------------- gimbal ----
    def _gimbal_resume(self):
        try:
            self.ep.gimbal.resume()
        except Exception as e:
            print("[hal] gimbal resume: %s" % e)
        time.sleep(0.6)
        try:
            self.ep.set_robot_mode(mode="free")
        except Exception:
            pass

    def _gimbal_check(self):
        """Nudge each axis with a move action and watch the angle feed: learns the
        feed's sign (the ToF direction uses it) and whether the gimbal moves."""
        if time.monotonic() - self._feed_t > 0.5:
            print("[hal] gimbal check: NO angle feed - ToF directions cannot be told")
            return
        for axis, (dy, dp) in (("yaw", (20, 0)), ("pitch", (0, 10))):
            i = 1 if axis == "yaw" else 0
            a0 = self._gimbal_actual[i] * self._sign[axis]
            self._move(dy, dp)
            time.sleep(0.15)
            d = self._gimbal_actual[i] * self._sign[axis] - a0
            self._move(-dy, -dp)
            if abs(d) < 5.0:
                print("[hal] gimbal check: %s did NOT move (%.1f deg) - asleep, locked or "
                      "blocked?" % (axis, d))
            elif d < 0:
                self._sign[axis] = -1.0
                print("[hal] gimbal check: %s feed has the opposite sign - flipped" % axis)
        self.gimbal_yaw, self.gimbal_pitch = 0.0, 0.0
        self._cmd = (0.0, 0.0)
        if C.GIMBAL_PITCH:
            self.gimbal_to(0.0, C.GIMBAL_PITCH, wait=True)
        print("[hal] gimbal check: done")

    def _move(self, dyaw, dpitch):
        """One relative move action, waited for (Dhai_8: planned relative moves,
        never trimmed from the angle feedback - no chasing, no shaking)."""
        try:
            act = self.ep.gimbal.move(pitch=dpitch, yaw=dyaw, pitch_speed=C.GIMBAL_PITCH_SPEED,
                                      yaw_speed=C.GIMBAL_YAW_SPEED)
            if not act.wait_for_completed(timeout=C.GIMBAL_TIMEOUT):
                print("[hal] gimbal move (%+.0f, %+.0f) not confirmed in %.1f s"
                      % (dyaw, dpitch, C.GIMBAL_TIMEOUT))
        except Exception as e:
            print("[hal] gimbal move failed: %s" % e)

    def set_mode(self, mode):
        """"free" (gimbal holds still while the chassis turns - scans) or
        "chassis_lead" (gimbal follows the chassis - turning and driving)."""
        if mode == self._mode:
            return
        from robomaster import robot
        try:
            self.ep.set_robot_mode(mode=robot.FREE if mode == "free" else robot.CHASSIS_LEAD)
            self._mode = mode
        except Exception as e:
            print("[hal] robot mode %s failed: %s" % (mode, e))

    def _on_tof(self, mm):
        v = mm[C.TOF_INDEX - 1]
        if v:
            self._tof_seen = True                  # 0 = no sensor on that port
        pitch, rel = self._gimbal_actual
        chassis = self._yaw - getattr(self, "yaw0", self._yaw)
        d = tof_dir(chassis + rel, pitch)
        if d is None or not v:
            return                                 # gimbal between directions / no data
        r = tof_to_centre(rel, v / 1000.0) if v < C.TOF_MAX * 1000 else None
        self._tof[d].append((time.monotonic(), r))

    def _on_adapter(self, info):
        io, ad = info
        for d, (aid, port) in C.SHARP.items():
            i = (aid - 1) * 2 + port - 1
            if i < len(ad):
                self._sharp_nodata[d] = not ad[i]      # the SDK reports 0 for a missing board
                self._sharp[d].append(self._sharp_m(ad[i], d))
        for n, spec in C.IR.items():
            aid, port = spec["port"]
            i = (aid - 1) * 2 + port - 1
            if i < len(io):
                self._set_io(n, io[i])

    def _start_adaptor_polling(self):
        self._adaptor_polling = True

        from robomaster import protocol
        sa = self.ep.sensor_adaptor
        ports = sorted(set(C.SHARP.values()) | {s["port"] for s in C.IR.values()})

        def read(aid, port):
            """One request returns both the ADC and the IO level of a port."""
            proto = protocol.ProtoSensorGetData()
            proto._port = port
            msg = protocol.Msg(sa._client.hostbyte, protocol.host2byte(22, aid), proto)
            resp = sa._client.send_sync_msg(msg)
            if not resp:
                return None, None
            p = resp.get_proto()
            return p._adc, p._io

        def loop():
            while self._adaptor_polling:
                vals = {}
                for aid, port in ports:
                    try:
                        vals[(aid, port)] = read(aid, port)
                    except Exception:
                        vals[(aid, port)] = (None, None)
                for d, key in C.SHARP.items():
                    self._sharp_nodata[d] = not vals[key][0]
                    self._sharp[d].append(self._sharp_m(vals[key][0], d))
                for n, spec in C.IR.items():
                    if vals[spec["port"]][1] is not None:
                        self._set_io(n, vals[spec["port"]][1])
                time.sleep(0.005)
        threading.Thread(target=loop, daemon=True).start()

    @staticmethod
    def _sharp_m(adc, d):
        """Raw ADC -> metres from the robot centre (None = outside 4-30 cm)."""
        if not adc or adc <= 0:
            return None
        cm = sharp_cm(adc, C.SHARP_CAL[d])
        if not (C.SHARP_MIN <= cm / 100.0 <= C.SHARP_MAX):
            return None
        return cm / 100.0 + C.SHARP_OFFSET[d]

    def _start_camera(self):
        ok = self.ep.camera.start_video_stream(display=False, resolution=C.CAMERA_RES)
        if ok is False:
            raise RuntimeError("the robot refused the video stream")
        return ok

    def _camera_loop(self):
        """Newest frame -> detections.  Watchdog: the SDK silently keeps a dead
        video socket (e.g. 'connection refused' on port 40921) - no frame for 3 s
        means stop + start the stream again."""
        cam = self.ep.camera
        last, tries = time.monotonic(), 0
        while self._running:
            try:
                frame = cam.read_cv2_image(strategy="newest", timeout=1.0)
            except Exception:
                frame = None
            now = time.monotonic()
            if frame is None:
                if now - last > 3.0 and self._running:
                    tries += 1
                    print("[hal] camera: no frames for %.0f s - restarting the video stream (try %d)"
                          % (now - last, tries))
                    try:
                        cam.stop_video_stream()
                    except Exception:
                        pass
                    time.sleep(1.0 + min(tries, 4))
                    try:
                        cam.start_video_stream(display=False, resolution=C.CAMERA_RES)
                    except Exception as e:
                        print("[hal] camera restart failed: %s" % e)
                    last = time.monotonic()
                continue
            if tries:
                print("[hal] camera: frames arriving again")
                tries = 0
            last = now
            self.frame = frame
            self._det = (now, self._detector.detect(frame))

    def rezero(self):
        """Robot was (re)placed on the start tile facing North."""
        self.yaw0 = self._yaw
        self.odom_rot = 0.0
        print("[hal] round start: heading %.0f deg from power-on (heading at connect %.0f) - "
              "odometry turned by %.0f deg" % (self._yaw, self._odom_yaw if self._odom_yaw is not None
                                             else float("nan"), self.yaw0))

    # ------------------------------------------------------------ reads ----
    def odom(self):
        """(north m, east m, yaw deg relative to start).  The SDK position frame
        turned out to be the robot's POWER-ON frame (two runs: odometry 175-180 deg
        off with the robot not turned after connect), like the attitude yaw; North
        is set at the round start (rezero).  Rotate by the heading from power-on,
        plus motion's measured correction odom_rot.  Wrong, this read driving N as
        S, or a few deg of false sideways drift every cell."""
        yaw = self._yaw - self.yaw0
        yaw = (yaw + 180.0) % 360.0 - 180.0
        a = math.radians(((self.yaw0 + 180.0) % 360.0 - 180.0) + self.odom_rot)
        x, y = self._odom
        return x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a), yaw

    def side_bias_check(self, ref=None, strict=False):
        """A side Sharp that sees a wall should read ref[side] (the gimbal ToF on
        that wall), or SIDE_NOMINAL without one (robot on the tile centre).
        Stores the difference (a calibration offset), ignoring anything larger
        than SHARP_BIAS_MAX (a target plate)."""
        out = {}
        for side, q in self._sharp.items():
            vals = [v for v in list(q)[-5:] if v is not None]
            if len(vals) < 3 or self._sharp_nodata[side]:
                continue
            if strict and side not in (ref or {}):
                continue                                # only against a ToF measurement
            r = statistics.median(vals)
            if r > C.SIDE_WALL_MAX:
                continue                                # no wall that side
            bias = r - (ref or {}).get(side, C.SIDE_NOMINAL)
            if abs(bias) <= C.SHARP_BIAS_MAX:
                self.sharp_bias[side] = bias
                out[side] = bias
        return out

    def heading(self):
        """Map direction the chassis faces now (None while turning)."""
        return chassis_dir(self.odom()[2])

    def ranges_ex(self):
        """{map dir: (metres from robot centre | None = beyond range, "tof" | "sharp")}
        for the directions measurable RIGHT NOW (missing key = unknown).  The side
        Sharps follow the chassis heading; the ToF is stored by world direction."""
        out = {}
        h = self.heading()
        if h is not None:
            for side, q in self._sharp.items():
                if self._sharp_nodata[side]:
                    continue
                vals = list(q)[-3:]
                good = [v - self.sharp_bias[side] for v in vals if v is not None]
                out[side_dir(h, side)] = (statistics.median(good) if len(good) * 2 > len(vals)
                                          else None, "sharp")
        t = time.monotonic() - C.TOF_FRESH
        for d, q in self._tof.items():
            fresh = [r for (ts, r) in list(q)[-3:] if ts >= t]
            if fresh:                                   # the ToF beats a Sharp
                good = [v for v in fresh if v is not None]
                out[d] = (statistics.median(good) if len(good) * 2 > len(fresh) else None, "tof")
        return out

    def ranges(self):
        return {d: r for d, (r, _) in self.ranges_ex().items()}

    def side_raw(self):
        """(left, right) side Sharps, m from the robot centre, any heading (also
        while turning); None = no reading."""
        out = []
        for side in ("L", "R"):
            q = list(self._sharp.get(side, ()))[-3:]
            vals = [v - self.sharp_bias[side] for v in q if v is not None]
            out.append(None if self._sharp_nodata.get(side, True) or len(vals) * 2 <= len(q)
                       else statistics.median(vals))
        return tuple(out)

    def _set_io(self, n, v):
        self._io[n] = v
        if v != C.IR_ACTIVE_LEVEL:
            self._ir_ok[n] = True                      # it can read "clear": it is real

    def sensor_ok(self, d):
        """A range reading for direction d is available right now."""
        return d in self.ranges_ex()

    def ir_state(self):
        """{name: True obstacle / False clear / None unverified or no data}"""
        return {n: (self._io.get(n) == C.IR_ACTIVE_LEVEL) if self._ir_ok[n] else None
                for n in C.IR}

    def bump(self, d, strict=False):
        """An IR module guarding map direction d sees something close.  Modules
        guard chassis sides ("F" = forward); strict = only a module that faces
        straight along d (45-degree corner modules never count as walls)."""
        h = self.heading()
        if h is None:
            return False
        r = None
        for n, on in self.ir_state().items():
            spec = C.IR[n]
            if not on or side_dir(h, spec["guards"]) != d or (strict and spec["ang"] % 90):
                continue
            if r is None:
                r = self.ranges().get(d)
            if r is not None and r > C.IR_TRUST_MAX:
                continue                               # a range sensor sees open space
            return True
        return False

    def sensor_report(self):
        """One line per sensor, printed after connecting."""
        ex = self.ranges_ex()
        lines = []
        d = tof_dir(self._gimbal_actual[1], self._gimbal_actual[0])
        if not self._tof_seen:
            st = "NO DATA - check TOF_INDEX in config.py / the cable"
        elif d in ex and ex[d][1] == "tof":
            st = "facing %s: %s" % (d, "far" if ex[d][0] is None else "%.2f m" % ex[d][0])
        else:
            st = "ok (gimbal not along N/E/S/W right now)"
        lines.append("ToF on gimbal #%d   %s" % (C.TOF_INDEX, st))
        h = self.heading() or "N"
        for side, port in C.SHARP.items():
            if self._sharp_nodata[side]:
                st = "NO DATA - check SHARP in config.py / the cable"
            else:
                r = self.ranges().get(side_dir(h, side))
                st = "far" if r is None else "%.2f m" % r
            lines.append("Sharp %s %-9s %s" % ({"R": "right", "L": "left"}[side], port, st))
        for n, spec in C.IR.items():
            if n not in self._io:
                st = "NO DATA"
            elif not self._ir_ok[n]:
                st = ("stuck at 'obstacle' since connect -> ignored until it reads clear "
                      "(missing board? wrong IR port / IR_ACTIVE_LEVEL?)")
            else:
                st = "obstacle" if self._io[n] == C.IR_ACTIVE_LEVEL else "clear"
            lines.append("IR %s %-9s %s" % (n, spec["port"], st))
        return lines

    def detections(self):
        return self._det

    # ---------------------------------------------------------- actions ----
    def drive(self, vx, vy, wz):
        """Chassis frame: vx forward, vy right (m/s), wz deg/s clockwise."""
        self.ep.chassis.drive_speed(x=vx, y=vy, z=wz, timeout=0.5)

    def stop(self):
        self.ep.chassis.drive_speed(x=0, y=0, z=0)

    def _gimbal_run(self, yaw, pitch):
        with self._gimbal_lock:
            dy = yaw - self._cmd[0]
            dp = pitch - self._cmd[1]
            while abs(dy) > 0.05 or abs(dp) > 0.05:
                step = max(-C.GIMBAL_STEP_DEG, min(C.GIMBAL_STEP_DEG, dy))
                self._move(step, dp)
                self._cmd = (self._cmd[0] + step, self._cmd[1] + dp)
                dy, dp = yaw - self._cmd[0], 0.0
            time.sleep(C.GIMBAL_SETTLE)             # stand still: sharp frames, fresh ToF
            self._trim(yaw, pitch)

    def _trim(self, yaw, pitch):
        """ONE correction move from the angle feed once the gimbal stands still.
        Relative moves add up small errors; unchecked, the gimbal ends a few degrees
        off and the ToF can no longer tell which wall it measures.  A single move
        (not a loop) cannot shake."""
        if time.monotonic() - self._feed_t > 0.3:
            return
        p, y = self._gimbal_actual
        ey, ep = yaw - y, pitch - p
        if (C.GIMBAL_TRIM_MIN < abs(ey) < 30.0) or (C.GIMBAL_TRIM_MIN < abs(ep) < 20.0):
            self._move(ey if abs(ey) < 30.0 else 0.0, ep if abs(ep) < 20.0 else 0.0)
            time.sleep(C.GIMBAL_SETTLE)

    def gimbal_error(self):
        """Commanded minus measured gimbal yaw (deg), None without a fresh feed."""
        if time.monotonic() - self._feed_t > 0.3:
            return None
        return self.gimbal_yaw - self._gimbal_actual[1]

    def gimbal_to(self, yaw, pitch=None, wait=True):
        """Point the gimbal (degrees vs the chassis): relative move actions of at
        most GIMBAL_STEP_DEG from the commanded position (Dhai_8), then settle."""
        pitch = self.gimbal_pitch if pitch is None else pitch
        if not hasattr(self, "_cmd"):
            self._cmd = (0.0, C.GIMBAL_PITCH)
        if self._gimbal_thread is not None and self._gimbal_thread.is_alive():
            self._gimbal_thread.join(C.GIMBAL_TIMEOUT * 4)
        self.gimbal_yaw, self.gimbal_pitch = yaw, pitch
        if getattr(self, "_detector", None) is not None:
            self._detector.pitch = pitch        # horizon for the segmentation
        if wait:
            self._gimbal_run(yaw, pitch)
        else:
            self._gimbal_thread = threading.Thread(target=self._gimbal_run, args=(yaw, pitch),
                                                   daemon=True)
            self._gimbal_thread.start()

    def gimbal_wait(self):
        if self._gimbal_thread is not None and self._gimbal_thread.is_alive():
            self._gimbal_thread.join(C.GIMBAL_TIMEOUT * 4)

    def gimbal_by(self, dyaw, dpitch):
        self.gimbal_to(self.gimbal_yaw + dyaw, self.gimbal_pitch + dpitch, wait=True)

    def gimbal_front(self):
        """Face forward and re-anchor the commanded position (before driving)."""
        self.gimbal_to(0.0, C.GIMBAL_PITCH, wait=True)
        if time.monotonic() - self._feed_t < 0.5 and abs(self._gimbal_actual[1]) > 5.0:
            print("[hal] gimbal %.0f deg off the front - recenter" % self._gimbal_actual[1])
            try:
                self.ep.gimbal.recenter(pitch_speed=C.GIMBAL_PITCH_SPEED,
                                        yaw_speed=C.GIMBAL_YAW_SPEED).wait_for_completed(
                    timeout=C.GIMBAL_TIMEOUT)
            except Exception as e:
                print("[hal] recenter failed: %s" % e)
            self._cmd = (0.0, 0.0)
            if C.GIMBAL_PITCH:
                self.gimbal_to(0.0, C.GIMBAL_PITCH, wait=True)

    def fire(self, times):
        if not self.fire_enabled:
            print("[hal] (fire disabled)")
            return
        ft = self._blaster_mod.WATER_FIRE if C.FIRE_TYPE == "water" else self._blaster_mod.INFRARED_FIRE
        for i in range(times):
            self.ep.blaster.fire(fire_type=ft, times=1)
            if i + 1 < times:
                time.sleep(C.FIRE_GAP)

    def close(self):
        self._running = False
        self._adaptor_polling = False
        try:
            self.stop()
            self.set_mode("free")
            self.ep.chassis.unsub_position()
            self.ep.chassis.unsub_attitude()
            self.ep.gimbal.unsub_angle()
            self.ep.sensor.unsub_distance()
            if self._adaptor_sub:
                self.ep.sensor_adaptor.unsub_adapter()
            self.ep.camera.stop_video_stream()
        finally:
            self.ep.close()
