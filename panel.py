"""Mission control panel (pygame) - Connect screen, live map, camera, sensors,
aim, targets, log and grouped mission controls.

Style follows the Final_Robomaster mission panel (light / dark themes,
immediate-mode widgets); the layout is reorganised so the map gets the most
room, the header only shows status, every control lives in one sidebar and
no two labels share the same space.

  python panel.py                  # Connect screen (robot or simulator)
  python panel.py --sim --connect  # straight into the simulator

Keys
  Space   start round / STOP            P      pause / resume
  1 / 2   pick round (idle)             M      camera: overlay / mask / raw
  WASD    strafe (idle, heading held)   J L I K  gimbal
  F       fire once                     C      centre gimbal
  S       screenshot                    T      light / dark theme
  V       simulator truth               Q      quit      Enter  connect
"""
import argparse
import collections
import math
import os
import sys
import threading
import time
import traceback
from datetime import datetime

import pygame

import config as C
import mapdraw
from mapdraw import layout
from hal import wrap
from maze import DIRS, WALL, Maze
from mission import Mission
from vision import distance, kind_label, label_parts, plate_size

W, H = 1280, 800
HEADER = 56
PAD = 10

PALETTES = {
    "light": dict(
        bg=(241, 243, 246), panel=(255, 255, 255), panel2=(247, 248, 250), line=(222, 226, 232),
        grid=(233, 236, 241), text=(22, 25, 29), muted=(104, 111, 122), accent=(37, 99, 235),
        ok=(22, 163, 74), warn=(217, 119, 6), bad=(220, 38, 38), btn=(237, 239, 243),
        hover=(225, 229, 235), onbg=(219, 234, 254), white=(255, 255, 255), camera=(226, 229, 234),
        visited=(226, 236, 252), startc=(220, 245, 228), wall=(40, 44, 52), path=(245, 140, 30)),
    "dark": dict(
        bg=(13, 16, 21), panel=(22, 26, 33), panel2=(28, 33, 42), line=(44, 50, 62),
        grid=(36, 41, 51), text=(230, 233, 238), muted=(142, 151, 164), accent=(96, 145, 255),
        ok=(60, 207, 122), warn=(245, 165, 36), bad=(255, 95, 95), btn=(34, 40, 52),
        hover=(44, 52, 66), onbg=(35, 54, 96), white=(255, 255, 255), camera=(28, 33, 42),
        visited=(33, 45, 70), startc=(28, 58, 42), wall=(226, 230, 237), path=(255, 160, 60)),
}
CARD_RGB = {"red": (232, 62, 58), "blue": (60, 120, 235), "green": (52, 180, 84),
            "yellow": (240, 200, 30)}
PHASE_COL = {"AIMING": "warn", "LOCKED": "ok", "FIRE": "bad", "DRY RUN": "warn",
             "LOST": "muted", "TOO FAR": "muted", "IDLE": "muted"}
SIM_CAM_H, SIM_WALL_H, SIM_PLATE_H = 0.25, 0.45, 0.145    # m, simulator camera view
MANUAL_V = 0.4


# =============================================================================
# immediate-mode widgets (adapted from Final_Robomaster src/panel_pygame.py)
# =============================================================================
class UI:
    def __init__(self, theme="light"):
        pygame.font.init()
        sans = pygame.font.match_font("helveticaneue,helvetica,segoeui,arial,dejavusans")
        mono = pygame.font.match_font("menlo,consolas,dejavusansmono,monospace")
        self.fonts = {"xs": pygame.font.Font(sans, 11), "s": pygame.font.Font(sans, 13),
                      "sb": pygame.font.Font(sans, 13), "h": pygame.font.Font(sans, 16),
                      "t": pygame.font.Font(sans, 22), "big": pygame.font.Font(sans, 28),
                      "m": pygame.font.Font(mono, 12)}
        for k in ("sb", "h", "t", "big"):
            self.fonts[k].set_bold(True)
        self.set_theme(theme)
        self.screen = None
        self.mouse = (0, 0)
        self.clicked = False
        self.cursor = pygame.SYSTEM_CURSOR_ARROW

    def set_theme(self, name):
        self.theme, self.pal, self._cache = name, PALETTES[name], {}

    def col(self, c):
        return self.pal[c] if isinstance(c, str) else c

    def begin(self, screen, events):
        self.screen = screen
        self.mouse = pygame.mouse.get_pos()
        self.clicked = False
        for e in events:
            if e.type == pygame.MOUSEBUTTONUP and getattr(e, "button", 1) == 1:
                self.clicked, self.mouse = True, e.pos
        self.cursor = pygame.SYSTEM_CURSOR_ARROW

    def text(self, s, font="s", color="text"):
        key = (str(s), font, color if isinstance(color, str) else tuple(color), self.theme)
        surf = self._cache.get(key)
        if surf is None:
            surf = self.fonts[font].render(str(s), True, self.col(color))
            if len(self._cache) > 4000:
                self._cache.clear()
            self._cache[key] = surf
        return surf

    def label(self, s, x, y, font="s", color="text", align="left", maxw=None):
        s = str(s)
        f = self.fonts[font]
        if maxw and f.size(s)[0] > maxw:
            while s and f.size(s + "…")[0] > maxw:
                s = s[:-1]
            s += "…"
        surf = self.text(s, font, color)
        if align == "right":
            x -= surf.get_width()
        elif align == "center":
            x -= surf.get_width() // 2
        self.screen.blit(surf, (x, y))
        return surf.get_width()

    def hit(self, rect):
        return rect.collidepoint(self.mouse)

    def rrect(self, rect, color, radius=6, width=0):
        pygame.draw.rect(self.screen, self.col(color), rect, width, border_radius=radius)

    def card(self, rect, title=None, right=None, right_col="muted"):
        self.rrect(rect, "panel", 10)
        self.rrect(rect, "line", 10, 1)
        if title:
            self.label(title, rect.x + 12, rect.y + 9, "sb", "muted")
        if right:
            self.label(right, rect.right - 12, rect.y + 9, "sb", right_col, "right")

    def button(self, rect, label, kind="normal", on=False, enabled=True, font="s"):
        hover = enabled and self.hit(rect)
        if not enabled:
            bg, fg, border = "btn", "muted", "line"
        elif kind in ("primary", "danger"):
            bg = "accent" if kind == "primary" else "bad"
            fg, border = "white", bg
        elif on:
            bg, fg, border = "onbg", "accent", "accent"
        else:
            bg, fg, border = ("hover" if hover else "btn"), "text", "line"
        self.rrect(rect, bg, 7)
        self.rrect(rect, border, 7, 1)
        if hover and kind != "normal":
            self.rrect(rect, "white", 7, 1)
        if label:
            f = "sb" if (kind != "normal" or on) and font == "s" else font
            surf = self.text(label, f, fg)
            self.screen.blit(surf, surf.get_rect(center=rect.center))
        if hover:
            self.cursor = pygame.SYSTEM_CURSOR_HAND
        return enabled and hover and self.clicked

    def seg(self, rect, options, value, enabled=True, font="s"):
        """Segmented control. options = [(value, label)]. Returns the new value."""
        n = len(options)
        w = (rect.w - (n - 1) * 4) / float(n)
        for i, (v, lbl) in enumerate(options):
            r = pygame.Rect(int(rect.x + i * (w + 4)), rect.y, int(w), rect.h)
            if self.button(r, lbl, on=v == value, enabled=enabled, font=font):
                value = v
        return value

    def checkbox(self, rect, value, label=None, enabled=True, color="accent"):
        box = pygame.Rect(rect.x, rect.y + (rect.h - 16) // 2, 16, 16)
        self.rrect(box, color if value else "panel2", 4)
        self.rrect(box, color if value else "line", 4, 1)
        if value:
            pygame.draw.lines(self.screen, self.pal["white"], False,
                              [(box.x + 3, box.y + 8), (box.x + 7, box.y + 12),
                               (box.x + 13, box.y + 4)], 2)
        if label:
            self.label(label, box.right + 8, rect.y + (rect.h - 16) // 2, "s",
                       "text" if enabled else "muted")
        if enabled and self.hit(rect):
            self.cursor = pygame.SYSTEM_CURSOR_HAND
            if self.clicked:
                return not value
        return value

    def pill(self, x, y, text, color, font="sb"):
        w = self.fonts[font].size(text)[0] + 20
        h = 24 if font != "xs" else 18
        r = pygame.Rect(x, y, w, h)
        self.rrect(r, color, h // 2, 1)
        self.label(text, r.centerx, y + (4 if font != "xs" else 2), font, color, "center")
        return r

    def icon(self, c, color, shape, r, hollow=False):
        col = CARD_RGB.get(color, self.pal["muted"]) if color else self.pal["muted"]
        w, (x, y) = (2 if hollow else 0), c
        if shape == "circle":
            pygame.draw.circle(self.screen, col, c, r, w)
        elif shape == "wide":
            pygame.draw.rect(self.screen, col, (x - r, y - int(r * .6), 2 * r, int(r * 1.2)), w)
        elif shape == "tall":
            pygame.draw.rect(self.screen, col, (x - int(r * .6), y - r, int(r * 1.2), 2 * r), w)
        else:
            pygame.draw.rect(self.screen, col, (x - int(r * .8), y - int(r * .8),
                                                int(r * 1.6), int(r * 1.6)), w)


def bgr_to_surface(img):
    import cv2
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")


def pretty(label):
    color, shape = label_parts(label)
    return "%s %s" % (color, {"wide": "wide rect", "tall": "tall rect"}.get(shape, shape))


def fmt(t):
    t = max(0.0, t)
    return "%02d:%02d" % (int(t) // 60, int(t) % 60)


class LogTee:
    """Mirror stdout into a ring buffer the panel draws."""

    def __init__(self, n=500):
        self.lines = collections.deque(maxlen=n)
        self.orig = sys.stdout
        self._buf = ""
        self.round_log = None                  # list of text while a round runs
        try:                                   # everything also goes to out/panel_log.txt
            os.makedirs(C.OUT_DIR, exist_ok=True)
            self.file = open(os.path.join(C.OUT_DIR, "panel_log.txt"), "a", buffering=1)
            self.file.write("\n===== panel started %s =====\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
        except OSError:
            self.file = None

    def write(self, s):
        self.orig.write(s)
        if self.round_log is not None:
            self.round_log.append(s)
        if self.file:
            try:
                self.file.write(s)
            except (OSError, ValueError):
                pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip() and not line.startswith(("+", "|", "round ")):
                self.lines.append((time.strftime("%H:%M:%S"), line))

    def flush(self):
        self.orig.flush()


# =============================================================================
# the app
# =============================================================================
class App:
    # layout in logical pixels (the window scales)
    MAP_R = pygame.Rect(PAD, HEADER + PAD, 450, 486)
    TGT_R = pygame.Rect(PAD, HEADER + 2 * PAD + 486, 450, H - HEADER - 3 * PAD - 486)
    CAM_R = pygame.Rect(470, HEADER + PAD, 520, 322)
    SEN_R = pygame.Rect(470, HEADER + 2 * PAD + 322, 255, 200)
    AIM_R = pygame.Rect(735, HEADER + 2 * PAD + 322, 255, 200)
    LOG_R = pygame.Rect(470, HEADER + 3 * PAD + 522, 520, H - HEADER - 4 * PAD - 522)
    SIDE_R = pygame.Rect(1000, HEADER + PAD, 270, H - HEADER - 2 * PAD)

    def __init__(self, args, headless=False):
        self.args = args
        self.headless = headless
        self.log = LogTee()
        sys.stdout = self.log
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        flags = 0 if headless else (pygame.SCALED | pygame.RESIZABLE)
        self.screen = pygame.display.set_mode((W, H), flags)
        pygame.display.set_caption("RoboMaster Rescue - Mission Control")
        self.ui = UI(args.theme)
        self.clock = pygame.time.Clock()
        self.running = True
        # connection
        self.source = "sim" if args.sim else "robot"
        self.connection = C.CONN_TYPE
        self.seed, self.speed = args.seed, args.speed
        self.armed = not args.no_fire
        self.hal = None
        self.conn = "idle"            # idle / connecting / connected
        self.error = ""
        # mission
        self.round_no = 1
        self.mission = None
        self.thread = None
        self.last = {1: None, 2: None}
        self.shoot = set(C.SHOOT_CLASSES)
        # ui state
        self.view = "overlay"          # camera: overlay / mask / raw
        self.show_truth = False
        self.confirm = None
        self.manual_busy = False
        self._was_driving = False
        self._fps = collections.deque(maxlen=30)
        self._last_det_t = None
        if args.connect:
            self.connect()

    # ============================================================ control
    @property
    def prefix(self):
        return "sim_" if self.source == "sim" else ""

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def state(self):
        m = self.mission
        if self.busy():
            return "PAUSED" if m.pause.is_set() else "RUNNING"
        if m is None:
            return "IDLE"
        return "STOPPED" if m.abort.is_set() else "DONE"

    def connect(self):
        if self.conn == "connecting":
            return
        self.conn, self.error = "connecting", ""
        self._attempt = getattr(self, "_attempt", 0) + 1
        self._conn_t0 = self._step_t0 = time.time()
        self.conn_step = "starting"
        threading.Thread(target=self._connect, args=(self._attempt,), daemon=True).start()

    def cancel_connect(self):
        self._attempt = getattr(self, "_attempt", 0) + 1     # a late result is dropped
        self.conn, self.error = "idle", "Connect cancelled."

    @staticmethod
    def local_ip_towards(ip):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((ip, 9))          # UDP connect sends nothing; it only picks a route
            return s.getsockname()[0]
        except OSError:
            return None
        finally:
            s.close()

    def _make_robot_hal(self):
        try:
            import robomaster  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "The robomaster SDK is not installed in this Python. On a Mac run:  "
                "bash tools/macos/setup_robomaster_mac.sh  (then restart the panel), "
                "or pick Simulator.")
        if self.connection == "ap":
            ip = self.local_ip_towards(C.ROBOT_AP_IP)
            if not (ip and ip.startswith(C.ROBOT_AP_IP.rsplit(".", 1)[0] + ".")):
                raise RuntimeError(
                    "This computer is not on the robot's Wi-Fi (its address is %s; on the "
                    "robot's Wi-Fi it would be 192.168.2.x). Set the robot's switch to Wi-Fi "
                    "direct, join the RMEP-xxxxxx network (password on the sticker), then "
                    "Connect again." % (ip or "unknown"))
        from hal import RoboMasterHAL
        C.CONN_TYPE = self.connection
        box = {}

        def progress(text):
            self.conn_step, self._step_t0 = text, time.time()

        def make():
            try:
                box["hal"] = RoboMasterHAL(fire_enabled=self.armed, progress=progress)
            except BaseException as e:      # noqa: B902 - report anything the SDK raises
                box["err"] = e
        t = threading.Thread(target=make, daemon=True)
        t.start()
        t.join(C.CONNECT_TIMEOUT + 45)      # every step has its own limit inside
        if t.is_alive():
            raise RuntimeError("Connecting stuck at '%s'. Details: out/panel_log.txt and "
                               "out/sdk_log.txt" % self.conn_step)
        if "err" in box:
            raise RuntimeError(
                "Robot connection failed at '%s': %s.  Check the robot is on, its switch "
                "matches (%s), and no phone/PC app is connected to it (power-cycle it). "
                "Details: out/panel_log.txt, out/sdk_log.txt"
                % (self.conn_step, box["err"], {"ap": "Wi-Fi direct", "sta": "router",
                                                "rndis": "USB"}[self.connection]))
        return box["hal"]

    def _connect(self, attempt):
        try:
            if self.source == "sim":
                from sim import SimHAL
                hal = SimHAL(seed=self.seed, speed=self.speed)
            else:
                hal = self._make_robot_hal()
            if attempt != self._attempt:                # cancelled meanwhile
                hal.close()
                return
            hal.fire_enabled = self.armed
            self.hal = hal
            self.conn = "connected"
            print("[panel] connected: %s" % self.source_label())
        except Exception as e:
            if attempt == self._attempt:
                self.conn, self.error = "idle", str(e)
            print("[panel] connect failed: %s" % e)

    def disconnect(self):
        self.stop()
        if self.thread is not None:
            self.thread.join(timeout=3)
        if self.hal is not None:
            try:
                self.hal.close()
            except Exception:
                pass
        self.hal, self.mission, self.conn = None, None, "idle"
        print("[panel] disconnected")

    def source_label(self):
        if self.source == "sim":
            return "SIMULATOR  seed %d  x%g" % (self.seed, self.speed)
        return "ROBOT  %s" % {"ap": "Wi-Fi AP", "sta": "router", "rndis": "USB"}[self.connection]

    def round1_data(self):
        if self.last[1] is not None:
            return self.last[1]
        path = os.path.join(C.OUT_DIR, self.prefix + "round1.json")
        return mapdraw.load(path) if os.path.exists(path) else None

    def can_start(self):
        return (self.hal is not None and not self.busy() and bool(self.shoot)
                and (self.round_no == 1 or self.round1_data() is not None))

    def start(self):
        if not self.can_start():
            if self.round_no == 2 and self.round1_data() is None:
                print("[panel] no round-1 map yet - run round 1 first")
            return
        maze = targets = None
        if self.round_no == 2:
            data = self.round1_data()
            maze, targets = Maze.from_dict(data["maze"]), data["targets"]
        self.hal.stop()
        self.hal.rezero()
        self.hal.fire_enabled = self.armed
        self.log.round_log = ["===== round %d started %s =====\n" % (
            self.round_no, time.strftime("%Y-%m-%d %H:%M:%S"))]      # this round's own log
        self.mission = Mission(self.hal, self.round_no, sorted(self.shoot), maze, targets)
        self.thread = threading.Thread(target=self._run, args=(self.mission,), daemon=True)
        self.thread.start()

    def _run(self, m):
        try:
            m.run_round1() if m.round == 1 else m.run_round2()
        except Exception:
            traceback.print_exc(file=sys.stdout)
        finally:
            self.hal.stop()
            m.finish()                      # also on STOP / error: the timer freezes
            data = m.to_dict()
            self.last[m.round] = data
            base = mapdraw.save(data, C.OUT_DIR, "%sround%d" % (self.prefix, m.round))
            print("[panel] round %d saved -> %s.png/.svg/.json" % (m.round, base))
            try:                                # its own folder: map + this round's log
                text = "".join(self.log.round_log or [])
                folder = mapdraw.save_run(data, C.OUT_DIR, "%sround%d" % (self.prefix, m.round), text)
                print("[panel] round %d map + log -> %s" % (m.round, folder))
            except OSError as e:
                print("[panel] could not save the round folder: %s" % e)
            finally:
                self.log.round_log = None

    def stop(self):
        if self.mission is not None:
            if self.busy():
                self.mission.finish()          # the timer stops the moment STOP is pressed
            self.mission.pause.clear()
            self.mission.abort.set()
        if self.hal is not None:
            self.hal.stop()

    def toggle_pause(self):
        if self.busy():
            p = self.mission.pause
            p.clear() if p.is_set() else p.set()
            print("[panel] %s" % ("paused" if p.is_set() else "resumed"))

    def save_map(self):
        if self.mission is None:
            print("[panel] nothing to save yet")
            return
        base = mapdraw.save(self.mission.to_dict(), C.OUT_DIR,
                            "%sround%d_snapshot" % (self.prefix, self.mission.round))
        print("[panel] map snapshot -> %s.png" % base)

    def screenshot(self):
        os.makedirs(C.OUT_DIR, exist_ok=True)
        path = os.path.join(C.OUT_DIR, "panel_%s.png" % datetime.now().strftime("%Y%m%d_%H%M%S"))
        pygame.image.save(self.screen, path)
        print("[panel] screenshot -> %s" % path)

    def set_armed(self, v):
        self.armed = v
        if self.hal is not None:
            self.hal.fire_enabled = v

    def bg(self, fn):
        """Blocking manual robot action, off the UI thread."""
        if self.hal is None or self.busy() or self.manual_busy:
            return
        self.manual_busy = True

        def run():
            try:
                fn()
            except Exception as e:
                print("[panel] %s" % e)
            finally:
                self.manual_busy = False
        threading.Thread(target=run, daemon=True).start()

    def fire_once(self):
        def f():
            print("[panel] manual fire%s" % ("" if self.armed else " (dry run)"))
            self.hal.fire(C.FIRE_TIMES)
        self.bg(f)

    def gimbal(self, dyaw=0.0, dpitch=0.0, centre=False):
        if centre:
            self.bg(lambda: self.hal.gimbal_to(0.0, C.GIMBAL_PITCH, wait=True))
        else:
            self.bg(lambda: self.hal.gimbal_by(dyaw, dpitch))

    def set_start(self, cell):
        C.START_CELL = tuple(cell)
        self.mission = None
        if self.hal is not None:
            self.hal.rezero()
        print("[panel] start cell -> %s (robot faces North)" % (cell,))

    def ask(self, message, callback):
        self.confirm = (message, callback)

    def quit(self):
        if self.busy():
            self.ask("A round is running. Stop it and quit?", self._quit)
        else:
            self._quit()

    def _quit(self):
        self.running = False

    def manual(self, keys, dt):
        """Keyboard driving in the chassis frame (W forward, S back, A / D sideways),
        heading held on the nearest grid direction (idle only) + simulator physics."""
        if self.hal is None or self.busy():
            return
        vn = (keys[pygame.K_w] or keys[pygame.K_UP]) - (keys[pygame.K_s] or keys[pygame.K_DOWN])
        ve = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        driving = bool(vn or ve)
        k = MANUAL_V / (math.hypot(vn, ve) or 1)

        def command():
            if driving:
                yaw = self.hal.odom()[2]
                hold = round(yaw / 90.0) * 90.0
                wz = max(-C.YAW_WMAX, min(C.YAW_WMAX, -C.YAW_KP * wrap(yaw - hold)))
                self.hal.drive(vn * k, ve * k, wz)

        if driving or self._was_driving:
            command() if driving else self.hal.stop()
        self._was_driving = driving
        if hasattr(self.hal, "tick"):
            left = dt * self.speed
            while left > 1e-6:
                h = min(1.0 / 30, left)
                command()
                self.hal.tick(h)
                left -= h

    # ============================================================ input
    def on_key(self, k):
        if self.confirm:
            if k in (pygame.K_ESCAPE, pygame.K_n):
                self.confirm = None
            elif k in (pygame.K_RETURN, pygame.K_y):
                cb, self.confirm = self.confirm[1], None
                cb()
            return
        if k == pygame.K_q:
            self.quit()
        elif k == pygame.K_t:
            self.ui.set_theme("dark" if self.ui.theme == "light" else "light")
        elif self.conn != "connected":
            if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.connect()
            elif k == pygame.K_ESCAPE and self.conn == "connecting":
                self.cancel_connect()
        elif k == pygame.K_SPACE:
            self.stop() if self.busy() else self.start()
        elif k == pygame.K_ESCAPE:
            self.stop()
        elif k == pygame.K_p:
            self.toggle_pause()
        elif k in (pygame.K_1, pygame.K_2) and not self.busy():
            self.round_no = 1 if k == pygame.K_1 else 2
        elif k == pygame.K_m:
            modes = ("overlay", "mask", "raw")
            self.view = modes[(modes.index(self.view) + 1) % len(modes)]
        elif k == pygame.K_s and not self._driving_keys():
            self.screenshot()
        elif k == pygame.K_v and self.source == "sim":
            self.show_truth = not self.show_truth
        elif k == pygame.K_f:
            self.fire_once()
        elif k == pygame.K_c:
            self.gimbal(centre=True)
        elif k in (pygame.K_j, pygame.K_l):
            self.gimbal(dyaw=-15 if k == pygame.K_j else 15)
        elif k in (pygame.K_i, pygame.K_k):
            self.gimbal(dpitch=5 if k == pygame.K_i else -5)

    @staticmethod
    def _driving_keys():
        p = pygame.key.get_pressed()
        return p[pygame.K_w] or p[pygame.K_a] or p[pygame.K_d]

    # ============================================================ loop
    def step(self, events, dt=1 / 30.0):
        for e in events:
            if e.type == pygame.QUIT:
                self.quit()
            elif e.type == pygame.KEYDOWN:
                self.on_key(e.key)
        self.ui.begin(self.screen, events)
        if self.conn == "connected":
            self.manual(pygame.key.get_pressed(), dt)
        try:
            self.draw()
        except RuntimeError:      # a container changed under us mid-frame
            pass
        if not self.headless:
            pygame.mouse.set_cursor(self.ui.cursor)
        pygame.display.flip()

    def run(self):
        while self.running:
            dt = self.clock.tick(C.PANEL_FPS) / 1000.0
            self.step(pygame.event.get(), dt)
        self.shutdown()

    def shutdown(self):
        self.stop()
        if self.thread is not None:
            self.thread.join(timeout=3)
        if self.hal is not None:
            self.hal.close()
        sys.stdout = self.log.orig
        pygame.quit()

    # ============================================================ draw
    def draw(self):
        ui = self.ui
        self.screen.fill(ui.pal["bg"])
        modal = self.confirm is not None
        if modal:                              # the dialog eats the clicks
            clicked, ui.clicked = ui.clicked, False
        if self.conn != "connected":
            self.draw_connect()
        else:
            self.draw_header()
            self.draw_map(self.MAP_R)
            self.draw_targets(self.TGT_R)
            self.draw_camera(self.CAM_R)
            self.draw_sensors(self.SEN_R)
            self.draw_aim(self.AIM_R)
            self.draw_log(self.LOG_R)
            self.draw_sidebar(self.SIDE_R)
        if modal:
            ui.clicked = clicked
            self.draw_confirm()

    # ------------------------------------------------------------ connect
    def draw_connect(self):
        ui = self.ui
        self.draw_bar("not connected", "muted")
        card = pygame.Rect(0, 0, 580, 520)
        card.center = (W // 2, HEADER + (H - HEADER) // 2)
        ui.card(card)
        x, y, cw = card.x + 30, card.y + 26, card.w - 60
        ui.label("Connect", x, y, "big")
        y += 42
        ui.label("Round 1 explores the maze, maps it and shoots.  Round 2 replays the best route.",
                 x, y, "s", "muted")
        y = self.section("Source", x, y + 34)
        self.source = ui.seg(pygame.Rect(x, y, cw, 36),
                             [("robot", "RoboMaster robot"), ("sim", "Simulator (no robot)")],
                             self.source)
        y += 50
        if self.source == "robot":
            y = self.section("Connection", x, y)
            self.connection = ui.seg(pygame.Rect(x, y, cw, 32),
                                     [("ap", "Wi-Fi direct (AP)"), ("sta", "Router (STA)"),
                                      ("rndis", "USB")], self.connection)
            now = time.time()
            if now - getattr(self, "_ip_t", 0) > 1.0:         # refresh once a second
                self._ip, self._ip_t = self.local_ip_towards(C.ROBOT_AP_IP), now
            ip = self._ip if self.connection == "ap" else None
            on_ap = bool(ip and ip.startswith("192.168.2."))
            hint = {"ap": ("On the robot's Wi-Fi (%s)." % ip) if on_ap else
                          ("Not on the robot's Wi-Fi yet (this computer is %s) - join RMEP-xxxxxx."
                           % (ip or "offline")),
                    "sta": "Robot and computer on the same router. Discovery blocked? "
                           "Set ROBOT_IP in config.py.",
                    "rndis": "USB cable to the robot's intelligent controller."}[self.connection]
            ui.label(hint, x, y + 40, "xs", "ok" if on_ap else ("warn" if self.connection == "ap"
                                                                 else "muted"), maxw=cw)
        else:
            y = self.section("Maze seed", x, y)
            if ui.button(pygame.Rect(x, y, 34, 32), "-"):
                self.seed = max(1, self.seed - 1)
            ui.rrect(pygame.Rect(x + 40, y, 60, 32), "panel2", 6)
            ui.label(self.seed, x + 70, y + 7, "h", "text", "center")
            if ui.button(pygame.Rect(x + 106, y, 34, 32), "+"):
                self.seed += 1
            ui.label("Speed", x + 170, y + 9, "xs", "muted")
            self.speed = ui.seg(pygame.Rect(x + 214, y, cw - 214, 32),
                                [(1.0, "1x"), (2.0, "2x"), (5.0, "5x"), (20.0, "20x")], self.speed)
            ui.label("Random 6x6 maze with the 4 sheet cards + 1 red circle that is not in the set.",
                     x, y + 40, "xs", "muted")
        y += 68
        y = self.section("Start", x, y)
        ui.label("cell %s facing North   (click a map cell after Connect to change it)"
                 % (tuple(C.START_CELL),), x, y, "s")
        y += 30
        self.set_armed(ui.checkbox(pygame.Rect(x, y, cw, 24), self.armed,
                                   "Blaster armed   (off = dry run: aims, never fires)"))
        y += 32
        sel = ", ".join(pretty(k) for k in sorted(self.shoot)) or "NONE"
        ui.label("shoot: " + sel, x, y, "s", "bad" if not self.shoot else "accent", maxw=cw)
        busy = self.conn == "connecting"
        r = pygame.Rect(x, card.bottom - 74, cw, 46)
        if self.error:                               # grow the card for the message
            for i, ln in enumerate(self.wrap(self.error, "s", cw)[:3]):
                ui.label(ln, x, y + 26 + i * 17, "s", "bad")
        if busy:
            ui.button(pygame.Rect(r.x, r.y, r.w - 130, r.h), "Connecting…  %s  %.0f s" % (
                getattr(self, "conn_step", ""), time.time() - self._step_t0),
                enabled=False, font="s")
            if ui.button(pygame.Rect(r.right - 122, r.y, 122, r.h), "Cancel", font="h"):
                self.cancel_connect()
        elif ui.button(r, "Connect", "primary", font="h"):
            self.connect()
        ui.label("Enter = Connect    T = theme    Q = quit", card.right - 24, card.bottom - 22,
                 "xs", "muted", "right")

    def section(self, title, x, y):
        self.ui.label(title.upper(), x, y, "xs", "muted")
        return y + 18

    # ------------------------------------------------------------ header
    def draw_bar(self, status, col):
        ui = self.ui
        ui.rrect(pygame.Rect(0, 0, W, HEADER), "panel", 0)
        pygame.draw.line(self.screen, ui.pal["line"], (0, HEADER - 1), (W, HEADER - 1))
        ui.label("RoboMaster Rescue", 16, 9, "h")
        ui.label("assignment 3-1-69  ·  6x6 maze", 16, 31, "xs", "muted")
        pygame.draw.circle(self.screen, ui.col(col), (212, 18), 5)
        ui.label(status, 224, 10, "sb", col)
        if ui.button(pygame.Rect(W - 72, 13, 60, 30), "Theme"):
            ui.set_theme("dark" if ui.theme == "light" else "light")

    def draw_header(self):
        ui = self.ui
        self.draw_bar(self.source_label(), "ok")
        x = 224
        for txt, c in ((("DRY RUN", "warn"),) if not self.armed else ()) + \
                ((("manual…", "accent"),) if self.manual_busy else ()):
            x += ui.pill(x, 30, txt, c, "xs").w + 6
        st = self.state()
        scol = {"RUNNING": "ok", "PAUSED": "warn", "DONE": "accent", "STOPPED": "bad"}.get(st, "muted")
        m = self.mission
        rnd = m.round if m else self.round_no
        ui.label("ROUND %d" % rnd, 560, 15, "t")
        ui.pill(678, 16, st, scol)
        lim = C.ROUND_LIMIT[rnd]
        el = m.elapsed() if m else 0.0
        left = lim - el
        if m is None:
            col = "muted"
        elif left <= 30:
            col = "bad" if (self.busy() and int(time.time() * 2) % 2) else "warn"
        elif left <= 60:
            col = "warn"
        else:
            col = "ok"
        ui.label("LEFT " + fmt(left), 900, 6, "t", col)
        ui.label("%s / %s" % (fmt(el), fmt(lim)), 900, 33, "xs", "text" if m else "muted")
        if ui.button(pygame.Rect(W - 186, 13, 104, 30), "Disconnect"):
            self.ask("Disconnect from the %s?" % ("simulator" if self.source == "sim" else "robot"),
                     self.disconnect)
        frac = min(1.0, el / lim) if lim else 0.0
        pygame.draw.rect(self.screen, ui.pal["line"], (0, HEADER - 3, W, 3))
        pygame.draw.rect(self.screen, ui.col(col), (0, HEADER - 3, int(W * frac), 3))

    # ------------------------------------------------------------ map
    def map_geom(self, r, m):
        size = min(r.w - 48, r.h - 76)
        cs = size / float(max(m.w, m.h))
        ox = r.x + 32 + (r.w - 48 - cs * m.w) / 2
        oy = r.y + 38

        def g2p(x, y):
            return ox + (x + 0.5) * cs, oy + (m.h - 0.5 - y) * cs
        return cs, ox, oy, g2p

    def draw_map(self, r):
        ui, s = self.ui, self.screen
        m = self.mission.maze if self.mission else Maze(C.GRID_W, C.GRID_H, C.ASSUME_BOUNDARY_WALLS)
        ui.card(r, "MAP", "north up  ·  %dx%d  ·  tile %.2f m" % (m.w, m.h, C.TILE))
        cs, ox, oy, g2p = self.map_geom(r, m)
        area = pygame.Rect(int(ox), int(oy), int(cs * m.w), int(cs * m.h))
        ui.rrect(area.inflate(10, 10), "panel2", 6)
        visited = set(m.visited)
        for c in range(m.w):
            for rr in range(m.h):
                x, y = g2p(c - 0.5, rr + 0.5)
                cell = pygame.Rect(int(x), int(y), math.ceil(cs), math.ceil(cs))
                if (c, rr) == tuple(C.START_CELL):
                    ui.rrect(cell, "startc", 0)
                elif (c, rr) in visited:
                    ui.rrect(cell, "visited", 0)
        for i in range(m.w + 1):
            x = ox + i * cs
            pygame.draw.line(s, ui.pal["grid"], (x, oy), (x, oy + cs * m.h))
        for i in range(m.h + 1):
            y = oy + i * cs
            pygame.draw.line(s, ui.pal["grid"], (ox, y), (ox + cs * m.w, y))
        for c in range(m.w):
            ui.label(c, g2p(c, 0)[0], oy + cs * m.h + 8, "xs", "muted", "center")
        for rr in range(m.h):
            ui.label(rr, ox - 16, g2p(0, rr)[1] - 7, "xs", "muted", "center")
        # walls
        for c in range(m.w):
            for rr in range(m.h):
                for d in DIRS:
                    if (d == "S" and rr > 0) or (d == "W" and c > 0) or m.get((c, rr), d) != WALL:
                        continue
                    a, b = {"N": ((c - .5, rr + .5), (c + .5, rr + .5)),
                            "E": ((c + .5, rr - .5), (c + .5, rr + .5)),
                            "S": ((c - .5, rr - .5), (c + .5, rr - .5)),
                            "W": ((c - .5, rr - .5), (c - .5, rr + .5))}[d]
                    pygame.draw.line(s, ui.pal["wall"], g2p(*a), g2p(*b), 5)
                    for p in (a, b):
                        pygame.draw.circle(s, ui.pal["wall"], [int(v) for v in g2p(*p)], 2)
        sx, sy = g2p(*C.START_CELL)
        ui.label("S", sx - cs * 0.40, sy - cs * 0.44, "sb", "ok")
        pose = self.pose()
        if pose is not None:                   # camera field of view
            rx, ry = g2p(pose[0] / C.TILE, pose[1] / C.TILE)
            head = pose[2] + self.hal.gimbal_yaw
            L = cs * 1.8
            pts = [(rx, ry)]
            for k in range(9):
                a = math.radians(head - C.CAMERA_HFOV / 2 + k * C.CAMERA_HFOV / 8)
                pts.append((rx + math.sin(a) * L, ry - math.cos(a) * L))
            cone = pygame.Surface((W, H), pygame.SRCALPHA)
            pygame.draw.polygon(cone, (*ui.pal["warn"], 38), pts)
            clip = s.get_clip()
            s.set_clip(area)
            s.blit(cone, (0, 0))
            s.set_clip(clip)
        if self.mission is not None:           # odometry trace + cell path
            tr = list(self.mission.motion.trace)
            if len(tr) > 1:
                pygame.draw.lines(s, ui.pal["accent"], False,
                                  [g2p(x / C.TILE, y / C.TILE) for x, y in tr], 1)
            path = list(self.mission.path)
            if len(path) > 1:
                pygame.draw.lines(s, ui.pal["path"], False, [g2p(*c) for c in path], 3)
            for c in set(path):
                pygame.draw.circle(s, ui.pal["path"], [int(v) for v in g2p(*c)], 3)
        for t, p in layout(self.targets_now()):
            self.draw_map_target(t, g2p(*p), cs)
        if self.show_truth and hasattr(self.hal, "targets"):
            for t, p in layout(self.hal.targets):
                x, y = g2p(*p)
                color, _ = label_parts(t["label"])
                pygame.draw.circle(s, CARD_RGB.get(color, (128, 128, 128)), (int(x), int(y)),
                                   int(cs * 0.2), 1)
        if pose is not None:                   # robot + gimbal direction
            a = math.radians(head)
            tip = (rx + math.sin(a) * cs * 0.36, ry - math.cos(a) * cs * 0.36)
            pygame.draw.line(s, ui.pal["bad"], (rx, ry), tip, 3)
            pygame.draw.circle(s, ui.pal["bad"], (int(rx), int(ry)), int(cs * 0.19))
            pygame.draw.circle(s, ui.pal["white"], (int(rx), int(ry)), int(cs * 0.07))
        strip_y = r.bottom - 26
        if not self.busy():
            if ui.clicked and area.collidepoint(ui.mouse):
                c = (int((ui.mouse[0] - ox) // cs), int(m.h - 1 - (ui.mouse[1] - oy) // cs))
                if 0 <= c[0] < m.w and 0 <= c[1] < m.h and c != tuple(C.START_CELL):
                    self.set_start(c)
            ui.label("Start %s facing N  ·  click a cell to move the start" % (tuple(C.START_CELL),),
                     r.x + 14, strip_y, "xs", "muted")
        else:
            mm = self.mission
            ui.label("visited %d/%d   ·   orange = cell path   ·   blue = odometry" % (
                len(mm.maze.visited), m.w * m.h), r.x + 14, strip_y, "xs", "muted")

    def draw_map_target(self, t, pos, cs):
        """A target against its wall: plate icon + status ring (several per cell)."""
        ui, s = self.ui, self.screen
        color, shape = label_parts(t["label"])
        x, y = int(pos[0]), int(pos[1])
        chosen = t["label"] in self.shoot
        if t.get("shot"):
            ring = ui.pal["ok"]
        elif chosen and not t.get("round1"):
            ring = ui.pal["accent"]
        else:
            ring = ui.pal["muted"]
        pygame.draw.circle(s, ui.pal["panel"], (x, y), int(cs * 0.17))
        pygame.draw.circle(s, ring, (x, y), int(cs * 0.17), 3 if t.get("shot") else (2 if chosen else 1))
        ui.icon((x, y), color, shape, int(cs * 0.10), hollow=t.get("round1", False))
        if t.get("shot"):
            ui.label("HIT", x, y + int(cs * 0.17) + 1 if t.get("side") != "S" else y - int(cs * 0.17) - 13,
                     "xs", "ok", "center")

    def targets_now(self):
        """Targets of the current mission, or the round-1 cards before round 2."""
        if self.mission is not None:
            return [dict(t, cell=tuple(t["cell"])) for t in list(self.mission.targets)]
        if self.round_no == 2:
            data = self.round1_data()
            if data:
                return [dict(t, cell=tuple(t["cell"]), shot=False, round1=True)
                        for t in data["targets"]]
        return []

    def pose(self):
        if self.hal is None:
            return None
        if self.mission is not None:
            return self.mission.motion.pose()
        n, e, yaw = self.hal.odom()
        return C.START_CELL[0] * C.TILE + e, C.START_CELL[1] * C.TILE + n, yaw

    # ------------------------------------------------------------ targets
    def draw_targets(self, r):
        ui = self.ui
        rows = self.targets_now()
        hits = {(s["label"], tuple(s["cell"]), s.get("side")): s["t"]
                for s in (self.mission.shots if self.mission else [])}
        n_hit = sum(1 for t in rows if t.get("shot"))
        n_sel = sum(1 for t in rows if t["label"] in self.shoot)
        ui.card(r, "TARGETS", "hit %d / %d to shoot   ·   %d found" % (n_hit, n_sel, len(rows)),
                "ok" if rows and n_hit == n_sel else "muted")
        y0 = r.y + 38
        if not rows:
            ui.label("no cards found yet", r.x + 14, y0, "s", "muted")
            sel = ", ".join(pretty(k) for k in sorted(self.shoot)) or "none - pick them under SHOOT"
            for i, ln in enumerate(self.wrap("shooting: " + sel, "s", r.w - 28)[:3]):
                ui.label(ln, r.x + 14, y0 + 22 + i * 18, "s")
            return
        rows.sort(key=lambda t: (t["label"] not in self.shoot, not t.get("shot"), t["label"]))
        rh = 25
        nmax = (r.h - 46) // rh
        for i, t in enumerate(rows[:nmax]):
            y = y0 + i * rh
            if i % 2 == 0:
                ui.rrect(pygame.Rect(r.x + 8, y - 5, r.w - 16, rh - 1), "panel2", 6)
            color, shape = label_parts(t["label"])
            ui.icon((r.x + 26, y + 8), color, shape, 9, hollow=t.get("round1", False))
            chosen = t["label"] in self.shoot
            ui.label(pretty(t["label"]), r.x + 46, y, "s", "text" if chosen else "muted")
            ui.label("cell %s  %s wall" % (tuple(t["cell"]), t.get("side") or "?"), r.x + 186, y,
                     "s", "text" if chosen else "muted")
            if t.get("shot"):
                st, c = "HIT " + fmt(hits.get((t["label"], tuple(t["cell"]), t.get("side")), 0)), "ok"
            elif t.get("round1"):
                st, c = "from round 1", "muted"
            elif chosen:
                st, c = "to shoot", "accent"
            else:
                st, c = "not selected", "muted"
            ui.label(st, r.right - 16, y, "sb", c, "right")
        if len(rows) > nmax:
            ui.label("+%d more on the map" % (len(rows) - nmax), r.x + 14, r.bottom - 18, "xs", "muted")

    # ------------------------------------------------------------ camera
    def draw_camera(self, r):
        ui, s = self.ui, self.screen
        ui.card(r, "CAMERA")
        self.view = ui.seg(pygame.Rect(r.right - 204, r.y + 5, 192, 24),
                           [("overlay", "Overlay"), ("mask", "Mask"), ("raw", "Raw")],
                           self.view, font="xs")
        view = pygame.Rect(r.x + 10, r.y + 34, r.w - 20, r.h - 44)
        frame = getattr(self.hal, "frame", None)
        t_det, dets = self.hal.detections()
        if self._last_det_t != t_det:
            self._fps.append(time.time())
            self._last_det_t = t_det
        clip = s.get_clip()
        s.set_clip(view)
        det = getattr(self.hal, "_detector", None)
        if frame is not None:
            try:
                import cv2
                img = cv2.resize(frame, view.size)
                if det is not None and self.view == "mask":
                    img = det.mask_view(img)      # wall grey, ignored blue, cards coloured
                elif det is not None and self.view == "overlay":
                    img = det.draw_ignored(img)   # dim the room + barrel, wall line
                s.blit(bgr_to_surface(img), view.topleft)
            except Exception:
                ui.rrect(view, "camera", 6)
        elif self.source == "sim":
            self.render_sim_view(view)
            if self.view == "mask":
                ui.label("Mask view needs the robot camera (the simulator has no pixels)",
                         view.centerx, view.bottom - 40, "xs", "warn", "center")
        else:
            ui.rrect(view, "camera", 6)
            ui.label("waiting for camera…", view.centerx, view.centery - 8, "h", "muted", "center")
        if self.view != "raw":
            placed = []                            # label boxes already drawn (no overlaps)
            for d in sorted(dets, key=lambda d: d.cx):
                self.draw_detection(view, d, placed)
        cx, cy = view.center                       # boresight
        col = ui.pal["bad"]
        pygame.draw.circle(s, col, (cx, cy), 9, 1)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            pygame.draw.line(s, col, (cx + dx * 5, cy + dy * 5), (cx + dx * 14, cy + dy * 14), 1)
        fps = 0.0
        if len(self._fps) > 1:
            fps = (len(self._fps) - 1) / max(1e-3, self._fps[-1] - self._fps[0])
        stale = frame is not None and self.hal.now() - t_det > 1.0
        badge = pygame.Rect(view.x + 8, view.y + 8, 236, 22)
        sh = pygame.Surface(badge.size, pygame.SRCALPHA)
        sh.fill((*ui.pal["panel"], 220))
        s.blit(sh, badge.topleft)
        if not stale and int(time.time() * 2) % 2:
            pygame.draw.circle(s, ui.pal["bad"], (badge.x + 11, badge.centery), 4)
        ui.label(("NO SIGNAL" if stale else ("LIVE" if frame is not None else "SIM VIEW")) +
                 "   det %4.1f fps   %d in view" % (fps, len(dets)), badge.x + 22, badge.y + 4,
                 "xs", "bad" if stale else "ok")
        a = self.mission.aim if self.mission else None
        if a and time.time() - a["wall"] < 1.2 and a["phase"] != "IDLE":
            txt = {"FIRE": "FIRE!", "DRY RUN": "FIRE (dry run)"}.get(a["phase"], a["phase"])
            b = ui.text(txt, "t", "white")
            box = b.get_rect(midtop=(view.centerx, view.y + 40)).inflate(24, 8)
            pygame.draw.rect(s, ui.col(PHASE_COL.get(a["phase"], "muted")), box, border_radius=8)
            s.blit(b, b.get_rect(center=box.center))
        if not self.armed:
            ui.label("DRY RUN – blaster off", view.right - 10, view.bottom - 22, "sb", "warn", "right")
        s.set_clip(clip)
        pygame.draw.rect(s, ui.pal["line"], view, 1, border_radius=4)

    def draw_detection(self, view, d, placed):
        ui, s = self.ui, self.screen
        sx, sy = view.w / float(d.fw), view.h / float(d.fh)
        rect = pygame.Rect(view.x + (d.cx - d.w / 2) * sx, view.y + (d.cy - d.h / 2) * sy,
                           max(3, d.w * sx), max(3, d.h * sy))
        chosen = d.label in self.shoot
        dist = distance(d)
        in_range = dist is not None and dist <= C.FIRE_RANGE_M
        col = ui.pal["ok"] if chosen and in_range else (ui.pal["warn"] if chosen else ui.pal["muted"])
        pygame.draw.rect(s, col, rect.inflate(6, 6), 2, border_radius=3)
        tag = "%s  %s  %s" % (pretty(d.label), "%.2f m" % dist if dist else "--",
                              ("IN RANGE" if in_range else "far") if chosen else "not selected")
        t = ui.text(tag, "xs", "white")
        box = t.get_rect(bottomleft=(rect.x - 3, rect.y - 6)).inflate(12, 6)
        box.clamp_ip(view)
        while box.collidelist(placed) >= 0 and box.y > view.y + 34:
            box.y -= box.h + 2                     # stack upwards past earlier labels
        placed.append(box)
        pygame.draw.line(s, col, (rect.x, rect.y - 3), (box.x + 1, box.bottom), 1)
        sh = pygame.Surface(box.size, pygame.SRCALPHA)
        sh.fill((20, 22, 26, 205))
        s.blit(sh, box.topleft)
        pygame.draw.rect(s, col, (box.x, box.y, 3, box.h))
        s.blit(t, (box.x + 7, box.y + 3))

    def render_sim_view(self, r):
        """Pseudo-3D simulator camera: ray-cast walls + cards on sticks."""
        hal, s, ui = self.hal, self.screen, self.ui
        f = (r.w / 2.0) / math.tan(math.radians(C.CAMERA_HFOV) / 2)
        head = hal.yaw + hal.gimbal_yaw
        hz = r.centery + f * math.tan(math.radians(hal.gimbal_pitch))
        dark = ui.theme == "dark"
        room, floor = ((52, 56, 64), (36, 38, 44)) if dark else ((216, 214, 210), (160, 158, 154))
        top_h = max(0, min(r.h, int(hz - r.y)))
        pygame.draw.rect(s, room, (r.x, r.y, r.w, top_h))
        pygame.draw.rect(s, floor, (r.x, r.y + top_h, r.w, r.h - top_h))
        n = r.w // 4
        cw = r.w / float(n)
        zbuf = []
        base = (190, 194, 202) if dark else (240, 241, 244)
        for i in range(n):
            px = r.x + (i + 0.5) * cw
            ang = head + math.degrees(math.atan((px - r.centerx) / f))
            dist, face = self._cast(hal.x, hal.y, ang)
            z = max(0.05, dist * math.cos(math.radians(ang - head)))
            zbuf.append(z)
            top = hz - f * (SIM_WALL_H - SIM_CAM_H) / z
            bot = hz + f * SIM_CAM_H / z
            k = max(0.55, 1.0 - z / 6.0) * (0.9 if face == "x" else 1.0)
            c = tuple(int(v * k) for v in base)
            y0, y1 = max(r.y, int(top)), min(r.bottom, int(bot))
            if y1 > y0:
                pygame.draw.rect(s, c, (int(r.x + i * cw), y0, math.ceil(cw), y1 - y0))
                pygame.draw.rect(s, tuple(int(v * 0.8) for v in c), (int(r.x + i * cw), y0, math.ceil(cw), 2))
        a = math.radians(head)
        items = []
        for tg in hal.targets:
            dx, dy = tg["x"] - hal.x, tg["y"] - hal.y
            z = dx * math.sin(a) + dy * math.cos(a)
            lat = dx * math.cos(a) - dy * math.sin(a)
            if z > 0.12:
                items.append((z, lat, tg))
        for z, lat, tg in sorted(items, key=lambda it: -it[0]):
            px = r.centerx + f * lat / z
            col = int((px - r.x) / cw)
            if not 0 <= col < n or zbuf[col] < z - 0.05:
                continue
            color, shape = label_parts(tg["label"])
            pw, ph = plate_size(tg["label"])
            w, h = f * pw / z, f * ph / z
            pcy = hz - f * (SIM_PLATE_H - SIM_CAM_H) / z
            ground = hz + f * SIM_CAM_H / z
            pygame.draw.line(s, (40, 40, 44), (px, pcy + h / 2), (px, ground), max(1, int(f * 0.012 / z)))
            rect = pygame.Rect(0, 0, max(2, int(w)), max(2, int(h)))
            rect.center = (int(px), int(pcy))
            rgb = CARD_RGB.get(color, (128, 128, 128))
            if shape == "circle":
                pygame.draw.ellipse(s, rgb, rect)
            else:
                pygame.draw.rect(s, rgb, rect)

    def _cast(self, x, y, ang):
        """DDA ray-cast on the true simulator maze -> (distance m, 'x'|'y' face)."""
        m, T = self.hal.maze, C.TILE
        dx, dy = math.sin(math.radians(ang)), math.cos(math.radians(ang))
        u, v = x / T + 0.5, y / T + 0.5
        cx, cy = int(math.floor(u)), int(math.floor(v))
        sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
        big = 1e9
        tx = ((cx + 1 - u) / dx if dx > 0 else (u - cx) / -dx) if abs(dx) > 1e-9 else big
        ty = ((cy + 1 - v) / dy if dy > 0 else (v - cy) / -dy) if abs(dy) > 1e-9 else big
        ddx = abs(1 / dx) if abs(dx) > 1e-9 else big
        ddy = abs(1 / dy) if abs(dy) > 1e-9 else big
        for _ in range(40):
            if tx < ty:
                if m.get((cx, cy), "E" if sx > 0 else "W") == WALL:
                    return tx * T, "x"
                cx += sx
                tx += ddx
            else:
                if m.get((cx, cy), "N" if sy > 0 else "S") == WALL:
                    return ty * T, "y"
                cy += sy
                ty += ddy
        return 20.0, "x"

    # ------------------------------------------------------------ sensors
    def draw_sensors(self, r):
        """Robot sketch: gimbal ToF ray (turns with the gimbal), side Sharp bars,
        front-corner IR rays; table on the right."""
        ui, s = self.ui, self.screen
        ui.card(r, "SENSORS")
        ex = self.hal.ranges_ex() if hasattr(self.hal, "ranges_ex") else {}
        thr = C.TILE / 2 + C.WALL_TOL
        cx, cy = r.x + 62, r.y + 108
        full = 44.0                                  # px per metre
        body = pygame.Rect(0, 0, 24, 32)
        body.center = (cx, cy)

        def col(v):
            return ui.pal["ok"] if v is None or v > thr else ui.pal["bad"]
        from hal import side_dir
        hd = (self.hal.heading() if hasattr(self.hal, "heading") else None) or "N"
        for side, ux in (("R", 1), ("L", -1)):       # side Sharps (robot drawn facing up)
            d = side_dir(hd, side)
            x0 = cx + ux * 13
            pygame.draw.line(s, ui.pal["line"], (x0, cy), (x0 + ux * full, cy), 7)
            if d in ex and ex[d][1] == "sharp":
                v = ex[d][0]
                pygame.draw.line(s, col(v), (x0, cy), (x0 + ux * min(full, (v or 9) * full), cy), 7)
            tk = x0 + ux * thr * full
            pygame.draw.line(s, ui.pal["text"], (tk, cy - 6), (tk, cy + 6), 1)
        tof = [(d, v) for d, (v, k) in ex.items() if k == "tof"]
        a = math.radians(self.hal.gimbal_yaw)        # gimbal ToF ray
        L = full * 1.25
        dx, dy = math.sin(a), -math.cos(a)
        pygame.draw.line(s, ui.pal["line"], (cx, cy), (cx + dx * L, cy + dy * L), 3)
        if tof:
            v = tof[0][1]
            Lv = min(L, (v or 9) * full)
            pygame.draw.line(s, col(v), (cx, cy), (cx + dx * Lv, cy + dy * Lv), 4)
        ui.rrect(body, "text", 4)
        pygame.draw.polygon(s, ui.pal["panel"], [(cx, cy - 9), (cx - 6, cy + 3), (cx + 6, cy + 3)])
        irs = self.hal.ir_state() if hasattr(self.hal, "ir_state") else {}
        for n, spec in C.IR.items():                 # IR rays from the front corners
            px, py = spec["pos"]
            x0, y0 = cx + px * 100, cy - py * 100
            aa = math.radians(spec["ang"])
            x1, y1 = x0 + math.sin(aa) * 16, y0 - math.cos(aa) * 16
            on = irs.get(n)
            c = ui.pal["warn"] if on is None else (ui.pal["bad"] if on else ui.pal["line"])
            pygame.draw.line(s, c, (x0, y0), (x1, y1), 4)
        ui.label("bar 1 m · tick = wall · ray = ToF", r.x + 12, r.bottom - 20, "xs", "muted")
        x, y = r.x + 124, r.y + 34

        def row(k, v, c="text"):
            ui.label(k, x, y, "xs", "muted")
            ui.label(v, r.right - 12, y, "xs", c, "right")
        if tof:
            d, v = tof[0]
            row("ToF (%s)" % d, "far" if v is None else "%.2f m" % v,
                "muted" if v is None else ("bad" if v <= thr else "ok"))
        else:
            row("ToF", "turning / no data", "muted")
        y += 17
        for side, name in (("R", "Sharp R"), ("L", "Sharp L")):
            d = side_dir(hd, side)
            if d not in ex:
                row(name, "NO DATA", "bad")
            else:
                v, kind = ex[d]
                txt = "far / close" if v is None else "%.2f m" % v
                row(name + (" (ToF)" if kind == "tof" else ""), txt,
                    "muted" if v is None else ("bad" if v <= thr else "ok"))
            y += 17
        row("IR", "  ".join("%s %s" % (n, "?" if irs.get(n) is None else ("ON" if irs[n] else "–"))
                            for n in C.IR), "warn" if None in irs.values() else "text")
        y += 21
        pygame.draw.line(s, ui.pal["line"], (x, y - 3), (r.right - 12, y - 3))
        p = self.pose()
        n_, e, yaw = self.hal.odom()
        for k, v in (("cell / facing", "(%d, %d)  %s" % (round(p[0] / C.TILE), round(p[1] / C.TILE), hd)),
                     ("yaw", "%+.1f°" % yaw),
                     ("gimbal", "%+.0f° / %+.0f°" % (self.hal.gimbal_yaw, self.hal.gimbal_pitch)),
                     ("odom", "%+.2f  %+.2f" % (n_, e))):
            row(k, v)
            y += 17

    # ------------------------------------------------------------ aim
    def draw_aim(self, r):
        ui, s = self.ui, self.screen
        a = self.mission.aim if self.mission else None
        active = a is not None and time.time() - a["wall"] < 3.0
        phase = a["phase"] if active else "IDLE"
        ui.card(r, "AIM", "tol ±%.1f°" % C.AIM_TOL)
        span, sz = 6.0, 88
        box = pygame.Rect(r.x + 12, r.y + 34, sz, sz)
        ui.rrect(box, "panel2", 4)
        ui.rrect(box, "line", 4, 1)
        pygame.draw.line(s, ui.pal["line"], (box.x, box.centery), (box.right, box.centery))
        pygame.draw.line(s, ui.pal["line"], (box.centerx, box.y), (box.centerx, box.bottom))
        t = int(sz / 2 * C.AIM_TOL / span)
        pygame.draw.rect(s, ui.pal["ok"], (box.centerx - t, box.centery - t, 2 * t, 2 * t), 1)

        def to_px(yaw, pitch):
            k = sz / 2 / span
            return (int(box.centerx + max(-span, min(span, yaw)) * k),
                    int(box.centery - max(-span, min(span, pitch)) * k))
        pc = PHASE_COL.get(phase, "muted")
        now = self.hal.now()
        if active and a["yaw"] is not None:
            trail = [h for h in a["hist"] if h[0] > now - 4][-12:]
            if len(trail) > 1:
                pygame.draw.lines(s, ui.pal["muted"], False, [to_px(h[1], h[2]) for h in trail], 1)
            pygame.draw.circle(s, ui.col(pc), to_px(a["yaw"], a["pitch"]), 5)
        tx = box.right + 12
        ui.label(phase, tx, r.y + 34, "t", pc, maxw=r.right - tx - 8)
        if active and a["label"]:
            color, _ = label_parts(a["label"])
            ui.label(pretty(a["label"]), tx, r.y + 62, "s", CARD_RGB.get(color, ui.pal["text"]),
                     maxw=r.right - tx - 8)
        if active and a["yaw"] is not None:
            ui.label("yaw    %+.2f°" % a["yaw"], tx, r.y + 84, "xs")
            ui.label("pitch  %+.2f°" % a["pitch"], tx, r.y + 100, "xs")
        g = pygame.Rect(r.x + 12, r.y + 132, r.w - 24, r.h - 142)     # error trace, 10 s
        ui.rrect(g, "panel2", 4)
        band = max(1, int(g.h / 2 * C.AIM_TOL / span))
        pygame.draw.rect(s, ui.pal["onbg"], (g.x, g.centery - band, g.w, 2 * band))
        if a is not None:
            pts = [h for h in a["hist"] if h[0] > now - 10]
            for idx, c in ((1, "accent"), (2, "ok")):
                poly = [(int(g.right - (now - h[0]) / 10.0 * g.w),
                         int(g.centery - max(-span, min(span, h[idx])) / span * g.h / 2)) for h in pts]
                if len(poly) > 1:
                    pygame.draw.lines(s, ui.col(c), False, poly, 1)
        ui.label("yaw", g.x + 6, g.y + 3, "xs", "accent")
        ui.label("pitch", g.x + 32, g.y + 3, "xs", "ok")
        ui.label("last 10 s", g.right - 6, g.y + 3, "xs", "muted", "right")

    # ------------------------------------------------------------ log
    def draw_log(self, r):
        ui = self.ui
        ui.card(r, "LOG")
        lines = list(self.log.lines)
        n = (r.h - 38) // 17
        y = r.y + 32
        for ts, line in lines[-n:]:
            c = "ok" if "SHOT" in line else "warn" if "TARGET" in line else \
                "bad" if any(w in line for w in ("miss", "fail", "Error", "abort", "Traceback")) else "text"
            ui.label(ts, r.x + 12, y, "m", "muted")
            ui.label(line.strip(), r.x + 80, y, "m", c, maxw=r.w - 92)
            y += 17

    # ------------------------------------------------------------ sidebar
    def draw_sidebar(self, r):
        ui, s = self.ui, self.screen
        ui.card(r)
        x, w = r.x + 12, r.w - 24
        half = (w - 6) // 2
        busy, st = self.busy(), self.state()
        # ---- mission
        y = self.section("Mission", x, r.y + 12)
        self.round_no = ui.seg(pygame.Rect(x, y, w, 30), [(1, "Round 1 · explore"),
                                                          (2, "Round 2 · replay")],
                               self.round_no, enabled=not busy, font="xs")
        y += 36
        if self.round_no == 2:
            ok = self.round1_data() is not None
            ui.label("round-1 map ready" if ok else "no round-1 map yet - run round 1 first",
                     x, y, "xs", "ok" if ok else "warn")
        else:
            ui.label("put the robot on S, facing North", x, y, "xs", "muted")
        y += 20
        if ui.button(pygame.Rect(x, y, w, 42), "Start round %d" % self.round_no, "primary",
                     enabled=self.can_start(), font="h"):
            self.start()
        y += 48
        paused = st == "PAUSED"
        if ui.button(pygame.Rect(x, y, half, 32), "Resume" if paused else "Pause",
                     on=paused, enabled=busy):
            self.toggle_pause()
        if ui.button(pygame.Rect(x + half + 6, y, half, 32), "Save map",
                     enabled=self.mission is not None):
            self.save_map()
        y += 38
        if ui.button(pygame.Rect(x, y, w, 42), "STOP", "danger", enabled=busy, font="h"):
            self.stop()
        y += 58
        # ---- which cards to shoot
        pygame.draw.line(s, ui.pal["line"], (x, y - 9), (x + w, y - 9))
        n = len(self.shoot)
        ui.label("SHOOT", x, y, "xs", "muted")
        ui.label("%d kind%s%s" % (n, "" if n == 1 else "s", "  - more than the sheet!" if n > 4 else ""),
                 x + w, y, "xs", "bad" if n == 0 else ("warn" if n > 4 else "accent"), "right")
        y += 18
        gx, cwid = x + 58, (w - 58) // 4
        for j, sh in enumerate(C.SHAPES):
            ui.label(sh, gx + j * cwid + cwid // 2, y, "xs", "muted", "center")
        y += 16
        for col in C.COLORS:
            pygame.draw.rect(s, CARD_RGB[col], (x, y + 8, 10, 10), border_radius=3)
            ui.label(col, x + 15, y + 6, "xs")
            for j, sh in enumerate(C.SHAPES):
                kind = kind_label(col, sh)
                on = kind in self.shoot
                cell = pygame.Rect(gx + j * cwid + 2, y + 2, cwid - 4, 24)
                if ui.button(cell, "", on=on, enabled=not busy):
                    self.shoot.symmetric_difference_update({kind})
                ui.icon(cell.center, col if on else None, sh, 6, hollow=not on)
                if kind in C.TARGET_CLASSES:
                    pygame.draw.circle(s, ui.pal["warn"], (cell.right - 6, cell.y + 6), 2)
            y += 28
        y += 4
        third = (w - 12) // 3
        for i, (lbl, fn) in enumerate((("Sheet set", lambda: set(C.TARGET_CLASSES)),
                                       ("All", lambda: {kind_label(c, s_) for c in C.COLORS
                                                        for s_ in C.SHAPES}),
                                       ("None", set))):
            if ui.button(pygame.Rect(x + i * (third + 6), y, third, 26), lbl, enabled=not busy,
                         font="xs"):
                self.shoot = fn()
        y += 32
        ui.label("orange dot = sheet card; others: map only", x, y, "xs", "muted", maxw=w)
        y += 28
        # ---- manual actions
        pygame.draw.line(s, ui.pal["line"], (x, y - 9), (x + w, y - 9))
        ui.label("ACTIONS", x, y, "xs", "muted")
        y += 18
        self.set_armed(ui.checkbox(pygame.Rect(x, y, half, 26), self.armed, "Blaster armed",
                                   color="bad"))
        idle = not busy and not self.manual_busy
        if ui.button(pygame.Rect(x + half + 6, y, half, 26), "Fire once", enabled=idle, font="xs"):
            self.fire_once()
        y += 32
        ui.label("Gimbal", x, y + 6, "xs", "muted")
        bw = (w - 52 - 4 * 4) // 5
        for i, (lbl, fn) in enumerate((("<", lambda: self.gimbal(dyaw=-15)),
                                       ("^", lambda: self.gimbal(dpitch=5)),
                                       ("v", lambda: self.gimbal(dpitch=-5)),
                                       (">", lambda: self.gimbal(dyaw=15)),
                                       ("C", lambda: self.gimbal(centre=True)))):
            if ui.button(pygame.Rect(x + 52 + i * (bw + 4), y, bw, 26), lbl, enabled=idle, font="sb"):
                fn()
        y += 32
        if ui.button(pygame.Rect(x, y, half, 26), "Screenshot", font="xs"):
            self.screenshot()
        if self.source == "sim":
            if ui.button(pygame.Rect(x + half + 6, y, half, 26), "Show sim truth",
                         on=self.show_truth, font="xs"):
                self.show_truth = not self.show_truth
        y = r.bottom - 44
        for line in ("Space start / STOP  ·  P pause  ·  1 2 round",
                     "WASD drive  ·  J L I K gimbal  ·  F fire  ·  S shot"):
            ui.label(line, x, y, "xs", "muted", maxw=w)
            y += 16

    # ------------------------------------------------------------ dialog
    def draw_confirm(self):
        ui = self.ui
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 110))
        self.screen.blit(shade, (0, 0))
        box = pygame.Rect(0, 0, 440, 150)
        box.center = (W // 2, H // 2)
        ui.card(box)
        msg, cb = self.confirm
        ui.label(msg, box.centerx, box.y + 34, "h", "text", "center")
        if ui.button(pygame.Rect(box.centerx - 130, box.bottom - 58, 120, 36), "Yes", "danger"):
            self.confirm = None
            cb()
        elif ui.button(pygame.Rect(box.centerx + 10, box.bottom - 58, 120, 36), "No"):
            self.confirm = None

    def wrap(self, s, font, width):
        f = self.ui.fonts[font]
        lines, cur = [], ""
        for word in str(s).split():
            nxt = (cur + " " + word).strip()
            if f.size(nxt)[0] <= width:
                cur = nxt
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
        return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="pre-select the simulator")
    ap.add_argument("--connect", action="store_true", help="connect straight away")
    ap.add_argument("--seed", type=int, default=1, help="simulator maze seed")
    ap.add_argument("--speed", type=float, default=2.0, help="simulator speed (x real time)")
    ap.add_argument("--no-fire", action="store_true", help="start with the blaster disarmed")
    ap.add_argument("--theme", choices=("light", "dark"), default="light")
    args = ap.parse_args()
    C.check_sensor_budget()
    App(args).run()


if __name__ == "__main__":
    main()
