"""Card detector - Final_Robomaster's segmentation (src/target_vision.py).

Per frame (processed at PROC_WIDTH px):
  1. white-balance on the white foam walls, blur, HSV colour masks (tuned ranges)
  2. ignore the room above the walls (top of the white wall band traced per
     column, card-sized gaps bridged) and the blaster barrel at the bottom
  3. per blob: shape from the contour (circle / wide / tall / square / unknown),
     and it is a CARD only if it is whole (not cut by the frame or the wall
     line), solid, not part of a bigger patch of that colour, has neutral wall
     around it, and sits at card height above the floor
Distance comes from the card HEIGHT only (a slanted card looks narrower).
"""
import json
import math
import os
from collections import namedtuple

import config as C

# cx, cy, w, h are in pixels of the ORIGINAL frame; fw, fh = frame size.
Detection = namedtuple("Detection", "label color shape cx cy w h area fw fh partial")

CLASS_BY_KEY = {(v["color"], v["shape"]): k for k, v in C.TARGET_CLASSES.items()}


def kind_label(color, shape):
    """Label for any colour x shape: the sheet name if it has one."""
    return CLASS_BY_KEY.get((color, shape), "%s_%s" % (color, shape))


def label_parts(label):
    """(color, shape) of a label."""
    spec = C.TARGET_CLASSES.get(label)
    if spec:
        return spec["color"], spec["shape"]
    color, _, shape = label.partition("_")
    return color, shape


def plate_size(label):
    """(w, h) of the coloured plate in metres."""
    spec = C.TARGET_CLASSES.get(label)
    if spec:
        return spec["w"], spec["h"]
    return C.SHAPE_SIZE.get(label_parts(label)[1], (0.10, 0.10))


def classify_contour(cnt):
    """Final_Robomaster's contour rules -> (shape, info). shape in circle / wide /
    tall / square / unknown."""
    import cv2
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if area <= 0 or peri <= 0:
        return "unknown", {}
    circularity = 4.0 * math.pi * area / (peri * peri)
    (_, _), radius = cv2.minEnclosingCircle(cnt)
    circle_fill = area / (math.pi * radius * radius) if radius > 0 else 0.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    rect_fill = area / (rw * rh) if rw * rh > 0 else 0.0
    _, _, bw, bh = cv2.boundingRect(cnt)
    aspect = bw / float(bh) if bh else 0.0      # plates stand upright: the upright box
    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
    info = {"circularity": round(circularity, 2), "circle_fill": round(circle_fill, 2),
            "rect_fill": round(rect_fill, 2), "aspect": round(aspect, 2), "vertices": len(approx)}
    if (circularity >= 0.70 and circle_fill >= 0.58 and rect_fill < 0.90
            and 0.65 <= aspect <= 1.40 and len(approx) >= 6):
        return "circle", info
    if rect_fill >= 0.82 and 4 <= len(approx) <= 6:
        if aspect > 2.4 or aspect < 0.3:
            info["why"] = "long thin strip (tape)"
            return "unknown", info
        if aspect > 1.28:
            return "wide", info
        if aspect < 0.78:
            return "tall", info
        if rect_fill >= 0.90 and len(approx) <= 5 and circle_fill < 0.72:
            return "square", info
        info["why"] = "round or square?"
    return "unknown", info


def bearing(det):
    """(yaw, pitch) in degrees from the image centre to the blob centre.
    yaw + = right, pitch + = up."""
    f = (det.fw / 2.0) / math.tan(math.radians(C.CAMERA_HFOV) / 2.0)
    yaw = math.degrees(math.atan((det.cx - det.fw / 2.0) / f))
    pitch = -math.degrees(math.atan((det.cy - det.fh / 2.0) / f))
    return yaw, pitch


def distance(det):
    """Pinhole range (m) from the card HEIGHT (a card seen at a slant looks
    narrower, its height does not change)."""
    if det.label is None:
        return None
    _, ph = plate_size(det.label)
    f = (det.fw / 2.0) / math.tan(math.radians(C.CAMERA_HFOV) / 2.0)
    return f * ph / max(1.0, det.h)


BARREL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "barrel_box.json")


def barrel_box():
    """(x0, y0, x1, y1) frame fractions of the barrel tip: barrel_box.json
    (written by  python main.py --vision-test, key B) or config.BARREL_BOX."""
    try:
        with open(BARREL_FILE) as f:
            return tuple(json.load(f)["box"])
    except (OSError, ValueError, KeyError):
        return C.BARREL_BOX


class Detector:
    def __init__(self):
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        self.kernel = np.ones((3, 3), np.uint8)
        self.ranges = {c: [(np.array(lo, np.uint8), np.array(hi, np.uint8))
                           for lo, hi in rs] for c, rs in C.HSV.items()}
        self.pitch = C.GIMBAL_PITCH        # gimbal pitch (deg), set by the HAL
        self.barrel = barrel_box()
        self.last = None                   # segmentation of the last frame (for drawing)

    # ------------------------------------------------------------ helpers
    def white_mask(self, hsv):
        """Foam wall pixels: low saturation, bright."""
        np = self.np
        return (hsv[:, :, 1] <= C.WALL_S_MAX) & (hsv[:, :, 2] >= C.WALL_V_MIN)

    def wall_top(self, white):
        """Row of the top of the white wall band per processing column (NaN =
        no wall).  Each column is walked up from the lowest long white run;
        runs above it join while the gap is card-sized (a card in front of the
        wall), and the walk stops at a bigger gap - the room."""
        cv2, np = self.cv2, self.np
        ph, pw = white.shape
        cols = 80
        rows = max(20, int(round(ph * cols / float(pw))))
        g = cv2.resize(white.astype(np.uint8) * 255, (cols, rows),
                       interpolation=cv2.INTER_AREA) > 127
        g = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
        max_gap = max(2, int(rows * C.WALL_MAX_GAP))
        low = int(rows * 0.5)                  # the wall base is in the lower half
        pad = np.zeros((1, cols), np.int8)
        edges = np.diff(np.vstack([pad, g.astype(np.int8), pad]), axis=0)
        top = np.full(cols, np.nan)
        for c in range(cols):
            st = np.flatnonzero(edges[:, c] == 1)
            en = np.flatnonzero(edges[:, c] == -1)
            base = [k for k in range(len(st)) if en[k] > low and en[k] - st[k] >= 2]
            if not base:
                continue
            k = max(base, key=lambda i: en[i])
            t = st[k]
            for j in range(k - 1, -1, -1):
                if t - en[j] > max_gap:
                    break
                t = st[j]
            top[c] = t
        half = 3                                # median over neighbouring columns
        sm = np.full(cols, np.nan)
        for c in range(cols):
            win = top[max(0, c - half):c + half + 1]
            ok = win[~np.isnan(win)]
            if len(ok) * 2 >= len(win):
                sm[c] = np.median(ok)
        xs = (np.arange(cols) + 0.5) * pw / cols
        valid = ~np.isnan(sm)
        if not valid.any():
            return np.full(pw, np.nan)
        out = np.interp(np.arange(pw), xs[valid], sm[valid] * ph / rows)
        near = np.interp(np.arange(pw), xs, valid.astype(float))
        out[near < 0.5] = np.nan
        return out

    def ignore_line(self, white):
        """Per processing column: blobs entirely above this row are not cards."""
        np = self.np
        ph, pw = white.shape
        f = (pw / 2.0) / math.tan(math.radians(C.CAMERA_HFOV) / 2.0)
        horizon = ph / 2.0 - f * math.tan(math.radians(C.MAX_ELEVATION_DEG - self.pitch))
        if not C.IGNORE_ABOVE_WALL:
            return np.full(pw, -1.0)
        top = self.wall_top(white)
        return np.where(np.isnan(top), horizon, top - C.WALL_MARGIN * ph)

    # ------------------------------------------------------------ detect
    def balance(self, img):
        """Grey-world on the walls: bright unsaturated pixels are the white foam;
        scale B, G, R so they come out neutral (Final_Robomaster)."""
        cv2, np = self.cv2, self.np
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ref = (hsv[..., 1] < 45) & (hsv[..., 2] > 140)
        if np.count_nonzero(ref) < 0.05 * ref.size:
            return img
        means = img[ref].reshape(-1, 3).mean(axis=0)
        gains = np.clip(means.mean() / np.maximum(means, 1.0), 0.75, 1.33)
        if np.all(np.abs(gains - 1.0) < 0.03):
            return img
        return np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8)

    def detect(self, frame):
        cv2, np = self.cv2, self.np
        fh, fw = frame.shape[:2]
        pw = min(C.PROC_WIDTH, fw)
        ph = int(round(fh * pw / float(fw)))
        s = fw / float(pw)
        small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA) if pw < fw else frame
        if C.WHITE_BALANCE:
            small = self.balance(small)
        blurred = cv2.GaussianBlur(small, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        white = self.white_mask(hsv)
        line = self.ignore_line(white)
        neutral = (hsv[..., 1] < C.NEUTRAL_SAT).astype(np.uint8)
        nint = cv2.integral(neutral)
        y_bot = int(ph * (1.0 - C.BOTTOM_IGNORE))
        bx0, by0, bx1, by1 = self.barrel
        barrel = (int(bx0 * pw), int(by0 * ph), int(math.ceil(bx1 * pw)), int(math.ceil(by1 * ph)))
        f = (fw / 2.0) / math.tan(math.radians(C.CAMERA_HFOV) / 2.0)
        min_a = C.MIN_AREA_PX * (pw / 640.0) ** 2
        max_a = C.MAX_AREA_FRAC * pw * ph
        out, rejected, masks = [], [], {}
        for color, rs in self.ranges.items():
            mask = cv2.inRange(hsv, rs[0][0], rs[0][1])
            for lo, hi in rs[1:]:
                mask |= cv2.inRange(hsv, lo, hi)
            mask[y_bot:, :] = 0                                    # the blaster barrel
            mask[barrel[1]:barrel[3], barrel[0]:barrel[2]] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            masks[color] = mask
            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            for cnt in cnts:
                a = cv2.contourArea(cnt)
                if a < min_a or a > max_a:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                shape, info = classify_contour(cnt)
                label = kind_label(color, shape) if shape != "unknown" else None
                why = self._reject(cnt, a, x, y, w, h, pw, ph, y_bot, line, mask, nint, shape, info)
                if why is None and label:            # card height above the floor
                    _, plate_h = plate_size(label)
                    dist = f * plate_h / (h * s)
                    elev = -math.degrees(math.atan(((y + h / 2.0) * s - fh / 2.0) / f))
                    height = C.CAMERA_HEIGHT + dist * math.tan(math.radians(elev + self.pitch))
                    if not (C.CARD_MIN_H <= height <= C.CARD_MAX_H):
                        why = "not at card height (%.2f m)" % height
                if why or not label:
                    rejected.append((why or info.get("why", "shape unclear"), color, x, y, w, h))
                    continue
                out.append(Detection(label, color, shape, (x + w / 2.0) * s, (y + h / 2.0) * s,
                                     w * s, h * s, a * s * s, fw, fh, False))
        out.sort(key=lambda d: -d.area)
        self.last = dict(size=(pw, ph), scale=s, line=line, white=white, barrel=barrel,
                         masks=masks, rejected=rejected, y_bot=y_bot)
        return out

    def _reject(self, cnt, a, x, y, w, h, pw, ph, y_bot, line, mask, nint, shape, info):
        """Why a blob is not a card (None = it is one)."""
        cv2, np = self.cv2, self.np
        if x <= 2 or y <= 2 or x + w >= pw - 2 or y + h >= y_bot - 2:
            return "cut by the picture edge"
        if y <= line[x:x + w].max() + 2:
            return "above / at the wall top (the room)"
        hull = cv2.contourArea(cv2.convexHull(cnt))
        if hull <= 0 or a / hull < C.MIN_SOLIDITY:
            return "not solid"
        mx, my = max(3, int(w * 0.35)), max(3, int(h * 0.35))
        x0, y0, x1, y1 = max(0, x - mx), max(0, y - my), min(pw, x + w + mx), min(ph, y + h + my)
        ring_area = (x1 - x0) * (y1 - y0) - w * h
        if ring_area <= 0:
            return None
        ring = mask[y0:y1, x0:x1]
        same = int(np.count_nonzero(ring)) - int(np.count_nonzero(mask[y:y + h, x:x + w]))
        if same / float(ring_area) > C.MAX_RING_FILL:
            return "part of a bigger patch (wall / tape)"
        tot = int(nint[y1, x1] - nint[y0, x1] - nint[y1, x0] + nint[y0, x0])
        inner = int(nint[y + h, x + w] - nint[y, x + w] - nint[y + h, x] + nint[y, x])
        if (tot - inner) / float(ring_area) < C.MIN_NEUTRAL:
            return "not on a white wall"
        return None

    # ------------------------------------------------------------ drawing
    def draw_ignored(self, frame):
        """Dim what the detector ignores, draw the wall line and barrel box."""
        cv2, np = self.cv2, self.np
        if not self.last:
            return frame
        L = self.last
        fh, fw = frame.shape[:2]                 # any size: scale from the processing grid
        pw, ph = L["size"]
        sx, sy = fw / float(pw), fh / float(ph)
        ys = np.interp(np.arange(fw) / sx, np.arange(pw), L["line"]) * sy
        ys = np.clip(ys, -1, fh - 1).astype(int)
        above = np.arange(fh)[:, None] <= ys[None, :]
        frame[above] = (frame[above] * 0.35 + np.array((70, 45, 30)) * 0.65).astype(np.uint8)
        pts = np.stack([np.arange(fw), ys], axis=1)[ys >= 0].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(frame, [pts], False, (255, 200, 0), 2, cv2.LINE_AA)
        bx0, by0, bx1, by1 = L["barrel"]
        x0, y0, x1, y1 = int(bx0 * sx), int(by0 * sy), int(bx1 * sx), int(by1 * sy)
        sub = frame[y0:y1, x0:x1]
        sub[:] = (sub * 0.35 + np.array((40, 40, 160)) * 0.65).astype(np.uint8)
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (60, 60, 255), 2)
        cv2.putText(frame, "barrel", (x0 + 4, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (60, 60, 255), 1)
        for why, _, x, y, w, h in L["rejected"]:
            p1, p2 = (int(x * sx), int(y * sy)), (int((x + w) * sx), int((y + h) * sy))
            cv2.rectangle(frame, p1, p2, (128, 128, 128), 1)
            cv2.putText(frame, why, (p1[0], min(fh - 4, p2[1] + 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (170, 170, 170), 1)
        return frame

    def mask_view(self, frame):
        """Segmentation picture: wall = light grey, ignored = dark blue, barrel =
        red box, card colours as detected, everything else black."""
        cv2, np = self.cv2, self.np
        if not self.last:
            return frame
        L = self.last
        pw, ph = L["size"]
        img = np.zeros((ph, pw, 3), np.uint8)
        img[L["white"]] = (200, 200, 200)
        paint = {"red": (40, 40, 230), "blue": (230, 110, 40), "green": (60, 190, 60),
                 "yellow": (40, 210, 240)}
        for color, m in L["masks"].items():
            img[m > 0] = paint.get(color, (255, 255, 255))
        ys = np.clip(L["line"], -1, ph - 1).astype(int)
        above = np.arange(ph)[:, None] <= ys[None, :]
        img[above] = (img[above] * 0.3 + np.array((90, 40, 20)) * 0.7).astype(np.uint8)
        x0, y0, x1, y1 = L["barrel"]
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (60, 60, 255), 1)
        return cv2.resize(img, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    def calibrate_barrel(self, frame, pad=0.04):
        """Find the red barrel tip in the lower part of the picture (nothing red
        in front of the robot!) and save its box to barrel_box.json."""
        cv2, np = self.cv2, self.np
        fh, fw = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rs = self.ranges["red"]
        m = cv2.inRange(hsv, rs[0][0], rs[0][1])
        for lo, hi in rs[1:]:
            m |= cv2.inRange(hsv, lo, hi)
        m[: int(fh * 0.55)] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
        cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if not cnts:
            return None
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        box = (max(0.0, x / fw - pad), max(0.0, y / fh - pad),
               min(1.0, (x + w) / fw + pad), 1.0)
        box = tuple(round(v, 3) for v in box)
        with open(BARREL_FILE, "w") as f:
            json.dump({"box": box}, f)
        self.barrel = box
        return box

    def draw(self, frame, dets):
        cv2 = self.cv2
        for d in dets:
            p1 = (int(d.cx - d.w / 2), int(d.cy - d.h / 2))
            p2 = (int(d.cx + d.w / 2), int(d.cy + d.h / 2))
            cv2.rectangle(frame, p1, p2, (255, 255, 255), 2)
            dist = distance(d)
            txt = "%s %s" % (d.label or "?%s/%s" % (d.color, d.shape),
                             "%.2fm" % dist if dist else "")
            cv2.putText(frame, txt, (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame
