"""Map output: JSON (for round 2), ASCII (terminal), SVG (always), PNG (if cv2)."""
import json
import os

from maze import DV, WALL, OPEN, Maze
from vision import label_parts

COLORS = {"blue": "#2f5d9a", "red": "#e53525", "yellow": "#e8d400", "green": "#43a047"}


def save(data, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, name)
    with open(base + ".json", "w") as f:
        json.dump(data, f, indent=1)
    maze = Maze.from_dict(data["maze"])
    txt = ascii_map(data, maze)
    print(txt)
    with open(base + ".txt", "w") as f:
        f.write(txt + "\n")
    with open(base + ".svg", "w") as f:
        f.write(svg_map(data, maze))
    try:
        png_map(data, maze, base + ".png")
    except ImportError:
        pass
    return base


def load(path):
    with open(path) as f:
        return json.load(f)


def _tmark(data):
    """cell -> up to 3 letters, one per target in that cell."""
    marks = {}
    for t in data["targets"]:
        ch = t["label"][0].upper() if t.get("shot") else t["label"][0].lower()
        marks[tuple(t["cell"])] = (marks.get(tuple(t["cell"]), "") + ch)[:3]
    return marks


def tpos(t, k=0.32):
    """Grid position of a target: against its wall, not the cell centre."""
    dx, dy = DV.get(t.get("side") or "", (0, 0))
    return t["cell"][0] + dx * k, t["cell"][1] + dy * k


def ascii_map(data, m):
    """Top row printed first.  Upper case letter = target shot, * = path."""
    marks, path = _tmark(data), {tuple(c) for c in data["path"]}
    lines = []
    for r in range(m.h - 1, -1, -1):
        top = "+"
        mid = ""
        for c in range(m.w):
            e = m.get((c, r), "N")
            top += ("---" if e == WALL else "   " if e == OPEN else " . ") + "+"
            w = m.get((c, r), "W")
            ch = marks.get((c, r)) or ("S" if [c, r] == data["start"] else
                                        "*" if (c, r) in path else " ")
            mid += ("|" if w == WALL else " " if w == OPEN else ":") + ch.center(3)
        mid += "|" if m.get((m.w - 1, r), "E") == WALL else ":"
        lines += [top, mid]
    lines.append("+" + "---+" * m.w)
    lines.append("round %s  %.1fs   b/r/y/g = target (UPPER = shot)   S = start   * = path"
                 % (data["round"], data["elapsed"]))
    return "\n".join(lines)


def _geom(m, px=90, pad=30):
    W, H = m.w * px + 2 * pad, m.h * px + 2 * pad + 40

    def xy(x, y):                    # grid units (cell centre = int) -> pixels
        return pad + (x + 0.5) * px, pad + (m.h - 0.5 - y) * px
    return W, H, px, xy


def svg_map(data, m):
    W, H, px, xy = _geom(m)
    tile = data["tile"]
    o = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
         'font-family="sans-serif"><rect width="100%%" height="100%%" fill="#fff"/>' % (W, H)]
    for c, r in m.visited:
        x, y = xy(c - 0.5, r + 0.5)
        o.append('<rect x="%.1f" y="%.1f" width="%d" height="%d" fill="#f2f5f8"/>' % (x, y, px, px))
    for c in range(m.w + 1):                    # light grid
        x0, y0 = xy(c - 0.5, -0.5)
        x1, y1 = xy(c - 0.5, m.h - 0.5)
        o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#ddd"/>' % (x0, y0, x1, y1))
    for r in range(m.h + 1):
        x0, y0 = xy(-0.5, r - 0.5)
        x1, y1 = xy(m.w - 0.5, r - 0.5)
        o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#ddd"/>' % (x0, y0, x1, y1))
    for (a, b, p0, p1) in _walls(m):
        x0, y0 = xy(*p0)
        x1, y1 = xy(*p1)
        o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#222" '
                 'stroke-width="5" stroke-linecap="round"/>' % (x0, y0, x1, y1))
    if data.get("trace"):
        pts = " ".join("%.1f,%.1f" % xy(x / tile, y / tile) for x, y in data["trace"])
        o.append('<polyline points="%s" fill="none" stroke="#8ab4f8" stroke-width="2"/>' % pts)
    pts = " ".join("%.1f,%.1f" % xy(*c) for c in data["path"])
    o.append('<polyline points="%s" fill="none" stroke="#1a73e8" stroke-width="3" '
             'stroke-dasharray="6 4"/>' % pts)
    sx, sy = xy(*data["start"])
    o.append('<circle cx="%.1f" cy="%.1f" r="9" fill="#1a73e8"/>'
             '<text x="%.1f" y="%.1f" font-size="11" fill="#1a73e8">START</text>'
             % (sx, sy, sx - 16, sy + 24))
    ex, ey = xy(*data["path"][-1])
    o.append('<rect x="%.1f" y="%.1f" width="14" height="14" fill="#555"/>' % (ex - 7, ey - 7))
    for t in data["targets"]:
        cx, cy = xy(*tpos(t))
        col = COLORS.get(t["label"].split("_")[0], "#888")
        sh = label_parts(t["label"])[1]
        w, h = {"circle": (18, 18), "wide": (24, 14), "tall": (14, 24)}.get(sh, (18, 18))
        if sh == "circle":
            o.append('<circle cx="%.1f" cy="%.1f" r="9" fill="%s"/>' % (cx, cy, col))
        else:
            o.append('<rect x="%.1f" y="%.1f" width="%d" height="%d" fill="%s"/>'
                     % (cx - w / 2, cy - h / 2, w, h, col))
        if t.get("shot"):
            o.append('<circle cx="%.1f" cy="%.1f" r="15" fill="none" stroke="#1e9e4a" '
                     'stroke-width="3"/>' % (cx, cy))
        tag = "HIT" if t.get("shot") else ("target" if t.get("designated") else "ignore")
        o.append('<title>%s %s %s wall</title>' % (t["label"], tuple(t["cell"]), t.get("side")))
        ty = cy + (22 if (t.get("side") or "S") != "S" else -16)
        o.append('<text x="%.1f" y="%.1f" font-size="10" text-anchor="middle">%s</text>'
                 % (cx, ty, tag))
    o.append('<text x="30" y="%d" font-size="14">Round %s — %.1f s — %d/%d targets hit — '
             'blue dashed = cell path, light = odometry</text>'
             % (H - 14, data["round"], data["elapsed"],
                sum(1 for t in data["targets"] if t.get("shot")), len(data["targets"])))
    o.append("</svg>")
    return "\n".join(o)


def _walls(m):
    """Wall segments in grid units (cell centres at integers)."""
    out = []
    for c in range(m.w):
        for r in range(m.h):
            if m.get((c, r), "N") == WALL:
                out.append(((c, r), "N", (c - 0.5, r + 0.5), (c + 0.5, r + 0.5)))
            if m.get((c, r), "E") == WALL:
                out.append(((c, r), "E", (c + 0.5, r - 0.5), (c + 0.5, r + 0.5)))
            if r == 0 and m.get((c, r), "S") == WALL:
                out.append(((c, r), "S", (c - 0.5, r - 0.5), (c + 0.5, r - 0.5)))
            if c == 0 and m.get((c, r), "W") == WALL:
                out.append(((c, r), "W", (c - 0.5, r - 0.5), (c - 0.5, r + 0.5)))
    return out


def png_map(data, m, fname):
    import cv2
    import numpy as np
    W, H, px, xy = _geom(m)
    img = np.full((H, W, 3), 255, np.uint8)
    ip = lambda p: (int(p[0]), int(p[1]))
    for c, r in m.visited:
        cv2.rectangle(img, ip(xy(c - 0.5, r + 0.5)), ip(xy(c + 0.5, r - 0.5)), (248, 245, 242), -1)
    for (_, _, p0, p1) in _walls(m):
        cv2.line(img, ip(xy(*p0)), ip(xy(*p1)), (34, 34, 34), 5)
    tile = data["tile"]
    if data.get("trace"):
        pts = np.array([ip(xy(x / tile, y / tile)) for x, y in data["trace"]], np.int32)
        cv2.polylines(img, [pts], False, (248, 180, 138), 2)
    pts = np.array([ip(xy(*c)) for c in data["path"]], np.int32)
    cv2.polylines(img, [pts], False, (232, 115, 26), 3)
    cv2.circle(img, ip(xy(*data["start"])), 9, (232, 115, 26), -1)
    for t in data["targets"]:
        hexcol = COLORS.get(t["label"].split("_")[0], "#888888").lstrip("#")
        bgr = tuple(int(hexcol[i:i + 2], 16) for i in (4, 2, 0))
        c = ip(xy(*tpos(t)))
        cv2.circle(img, c, 9, bgr, -1)
        if t.get("shot"):
            cv2.circle(img, c, 14, (74, 158, 30), 2)
    cv2.putText(img, "Round %s  %.1fs" % (data["round"], data["elapsed"]), (30, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(fname, img)
