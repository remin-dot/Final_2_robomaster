"""Grid map with tri-state walls and a turn-aware shortest-path planner."""
import heapq

DIRS = "NESW"
DV = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
UNKNOWN, OPEN, WALL = 0, 1, 2


def step(c, d, n=1):
    dx, dy = DV[d]
    return (c[0] + dx * n, c[1] + dy * n)


def direction(a, b):
    """Direction from cell a to cell b (same row or column)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0:
        return "N" if dy > 0 else "S"
    return "E" if dx > 0 else "W"


def compress(path):
    """Cell path -> [(dir, n_tiles), ...] straight segments."""
    segs = []
    for a, b in zip(path, path[1:]):
        d = direction(a, b)
        if segs and segs[-1][0] == d:
            segs[-1] = (d, segs[-1][1] + 1)
        else:
            segs.append((d, 1))
    return segs


class Maze:
    def __init__(self, w, h, boundary=True):
        self.w, self.h = w, h
        self.boundary = boundary
        self.edges = {}         # (col, row, "N"|"E") -> OPEN / WALL
        self.visited = set()
        self.observed = set()   # cells the camera has looked into
        self.faces = set()      # (cell, dir) wall faces the camera has checked for targets
        self.driven = set()     # edge keys the robot has driven through: never a wall

    # ---------------------------------------------------------- edges ----
    def inside(self, c):
        return 0 <= c[0] < self.w and 0 <= c[1] < self.h

    @staticmethod
    def _key(c, d):
        if d in "SW":
            c, d = step(c, d), OPP[d]
        return (c[0], c[1], d)

    def get(self, c, d):
        if self.boundary and not self.inside(step(c, d)):
            return WALL
        return self.edges.get(self._key(c, d), UNKNOWN)

    def set(self, c, d, v, force=True):
        if self.boundary and not self.inside(step(c, d)):
            return
        k = self._key(c, d)
        if v == WALL and k in self.driven:
            return                      # driven through it: a reading off-centre, not a wall
        if force or k not in self.edges:
            self.edges[k] = v

    def drove(self, c, d):
        """The robot drove from c towards d: that edge is open, for good."""
        if self.inside(c) and self.inside(step(c, d)):
            k = self._key(c, d)
            self.driven.add(k)
            self.edges[k] = OPEN

    def ray(self, c, d, maxlen, peek=True):
        """Cells seen looking from c towards d: through OPEN edges, plus (peek)
        one cell past an UNKNOWN edge (camera sees it even if unmapped)."""
        out = []
        while len(out) < maxlen:
            e = self.get(c, d)
            if e == WALL or (e == UNKNOWN and not peek):
                break
            c = step(c, d)
            if self.boundary and not self.inside(c):
                break
            out.append(c)
            if e == UNKNOWN:
                break
        return out

    def clear_line(self, a, b):
        """True if a and b share a row/col and every edge between is OPEN."""
        if a[0] != b[0] and a[1] != b[1]:
            return False
        d = direction(a, b)
        c = a
        while c != b:
            if self.get(c, d) != OPEN:
                return False
            c = step(c, d)
        return True

    # -------------------------------------------------------- planning ----
    def _search(self, start, cost_tile, cost_seg, start_dir=None, blocked=()):
        """Dijkstra over (cell, heading) with a penalty for each new straight
        segment (every direction change costs a stop + accelerate)."""
        dist = {(start, start_dir): 0.0}
        self._prev = prev = {}
        pq = [(0.0, 0, start, start_dir)]
        seq = 0
        while pq:
            g, _, c, h = heapq.heappop(pq)
            if g > dist.get((c, h), 1e18):
                continue
            yield c, g, (c, h)
            for d in DIRS:
                if self.get(c, d) != OPEN:
                    continue
                n = step(c, d)
                if n in blocked:
                    continue
                ng = g + cost_tile + (0.0 if d == h else cost_seg)
                if ng < dist.get((n, d), 1e18):
                    dist[(n, d)] = ng
                    prev[(n, d)] = (c, h)
                    seq += 1
                    heapq.heappush(pq, (ng, seq, n, d))

    def _unwind(self, state):
        path = [state[0]]
        while state in self._prev:
            state = self._prev[state]
            path.append(state[0])
        return path[::-1]

    def plan(self, start, goals, cost_tile=1.0, cost_seg=1.0, start_dir=None, blocked=()):
        """Cheapest path (list of cells) from start to the nearest goal."""
        goals = set(goals)
        for c, g, state in self._search(start, cost_tile, cost_seg, start_dir, blocked):
            if c in goals:
                return self._unwind(state), g
        return None, None

    def costs_from(self, start, cost_tile=1.0, cost_seg=1.0, blocked=(), start_dir=None):
        out = {}
        for c, g, _ in self._search(start, cost_tile, cost_seg, start_dir, blocked):
            out.setdefault(c, g)
        return out

    # ------------------------------------------------------------- I/O ----
    def to_dict(self):
        return {
            "w": self.w, "h": self.h, "boundary": self.boundary,
            "edges": [[k[0], k[1], k[2], v] for k, v in sorted(self.edges.items())],
            "visited": sorted(self.visited),
            "observed": sorted(self.observed),
            "faces": sorted([list(c), d] for c, d in self.faces),
            "driven": sorted(list(k) for k in self.driven),
        }

    @classmethod
    def from_dict(cls, d):
        m = cls(d["w"], d["h"], d.get("boundary", True))
        m.edges = {(e[0], e[1], e[2]): e[3] for e in d["edges"]}
        m.visited = {tuple(c) for c in d.get("visited", [])}
        m.observed = {tuple(c) for c in d.get("observed", [])}
        m.faces = {(tuple(c), s) for c, s in d.get("faces", [])}
        m.driven = {tuple(k) for k in d.get("driven", [])}
        return m
