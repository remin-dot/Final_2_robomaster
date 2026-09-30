"""All tunables in one place.  Measure / calibrate the values marked  # CAL.

Coordinate conventions used everywhere
--------------------------------------
* The chassis NEVER rotates.  Its heading at power-on is "North" on the map.
  Motion in other directions is done by strafing (mecanum), which is much
  faster than turning and keeps every sensor pointing a fixed map direction.
* Cells are (col, row).  col grows East, row grows North.  (0, 0) = south-west.
* Metric pose (X east, Y north) in metres; cell (c, r) centre = (c*TILE, r*TILE).
"""

# ----------------------------------------------------------------- maze ----
GRID_W = 6
GRID_H = 6
TILE = 0.60                 # m, tile edge length                         # CAL
START_CELL = (0, 0)         # cell the robot starts in (facing North)     # CAL
ASSUME_BOUNDARY_WALLS = True

# -------------------------------------------------------------- targets ----
# Colour + shape identify a class.  w/h = size of the coloured plate (m).  # CAL
# Sizes measured on the lab cards (Final_Robomaster settings.yaml).
TARGET_CLASSES = {
    "blue_circle":  dict(color="blue",   shape="circle", w=0.07, h=0.07),
    "red_rect":     dict(color="red",    shape="wide",   w=0.09, h=0.06),
    "yellow_rect":  dict(color="yellow", shape="tall",   w=0.06, h=0.09),
    "green_square": dict(color="green",  shape="square", w=0.07, h=0.07),
}
# Every colour x shape card is recognised and mapped; one that is not in the
# table above is named "<color>_<shape>" and sized from SHAPE_SIZE.
COLORS = ("blue", "red", "yellow", "green")
SHAPES = ("circle", "wide", "tall", "square")
SHAPE_SIZE = {"circle": (0.07, 0.07), "wide": (0.09, 0.06),
              "tall": (0.06, 0.09), "square": (0.07, 0.07)}                # CAL
# Classes to shoot (the "designated" targets).  Override with --shoot.
SHOOT_CLASSES = ["red_rect", "yellow_rect", "blue_circle", "green_square"]
# Targets stand against a wall of a cell, facing into it.  One cell can hold
# several (one per wall).  A target is (cell, side): side = the wall it is on.
TARGET_WALL_GAP = 0.08      # m, wall surface -> target plate               # CAL
# Round 1 ends early once this many targets are mapped and every designated
# one is shot.  0 = always check every wall of the maze (safest).
EXPECTED_TARGETS = 0
FIRE_RANGE_TILES = 2        # rule: shoot from <= 2 tiles ...
FIRE_RANGE_M = FIRE_RANGE_TILES * TILE   # ... measured camera -> target plate
AVOID_TARGET_CELLS = False  # True: never drive into a cell holding a target
VIEW_TILES = 5              # how far down a corridor the camera looks
VIEW_RANGE_M = 1.0          # a wall face counts as checked when seen this close
                            # (7 cm cards: Final_Robomaster see_range_m)

# -------------------------------------------------------------- sensors ----
# Budget (rules): Sharp <= 2, obstacle <= 4, total <= 7.  On-robot ToF allowed.
# This robot: 1 ToF on the gimbal + 2 Sharp (left, right) + 2 IR (front corners) = 5.

# ToF on the gimbal: it measures wherever the camera looks.  A reading is used
# only while the gimbal points (within TOF_ALIGN_DEG) along N / E / S / W.
TOF_INDEX = 1               # EP ToF port (1-4)                              # CAL
TOF_OFFSET = 0.13           # m, gimbal yaw axis -> ToF face (reads 170 mm centred)  # CAL
GIMBAL_AXIS_N = 0.0         # m, gimbal yaw axis ahead (+) of chassis centre # CAL
TOF_ALIGN_DEG = 4.0
TOF_FRESH = 0.35            # s, a ToF reading is used this long
TOF_MAX = 2.5               # m, ignore readings beyond this

# Sharp GP2Y0A41 (4-30 cm) on the sides.  Ports + calibration from Final_Robomaster
# (settings.yaml, calibrated 2026-09-30 with src/sharp_calibrate.py).
# The chassis turns to face each move, so the Sharps are "R" (right) and "L" (left)
# of the chassis; the map direction they measure follows its heading.
SHARP = {"R": (2, 1), "L": (1, 2)}       # sensor-adaptor (id, port)          # CAL
SHARP_OFFSET = {"R": 0.16, "L": 0.16}    # m, robot centre -> sensor face (reads 14 cm centred)  # CAL
SHARP_MIN, SHARP_MAX = 0.04, 0.30        # m, trusted range (GP2Y0A41)
# per side: cm = A * raw ** B, and log-log interpolation between (cm, raw) points
SHARP_CAL = {
    "L": dict(A=4781.791017, B=-1.0469, points=[(5.0, 690), (10.0, 370), (15.0, 254), (20.0, 181)]),
    "R": dict(A=2153519.156214, B=-1.9317, points=[(5.0, 728), (10.0, 663), (15.0, 477), (20.0, 389)]),
}
ADAPTOR_POLL = True         # read the sensor board by polling (what works on this robot)

# IR obstacle modules: adaptor (id, port), the map directions each one guards,
# and (for the simulator) mount position (east, north m from centre) + angle.
IR = {
    "FL": dict(port=(1, 1), guards="F", pos=(-0.11, 0.15), ang=-45.0),  # front-left 45 deg  # CAL
    "FR": dict(port=(2, 2), guards="F", pos=(0.11, 0.15), ang=45.0),    # front-right 45 deg # CAL
}
IR_ACTIVE_LEVEL = 0         # io value when an obstacle is present (active low)  # CAL
IR_TRUST_MAX = 0.35         # m: IR "obstacle" is overruled when a range reading
                            # in that direction is farther than this
# The corner modules guard driving forward (the robot always drives forward).
# At 45 deg they are collision guards, never wall sensors.  Screws set to 8 cm.

SENSOR_HZ = 20

# --------------------------------------------------------------- motion ----
# Grid by grid (Dhai_8): the chassis turns in place to face the next cell, drives
# forward ONE cell holding its heading and centring on the side Sharps, stops on
# the cell centre (front ToF / odometry) and settles before the next step.
GRID_STEP = True            # stop on every cell (False: straight runs in one go)
TURN_SPEED = 90.0           # deg/s top turn speed
TURN_KP = 3.0               # deg/s per deg of heading error
TURN_TOL = 1.5              # deg: turned
TURN_TIMEOUT = 4.0          # s
CELL_PAUSE = 0.15           # s standing still on each cell
CTRL_HZ = 30
V_MAX = 0.40                # m/s  forward (Dhai_8 base speed); raise once it behaves
A_MAX = 0.8                 # m/s^2
V_MIN = 0.06                # m/s  creep speed near goal
POS_TOL = 0.015             # m    arrival tolerance
MOTION_LAG = 0.08           # s    command -> wheel delay; braking starts this early  # CAL
LAT_KP = 3.0                # 1/s  lateral centring gain (pose fallback, no side walls)
LAT_VMAX = 0.30             # m/s
# Centring on the walls (Dhai_8 wall PID): distances from the robot centre
SIDE_NOMINAL = 0.30         # m, wall distance when centred in a 0.60 m cell
SIDE_WALL_MAX = 0.42        # m, a side reading closer than this = a wall to centre on
WALL_KP = 1.5               # 1/s  sideways speed per metre of centring error
WALL_VMAX = 0.10            # m/s
WALL_DEADBAND = 0.015       # m
TOO_CLOSE = 0.20            # m from the centre (~8 cm from the chassis side): push away
IR_STEER = 0.08             # m/s sideways away from a front-corner IR that is on
IR_SLOW = 0.5               # forward speed share while a corner IR is on
CENTER_TIME = 1.2           # s, centring in place after each cell scan (stops once centred)
SHARP_BIAS_MAX = 0.06       # m: start-of-round Sharp bias check accepts at most this
                            # (robot placed on the CENTRE of the start tile)
CENTER_KP = 1.5             # 1/s
CENTER_VMAX = 0.08          # m/s
CENTER_TOL = 0.02           # m
YAW_KP = 4.0                # deg/s per deg heading error
YAW_WMAX = 90.0             # deg/s
SETTLE_TIME = 0.12          # s    still-time for fresh sensor samples
SEG_TIMEOUT = 4.0           # s    per tile before a segment is aborted

# wall snapping (localisation): a range reading that lands within SNAP_TOL of
# a tile boundary is taken as a wall there and pulls the pose onto the grid.
SNAP_TOL = 0.04             # well below TARGET_WALL_GAP: a plate is not a wall
SNAP_TOL_CHECKED = 0.10     # on wall faces the camera checked (plate or not is known)
SNAP_TOL_FAR = 0.10         # ToF reading LONGER than expected: a plate cannot cause that
                            # (not for Sharps: closer than 4 cm they fold back and read LONG)
SNAP_GAIN = 0.35
SNAP_STEP_MAX = 0.01        # m: one reading moves the pose at most this much (a few
                            # bad readings cannot throw it 9 cm - last run's wall hit)
SNAP_MAX_SPEED = 0.35       # m/s, no along-track snapping faster than this
SNAP_MAX_RANGE = 2.5        # m, readings farther than this never move the pose
WALL_TOL = 0.15             # m, reading <= TILE/2 + WALL_TOL  ->  wall

# planning cost (seconds-ish): per tile, per extra straight segment
COST_TILE = 1.0
COST_SEGMENT = 1.0
COST_CLOSE_SPOT = 1.5       # round 2: extra cost to shoot from inside the target's cell
ROUTE_EXACT_MAX = 12        # round 2: exact best order up to this many targets (else nearest-first)

# --------------------------------------------------------- gimbal / gun ----
# Gimbal (Dhai_8): relative move actions of at most GIMBAL_STEP_DEG, planned from
# the commanded position, never trimmed from the angle feed (no chasing = no shaking)
GIMBAL_YAW_SPEED = 180      # deg/s
GIMBAL_PITCH_SPEED = 60
GIMBAL_STEP_DEG = 90
GIMBAL_SETTLE = 0.25        # s standing still after a move (sharp frames, fresh ToF)
GIMBAL_TIMEOUT = 3.0        # s per move action
GIMBAL_TRIM_MIN = 1.5       # deg: off by more than this after a move -> one correction move
GIMBAL_PITCH = 0.0          # deg, default look pitch                      # CAL
AIM_YAW_OFFSET = 0.0        # deg, barrel vs camera                         # CAL
AIM_PITCH_OFFSET = 2.0      # deg, + = aim higher (hit at ~+2 deg on this robot)  # CAL
BARREL_BELOW_CAMERA_M = 0.03  # barrel under the camera: aim up atan(this / distance)
AIM_TOL = 1.5               # deg
MIN_SHOOT_M = 0.30          # closer than this: back off first if there is room (0.21 m missed)
AIM_ITERS = 4
FIRE_TYPE = "water"         # "water" (gel beads) or "ir"
FIRE_TIMES = 2
FIRE_GAP = 0.15             # s between shots

# --------------------------------------------------------------- camera ----
CONN_TYPE = "ap"            # "ap" (robot Wi-Fi) or "sta" (router) or "rndis"
ROBOT_IP = None             # "sta" only: robot IP, skips discovery (often blocked on campus Wi-Fi)
CONNECT_TIMEOUT = 20.0      # s, give up connecting after this (the SDK itself never does)
ROBOT_AP_IP = "192.168.2.1" # the robot's address in Wi-Fi direct (AP) mode
CAMERA_HFOV = 96.0          # deg horizontal FOV of the video stream        # CAL
CAMERA_RES = "540p"         # 7 cm cards at 1.2 m are too small at 360p
VIDEO_LATENCY = 0.10        # s, wait this long after the gimbal stops (measured 0.06)
PROC_WIDTH = 640            # frames are processed at this width (Final_Robomaster)
# Colour ranges tuned on the arena (Final_Robomaster config/color_config.json).
# OpenCV HSV (H 0-179).  Tune with  python main.py --vision-test            # CAL
HSV = {
    "red":    [((0, 120, 36), (9, 255, 255)), ((173, 120, 36), (179, 255, 255))],
    "yellow": [((13, 120, 36), (29, 255, 255))],
    "green":  [((66, 120, 30), (82, 255, 255))],
    "blue":   [((111, 120, 30), (127, 255, 255))],
}
WHITE_BALANCE = True        # scale B, G, R so the white walls come out neutral
MIN_AREA_PX = 80            # smallest blob (px at PROC_WIDTH; 7 cm card at 1.2 m ~ 250)
MAX_AREA_FRAC = 0.25        # largest blob, share of the picture
# --- segmentation: what counts as a card (Final_Robomaster target_vision) - # CAL
MIN_SOLIDITY = 0.85         # ragged blobs are not cards
MAX_RING_FILL = 0.25        # same colour all around the blob = a wall / tape / floor patch
NEUTRAL_SAT = 70            # saturation below this = white wall / grey floor
MIN_NEUTRAL = 0.45          # share of neutral pixels around a card (clothes, robot: less)
CAMERA_HEIGHT = 0.25        # m, lens above the floor
CARD_MIN_H, CARD_MAX_H = 0.06, 0.45   # m, height of a card's centre above the floor
BOTTOM_IGNORE = 0.15        # bottom share of the picture: the blaster barrel
IGNORE_ABOVE_WALL = True    # the room above the white walls is never a card
WALL_S_MAX = 60             # white foam wall: saturation at most this ...
WALL_V_MIN = 120            # ... and brightness at least this (HSV 0-255; tuned on the arena)
WALL_MAX_GAP = 0.25         # non-white gap (share of picture height) bridged in a wall
WALL_MARGIN = 0.03          # keep this much of the picture height above the wall line
MAX_ELEVATION_DEG = 2.0     # no wall in a column: nothing higher than this above level
                            # (cards on 15-20 cm sticks hang below the 0.25 m camera)
BARREL_BOX = (0.30, 0.85, 0.70, 1.0)   # barrel tip, frame fractions x0 y0 x1 y1
                                       # (python main.py --vision-test, key B, measures it)
CONFIRM_FRAMES = 2          # a card must show in this many frames in a row to count

# --------------------------------------------------------------- timing ----
ROUND_LIMIT = {1: 600.0, 2: 300.0}
TIME_MARGIN = 10.0          # s, stop exploring this long before the limit
OUT_DIR = "out"
PANEL_FPS = 60              # control panel redraw rate (the robot camera itself sends <= 30 fps)


def check_sensor_budget():
    n_sharp, n_ir, n_tof = len(SHARP), len(IR), 1
    assert n_sharp <= 2, "rule 1: max 2 Sharp sensors"
    assert n_ir <= 4, "rule 3: max 4 obstacle sensors"
    assert n_sharp + n_ir + n_tof <= 7, "rule 4: max 7 sensors in total"
