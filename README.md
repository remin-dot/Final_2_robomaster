# Final_2_robomaster — maze hostage rescue (Assignment 2)

This code runs a RoboMaster EP through a 6×6-tile maze. The robot finds the
acrylic targets (blue circle, red wide rectangle, yellow tall rectangle and
green square) and shoots the designated ones from ≤ 2 tiles. It draws the
path and the target positions after each round.

| Round | What the robot does | Limit | Typical in the simulator |
|---|---|---|---|
| 1 | Explores, maps walls, finds and shoots targets | 10 min | 200–270 s |
| 2 | Loads the round-1 map, drives an optimal route and shoots | 5 min | 40–110 s |

## How it moves: grid by grid

This follows robomaster-assignment2-4x4_Dhai_8.

* **Turn to face the next cell.** The chassis turns in place, closed-loop on
  the IMU yaw. If the error grows, the turn-command sign is flipped once. The
  robot always drives forward, so the gimbal ToF and the two front-corner IR
  modules guard the way ahead, and the side Sharps keep it centred.
* **Drive exactly one cell and stop on its centre.** Heading is held, and the
  Sharps pull the robot back to the middle between the walls. The ToF ahead
  and the wall-snap localisation place the stop. The robot stands still
  briefly before the next step. Set `GRID_STEP = False` in `config.py` to
  drive straight corridors in one run instead.
* **Gimbal: relative move actions of at most 90°.** Each move is planned from
  the commanded position and never corrected from the angle feed, then the
  gimbal settles for 0.25 s. This avoids the "snake" shaking that a speed loop
  chasing a delayed angle feed caused. While turning and driving, the robot is
  in chassis-lead mode with the gimbal forward; scans run in free mode.
* **Turn-aware planner.** Dijkstra runs over (cell, heading) states and charges
  a penalty for each change of direction.
* **Round 2 plans the whole route.** It tries every target order and every
  legal firing cell, then drives the cheapest route.

## What counts as a card (segmentation)

This follows Final_Robomaster `src/target_vision.py`, with its tuned HSV ranges.

1. The frame is white-balanced on the white foam walls, blurred, and split into
   HSV colour masks.
2. Two regions are ignored:
   * **The room above the walls.** The top of the white wall band is traced in
     each column, and card-sized gaps are bridged. Blobs at or above that line
     are dropped.
   * **The blaster barrel.** It sits in the bottom 15 % of the picture, plus
     `BARREL_BOX`.
3. The shape comes from the contour: circle, wide rectangle, tall rectangle,
   square, or unknown. Long thin strips count as tape.
4. A blob counts as a **card** only if all of these hold:
   * it is whole, not cut by the picture edge
   * it is solid (≥ 0.85)
   * it is not part of a bigger patch of the same colour (≤ 0.25 of the ring
     around it). This catches yellow walls and tape.
   * it has neutral wall around it (≥ 0.45)
   * it sits at card height above the floor (0.06–0.45 m, from its distance and
     elevation with the camera 0.25 m up)
5. A card must also appear in **2 frames in a row**.
6. Distance is measured from the card's height only, because a card seen at a
   slant looks narrower.

## Sensors (rules: Sharp ≤ 2, obstacle ≤ 4, total ≤ 7)

This robot uses 5 sensors. The ports and calibration come from
Final_Robomaster's `config/settings.yaml`.

| Sensor | Port | Used for |
|---|---|---|
| EP ToF, on the gimbal | ToF #1 | Measures wherever the camera points. A reading counts only while chassis heading plus gimbal angle is lined up with N/E/S/W, using the gimbal's actual angle. It maps walls, including long corridors, and gives the distance ahead while driving. |
| Sharp GP2Y0A41 (4–30 cm), right side | board 2, port 1 | Wall on the chassis's right, centring, localisation |
| Sharp GP2Y0A41 (4–30 cm), left side | board 1, port 2 | Wall on the chassis's left, centring, localisation |
| IR obstacle, front-left at 45° | board 1, port 1 | Collision guard while driving forward |
| IR obstacle, front-right at 45° | board 2, port 2 | Collision guard while driving forward |

**Walls.** On arriving at a cell, the Sharps map the walls on the chassis's
left and right at once. The gimbal then turns only toward sides that are still unknown, or wall
faces not yet checked. At each stop it reads the ToF and looks with the camera.
A Sharp that reads nothing is ambiguous: the space is either open or closer
than 4 cm. The ToF decides those cases.

**IR modules.** They trigger at 7 cm and read 0 when they detect something.
Angled at 45°, they can't tell "ahead" from "beside", so they only guard forward
motion. A module counts only after it has read "clear" at least once, and it is
overruled when a range sensor sees open space that way.

**Shooting.** When a card is closer than 0.30 m, the robot first backs off, but
only toward a wall it has already checked and found free of plates. It aims
2° higher, plus a small extra angle because the barrel is 3 cm below the camera.

`config.check_sensor_budget()` checks the rules at start-up. After Connect, the
log prints one line per sensor: its reading, NO DATA, or "stuck". The panel's
Sensors card draws the ToF ray, the Sharp bars and the IR rays.

## Setup

On macOS (Apple Silicon or Intel), run the setup script once:

```bash
bash tools/macos/setup_robomaster_mac.sh
```

DJI ships no macOS build of the SDK, so the script does this instead:
* It installs the dependencies into `.venv`.
* It extracts the SDK's pure-Python code from DJI's wheel.
* It patches several DJI bugs.
* It installs `libmedia_codec.py`, a PyAV replacement for DJI's native video
  decoder.
* It runs a self-test.

The script was adapted from Final_Robomaster / RoboFinal.

On Windows or Linux x86_64 with Python 3.6–3.8, install the requirements
instead:

```bash
pip install -r requirements.txt
```

To connect to the robot:

1. Turn the robot on and set its connection switch to Wi-Fi direct (AP).
2. Join the robot's Wi-Fi (`RMEP-xxxxxx`). The password is on the sticker.
3. Run the connection check:

   ```bash
   .venv/bin/python tools/macos/check_robomaster.py
   ```

   It prints OK/FAIL for the Wi-Fi, the SDK connection, the version and
   battery, and the camera stream.

4. Start the panel:

   ```bash
   .venv/bin/python panel.py
   ```

   Pick **RoboMaster robot**, then **Wi-Fi direct**, then **Connect**.

## Before the run (calibration)

1. Measure `TILE`. Also set `START_CELL`: (0,0) is the south-west corner, and
   the robot must start facing North (map up).
2. Run `python main.py --selftest` and hold a board in front of each sensor.
   Check these:
   * Each direction letter reacts to the correct sensor. If not, fix `TOF`,
     `SHARP` or `IR_BUMP`.
   * With the robot at the centre of a tile, the reading to a wall is about
     `TILE/2`. If not, adjust `*_OFFSET`, or re-fit `SHARP_A/B`.
3. Run `python main.py --vision-test` and point the camera at the maze:
   * Tune `HSV` until every card shows the correct label.
   * Place a card at a measured distance and adjust `CAMERA_HFOV` until the
     displayed distance matches. Set the plate sizes in `TARGET_CLASSES`.
   * Press `M` to switch between overlay, mask and raw views. The segmentation
     ignores three kinds of blob:
     * **The room above the white walls.** It is dimmed, with a cyan line
       along the wall top. If walls are missed, lower `WALL_V_MIN`; if the
       floor counts as wall, raise it.
     * **Blobs with no white wall around them** ("not on wall", controlled by
       `WALL_RING_MIN`).
     * **The red barrel tip.** With nothing red in front of the robot, press
       `B` and the tip's box is measured and saved to `barrel_box.json`.
       Rejected blobs are outlined in grey with the reason.
   * The panel's camera card has the same Overlay / Mask / Raw switch.
4. Do a dry run of one short corridor with `--no-fire`. If the robot
   oscillates, lower `LAT_KP` or `YAW_KP`. Then raise `V_MAX` and `A_MAX` step
   by step, as far as it stays stable.
5. Test-fire at a target 1–2 tiles away. Adjust `AIM_PITCH_OFFSET` and
   `AIM_YAW_OFFSET` until it hits.

## Running

```bash
python main.py --round 1 --shoot red,yellow
```

This explores the maze, maps it and shoots the listed targets. It writes
`out/round1.json`, `.svg`, `.png` and `.txt`.

```bash
python main.py --round 2 --shoot red,yellow
```

Before round 2, put the robot back on the start tile, facing North. It reads
`out/round1.json` and writes the `out/round2.*` files.

Target names for `--shoot`: `red`, `blue`, `yellow`, `green`, the full class
names (`red_rect`), or shapes (`circle`, `square`, `wide`, `tall`). If you
leave it out, the robot shoots every class in `SHOOT_CLASSES`.

Ctrl-C stops the robot and still saves the map drawn so far.

## Control panel (pygame)

```bash
python panel.py
```

This opens the Connect screen. Pick the robot (Wi-Fi AP, router or USB) or the
simulator, then press Connect or Enter.

```bash
python panel.py --sim --connect
```

This goes straight into the simulator.

The panel follows the style of the Final_Robomaster mission panel, with light
and dark themes, but uses a cleaner layout:

| Area | Contents |
|---|---|
| Header | Status only: source, DRY RUN, round, state (IDLE / RUNNING / PAUSED / DONE / STOPPED) and the countdown with a progress line. It also has Disconnect (asks first) and Theme. |
| Map (largest view) | Walls, visited cells, orange cell path, blue odometry trace, the camera's field-of-view cone, the robot and its gimbal direction, and targets marked HIT, shoot or ignore. When idle, click a cell to move the start. |
| Targets | Every card found, with its cell and status: HIT with split time, to shoot, not selected, or from round 1. |
| Camera | The live stream. The simulator shows a ray-cast 3D view instead. Overlays show detection boxes with distance and IN RANGE or not selected, labels that never overlap, the boresight, a LIVE / fps badge and an AIMING / LOCKED / FIRE banner. |
| Sensors | A sensor cross (1 m bars with a wall tick), plus a table of ToF, Sharp and IR readings, cell, yaw, gimbal and odometry. |
| Aim | Error crosshair with the tolerance box, the phase, the target, and a 10 s yaw/pitch error trace. |
| Log | Timestamped and color-coded: TARGET, SHOT and miss. |
| Sidebar | **Mission:** Round 1 / Round 2, Start, Pause/Resume, Save map, STOP. **Shoot:** a 4 × 4 colour × shape grid with Sheet set / All / None. **Actions:** Blaster armed, Fire once, gimbal pad, Screenshot, Show sim truth. |

**Target selection:** every colour × shape card is recognised and mapped, but
only the kinds you select are shot. A card that is not selected is never fired
at, which avoids the −1 wrong-target penalty. The simulator adds one red circle
that is not in the sheet set, so you can test this.

| Key | Action | Key | Action |
|---|---|---|---|
| `Space` | Start round / STOP | `P` | Pause / resume |
| `1` / `2` | Pick round (idle) | `M` | Camera overlay / raw |
| WASD / arrows | Drive (W forward, A/D sideways) with heading hold, when idle | `J` `L` / `I` `K` | Gimbal yaw / pitch |
| `F` | Fire once | `C` | Centre gimbal |
| `S` | Screenshot to `out/` | `T` | Light / dark theme |
| `V` | Simulator truth | `Q` | Quit (asks if a round is running) |

Before each round, the panel treats the robot as standing on the start tile
facing North, and re-zeroes its heading.

## Offline simulator

`sim.py` provides the same interface as the real robot. It has a virtual
clock, random mazes, odometry drift, sensor noise and a pinhole camera model.
Use it to test any change to the logic without the robot:

```bash
python main.py --sim --round 1 --seed 7
python main.py --sim --round 2 --seed 7
```

Results over 30 random mazes:

* Every target that could legally be shot was hit.
* There were 0 wrong-target shots, 0 shots from more than 2 tiles and 0
  collisions.

## Files

| File | Contents |
|---|---|
| `config.py` | Every tunable, the sensor layout and the calibration values |
| `main.py` | Command-line interface: rounds, selftest, vision-test, sim |
| `panel.py` | pygame control panel |
| `mission.py` | Round 1 exploration, target bookkeeping, aiming and firing, round 2 route optimisation |
| `motion.py` | Grid-snapped localisation and the segment controller |
| `maze.py` | Tri-state wall map, rays, turn-aware Dijkstra, path compression |
| `vision.py` | HSV colour and shape detector, bearing and distance estimates |
| `hal.py` | RoboMaster EP hardware layer (subscriptions, gimbal, blaster, camera thread) |
| `mapdraw.py` | JSON, ASCII, SVG and PNG map output |
| `sim.py` | Offline simulator |

## Targets on walls

* **Placement:** a target stands against a wall of a cell and faces into the
  cell. One cell can hold several targets, one per wall. Each target is stored
  as its cell plus the wall it's on (`N`, `E`, `S` or `W`).
* **Exploration:** round 1 checks every wall face in the maze with the camera.
  A face counts as checked when the camera has looked straight at it from
  within `VIEW_RANGE_M`, and faces seen down a corridor count too. Round 1
  ends when every face is checked, or earlier once `EXPECTED_TARGETS` targets
  are mapped and every designated one is shot, if you set that number.
* **Shooting:** the robot always shoots facing the target's wall, from inside
  its cell or from a cell behind it. The camera-to-plate distance must be at
  most `FIRE_RANGE_M`, which is 2 × `TILE` = 1.2 m. Round 2 prefers the spot
  one cell back, which gives a clearer view and keeps the barrel off the plate.
* **Localisation:** a plate stands `TARGET_WALL_GAP` in front of its wall, so a
  range reading off a plate is never mistaken for the wall. Walls are only
  snapped to within `SNAP_TOL`, and known plates are modelled.
* **Look ahead:** before each move, the robot checks the wall it is driving
  toward. Any plate on it is then known before arrival.
* **IR modules:** set their pots so they trigger at about 10–11 cm from the
  chassis face. At that distance the Sharp sensors are below their minimum
  range, so the IR module is what reports a plate on the robot's own wall. If
  one fires, the robot steps back 6 cm, but only when there is room behind it.

Results in the simulator:

| Test | Round 1 | Round 2 |
|---|---|---|
| 30 mazes: 4 sheet cards + 1 extra sheet card + 1 red circle (not selected); two targets share a cell | 30/30 perfect, about 94 s | 30/30 perfect, about 26 s |
| 100 stress mazes: 10 targets each, 8 designated; odometry scale error up to ±4 % | all 800 designated targets hit, about 98 s | 800/800 hit, about 36 s |

Neither test had any wrong-target shots, any shots over range or any
collisions.

## Assumptions to check against the real course

* **Plate position:** measure `TARGET_WALL_GAP`, the distance from the wall to
  the plate, and each plate's size.
* **Blaster clearance:** check that the blaster barrel clears a plate when the
  robot is at the centre of the plate's own cell. If it doesn't, raise
  `COST_CLOSE_SPOT` so the robot shoots from one cell back whenever it can.
* **Sensor-adaptor layout:** the code takes the subscription layout as
  `index = (id-1)*2 + (port-1)`. Check it with `--selftest`. If the
  subscription fails, the code falls back to polling.
