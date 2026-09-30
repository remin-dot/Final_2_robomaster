"""RoboMaster EP maze hostage-rescue — entry point.

  python main.py --round 1 --shoot red,yellow          # explore, map, shoot
  python main.py --round 2 --shoot red,yellow          # replay optimal route
  python main.py --selftest                            # live sensor readout
  python main.py --vision-test                         # live detector window
  python main.py --sim --round 1 [--seed 3]            # offline simulation
"""
import argparse
import os
import sys
import time

import config as C
import mapdraw
from maze import Maze
from mission import Mission
from vision import kind_label


def parse_shoot(s):
    if not s:
        return list(C.SHOOT_CLASSES)
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        hit = [k for k in C.TARGET_CLASSES if k == tok or k.split("_")[0] == tok
               or C.TARGET_CLASSES[k]["shape"] == tok]
        if not hit and "_" in tok:                      # any colour x shape, e.g. red_circle
            color, _, shape = tok.partition("_")
            if color in C.COLORS and shape in C.SHAPES:
                hit = [kind_label(color, shape)]
        if not hit:
            sys.exit("unknown target '%s' (choose from %s, or <color>_<shape>)"
                     % (tok, ", ".join(C.TARGET_CLASSES)))
        out += hit
    return out


def make_hal(args):
    if args.sim:
        from sim import SimHAL
        return SimHAL(seed=args.seed)
    from hal import RoboMasterHAL
    return RoboMasterHAL(fire_enabled=not args.no_fire)


def run_round(args):
    hal = make_hal(args)
    shoot = parse_shoot(args.shoot)
    prefix = "sim_" if args.sim else ""
    m = None
    try:
        if args.round == 1:
            m = Mission(hal, 1, shoot)
            m.run_round1()
        else:
            path = args.map or os.path.join(C.OUT_DIR, prefix + "round1.json")
            data = mapdraw.load(path)
            m = Mission(hal, 2, shoot, Maze.from_dict(data["maze"]), data["targets"])
            m.run_round2()
    except KeyboardInterrupt:
        print("\n[main] interrupted - saving what we have")
    finally:
        hal.stop()
        if m is not None:
            m.finish()
            data = m.to_dict()
            base = mapdraw.save(data, C.OUT_DIR, "%sround%d" % (prefix, args.round))
            print("[main] map saved: %s.{json,svg,txt,png}" % base)
        if args.sim:
            print("[sim] truth:", hal.truth())
            print("[sim] shots:", hal.hits)
            wrong = [h for h in hal.hits if h["hit"] not in shoot]
            far = [h for h in hal.hits if h["dist_tiles"] and h["dist_tiles"] * C.TILE > C.FIRE_RANGE_M + 0.05]
            want = [t["id"] for t in hal.targets if t["label"] in shoot]
            got = {h["id"] for h in hal.hits if h["hit"] in shoot}
            print("[sim] correct hits: %d/%d   wrong/miss: %d   >2-tile shots: %d   "
                  "collisions: %s   time %.1fs" % (
                      len(got), len(want), len(wrong), len(far),
                      sorted(hal.collisions) or "none", hal.now()))
        hal.close()


def selftest(args):
    hal = make_hal(args)
    try:
        while True:
            n, e, yaw = hal.odom()
            r = hal.ranges()
            _, dets = hal.detections()
            print("odom N%+.3f E%+.3f yaw%+6.1f | %s | IR %s | %s" % (
                n, e, yaw,
                " ".join("%s=%s" % (d, "  -  " if v is None else "%.3f" % v)
                         for d, v in sorted(r.items())),
                " ".join("%s=%s" % (n, "?" if v is None else ("ON" if v else "-"))
                         for n, v in hal.ir_state().items()),
                ", ".join(d.label or "?" for d in dets[:4])))
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        hal.close()


def vision_test(args):
    import cv2
    from robomaster import robot, camera
    from vision import Detector
    ep = robot.Robot()
    if C.CONN_TYPE == "ap":
        ep.initialize(conn_type="ap", proto_type="udp")
    else:
        ep.initialize(conn_type=C.CONN_TYPE)
    ep.camera.start_video_stream(display=False, resolution=C.CAMERA_RES)
    det = Detector()
    modes, mode = ("overlay", "mask", "raw"), 0
    print("keys: M = overlay / mask / raw   B = learn barrel box (nothing red in front!)   Q = quit")
    try:
        while True:
            frame = ep.camera.read_cv2_image(strategy="newest", timeout=2)
            t = time.perf_counter()
            dets = det.detect(frame)
            ms = (time.perf_counter() - t) * 1000
            if modes[mode] == "mask":
                view = det.mask_view(frame)
            elif modes[mode] == "overlay":
                view = det.draw(det.draw_ignored(frame.copy()), dets)
            else:
                view = frame.copy()
            cv2.putText(view, "%.1f ms  %s" % (ms, modes[mode]), (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("detections (M mode, B barrel, Q quit)", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("m"):
                mode = (mode + 1) % len(modes)
            if k == ord("b"):
                box = det.calibrate_barrel(frame)
                print("barrel box -> %s (saved to barrel_box.json)" % (box,) if box
                      else "no red barrel tip found in the lower part of the picture")
    finally:
        ep.camera.stop_video_stream()
        ep.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, choices=(1, 2), default=1)
    ap.add_argument("--shoot", help="designated targets: e.g. red,yellow or red_rect,circle")
    ap.add_argument("--map", help="round-1 json for round 2 (default out/round1.json)")
    ap.add_argument("--sim", action="store_true", help="run in the offline simulator")
    ap.add_argument("--seed", type=int, default=1, help="simulator maze seed")
    ap.add_argument("--no-fire", action="store_true", help="dry run, blaster disabled")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--vision-test", action="store_true")
    args = ap.parse_args()
    C.check_sensor_budget()
    if args.selftest:
        selftest(args)
    elif args.vision_test:
        vision_test(args)
    else:
        run_round(args)


if __name__ == "__main__":
    main()
