"""
Live Type2 nut inspection with safety checks.

Keys
  b : capture EMPTY-tray background (remove all nuts first) -> enables leftover search
  s : full inspection (5-frame depth median + SAM 3) -> saves out/inspect_<n>.jpg/.json
  d : toggle debug view (residual / height map)
  m : print the measured size / height of every box YOLO is sure about (use it to set SAFETY_NUT_SIDE_MM,
      SAFETY_NUT_THICKNESS_MM: put ONE kind of nut on the tray, press m on a few frames)
  q : quit

Report status (also in the on-screen banner and in the .json):
  PASS        empty-tray background present and nothing to check
  CHECK       at least one alert (miss / large_blob / review / possible_stack / recovered_by_sam3)
  UNVERIFIED  no background: only YOLO looked. NOT a pass - press b with an empty tray
  BG_MISMATCH the scene does not match the stored background (old file, moved camera / tray, changed light):
              the leftover search is not trusted. Press b again with the tray empty, same camera and light

Set the area that is checked (important: everything else on the table is ignored):
  r : click the corners of the TRAY area (left click = add corner, Enter = finish, Esc = cancel) -> saved
  i : click an area to IGNORE (a clamp, a second box, a label ...), can be repeated -> saved
  c : clear the tray area and all ignore areas

Every frame runs YOLO + leftover search (cheap). SAM 3 only runs on 's'
(or every SAM_EVERY_N frames if set in config), because it is much heavier.

Optional settings in config/config.py (all have defaults):
  BACKGROUND_PATH = "background_empty_tray.npz"
  SAM_EVERY_N     = 0            # 0 = only on key 's'
  SAFETY_TRAY_ROI = [[x,y], ...]
  SAFETY_IGNORE_POLYS = [[[x,y],...], ...]
  SAFETY_NUT_SIDE_MM = (8, 30)
  SAFETY_NUT_THICKNESS_MM = 5.0
  SAFETY_CAM_TO_ROBOT = [[...4x4...]]
  ...any SafetyConfig field as SAFETY_<FIELD_NAME_UPPER>
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import fields

import cv2
import numpy as np
from ultralytics import YOLO

import config.config as cfg
from src.librealsense import initialize_realsense, get_frames
from src.nut_safety import (SafetyConfig, Intrinsics, Background, inspect, draw, load_sam3)


def build_safety_config():
    sc = SafetyConfig()
    for f in fields(sc):
        key = "SAFETY_" + f.name.upper()
        if hasattr(cfg, key):
            setattr(sc, f.name, getattr(cfg, key))
    return sc


def grab_stable(pipeline, align, depth_scale, n=5):
    """Median depth over n frames (RealSense depth is noisy at mm scale)."""
    colors, depths = [], []
    while len(colors) < n:
        c, d, _, _ = get_frames(pipeline, align, depth_scale)
        if c is None:
            continue
        colors.append(np.array(c, copy=True)); depths.append(np.array(d, copy=True))   # frames may be reused buffers
    D = np.stack(depths); D[D <= 0] = np.nan
    return colors[-1], np.nan_to_num(np.nanmedian(D, axis=0), nan=0.0).astype(np.float32)


def pick_polygon(title, frame, existing=()):
    """Click the corners of an area on the live picture. Left click = corner, Enter = done, Esc = cancel,
    Backspace = remove the last corner. Returns [[x, y], ...] (at least 3 corners) or None."""
    pts = []

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append([int(x), int(y)])

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        vis = frame.copy()
        for p in existing:
            cv2.polylines(vis, [np.asarray(p, np.int32)], True, (150, 150, 150), 1)
        if pts:
            cv2.polylines(vis, [np.asarray(pts, np.int32)], len(pts) >= 3, (0, 255, 255), 2)
            for p in pts:
                cv2.circle(vis, tuple(p), 4, (0, 255, 255), -1)
        cv2.putText(vis, "click corners | Enter = done | Esc = cancel | Backspace = undo", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow(title, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):
            if len(pts) >= 3:
                cv2.destroyWindow(title)
                return pts
        elif k == 27:
            cv2.destroyWindow(title)
            return None
        elif k in (8, 127) and pts:
            pts.pop()


def load_areas(path, sc):
    """tray area / ignore areas drawn with keys r / i (overrides the config values when the file exists)."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                d = json.load(f)
            if d.get("tray_roi"):
                sc.tray_roi = d["tray_roi"]
            if d.get("ignore_polys") is not None:
                sc.ignore_polys = d["ignore_polys"]
            print(f"Areas loaded from {path}: tray {'set' if sc.tray_roi else 'NOT set (whole picture is checked)'}, "
                  f"{len(sc.ignore_polys)} ignore area(s)")
        except Exception as e:
            print(f"[WARNING] could not read {path}: {e}")


def save_areas(path, sc):
    with open(path, "w") as f:
        json.dump({"tray_roi": sc.tray_roi, "ignore_polys": sc.ignore_polys}, f)


def print_sizes(items):
    """Calibration help: size (mm) and height (mm) of the boxes YOLO is sure about, per class."""
    rows = defaultdict(lambda: ([], []))
    for it in items:
        if it.source.startswith("yolo") and it.score >= 0.5:
            if it.size_mm:
                rows[it.label][0].append(it.size_mm)
            if it.height_mm:
                rows[it.label][1].append(it.height_mm)
    if not rows:
        print("[sizes] no confident YOLO boxes in this frame")
        return
    for lab, (sz, hh) in sorted(rows.items()):
        txt = f"[sizes] {lab:12s} n={len(sz):2d}  side mm: min {min(sz):.1f}  median {np.median(sz):.1f}  max {max(sz):.1f}" if sz else f"[sizes] {lab}"
        if hh:
            txt += f"   height mm (p90): median {np.median(hh):.1f}  max {max(hh):.1f}"
        print(txt)
    allsz = [v for sz, _ in rows.values() for v in sz]
    if allsz:
        print(f"[sizes] suggestion: SAFETY_NUT_SIDE_MM = ({0.7 * min(allsz):.0f}, {1.4 * max(allsz):.0f})  "
              "(sizes above are box sizes; the leftover search measures the real outline, which is ~10-25% smaller, "
              "so the range is wide on purpose)")


def debug_view(debug, shape):
    h = debug["height_mm"]
    hv = cv2.applyColorMap(np.clip(h * 8, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    if debug["residual"] is not None:
        hv[debug["residual"]] = (255, 0, 255)
    hv[debug["explained"]] = (hv[debug["explained"]] * 0.4).astype(np.uint8)
    return hv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sc = build_safety_config()
    bg_path = getattr(cfg, "BACKGROUND_PATH", "background_empty_tray.npz")
    sam_every = getattr(cfg, "SAM_EVERY_N", 0)

    model = YOLO(cfg.MODEL_PATH)
    print("Classes:", model.names)
    pipeline, align, depth_scale, color_stream = initialize_realsense()
    intr = Intrinsics.from_rs(color_stream.get_intrinsics())   # depth is aligned to color

    areas_path = getattr(cfg, "AREAS_PATH", "tray_areas.json")
    load_areas(areas_path, sc)
    if not sc.tray_roi:
        print("TIP: press r and click the corners of the tray, so the table around it is not checked")
    bg = Background.load(bg_path) if os.path.exists(bg_path) else None
    print("Background:", "loaded" if bg else "NOT SET - press 'b' with an empty tray")
    bg_checked = False
    sam = None
    show_debug, frame_i, shot = False, 0, 0

    try:
        while True:
            color, depth, _, _ = get_frames(pipeline, align, depth_scale)
            if color is None:
                continue
            frame_i += 1
            if bg is not None and not bg_checked:          # a background of another resolution would crash
                bg_checked = True
                if bg.color.shape[:2] != color.shape[:2]:
                    print(f"[WARNING] saved background is {bg.color.shape[1]}x{bg.color.shape[0]}, camera is "
                          f"{color.shape[1]}x{color.shape[0]}: ignoring it - press 'b' to capture a new one")
                    bg = None
            run_sam = (not args.no_sam) and sam_every and frame_i % sam_every == 0
            if run_sam and sam is None:
                sam = load_sam3()

            report, items, debug = inspect(color, depth, model, intr, sc, bg, sam if run_sam else None)
            vis = debug_view(debug, color.shape) if show_debug else draw(color, items, debug, sc, report)
            if bg is None:
                cv2.putText(vis, "NO BACKGROUND: leftover search off (press b)", (8, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            last_items = items
            cv2.imshow("Jinhap Metal Nuts Type2 - safety", vis)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("d"):
                show_debug = not show_debug
            if k == ord("m"):
                print_sizes(last_items)
            if k in (ord("r"), ord("i")):
                poly = pick_polygon("Tray area" if k == ord("r") else "Ignore area", color,
                                    ([sc.tray_roi] if sc.tray_roi else []) + list(sc.ignore_polys))
                if poly:
                    if k == ord("r"):
                        sc.tray_roi = poly
                    else:
                        sc.ignore_polys = list(sc.ignore_polys) + [poly]
                    save_areas(areas_path, sc)
                    print("Saved", areas_path)
            if k == ord("c"):
                sc.tray_roi, sc.ignore_polys = None, []
                save_areas(areas_path, sc)
                print("Tray area and ignore areas cleared")
            if k == ord("b"):
                print("Capturing empty-tray background (30 frames)... keep hands and arm out of the picture")
                try:
                    bg = Background.capture(lambda: get_frames(pipeline, align, depth_scale)[:2], n=30)
                    bg.save(bg_path)
                    bg_checked = True
                    print("Saved", bg_path)
                except RuntimeError as e:
                    print("[ERROR]", e)
            if k == ord("s"):
                if sam is None and not args.no_sam:
                    print("Loading SAM 3 ..."); sam = load_sam3()
                color_s, depth_s = grab_stable(pipeline, align, depth_scale)
                report, items, debug = inspect(color_s, depth_s, model, intr, sc, bg, sam)
                shot += 1
                stem = os.path.join(args.out, f"inspect_{shot:04d}")
                cv2.imwrite(stem + ".jpg", draw(color_s, items, debug, sc, report, show_rejected=True))
                with open(stem + ".json", "w") as f:
                    json.dump(report, f, indent=2)
                print(json.dumps(report["counts"]), report["status"], "->", stem)
                for w in report["warnings"]:
                    print("  [WARNING]", w)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()