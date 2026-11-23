"""
Find nuts on BOTH trays with pure computer vision (no AI models):

  Black tray : bright-blob detection + watershed + GrabCut        (from find_nuts_tray.py)
  Gray plate : dark center-hole detection                          (new here)
               An inserted nut has a small, round, very dark screw hole in its middle.
               Empty holes in the plate show the bright table, so they have no such dark dot.

Keep find_nuts_tray.py in the same folder. Install:  pip install opencv-python numpy pillow
Run:  python find_nuts_all.py                (edit the variables below first)
      python find_nuts_all.py my_image.webp

Limits: the gray-plate nut outline is a circle of fixed size (NUT_RADIUS_PX), so it is a
position detector there, not a size measurement. Nuts lying on their side on the plate
and nuts cut off by the image edge can be missed. Polygons and thresholds are tuned for
the 1280 x 720 example image, so adjust them for other cameras.
"""
import argparse, csv, os
import cv2
import numpy as np
from find_nuts_tray import (load_image, bright_blobs, split_merged, refine)

# ---------------------------------------------------------------
# PUT YOUR SETTINGS HERE
# ---------------------------------------------------------------
IMAGE = "tray.png"
OUTPUT_FOLDER = "all_results"
TRAY_POLYGON = [(345, 215), (1165, 25), (1250, 345), (420, 550)]          # black tray
PLATE_POLYGON = [(0, 140), (235, 103), (262, 115), (405, 700), (395, 720), (0, 720)]  # gray plate
TRAY_CONTRAST_THRESHOLD = 18   # black tray: lower = finds darker nuts, more false blobs
TRAY_MIN_AREA = 250
PLATE_DARK_THRESHOLD = 35      # gray plate: pixels darker than this count as a screw hole
HOLE_MIN_AREA, HOLE_MAX_AREA = 38, 220     # size of the dark center hole in pixels
NUT_RADIUS_PX = 29             # radius of a nut on the plate, used for the drawn circle
# ---------------------------------------------------------------


def plate_nuts(img, polygon, dark_thr):
    """Return a list of (x, y) centers of nuts on the gray plate."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    roi = np.zeros(g.shape, np.uint8)
    cv2.fillPoly(roi, [np.array(polygon, np.int32)], 255)
    dark = ((g < dark_thr) & (roi > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark)
    cands = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        fill = a / float(w * h)                      # round blobs fill ~0.7 of their box; the dark dot looks oval when the camera is tilted
        ratio = w / float(h)
        if HOLE_MIN_AREA <= a <= HOLE_MAX_AREA and fill >= 0.45 and 0.4 <= ratio <= 2.5:
            cands.append((float(cen[i][0]), float(cen[i][1]), int(a)))
    cands.sort(key=lambda c: -c[2])                  # keep the biggest when several are close
    keep = []
    for x, y, a in cands:
        if all((x - kx) ** 2 + (y - ky) ** 2 > 25 ** 2 for kx, ky in keep):
            keep.append((x, y))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=IMAGE)
    ap.add_argument("--out", default=OUTPUT_FOLDER)
    args = ap.parse_args()
    img = load_image(args.image)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    os.makedirs(args.out, exist_ok=True)
    vis = img.copy()
    rows = []

    # ---- black tray ----
    blobs = bright_blobs(img, TRAY_POLYGON, TRAY_CONTRAST_THRESHOLD, TRAY_MIN_AREA)
    big = [int(b.sum()) for b in blobs if b.sum() > 1000]
    typical = float(np.median(big)) if big else 1500.0
    seeds = [p for b in blobs for p in split_merged(b, typical)]
    for seed in seeds:
        mask = refine(img, seed)
        area = int(mask.sum())
        x, y, w, h = cv2.boundingRect(mask)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        rows.append(["black_tray", x + w // 2, y + h // 2, area, f"{np.sqrt(4 * area / np.pi):.1f}"])
    n_tray = len(rows)

    # ---- gray plate ----
    for x, y in plate_nuts(img, PLATE_POLYGON, PLATE_DARK_THRESHOLD):
        cv2.circle(vis, (int(x), int(y)), NUT_RADIUS_PX, (255, 160, 0), 2)
        rows.append(["gray_plate", int(x), int(y), "", ""])
    n_plate = len(rows) - n_tray

    for i, r in enumerate(rows, 1):
        cv2.putText(vis, str(i), (r[1] - 8, r[2] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    name = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out, f"{name}_result.jpg"), vis)
    with open(os.path.join(args.out, f"{name}_nuts.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["nut_id", "region", "center_x", "center_y", "area_px", "equiv_diameter_px"])
        for i, r in enumerate(rows, 1):
            wr.writerow([i] + r)
    print(f"Black tray: {n_tray} nut(s)   Gray plate: {n_plate} nut(s)")
    print(f"Results saved in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
