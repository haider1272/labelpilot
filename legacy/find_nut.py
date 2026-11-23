"""
Find nut(s) in an image with classical computer vision (no deep learning):
    1. HSV color threshold  -> pixels in the nut's color range
    2. Morphological closing -> fill small gaps
    3. Connected blobs       -> one blob per nut
    4. GrabCut (per blob)    -> refine each outline (removes shadow and table)
    5. Measure area and equivalent diameter  d = sqrt(4 * area / pi)

Works for nuts with a clear color (e.g. the blue nuts) on a plain background.
Nuts that touch each other merge into one blob. For silver nuts in the black
tray, or touching nuts, use auto_label_dino_sam.py instead.

Supports jpg, png, webp, bmp and more (Pillow is used as a fallback for formats OpenCV can't read).

Install:  pip install opencv-python numpy pillow

Easiest: edit the variables below, then run:   python find_nut.py
Or pass paths on the command line:             python find_nut.py small.png large.png
Outputs go to OUTPUT_FOLDER.
"""
import argparse, csv, os
import cv2
import numpy as np

# ---------------------------------------------------------------
# PUT YOUR IMAGE PATHS HERE (used when you run: python find_nut.py)
# Set LARGE_IMAGE = None to process only one image.
# ---------------------------------------------------------------
SMALL_IMAGE = "small.png"
LARGE_IMAGE = "large.png"
OUTPUT_FOLDER = "nut_results"
MULTIPLE_NUTS = False     # True = find every nut in the image, False = only the biggest one
SAVE_ENLARGED = True      # save the first image enlarged by X (needs two images)
MIN_NUT_AREA_FRAC = 0.0005  # multi mode: ignore blobs smaller than this share of the image


def load_image(path):
    """Read jpg, png, webp, bmp, ... as a BGR image. Falls back to Pillow if OpenCV can't read it."""
    img = cv2.imread(path)
    if img is None:
        try:
            from PIL import Image
            img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            img = None
    return img


def color_blobs(img, hue, sat_min, min_area):
    """Return a list of 0/1 masks, one per color blob (biggest first)."""
    h, s, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    color = ((h >= hue[0]) & (h <= hue[1]) & (s >= sat_min)).astype(np.uint8)
    color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(color)
    order = sorted(range(1, n), key=lambda i: -stats[i, 4])
    return [(lab == i).astype(np.uint8) for i in order if stats[i, 4] >= min_area]


def refine(img, seed, full_image=False):
    """GrabCut around one blob. Returns a filled 0/1 mask, or None."""
    H, W = seed.shape
    x, y, w, h = cv2.boundingRect(seed)
    m = max(30, int(0.3 * max(w, h)))
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
    if full_image:  # single-nut mode: run GrabCut on the whole image
        x0, y0, x1, y1 = 0, 0, W, H
    sub, sd = img[y0:y1, x0:x1], seed[y0:y1, x0:x1]

    gc = np.full(sd.shape, cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(sd, np.ones((25, 25), np.uint8)) > 0] = cv2.GC_PR_BGD
    gc[cv2.dilate(sd, np.ones((9, 9), np.uint8)) > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(sd, np.ones((9, 9), np.uint8)) > 0] = cv2.GC_FGD
    cv2.grabCut(sub, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    mask = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    mask = (lab == 1 + np.argmax(stats[1:, 4])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = filled
    return full


def find_nuts(img, hue, sat_min, multiple):
    H, W = img.shape[:2]
    blobs = color_blobs(img, hue, sat_min, MIN_NUT_AREA_FRAC * H * W if multiple else 1)
    if not multiple:
        blobs = blobs[:1]
    masks = [refine(img, b, full_image=not multiple) for b in blobs]
    return [m for m in masks if m is not None]


def measure(mask):
    area = int(mask.sum())
    x, y, w, h = cv2.boundingRect(mask)
    return dict(area=area, eq_diam=float(np.sqrt(4 * area / np.pi)),
                bbox=(x, y, w, h), center=(x + w / 2, y + h / 2))


def save_outputs(img, masks, ms, name, out):
    vis = img.copy()
    for i, (mask, m) in enumerate(zip(masks, ms), 1):
        x, y, w, h = m["bbox"]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 200, 0), 2)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(vis, f"#{i} d={m['eq_diam']:.0f}px", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        alpha = cv2.GaussianBlur(mask * 255, (5, 5), 0)
        cv2.imwrite(os.path.join(out, f"{name}_nut{i:02d}_cutout.png"),
                    np.dstack([img, alpha])[y:y + h, x:x + w])
    cv2.imwrite(os.path.join(out, f"{name}_result.jpg"), vis)
    with open(os.path.join(out, f"{name}_nuts.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["nut_id", "area_px", "equiv_diameter_px", "box_x", "box_y", "box_w", "box_h"])
        for i, m in enumerate(ms, 1):
            wr.writerow([i, m["area"], f"{m['eq_diam']:.1f}", *m["bbox"]])


def enlarge(img, factor, center):
    """Enlarge the whole image by `factor` around `center`, keeping the image size."""
    H, W = img.shape[:2]
    cx, cy = center
    M = np.float32([[factor, 0, cx - factor * cx], [0, factor, cy - factor * cy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="optional: overrides SMALL_IMAGE / LARGE_IMAGE")
    ap.add_argument("--out", default=OUTPUT_FOLDER)
    ap.add_argument("--multi", action="store_true", help="find every nut in the image")
    ap.add_argument("--hue-min", type=int, default=85, help="OpenCV hue 0-180 (blue is about 85-125)")
    ap.add_argument("--hue-max", type=int, default=125)
    ap.add_argument("--sat-min", type=int, default=70)
    ap.add_argument("--enlarge", action="store_true", help="with two images: enlarge the first by X and save it")
    args = ap.parse_args()
    if not args.images:
        args.images = [p for p in (SMALL_IMAGE, LARGE_IMAGE) if p]
    args.enlarge = args.enlarge or SAVE_ENLARGED
    multiple = args.multi or MULTIPLE_NUTS
    os.makedirs(args.out, exist_ok=True)

    results = []
    for path in args.images:
        img = load_image(path)
        if img is None:
            raise SystemExit(f"Cannot read image: {path}")
        masks = find_nuts(img, (args.hue_min, args.hue_max), args.sat_min, multiple)
        if not masks:
            raise SystemExit(f"No nut found in {path}. Try changing --hue-min/--hue-max/--sat-min.")
        ms = [measure(m) for m in masks]
        name = os.path.splitext(os.path.basename(path))[0]
        save_outputs(img, masks, ms, name, args.out)
        print(f"{name}: found {len(ms)} nut(s)")
        for i, m in enumerate(ms, 1):
            print(f"   #{i}: area = {m['area']} px, equivalent diameter = {m['eq_diam']:.1f} px, "
                  f"box = {m['bbox'][2]} x {m['bbox'][3]} px")
        diam = float(np.median([m["eq_diam"] for m in ms]))   # one number per image
        results.append((img, ms[0]["center"], diam, name))

    if len(results) == 2:
        (img1, c1, d1, n1), (_, _, d2, n2) = results
        X = d2 / d1
        which = "median diameter" if multiple else "diameter"
        print(f"\nX = {n2} {which} / {n1} {which} = {d2:.1f} / {d1:.1f} = {X:.3f}")
        if args.enlarge:
            out_path = os.path.join(args.out, f"{n1}_x{X:.3f}.jpg")
            cv2.imwrite(out_path, enlarge(img1, X, c1), [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Enlarged image saved to {os.path.abspath(out_path)}")
    print(f"\nResults saved in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
