"""
Find many silver nuts in a dark tray with the same classical approach as find_nut.py:
    1. Threshold      -> bright pixels (local contrast, because the nuts are silver, not blue)
    2. Morphology     -> clean noise, join broken pieces
    3. Connected blobs-> one blob per nut (merged blobs are split with a watershed)
    4. GrabCut        -> refine each nut outline
    5. Measure        -> area and equivalent diameter  d = sqrt(4 * area / pi)

Works well for nuts lying face-up (the round flower shape) on the black tray.
Nuts lying on their side are found but their outline is only approximate.
It does NOT work on the gray metal plate (too little contrast); use auto_label_dino_sam.py there.

Install:  pip install opencv-python numpy pillow
Run:      python find_nuts_tray.py            (edit the variables below first)
          python find_nuts_tray.py my_image.webp
"""
import argparse, csv, os
import cv2
import numpy as np

# ---------------------------------------------------------------
# PUT YOUR SETTINGS HERE
# ---------------------------------------------------------------
IMAGE = "tray.png"
OUTPUT_FOLDER = "tray_results"
# Polygon around the black tray, as x,y points in pixels (image 1280x720). None = whole image.
TRAY_POLYGON = [(345, 215), (1165, 25), (1250, 345), (420, 550)]
CONTRAST_THRESHOLD = 18   # lower = more sensitive (finds darker nuts, but more false blobs)
MIN_AREA = 250            # ignore blobs smaller than this many pixels
# ---------------------------------------------------------------


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        try:
            from PIL import Image
            img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            img = None
    return img


def bright_blobs(img, polygon, thr, min_area):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    local = cv2.subtract(g, cv2.GaussianBlur(g, (0, 0), 25))   # brightness minus surroundings
    roi = np.full(g.shape, 255, np.uint8)
    if polygon:
        roi[:] = 0
        cv2.fillPoly(roi, [np.array(polygon, np.int32)], 255)
    b = ((local > thr) & (roi > 0)).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    return [(lab == i).astype(np.uint8) for i in range(1, n) if st[i, 4] >= min_area]


def split_merged(mask, typical):
    """Split a blob that is much bigger than a normal nut (touching nuts) with a watershed."""
    if mask.sum() < 1.8 * typical:
        return [mask]
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, peaks = cv2.threshold(dist, 0.55 * dist.max(), 255, cv2.THRESH_BINARY)
    n, mk = cv2.connectedComponents(peaks.astype(np.uint8))
    if n <= 2:
        return [mask]
    markers = mk.astype(np.int32) + 1
    markers[mask == 0] = 0
    markers[(mask > 0) & (peaks == 0)] = 0
    ws = cv2.watershed(cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), markers)
    parts = [((ws == k) & (mask > 0)).astype(np.uint8) for k in range(2, n + 1)]
    parts = [p for p in parts if p.sum() > 150]
    return parts or [mask]


def refine(img, seed):
    """GrabCut around one blob."""
    H, W = seed.shape
    x, y, w, h = cv2.boundingRect(seed)
    m = max(15, int(0.25 * max(w, h)))
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
    sub, sd = img[y0:y1, x0:x1], seed[y0:y1, x0:x1]
    gc = np.full(sd.shape, cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(sd, np.ones((15, 15), np.uint8)) > 0] = cv2.GC_PR_BGD
    gc[cv2.dilate(sd, np.ones((5, 5), np.uint8)) > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(sd, np.ones((7, 7), np.uint8)) > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(sub, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return seed
    mm = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mm)
    if n <= 1:
        return seed
    mm = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = mm
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=IMAGE)
    ap.add_argument("--out", default=OUTPUT_FOLDER)
    ap.add_argument("--thr", type=int, default=CONTRAST_THRESHOLD)
    ap.add_argument("--whole-image", action="store_true", help="ignore TRAY_POLYGON and search everywhere")
    args = ap.parse_args()

    img = load_image(args.image)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    os.makedirs(args.out, exist_ok=True)

    blobs = bright_blobs(img, None if args.whole_image else TRAY_POLYGON, args.thr, MIN_AREA)
    if not blobs:
        raise SystemExit("No nuts found. Try a lower --thr or check TRAY_POLYGON.")
    big = [int(b.sum()) for b in blobs if b.sum() > 1000]
    typical = float(np.median(big)) if big else float(np.median([b.sum() for b in blobs]))
    seeds = [p for b in blobs for p in split_merged(b, typical)]

    vis = img.copy()
    name = os.path.splitext(os.path.basename(args.image))[0]
    rows = []
    for i, seed in enumerate(seeds, 1):
        mask = refine(img, seed)
        area = int(mask.sum())
        x, y, w, h = cv2.boundingRect(mask)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = cv2.arcLength(max(cnts, key=cv2.contourArea), True)
        circ = 4 * np.pi * area / (per * per + 1e-9)
        eq = float(np.sqrt(4 * area / np.pi))
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        cv2.putText(vis, str(i), (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        alpha = cv2.GaussianBlur(mask * 255, (5, 5), 0)
        cv2.imwrite(os.path.join(args.out, f"{name}_nut{i:02d}_cutout.png"), np.dstack([img, alpha])[y:y + h, x:x + w])
        rows.append([i, area, f"{eq:.1f}", x, y, w, h, f"{circ:.2f}"])
        print(f"#{i}: area = {area} px, equivalent diameter = {eq:.1f} px, box = {w} x {h} px, roundness = {circ:.2f}")

    cv2.imwrite(os.path.join(args.out, f"{name}_result.jpg"), vis)
    with open(os.path.join(args.out, f"{name}_nuts.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["nut_id", "area_px", "equiv_diameter_px", "box_x", "box_y", "box_w", "box_h", "roundness"])
        wr.writerows(rows)
    print(f"\nFound {len(rows)} nut(s). Results saved in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
