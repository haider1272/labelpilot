"""
Safety net around a trained YOLO model so that visible nuts are not silently missed.
It works with ANY number of nuts (0, 1 or many): no expected count is needed.

How it works (per frame)
  1. Normal YOLO pass (fast, low confidence threshold, nothing is thrown away too early).
  2. A cheap, non-AI "candidate finder" looks for anything that could be a nut:
       - "reference": difference from a photo of the EMPTY tray (camera must be fixed), or
       - "contrast" : objects brighter than their surroundings (silver nuts on a dark tray).
  3. Coverage check: every candidate must be covered by a YOLO box.
       - covered                   -> nothing to do
       - NOT covered               -> re-run YOLO on a zoomed crop around it (cheap, only these spots)
           - found                 -> "rescued" detection (orange)
           - still nothing         -> red UNKNOWN box = something is there that the model can't name
  4. (video) Persistence: a nut seen before that vanishes for a few frames stays as a blue ghost.
  If there are no candidates and no detections, the answer is simply "0 nuts".

Colors: green = confident, orange = low confidence or rescued, red = unknown object, blue = ghost.

Install:  pip install ultralytics opencv-python numpy
Run:      python robust_detect.py            (edit the variables below first)
          python robust_detect.py my_video.mp4
"""
import argparse, glob, os
import cv2
import numpy as np

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
MODEL_PATH = "best.pt"
SOURCE = "video.mp4"            # image, folder of images, or video file
OUTPUT_FOLDER = "robust_out"

IMGSZ = 1280                    # main pass size (larger helps small nuts; retrain at the same size)
CONF = 0.10                     # low on purpose; weak boxes are flagged orange, not dropped
CONFIDENT = 0.50
MERGE_IOU = 0.55

# --- candidate finder ---
CANDIDATE_METHOD = "contrast"   # "contrast" (bright nuts on dark tray) or "reference" (empty-tray photo)
EMPTY_REFERENCE = "empty_tray.png"   # used when CANDIDATE_METHOD = "reference" (fixed camera!)
WATCH_POLYGON = None            # e.g. [(345,215),(1165,25),(1250,345),(420,550)]; None = whole image
CONTRAST_THR = 18               # "contrast": brightness above the local background
DIFF_THR = 25                   # "reference": difference from the empty photo
MIN_CANDIDATE_AREA = 250        # ignore specks smaller than this many pixels

# --- rescue pass on uncovered candidates ---
CROP_IMGSZ = 640
CROP_CONF = 0.05

# --- optional heavy options (slow: they multiply inference time) ---
USE_TTA = False
USE_TILING = False
TILE, TILE_OVERLAP = 640, 0.25

PERSIST_FRAMES = 8              # video: keep a lost nut as a ghost this many frames (0 = off)
# ---------------------------------------------------------------


# ---------- model helpers ----------
def run_model(model, img, offset=(0, 0), imgsz=None, conf=None, augment=False):
    r = model.predict(img, imgsz=imgsz or IMGSZ, conf=CONF if conf is None else conf,
                      iou=0.6, augment=augment, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    b = r.boxes.xyxy.cpu().numpy().copy()
    b[:, [0, 2]] += offset[0]
    b[:, [1, 3]] += offset[1]
    return b, r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)


def nms(B, S, thr):
    order = np.argsort(-S)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        order = np.array([j for j in rest if iou(B[i], B[j]) <= thr], dtype=int)
    return np.array(keep, dtype=int)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge(parts):
    B = np.concatenate([p[0] for p in parts])
    S = np.concatenate([p[1] for p in parts])
    C = np.concatenate([p[2] for p in parts])
    if len(B) == 0:
        return B, S, C
    k = nms(B, S, MERGE_IOU)
    return B[k], S[k], C[k]


def detect(model, img):
    parts = [run_model(model, img, augment=USE_TTA)]
    if USE_TILING:
        H, W = img.shape[:2]
        step = int(TILE * (1 - TILE_OVERLAP))
        xs = list(range(0, max(1, W - TILE + 1), step)) or [0]
        ys = list(range(0, max(1, H - TILE + 1), step)) or [0]
        if xs[-1] + TILE < W: xs.append(W - TILE)
        if ys[-1] + TILE < H: ys.append(H - TILE)
        for y in ys:
            for x in xs:
                parts.append(run_model(model, img[y:y + TILE, x:x + TILE], (x, y)))
    return merge(parts)


# ---------- candidate finder (no AI) ----------
_ref_cache = {}

def candidates(img):
    """Boxes (x1,y1,x2,y2) of anything that could be a nut."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    if CANDIDATE_METHOD == "reference":
        if "g" not in _ref_cache:
            ref = cv2.imread(EMPTY_REFERENCE)
            if ref is None:
                raise SystemExit(f"Cannot read EMPTY_REFERENCE: {EMPTY_REFERENCE}")
            ref = cv2.resize(ref, (img.shape[1], img.shape[0]))
            _ref_cache["g"] = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mask = (cv2.absdiff(g, _ref_cache["g"]) > DIFF_THR)
    else:
        mask = (cv2.subtract(g, cv2.GaussianBlur(g, (0, 0), 25)) > CONTRAST_THR)
    mask = mask.astype(np.uint8)
    if WATCH_POLYGON:
        roi = np.zeros(g.shape, np.uint8)
        cv2.fillPoly(roi, [np.array(WATCH_POLYGON, np.int32)], 1)
        mask &= roi
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    return [(x, y, x + w, y + h) for x, y, w, h, a in st[1:] if a >= MIN_CANDIDATE_AREA]


def covered(cand, B):
    cx, cy = (cand[0] + cand[2]) / 2, (cand[1] + cand[3]) / 2
    area = max(1.0, (cand[2] - cand[0]) * (cand[3] - cand[1]))
    for b in B:
        if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
            return True
        ix = max(0, min(cand[2], b[2]) - max(cand[0], b[0]))
        iy = max(0, min(cand[3], b[3]) - max(cand[1], b[1]))
        if ix * iy / area > 0.3:
            return True
    return False


# ---------- per-frame logic ----------
def process(model, img, persist=None):
    """Returns items [(box, score, class, kind)], kind in: ok, low, rescued, unknown, ghost."""
    B, S, C = detect(model, img)
    items = [(b, s, c, "ok" if s >= CONFIDENT else "low") for b, s, c in zip(B, S, C)]
    H, W = img.shape[:2]
    for cand in candidates(img):
        if covered(cand, B):
            continue
        w, h = cand[2] - cand[0], cand[3] - cand[1]
        m = max(60, int(0.75 * max(w, h)))
        x0, y0 = max(0, int(cand[0]) - m), max(0, int(cand[1]) - m)
        x1, y1 = min(W, int(cand[2]) + m), min(H, int(cand[3]) + m)
        rb, rs, rc = run_model(model, img[y0:y1, x0:x1], (x0, y0), imgsz=CROP_IMGSZ, conf=CROP_CONF)
        found = False
        for b, s, c in zip(rb, rs, rc):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if cand[0] - m / 2 <= cx <= cand[2] + m / 2 and cand[1] - m / 2 <= cy <= cand[3] + m / 2:
                items.append((b, s, c, "rescued"))
                found = True
        if not found:
            items.append((np.array(cand, float), 0.0, -1, "unknown"))
    if persist is not None:
        real = [(b, s, c) for b, s, c, k in items if k != "unknown"]
        Bn = np.array([r[0] for r in real]) if real else np.zeros((0, 4))
        for t in persist.update(Bn, [r[1] for r in real], [r[2] for r in real]):
            if t["missed"] > 0:
                items.append((t["box"], t["score"], t["cls"], "ghost"))
    return items


class Persistence:
    def __init__(self, frames):
        self.frames, self.tracks = frames, []

    def update(self, B, S, C):
        used = set()
        for t in self.tracks:
            best, bi = 0.3, None
            for i, b in enumerate(B):
                if i not in used and iou(t["box"], b) > best:
                    best, bi = iou(t["box"], b), i
            if bi is None:
                t["missed"] += 1
            else:
                used.add(bi)
                t.update(box=B[bi], score=S[bi], cls=C[bi], missed=0)
        for i in range(len(B)):
            if i not in used:
                self.tracks.append(dict(box=B[i], score=S[i], cls=C[i], missed=0))
        self.tracks = [t for t in self.tracks if t["missed"] <= self.frames]
        return self.tracks


COLORS = {"ok": (0, 200, 0), "low": (0, 140, 255), "rescued": (0, 140, 255),
          "unknown": (0, 0, 255), "ghost": (255, 120, 0)}


def draw(img, items, names):
    vis = img.copy()
    for box, score, cls, kind in items:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS[kind], 2)
        label = "UNKNOWN" if kind == "unknown" else f"{names[int(cls)]} {score:.2f}" + (f" {kind}" if kind in ("rescued", "ghost") else "")
        cv2.putText(vis, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[kind], 1, cv2.LINE_AA)
    return vis


def summary(items):
    k = [it[3] for it in items]
    return {x: k.count(x) for x in ("ok", "low", "rescued", "unknown", "ghost")}


def main():
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=SOURCE)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--out", default=OUTPUT_FOLDER)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.model)
    names = [model.names[i] for i in sorted(model.names)]
    src = args.source

    if src.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(args.out, "result.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        persist = Persistence(PERSIST_FRAMES) if PERSIST_FRAMES > 0 else None
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            items = process(model, frame, persist)
            writer.write(draw(frame, items, names))
            s = summary(items)
            if s["unknown"] or s["rescued"]:
                print(f"frame {i}: {s}")
            i += 1
        cap.release(); writer.release()
        print(f"Saved video: {os.path.abspath(out_path)}")
        return

    paths = [src] if os.path.isfile(src) else sorted(
        p for ext in ("jpg", "jpeg", "png", "webp", "bmp") for p in glob.glob(os.path.join(src, f"*.{ext}")))
    if not paths:
        raise SystemExit(f"No images found at: {src}")
    for p in paths:
        img = cv2.imread(p)
        items = process(model, img)
        name = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(args.out, f"{name}_result.jpg"), draw(img, items, names))
        print(f"{name}: {summary(items)}")
    print(f"Results saved in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
