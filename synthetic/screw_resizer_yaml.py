"""
Screw Resizer (YAML version) - enlarge objects inside YOLO bounding boxes.

You give ONE thing: the dataset's data.yaml. The script reads the train / val / test image
folders and the class names from it, finds the label folders (images -> labels), enlarges the
selected classes, and writes a new dataset (images, labels and a new data.yaml) to the output folder.

For every selected bbox:
  1. the exact nut pixels are found inside the box: SAM 3 mask, else GrabCut (no AI model)
  2. only the nut is cut out and scaled up by X around its centre
  3. it is pasted back with soft edges; complete nuts are pasted last so they stay on top
  4. if no good mask can be made, the object is recovered with the old method (whole box + margin)
  5. the label is updated from the enlarged nut

Install:  pip install opencv-python numpy pillow pyyaml

Window (GUI):   python screw_resizer_yaml.py
Command line:   python screw_resizer_yaml.py --yaml path/to/data.yaml --out out_folder \
                    --scale-x 1.078 --mask-method auto --classes 0 1
Per-class factors in one pass:  --class-scale 0:1.37 1:1.17 2:1.37

Collisions (on by default): enlarged nuts must not run into their neighbours.
  - a nut that would collide gets the largest smaller factor that fits (down to --cap-floor of the intended growth)
  - nuts that already touch / overlap are enlarged TOGETHER as one group around a common centre, with one factor
    (the mean of their factors), so the overlap between them keeps its proportions
  - a nut that cannot be enlarged without a new collision stays at its original size
  --no-collision-check switches this off (every nut is enlarged around its own centre)
  --box-mode tight|margin|scaled decides how the label box of an enlarged nut is made (default: tight)
  Memory: SAM 3 gets the boxes in chunks (halved automatically if memory runs out), runs in half precision on the GPU
  (--sam-fp32 to turn that off), moves to the CPU if even one box does not fit, and the GPU cache and Python garbage
  are released after every image. Nut masks are kept as small crops, not full-size arrays.
  --drop-mixed does not write an image in which a nut had to stay at its original size
"""
import argparse
import gc
import os
import random
import sys
import threading

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    sys.exit("Please install PyYAML first:  pip install pyyaml")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
SPLIT_KEYS = ("train", "val", "valid", "test")


# --------------------------------------------------------------------------- #
#  Image / label helpers
# --------------------------------------------------------------------------- #
def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, img):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(path)


def read_labels(path):
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r") as f:
        for line in f:
            t = line.strip().split()
            if len(t) < 5:
                continue
            cls = int(float(t[0]))
            vals = [float(v) for v in t[1:]]
            if len(vals) == 4:
                cx, cy, w, h = vals
                items.append({"cls": cls, "kind": "box", "cx": cx, "cy": cy, "w": w, "h": h})
            else:  # polygon (segmentation)
                pts = np.array(vals, dtype=np.float64).reshape(-1, 2)
                mn, mx = pts.min(0), pts.max(0)
                items.append({"cls": cls, "kind": "poly", "pts": pts,
                              "cx": (mn[0] + mx[0]) / 2, "cy": (mn[1] + mx[1]) / 2,
                              "w": mx[0] - mn[0], "h": mx[1] - mn[1]})
    return items


def format_label(it):
    if it["kind"] == "box":
        return f"{it['cls']} {it['cx']:.6f} {it['cy']:.6f} {it['w']:.6f} {it['h']:.6f}"
    pts = " ".join(f"{x:.6f} {y:.6f}" for x, y in it["pts"])
    return f"{it['cls']} {pts}"


def feather_alpha(h, w, feather):
    feather = int(max(1, feather))
    a = np.zeros((h, w), np.float32)
    a[feather:h - feather, feather:w - feather] = 1.0
    if a.sum() == 0:
        a[:] = 1.0
    k = feather * 2 + 1
    a = cv2.GaussianBlur(a, (k, k), feather / 2.0)
    return a[..., None]


# ---------------- memory helpers ---------------- #
def is_oom(e):
    t = (type(e).__name__ + " " + str(e)).lower()
    return "out of memory" in t or "outofmemory" in t or "cuda error: out of" in t


def free_memory():
    """Give back what is no longer used: Python garbage and (if SAM 3 runs on the GPU) the CUDA cache."""
    gc.collect()
    torch = sys.modules.get("torch")                 # only if torch was ever loaded
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------- exact nut pixels: SAM 3 -> GrabCut -> (rectangle as last resort) ---------------- #
# "obj" = [model, processor, device]; "chunk" = boxes per SAM call that worked last time (0 = all boxes at once)
_SAM = {"obj": None, "failed": False, "chunk": 0, "fp16": True, "oom_images": 0}


def get_sam():
    """Load SAM 3 once. Returns None (and remembers) if it is not available."""
    if _SAM["obj"] is None and not _SAM["failed"]:
        try:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less GPU fragmentation
            import torch
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Sam3TrackerModel.from_pretrained("facebook/sam3").eval()
            try:
                model = model.to(device)
            except Exception as e:
                if device != "cuda" or not is_oom(e):
                    raise
                print("[info] not enough GPU memory for SAM 3 - running it on the CPU (slower)")
                model, device = model.to("cpu"), "cpu"
                free_memory()
            proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
            _SAM["obj"] = [model, proc, device]
        except Exception as e:
            _SAM["failed"] = True
            print(f"[info] SAM 3 not available ({type(e).__name__}); using GrabCut instead")
    return _SAM["obj"]


def _sam3_chunk(sam, pil, boxes):
    """SAM 3 masks for these boxes (one call). Returns a list of full-size bool arrays."""
    import torch
    model, proc, device = sam
    inputs = out = None
    try:
        inputs = proc(images=pil, input_boxes=[[list(map(float, b)) for b in boxes]], return_tensors="pt").to(device)
        with torch.inference_mode():
            if device == "cuda" and _SAM["fp16"]:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(**inputs, multimask_output=False)
            else:
                out = model(**inputs, multimask_output=False)
        m = proc.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"])[0].numpy()
        if m.ndim == 4:
            m = m[:, 0]
        return [x > 0 for x in m]
    finally:
        del inputs, out                          # release the GPU tensors before anything else happens


def sam3_cleaned(sam, img, boxes, min_fill):
    """Cleaned, cropped masks (crop, x, y) or None for every box.
    Memory: boxes are sent in chunks. If the GPU runs out of memory the chunk is halved (and remembered for the
    next images); at one box per call it is retried once, then SAM 3 moves to the CPU. A CPU failure raises."""
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    H, W = img.shape[:2]
    chunk = _SAM["chunk"] if 0 < _SAM["chunk"] < len(boxes) else len(boxes)
    chunk = max(1, min(chunk, int(3e8 // max(1, H * W))))        # also keep the full-size mask batch below ~300 MB
    out = [None] * len(boxes)
    i, retried = 0, False
    while i < len(boxes):
        part = boxes[i:i + chunk]
        try:
            full = _sam3_chunk(sam, pil, part)
            for k, b in enumerate(part):
                out[i + k] = clean_mask(full[k], 0, 0, b, min_fill)
            del full
            i += chunk
        except Exception as e:
            if not is_oom(e):
                raise
            free_memory()
            if chunk > 1:
                chunk = max(1, chunk // 2)
                _SAM["chunk"] = chunk
                print(f"[memory] GPU/CPU memory is full - SAM 3 now takes {chunk} box(es) per call")
            elif sam[2] == "cuda" and not retried:
                retried = True                                    # one more try after the cache was freed
            elif sam[2] == "cuda":
                print("[memory] SAM 3 moves to the CPU (slower, but it will not run out of GPU memory)")
                sam[0] = sam[0].to("cpu")
                sam[2] = "cpu"
                free_memory()
            else:
                raise
    return out


def grabcut_mask(img, box):
    """GrabCut started from the YOLO box (no AI model). Returns (mask of the working window, x0, y0) or None."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    mx, my = max(10, (x1 - x0) // 3), max(10, (y1 - y0) // 3)
    cx0, cy0, cx1, cy1 = max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)
    sub = img[cy0:cy1, cx0:cx1]
    gc = np.full(sub.shape[:2], cv2.GC_BGD, np.uint8)
    gc[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = cv2.GC_PR_FGD
    bw, bh = x1 - x0, y1 - y0
    gc[y0 - cy0 + bh // 4:y1 - cy0 - bh // 4, x0 - cx0 + bw // 4:x1 - cx0 - bw // 4] = cv2.GC_FGD
    try:
        cv2.grabCut(sub, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    return (((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8), cx0, cy0)


def clean_mask(mask, ox, oy, box, min_fill):
    """`mask` covers the image area whose top-left corner is (ox, oy). Keeps the part inside `box`, the biggest piece,
    and rejects masks that are too small. Returns the tight crop (crop, x, y) in image coordinates, or None.
    Only the box area is processed, so a nut costs a few KB instead of a full-size array."""
    if mask is None:
        return None
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = box
    bx0, by0 = max(ox, int(x0)), max(oy, int(y0))
    bx1, by1 = min(ox + w, int(x1) + 1), min(oy + h, int(y1) + 1)
    if bx1 <= bx0 or by1 <= by0:
        return None
    sub = (mask[by0 - oy:by1 - oy, bx0 - ox:bx1 - ox] > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub)
    if n <= 1:
        return None
    m = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
    if m.sum() / max(1.0, (x1 - x0) * (y1 - y0)) < min_fill:
        return None
    x, y, ww, hh = cv2.boundingRect(m)
    return (m[y:y + hh, x:x + ww].copy(), bx0 + x, by0 + y)


def get_masks(img, boxes, cfg):
    """One cleaned, cropped mask (crop, x, y) (or None = 'could not segment') per box, using the chosen method."""
    method = cfg["mask_method"]
    if method == "rect" or not boxes:
        return [None] * len(boxes)
    raw = None
    if method in ("auto", "sam3"):
        sam = get_sam()
        if sam is not None:
            try:
                raw = sam3_cleaned(sam, img, boxes, cfg["min_fill"])
            except Exception as e:
                print(f"[info] SAM 3 failed on this image ({type(e).__name__}); using GrabCut")
                free_memory()
    out = []
    for i, box in enumerate(boxes):
        m = raw[i] if raw is not None else None
        if m is None and method != "sam3":          # recover: GrabCut from the box
            g = grabcut_mask(img, box)
            m = clean_mask(g[0], g[1], g[2], box, cfg["min_fill"]) if g is not None else None
        out.append(m)
    return out


def paste_scaled(canvas, src, mask, scale, feather_px=1, centre=None):
    """Cut the masked nut out of `src`, scale it, paste it on `canvas` at the same centre. Returns new box.
    `mask` is the tight crop (crop, x, y) of the nut."""
    H, W = canvas.shape[:2]
    crop, x, y = mask
    h, w = crop.shape[:2]
    pad = 4
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    patch = src[y0:y1, x0:x1]
    mfull = np.zeros(patch.shape[:2], np.uint8)
    mfull[y - y0:y - y0 + h, x - x0:x - x0 + w] = crop.astype(np.uint8) * 255
    m = cv2.erode(mfull, np.ones((3, 3), np.uint8))
    nw, nh = max(2, int(round(patch.shape[1] * scale))), max(2, int(round(patch.shape[0] * scale)))
    interp = cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA
    patch_s = cv2.resize(patch, (nw, nh), interpolation=interp)
    m_s = cv2.resize(m, (nw, nh), interpolation=interp)
    m_s = cv2.GaussianBlur(m_s, (5, 5), 0).astype(np.float32) / 255.0
    cx, cy = centre if centre is not None else (x + w / 2.0, y + h / 2.0)     # the point everything grows away from
    px0 = int(round(cx + (x0 - cx) * scale))
    py0 = int(round(cy + (y0 - cy) * scale))
    sx0, sy0 = max(0, -px0), max(0, -py0)
    ex, ey = min(nw, W - px0), min(nh, H - py0)
    if ex <= sx0 or ey <= sy0:
        return None
    roi = canvas[py0 + sy0:py0 + ey, px0 + sx0:px0 + ex].astype(np.float32)
    a = m_s[sy0:ey, sx0:ex, None]
    canvas[py0 + sy0:py0 + ey, px0 + sx0:px0 + ex] = (patch_s[sy0:ey, sx0:ex] * a + roi * (1 - a)).astype(np.uint8)
    ys, xs = np.where(m_s[sy0:ey, sx0:ex] > 0.5)
    if len(xs) == 0:
        return None
    return (px0 + sx0 + xs.min(), py0 + sy0 + ys.min(), px0 + sx0 + xs.max() + 1, py0 + sy0 + ys.max() + 1)


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def rect_paste(out, src, rect, rects, i, s, cfg, centre=None):
    """Last resort (the original method): enlarge the whole box + margin. Returns new pixel box or None."""
    H, W = src.shape[:2]
    x0, y0, x1, y1 = rect
    bw, bh = x1 - x0, y1 - y0
    if bw < 2 or bh < 2:
        return None
    cx, cy = centre if centre is not None else ((x0 + x1) / 2, (y0 + y1) / 2)
    mx, my = bw * cfg["margin"], bh * cfg["margin"]
    sx0, sy0 = int(max(0, np.floor(x0 - mx))), int(max(0, np.floor(y0 - my)))
    sx1, sy1 = int(min(W, np.ceil(x1 + mx))), int(min(H, np.ceil(y1 + my)))
    crop = src[sy0:sy1, sx0:sx1]
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    dx0, dy0 = cx + (sx0 - cx) * s, cy + (sy0 - cy) * s
    dx1, dy1 = cx + (sx1 - cx) * s, cy + (sy1 - cy) * s
    nw, nh = max(2, int(round(dx1 - dx0))), max(2, int(round(dy1 - dy0)))
    dx0, dy0 = int(round(dx0)), int(round(dy0))
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    patch = cv2.resize(crop, (nw, nh), interpolation=interp).astype(np.float32)
    alpha = feather_alpha(nh, nw, max(2, int(min(mx, my) * s * cfg["feather"])))
    cx0, cy0, cx1, cy1 = max(0, dx0), max(0, dy0), min(W, dx0 + nw), min(H, dy0 + nh)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    p = patch[cy0 - dy0:cy1 - dy0, cx0 - dx0:cx1 - dx0]
    a = alpha[cy0 - dy0:cy1 - dy0, cx0 - dx0:cx1 - dx0]
    out[cy0:cy1, cx0:cx1] = (a * p + (1 - a) * out[cy0:cy1, cx0:cx1].astype(np.float32)).astype(np.uint8)
    return (cx + (x0 - cx) * s, cy + (y0 - cy) * s, cx + (x1 - cx) * s, cy + (y1 - cy) * s)


# ---------------- collision control ---------------- #
def _shape_of(mask, rect, W, H):
    """(crop, x0, y0): the pixels an object covers = its mask, or its label rectangle when there is no mask."""
    if mask is not None:                                   # mask = (tight crop, x, y)
        return (mask[0].astype(np.uint8), mask[1], mask[2])
    x0, y0 = max(0, int(np.floor(rect[0]))), max(0, int(np.floor(rect[1])))
    x1, y1 = min(W, int(np.ceil(rect[2]))), min(H, int(np.ceil(rect[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    return (np.ones((y1 - y0, x1 - x0), np.uint8), x0, y0)


def _union_shape(shapes):
    x0, y0 = min(s[1] for s in shapes), min(s[2] for s in shapes)
    x1, y1 = max(s[1] + s[0].shape[1] for s in shapes), max(s[2] + s[0].shape[0] for s in shapes)
    out = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for crop, sx, sy in shapes:
        out[sy - y0:sy - y0 + crop.shape[0], sx - x0:sx - x0 + crop.shape[1]] |= crop
    return (out, x0, y0)


def _overlap(a, b):
    ac, ax, ay = a
    bc, bx, by = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + ac.shape[1], bx + bc.shape[1]), min(ay + ac.shape[0], by + bc.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(ac[y0 - ay:y1 - ay, x0 - ax:x1 - ax] & bc[y0 - by:y1 - by, x0 - bx:x1 - bx]))


def _touching(a, b, gap=3):
    """True if the two shapes overlap or come within `gap` pixels of each other."""
    ac, ax, ay = a
    k = 2 * gap + 1
    ad = cv2.dilate(np.pad(ac, gap), np.ones((k, k), np.uint8))
    return _overlap((ad, ax - gap, ay - gap), b) > 0


def _scale_shape(shape, s, c):
    crop, x, y = shape
    nw, nh = max(1, int(round(crop.shape[1] * s))), max(1, int(round(crop.shape[0] * s)))
    return (cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST),
            int(round(c[0] + (x - c[0]) * s)), int(round(c[1] + (y - c[1]) * s)))


def plan_scales(n_items, sel, rects, masks, targets, W, H, cfg):
    """Decide the factor and the growth centre of every selectable object so that nothing new collides.
    Returns ({i: (factor, centre)}, stats). Objects missing from the plan stay at their original size."""
    shapes = {i: _shape_of(masks.get(i), rects[i], W, H) for i in range(n_items)}
    sel = [i for i in sel if shapes[i] is not None]
    parent = {i: i for i in sel}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(len(sel)):
        for b in range(a + 1, len(sel)):
            ia, ib = sel[a], sel[b]
            xa, ya, wa, ha = shapes[ia][1], shapes[ia][2], shapes[ia][0].shape[1], shapes[ia][0].shape[0]
            xb, yb, wb, hb = shapes[ib][1], shapes[ib][2], shapes[ib][0].shape[1], shapes[ib][0].shape[0]
            if xa - 4 > xb + wb or xb - 4 > xa + wa or ya - 4 > yb + hb or yb - 4 > ya + ha:
                continue
            if _touching(shapes[ia], shapes[ib]):
                parent[find(ia)] = find(ib)
    groups = {}
    for i in sel:
        groups.setdefault(find(i), []).append(i)

    plan, stats = {}, {"capped": 0, "grouped": 0, "skipped": 0}
    floor = float(cfg.get("cap_floor", 0.5))
    for members in sorted(groups.values(), key=min):
        U = _union_shape([shapes[i] for i in members])
        t = float(np.mean([targets[i] for i in members]))
        centre = (U[1] + U[0].shape[1] / 2.0, U[2] + U[0].shape[0] / 2.0)
        if t <= 1.0:                                           # shrinking never causes a new collision
            for i in members:
                plan[i] = (t, centre)
            continue
        others = [j for j in range(n_items) if j not in members and shapes[j] is not None]
        base = {j: _overlap(U, shapes[j]) for j in others}      # overlap that already exists stays allowed
        smin = 1.0 + floor * (t - 1.0)
        steps = max(1, int(np.ceil((t - smin) / 0.02)))
        chosen = None
        for k in range(steps + 1):
            f = t - k * (t - smin) / steps
            sc = _scale_shape(U, f, centre)
            if sc[1] < -1 or sc[2] < -1 or sc[1] + sc[0].shape[1] > W + 1 or sc[2] + sc[0].shape[0] > H + 1:
                continue                                       # would leave the picture
            ok = True
            for j in others:
                tol = max(4.0, 0.01 * min(float(np.count_nonzero(sc[0])), float(np.count_nonzero(shapes[j][0]))))
                if _overlap(sc, shapes[j]) > base[j] + tol:
                    ok = False
                    break
            if ok:
                chosen = f
                break
        if chosen is None:
            stats["skipped"] += len(members)
            continue
        if chosen < t - 1e-6:
            stats["capped"] += len(members)
        if len(members) > 1:
            stats["grouped"] += len(members)
        for i in members:
            plan[i] = (chosen, centre)
            shapes[i] = _scale_shape(shapes[i], chosen, centre)  # later groups must avoid the enlarged nut
    return plan, stats


def process_image_ex(img, items, cfg, rng):
    """Returns (new_image, new_items, n_scaled, n_skipped, n_rect_fallback, stats).
    Nut-only pasting with an exact mask (SAM 3, else GrabCut); where no good mask can be made the
    object is recovered with the rectangle method. Collisions are avoided (see plan_scales); hidden nuts are pasted first so complete ones stay on top."""
    H, W = img.shape[:2]
    out = img.copy()
    rects = [((it["cx"] - it["w"] / 2) * W, (it["cy"] - it["h"] / 2) * H,
              (it["cx"] + it["w"] / 2) * W, (it["cy"] + it["h"] / 2) * H) for it in items]
    sel = [i for i, it in enumerate(items) if it["cls"] in cfg["classes"]
           and not (cfg["max_size"] > 0 and max(it["w"], it["h"]) > cfg["max_size"])]
    masks_list = get_masks(img, [rects[i] for i in sel], cfg)
    masks = dict(zip(sel, masks_list))

    med = {}
    for c in set(items[i]["cls"] for i in sel):
        v = [float(np.count_nonzero(masks[i][0])) for i in sel if items[i]["cls"] == c and masks[i] is not None]
        med[c] = float(np.median(v)) if v else 1.0
    comp = {i: (np.count_nonzero(masks[i][0]) / med[items[i]["cls"]] if masks[i] is not None else 1.0) for i in sel}
    order = sorted(sel, key=lambda i: comp[i])                      # most hidden first, complete last = on top

    cs = cfg.get("class_scale") or {}
    targets = {}
    for i in order:
        if items[i]["cls"] in cs:                                   # a factor of its own for this class
            targets[i] = cs[items[i]["cls"]] * rng.uniform(cfg["scale_min"], cfg["scale_max"]) / max(1e-9, cfg["scale_min"])
        else:
            targets[i] = rng.uniform(cfg["scale_min"], cfg["scale_max"])
    if cfg.get("collision", True):
        plan, stats = plan_scales(len(items), sel, rects, masks, targets, W, H, cfg)
    else:
        plan, stats = {i: (targets[i], None) for i in sel}, {"capped": 0, "grouped": 0, "skipped": 0}

    new_box, n_scaled, n_skipped, n_rect = {}, 0, 0, 0
    for i in order:
        if i not in plan:                                           # would collide: stays at its original size
            n_skipped += 1
            continue
        s, centre = plan[i]
        m = masks[i]
        nb = None
        if m is not None:
            bx, by, bh, bw = m[1], m[2], m[0].shape[0], m[0].shape[1]
            ncx, ncy = centre if centre is not None else (bx + bw / 2, by + bh / 2)
            nb = paste_scaled(out, img, m, s, centre=centre)         # tight box of the enlarged nut
            if nb is not None:
                ox0, oy0, ox1, oy1 = rects[i]
                mode = cfg.get("box_mode", "tight")
                fill = (bw * bh) / max(1.0, (ox1 - ox0) * (oy1 - oy0))   # how much of your box the nut mask covers
                if mode == "margin" and fill >= 0.6:
                    # tight box of the enlarged nut + the space your original box had around the nut (not enlarged)
                    nb = (nb[0] - max(0.0, bx - ox0), nb[1] - max(0.0, by - oy0),
                          nb[2] + max(0.0, ox1 - (bx + bw)), nb[3] + max(0.0, oy1 - (by + bh)))
                elif mode == "tight" and fill >= 0.6:
                    pass                                             # keep the tight box of the enlarged nut
                else:   # "scaled", or the mask covers too little of the box to trust: original box scaled about the centre
                    nb = (ncx + (ox0 - ncx) * s, ncy + (oy0 - ncy) * s, ncx + (ox1 - ncx) * s, ncy + (oy1 - ncy) * s)
        if nb is None:                                              # recover with the rectangle method
            nb = rect_paste(out, img, rects[i], rects, i, s, cfg, centre=centre)
            if nb is None:
                n_skipped += 1
                continue
            n_rect += 1
        n_scaled += 1
        new_box[i] = (nb, s)

    new_items = []
    for i, it in enumerate(items):
        it = dict(it)
        if i in new_box:
            (x0, y0, x1, y1), s = new_box[i]
            x0, y0, x1, y1 = max(0, x0) / W, max(0, y0) / H, min(W, x1) / W, min(H, y1) / H
            ncx, ncy = (x0 + x1) / 2, (y0 + y1) / 2
            if it["kind"] == "box":
                it["cx"], it["cy"], it["w"], it["h"] = ncx, ncy, x1 - x0, y1 - y0
            else:
                c = np.array([it["cx"], it["cy"]])
                it["pts"] = np.clip(c + (it["pts"] - c) * s, 0, 1)
        new_items.append(it)
    return out, new_items, n_scaled, n_skipped, n_rect, stats


def process_image(img, items, cfg, rng):
    """Old 5-value form: (new_image, new_items, n_scaled, n_skipped, n_rect_fallback)."""
    return process_image_ex(img, items, cfg, rng)[:5]


# --------------------------------------------------------------------------- #
#  data.yaml handling
# --------------------------------------------------------------------------- #
def _resolve_dir(base_dirs, p):
    """Find an existing folder for a path written in data.yaml (Roboflow often writes ../train/images)."""
    if os.path.isabs(p) and os.path.isdir(p):
        return p
    stripped = p
    while stripped.startswith(("../", "..\\", "./", ".\\")):
        stripped = stripped[3:] if stripped.startswith("..") else stripped[2:]
    for base in base_dirs:
        for cand in (p, stripped):
            full = os.path.normpath(os.path.join(base, cand))
            if os.path.isdir(full):
                return full
    return None


def _labels_dir(img_dir):
    parent, last = os.path.split(os.path.normpath(img_dir))
    if last.lower() == "images":
        return os.path.join(parent, "labels")
    return img_dir.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")


def load_dataset(yaml_path):
    """Returns dict(names={id: name}, splits={split_name: (images_dir, labels_dir)}, raw=yaml_dict)."""
    yaml_path = os.path.abspath(yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ydir = os.path.dirname(yaml_path)
    bases = [ydir]
    if raw.get("path"):
        root = raw["path"] if os.path.isabs(raw["path"]) else os.path.join(ydir, raw["path"])
        bases.insert(0, os.path.normpath(root))

    names = raw.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): str(v) for k, v in names.items()}

    splits = {}
    for key in SPLIT_KEYS:
        v = raw.get(key)
        if not v:
            continue
        if isinstance(v, list):
            v = v[0]
        img_dir = _resolve_dir(bases, str(v))
        if img_dir is None:
            continue
        lbl_dir = _labels_dir(img_dir)
        if os.path.isdir(lbl_dir):
            splits[key] = (img_dir, lbl_dir)
    if not splits:
        raise ValueError("No usable train/val/test image folders (with matching 'labels' folders) found from this data.yaml")
    return {"names": names, "splits": splits, "raw": raw}


def run_dataset(ds, split_names, out_dir, cfg, suffix, seed, log=print, progress=None, stop=lambda: False):
    """Process the chosen splits and write a new dataset (images, labels, data.yaml)."""
    rng = random.Random(seed)
    jobs = []
    for sp in split_names:
        img_dir, lbl_dir = ds["splits"][sp]
        for f in sorted(os.listdir(img_dir)):
            if f.lower().endswith(IMG_EXTS):
                jobs.append((sp, img_dir, lbl_dir, f))
    if not jobs:
        raise ValueError("No images found in the selected splits")
    tot_s = tot_k = tot_r = tot_c = tot_g = tot_d = 0
    for n, (sp, img_dir, lbl_dir, name) in enumerate(jobs, 1):
        if stop():
            log("Stopped by user.")
            break
        out_img, out_lbl = os.path.join(out_dir, sp, "images"), os.path.join(out_dir, sp, "labels")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        try:
            stem, ext = os.path.splitext(name)
            img = read_image(os.path.join(img_dir, name))
            if img is None:
                log(f"[skip] cannot read {name}")
                continue
            items = read_labels(os.path.join(lbl_dir, stem + ".txt"))
            new, new_items, ns, nk, nr, stt = process_image_ex(img, items, cfg, rng)
            tot_c += stt["capped"]
            tot_g += stt["grouped"]
            if cfg.get("drop_mixed") and nk:
                tot_d += 1
                log(f"{sp}/{name}: dropped ({nk} nut(s) had to stay at the original size)")
                if progress:
                    progress(n, len(jobs))
                continue
            write_image(os.path.join(out_img, stem + suffix + ext), new)
            with open(os.path.join(out_lbl, stem + suffix + ".txt"), "w") as f:
                f.write("\n".join(format_label(it) for it in new_items))
            tot_s += ns
            tot_k += nk
            tot_r += nr
            if nk or nr:
                log(f"{sp}/{name}: scaled {ns}, left at original size {nk}, rectangle fallback {nr}")
        except Exception as e:
            log(f"[error] {name}: {e}")
            if is_oom(e):
                free_memory()
        img = new = items = new_items = None                 # let go of the big arrays before the next image
        if _SAM["obj"] is not None or n % 25 == 0:
            free_memory()                                    # garbage + CUDA cache (every image once SAM 3 is loaded)
        if progress:
            progress(n, len(jobs))

    # new data.yaml pointing at the new folders
    new_yaml = {"path": os.path.abspath(out_dir), "names": ds["names"], "nc": len(ds["names"])}
    for sp in split_names:
        new_yaml[sp] = f"{sp}/images"
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(new_yaml, f, sort_keys=False, allow_unicode=True)
    log(f"Done. Objects scaled: {tot_s} ({tot_r} with the rectangle fallback; {tot_c} with a smaller factor to avoid "
        f"collisions; {tot_g} enlarged as a touching group), left at original size: {tot_k}"
        + (f", images dropped: {tot_d}" if tot_d else "") + f". Output: {os.path.abspath(out_dir)}")
    return tot_s, tot_k


def make_cfg(scale_x, margin, feather, max_size, classes, mask_method="auto", min_fill=0.35, class_scale=None,
             collision=True, cap_floor=0.5, drop_mixed=False, box_mode="tight"):
    if scale_x <= 0:
        raise ValueError("Scale X must be greater than 0")
    if not classes:
        raise ValueError("Select at least one class")
    class_scale = {int(k): float(v) for k, v in (class_scale or {}).items()}
    if any(v <= 0 for v in class_scale.values()):
        raise ValueError("Every class scale must be greater than 0")
    return {"scale_min": scale_x, "scale_max": scale_x, "margin": margin, "feather": feather,
            "max_size": max_size, "class_scale": class_scale,
            "classes": set(classes), "mask_method": mask_method, "min_fill": min_fill,
            "box_mode": box_mode, "collision": bool(collision), "cap_floor": float(cap_floor), "drop_mixed": bool(drop_mixed)}


# --------------------------------------------------------------------------- #
#  GUI (tkinter is imported only when the window is opened)
# --------------------------------------------------------------------------- #
def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Screw Resizer (data.yaml) - scale objects inside bboxes")
            self.geometry("780x700")
            self.stop_flag = False
            self.ds = None
            self._preview_ref = None
            self.var_yaml, self.var_out = tk.StringVar(), tk.StringVar()
            self.var_x = tk.DoubleVar(value=1.37)
            self.var_margin, self.var_feather = tk.DoubleVar(value=0.25), tk.DoubleVar(value=0.6)
            self.var_maxsize = tk.DoubleVar(value=0.0)
            self.var_seed = tk.IntVar(value=42)
            self.var_suffix = tk.StringVar(value="")
            self.var_method = tk.StringVar(value="auto")
            self.var_collide = tk.BooleanVar(value=True)
            self.var_boxmode = tk.StringVar(value="tight")
            self.var_fp16 = tk.BooleanVar(value=True)
            self.var_cls, self.var_split, self.var_cscale = {}, {}, {}
            self._build()

        def _build(self):
            pad = {"padx": 6, "pady": 3}
            f = ttk.LabelFrame(self, text="Dataset")
            f.pack(fill="x", padx=10, pady=6)
            ttk.Label(f, text="data.yaml", width=14).grid(row=0, column=0, sticky="w", **pad)
            ttk.Entry(f, textvariable=self.var_yaml, width=62).grid(row=0, column=1, **pad)
            ttk.Button(f, text="Browse", command=self._browse_yaml).grid(row=0, column=2, **pad)
            ttk.Label(f, text="Output folder", width=14).grid(row=1, column=0, sticky="w", **pad)
            ttk.Entry(f, textvariable=self.var_out, width=62).grid(row=1, column=1, **pad)
            ttk.Button(f, text="Browse", command=self._browse_out).grid(row=1, column=2, **pad)

            self.fr_split = ttk.LabelFrame(self, text="Splits to process (found in data.yaml)")
            self.fr_split.pack(fill="x", padx=10, pady=6)
            self.fr_cls = ttk.LabelFrame(self, text="Classes to enlarge (names from data.yaml)")
            self.fr_cls.pack(fill="x", padx=10, pady=6)

            p = ttk.LabelFrame(self, text="Scaling")
            p.pack(fill="x", padx=10, pady=6)
            rows = [("Scale X", self.var_x, "Used for a class whose own factor (below) is empty. X = big size / small size"),
                    ("Context margin", self.var_margin, "Extra area copied with the object (fraction of bbox)"),
                    ("Edge feather", self.var_feather, "Blend softness at the patch border (0-1)"),
                    ("Only if size <=", self.var_maxsize, "Scale only boxes whose longest side (normalised) is <= this. 0 = all"),
                    ("Random seed", self.var_seed, ""),
                    ("Output suffix", self.var_suffix, "Appended to file names, e.g. _big (optional)")]
            for r, (txt, var, tip) in enumerate(rows):
                ttk.Label(p, text=txt, width=16).grid(row=r, column=0, sticky="w", **pad)
                ttk.Entry(p, textvariable=var, width=12).grid(row=r, column=1, sticky="w", **pad)
                ttk.Label(p, text=tip, foreground="gray").grid(row=r, column=2, sticky="w", **pad)
            ttk.Label(p, text="Mask method", width=16).grid(row=len(rows) + 1, column=0, sticky="w", **pad)
            ttk.Combobox(p, textvariable=self.var_method, values=["auto", "sam3", "grabcut", "rect"], width=10, state="readonly").grid(row=len(rows) + 1, column=1, sticky="w", **pad)
            ttk.Label(p, text="auto = SAM 3, then GrabCut, then rectangle; rect = old method (whole box)", foreground="gray").grid(row=len(rows) + 1, column=2, sticky="w", **pad)

            ttk.Checkbutton(p, text="Avoid collisions (smaller factor, touching nuts enlarged together, or left unchanged)",
                            variable=self.var_collide).grid(row=len(rows) + 2, column=0, columnspan=3, sticky="w", **pad)

            ttk.Label(p, text="Box labels", width=16).grid(row=len(rows) + 3, column=0, sticky="w", **pad)
            ttk.Combobox(p, textvariable=self.var_boxmode, values=["tight", "margin", "scaled"], width=10,
                         state="readonly").grid(row=len(rows) + 3, column=1, sticky="w", **pad)
            ttk.Label(p, text="tight = outline of the enlarged nut; margin = tight + your original spacing; "
                              "scaled = your box x factor", foreground="gray").grid(row=len(rows) + 3, column=2, sticky="w", **pad)

            ttk.Checkbutton(p, text="SAM 3 in half precision (about half the GPU memory)",
                            variable=self.var_fp16).grid(row=len(rows) + 4, column=0, columnspan=3, sticky="w", **pad)

            b = ttk.Frame(self)
            b.pack(fill="x", padx=10, pady=6)
            ttk.Button(b, text="Preview random image", command=self.preview).pack(side="left", padx=4)
            self.btn_run = ttk.Button(b, text="Run on all images", command=self.run)
            self.btn_run.pack(side="left", padx=4)
            ttk.Button(b, text="Stop", command=lambda: setattr(self, "stop_flag", True)).pack(side="left", padx=4)
            self.pb = ttk.Progressbar(self, mode="determinate")
            self.pb.pack(fill="x", padx=10, pady=4)
            self.log = tk.Text(self, height=10, state="disabled")
            self.log.pack(fill="both", expand=True, padx=10, pady=6)

        def _browse_out(self):
            d = filedialog.askdirectory()
            if d:
                self.var_out.set(d)

        def _browse_yaml(self):
            p = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")])
            if not p:
                return
            self.var_yaml.set(p)
            try:
                self.ds = load_dataset(p)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                return
            if not self.var_out.get():
                self.var_out.set(os.path.join(os.path.dirname(p), "output_scaled"))
            for fr in (self.fr_split, self.fr_cls):
                for w in fr.winfo_children():
                    w.destroy()
            self.var_split = {s: tk.BooleanVar(value=True) for s in self.ds["splits"]}
            for s, v in self.var_split.items():
                n = len([x for x in os.listdir(self.ds["splits"][s][0]) if x.lower().endswith(IMG_EXTS)])
                ttk.Checkbutton(self.fr_split, text=f"{s} ({n} images)", variable=v).pack(side="left", padx=10, pady=4)
            self.var_cls = {k: tk.BooleanVar(value=True) for k in self.ds["names"]}
            self.var_cscale = {}
            for k, name in self.ds["names"].items():
                low = name.lower()                 # measured: head width x1.37, length x1.17 (small -> large nut)
                guess = "1.17" if "loose" in low else ("1.37" if ("insert" in low or "occlu" in low) else "")
                self.var_cscale[k] = tk.StringVar(value=guess)
                cell = ttk.Frame(self.fr_cls)
                cell.pack(side="left", padx=10, pady=4)
                ttk.Checkbutton(cell, text=f"{k}: {name}", variable=self.var_cls[k]).pack(anchor="w")
                row = ttk.Frame(cell)
                row.pack(anchor="w")
                ttk.Label(row, text="factor").pack(side="left")
                ttk.Entry(row, textvariable=self.var_cscale[k], width=6).pack(side="left", padx=4)
            self._log(f"Loaded {p}: {len(self.ds['splits'])} split(s), {len(self.ds['names'])} class(es)")

        def _log(self, msg):
            def w():
                self.log.configure(state="normal")
                self.log.insert("end", msg + "\n")
                self.log.see("end")
                self.log.configure(state="disabled")
            self.after(0, w)

        def _cfg(self):
            if self.ds is None:
                raise ValueError("Please choose a data.yaml first")
            cscale = {}
            for k, v in self.var_cscale.items():
                t = v.get().strip()
                if t and self.var_cls[k].get():
                    try:
                        cscale[k] = float(t)
                    except ValueError:
                        raise ValueError(f"The factor of class {k} must be a number")
            _SAM["fp16"] = bool(self.var_fp16.get())
            return make_cfg(self.var_x.get(), self.var_margin.get(), self.var_feather.get(),
                            self.var_maxsize.get(),
                            [k for k, v in self.var_cls.items() if v.get()], self.var_method.get(),
                            class_scale=cscale, collision=self.var_collide.get(),
                            box_mode=self.var_boxmode.get())

        def _splits(self):
            s = [k for k, v in self.var_split.items() if v.get()]
            if not s:
                raise ValueError("Select at least one split")
            return s

        def preview(self):
            try:
                cfg, splits = self._cfg(), self._splits()
            except Exception as e:
                messagebox.showerror("Error", str(e))
                return
            sp = random.choice(splits)
            img_dir, lbl_dir = self.ds["splits"][sp]
            files = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)]
            name = random.choice(files)
            img = read_image(os.path.join(img_dir, name))
            items = read_labels(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"))
            new, new_items, n, sk, nr = process_image(img, items, cfg, random.Random())
            free_memory()
            self._log(f"Preview {sp}/{name}: scaled {n} ({nr} rectangle fallback), left at original size {sk}")
            both = np.hstack([draw_boxes(img, items), draw_boxes(new, new_items)])
            sc = min(1.0, 1400 / both.shape[1], 750 / both.shape[0])
            both = cv2.resize(both, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            top = tk.Toplevel(self)
            top.title(f"{sp}/{name}   |   left: original   right: scaled")
            self._preview_ref = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(both, cv2.COLOR_BGR2RGB)))
            tk.Label(top, image=self._preview_ref).pack()

        def run(self):
            try:
                cfg, splits = self._cfg(), self._splits()
                out = self.var_out.get()
                if not out:
                    raise ValueError("Please choose an output folder")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                return
            self.stop_flag = False
            self.btn_run.configure(state="disabled")

            def work():
                try:
                    run_dataset(self.ds, splits, out, cfg, self.var_suffix.get(), self.var_seed.get(),
                                log=self._log,
                                progress=lambda n, t: self.after(0, lambda: self.pb.configure(maximum=t, value=n)),
                                stop=lambda: self.stop_flag)
                except Exception as e:
                    self._log(f"[error] {e}")
                self.after(0, lambda: self.btn_run.configure(state="normal"))
            threading.Thread(target=work, daemon=True).start()

    App().mainloop()


def draw_boxes(img, items):
    im = img.copy()
    H, W = im.shape[:2]
    palette = [(0, 200, 0), (0, 140, 255), (255, 80, 80), (200, 0, 200), (0, 200, 200)]
    for it in items:
        x0, y0 = int((it["cx"] - it["w"] / 2) * W), int((it["cy"] - it["h"] / 2) * H)
        x1, y1 = int((it["cx"] + it["w"] / 2) * W), int((it["cy"] + it["h"] / 2) * H)
        cv2.rectangle(im, (x0, y0), (x1, y1), palette[it["cls"] % len(palette)], 2)
    return im


def main():
    ap = argparse.ArgumentParser(description="Enlarge objects inside YOLO boxes, using a data.yaml")
    ap.add_argument("--yaml", help="path to data.yaml (without it, the window opens)")
    ap.add_argument("--out", default="output_scaled")
    ap.add_argument("--splits", nargs="*", help="e.g. train val (default: all found)")
    ap.add_argument("--classes", nargs="*", type=int, help="class ids to enlarge (default: all)")
    ap.add_argument("--scale-x", type=float, default=1.078, help="enlargement factor X = big size / small size")
    ap.add_argument("--class-scale", nargs="*", metavar="ID:FACTOR",
                    help="a factor per class in ONE pass, e.g. 0:1.37 1:1.17 2:1.37 (classes listed here are enlarged; "
                         "others stay as they are)")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--feather", type=float, default=0.6)
    ap.add_argument("--max-size", type=float, default=0.0)
    ap.add_argument("--mask-method", choices=["auto", "sam3", "grabcut", "rect"], default="auto",
                    help="auto = SAM 3, then GrabCut, then rectangle fallback; rect = whole box (old method)")
    ap.add_argument("--min-fill", type=float, default=0.35, help="a mask must fill at least this share of its box")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-collision-check", action="store_true",
                    help="enlarge every nut around its own centre, even if it runs into its neighbours")
    ap.add_argument("--cap-floor", type=float, default=0.5,
                    help="a colliding nut may be enlarged by at least this share of the intended growth "
                         "(0.5 = half of it); below that it stays at its original size")
    ap.add_argument("--box-mode", choices=["tight", "margin", "scaled"], default="tight",
                    help="label box of an enlarged nut: tight = outline of the enlarged nut (default); "
                         "margin = tight + the space your box had around the nut (not enlarged); "
                         "scaled = your original box scaled by the factor")
    ap.add_argument("--sam-fp32", action="store_true",
                    help="run SAM 3 in full precision on the GPU (default: half precision, uses about half the GPU memory)")
    ap.add_argument("--sam-chunk", type=int, default=0,
                    help="most boxes sent to SAM 3 per call (0 = all at once; it shrinks by itself when memory runs out)")
    ap.add_argument("--drop-mixed", action="store_true",
                    help="do not write an image in which a nut had to stay at its original size")
    a = ap.parse_args()
    if not a.yaml:
        run_gui()
        return
    _SAM["fp16"] = not a.sam_fp32
    _SAM["chunk"] = max(0, a.sam_chunk)
    ds = load_dataset(a.yaml)
    cscale = {}
    for tok in a.class_scale or []:
        k, _, v = tok.partition(":")
        cscale[int(k)] = float(v)
    classes = list(cscale) if cscale else (a.classes if a.classes else list(ds["names"]))
    cfg = make_cfg(a.scale_x, a.margin, a.feather, a.max_size,
                   classes, a.mask_method, a.min_fill, cscale,
                   collision=not a.no_collision_check, cap_floor=a.cap_floor, drop_mixed=a.drop_mixed,
                   box_mode=a.box_mode)
    splits = a.splits or list(ds["splits"])
    print(f"Found splits: {list(ds['splits'])}  classes: {ds['names']}")
    run_dataset(ds, splits, a.out, cfg, a.suffix, a.seed)


if __name__ == "__main__":
    main()
