"""
Screw / nut annotation assistant  (tkinter)

What it does
  1. You define the classes (default: inserted screw, loose screw, occluded; add or remove any).
  2. You give a few SAMPLES of each class: sample image files, or boxes drawn on any example image.
  3. You choose the folder with your images.
  4. For every image the app
        - finds the objects with SAM 3 (text prompt) or with a classical mask method (no AI),
        - gives every object a class by comparing it with your samples,
        - draws a bounding box in the class colour.
  5. You ACCEPT or REJECT each image. Accepted images are saved in YOLO format.
     (You can also fix a box: click it, press 1-9 to change its class, Del to delete it,
      or drag on empty space to draw a new one.)

Output folder
    images/        copies of the accepted images
    labels/        YOLO .txt files  (class cx cy w h, normalised)
    classes.txt, data.yaml
    progress.json  accepted / rejected per image (the next run continues where you stopped)
    _setup/        your classes, samples and settings (use "Load saved setup" next time)

Install
    pip install opencv-python numpy pillow
    optional, for SAM 3 and the better classifier:
    pip install torch transformers
    optional, for a base model (your trained YOLO .pt):  pip install ultralytics
    (SAM 3 weights "facebook/sam3" need you to accept the licence on Hugging Face and log in once:
     huggingface-cli login)

Run
    python annotator_gui.py
    python annotator_gui.py --images path/to/images --output path/to/output

Keys (review screen)
    A           accept        Enter   deselect the box        R   reject        S / Right   skip        B / Left   back
    Ctrl/Cmd+C  copy box (all boxes if none selected)   Ctrl/Cmd+V  paste   Ctrl/Cmd+Z  undo   Ctrl/Cmd+Shift+Z or Y  redo
    1-9         set class of the selected box / class for new boxes
    F           find objects similar to the selected box (draw a box around one object first)
    X           selected box is NOT an object: delete it and learn to ignore similar ones
    wheel / + - / 0   zoom in, zoom out, fit;  pan with right/middle button or Shift+drag
    mouse       drag a box to move it, drag its handles to resize it
    Del         delete selected box              D   detect this image again
"""
import argparse
import contextlib
import datetime
import gc
import json
import os
import queue
import shutil
import threading
import sys
import traceback
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less GPU memory fragmentation
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")      # Apple GPUs: unsupported ops fall back to the CPU

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

PREFETCH = 3          # images detected in the background while you review the current one
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
DEFAULT_CLASSES = ["inserted screw", "loose screw", "occluded"]
PALETTE = ["#2ecc71", "#ff9f1c", "#e63946", "#4cc9f0", "#c77dff",
           "#ffd60a", "#06d6a0", "#ff7bac", "#8ecae6", "#b08968"]
SETUP_DIR = "_setup"
PROGRESS_FILE = "progress.json"
METHOD_SAM = "SAM 3 (text prompt)"
METHOD_MASK = "Mask (background subtraction, no AI)"
METHOD_SAMPLES = "Sample images (image search)"


# =============================================================================
#  Basic helpers
# =============================================================================
# =============================================================================
#  Safety: error log, atomic writes, memory helpers
# =============================================================================
def _log_path():
    for d in (os.path.dirname(os.path.abspath(__file__)), os.getcwd(), os.path.expanduser("~")):
        try:
            if os.access(d, os.W_OK):
                return os.path.join(d, "annotator_errors.log")
        except Exception:
            pass
    return None


def log_error(text):
    """Append an error to annotator_errors.log (never raises)."""
    try:
        lp = _log_path()
        if lp:
            with open(lp, "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n{text}\n")
    except Exception:
        pass


def atomic_write(path, text):
    """Write a text file so that a crash in the middle can never leave a broken file behind."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


_OOM_MARKERS = ("out of memory", "cuda error: out of memory", "mps backend out of memory", "insufficient memory",
                "unable to allocate", "std::bad_alloc", "defaultcpuallocator")


def is_oom(e):
    return "OutOfMemory" in type(e).__name__ or isinstance(e, MemoryError) or any(m in str(e).lower() for m in _OOM_MARKERS)


_last_free = [0.0]


def free_memory(force=False):
    """Release cached GPU memory and Python garbage. Throttled (a cleanup after EVERY image costs time);
    out-of-memory handlers call it with force=True."""
    import time
    now = time.time()
    if not force and now - _last_free[0] < 6.0:
        return
    _last_free[0] = now
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def gpu_free_gb():
    """Free GPU memory in GB (None when there is no CUDA GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0] / 1e9
    except Exception:
        pass
    return None


def pick_device(pref="auto"):
    try:
        import torch
        mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if pref == "cuda" and torch.cuda.is_available():
            return "cuda"
        if pref == "mps" and mps:
            return "mps"
        if pref == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if mps:
            return "mps"
    except Exception:
        pass
    return "cpu"


def shrink(img, max_side):
    """Downscale so that the longest side is <= max_side. Returns (image, scale) with scale <= 1."""
    H, W = img.shape[:2]
    s = min(1.0, float(max_side) / max(H, W)) if max_side else 1.0
    if s >= 0.999:
        return img, 1.0
    return cv2.resize(img, (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_AREA), s


def read_image(path):
    """Read an image as BGR (works with non-ASCII paths and with webp)."""
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        try:
            img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            img = None
    return img


def write_image(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(path)


def default_output_for(images_dir):
    """Output folder NEXT TO the images folder (never inside it)."""
    d = os.path.abspath(images_dir.rstrip("/\\") or images_dir)
    return os.path.join(os.path.dirname(d), os.path.basename(d) + "_annotated")


def list_images(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int = 0
    conf: float = 1.0
    poly: object = None          # outline of the segmentation mask (only used for drawing)


def apply_box_drag(orig, mode, dx, dy, W, H, min_size=4.0):
    """Move ('move') or resize ('n','se',...) a box by (dx, dy) image pixels. Returns the new (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = orig
    if mode == "move":
        w, h = x2 - x1, y2 - y1
        nx1 = min(max(x1 + dx, 0), W - w)
        ny1 = min(max(y1 + dy, 0), H - h)
        return nx1, ny1, nx1 + w, ny1 + h
    if "w" in mode:
        x1 += dx
    if "e" in mode:
        x2 += dx
    if "n" in mode:
        y1 += dy
    if "s" in mode:
        y2 += dy
    x1, x2 = sorted((min(max(x1, 0), W), min(max(x2, 0), W)))     # dragging past the opposite side flips the box
    y1, y2 = sorted((min(max(y1, 0), H), min(max(y2, 0), H)))
    if x2 - x1 < min_size:
        x2 = min(W, x1 + min_size)
    if y2 - y1 < min_size:
        y2 = min(H, y1 + min_size)
    return x1, y1, x2, y2


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(cands, thr=0.6):
    """cands: list of (x1, y1, x2, y2, score)."""
    keep = []
    for c in sorted(cands, key=lambda c: -c[4]):
        if all(iou(c, k) < thr for k in keep):
            keep.append(c)
    return keep


def crop_context(img, box, margin):
    """Crop a box with `margin` (fraction of its size) of surrounding context."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    X1, Y1 = max(0, int(np.floor(x1 - mx))), max(0, int(np.floor(y1 - my)))
    X2, Y2 = min(W, int(np.ceil(x2 + mx))), min(H, int(np.ceil(y2 + my)))
    if X2 - X1 < 2 or Y2 - Y1 < 2:
        return np.zeros((8, 8, 3), np.uint8)
    return img[Y1:Y2, X1:X2]


def read_yolo_boxes(path, W, H):
    boxes = []
    if not os.path.exists(path):
        return boxes
    for line in open(path).read().strip().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        v = [float(x) for x in p[1:]]
        if len(v) == 4:
            cx, cy, w, h = v
        else:  # polygon -> box
            xs, ys = v[0::2], v[1::2]
            cx, cy, w, h = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
        boxes.append(Box((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H, int(float(p[0])), 1.0))
    return boxes


def write_yolo_boxes(path, boxes, W, H):
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = max(0, b.x1), max(0, b.y1), min(W, b.x2), min(H, b.y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{b.cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    atomic_write(path, "\n".join(lines))


def write_dataset_files(out, names):
    os.makedirs(out, exist_ok=True)
    atomic_write(os.path.join(out, "classes.txt"), "\n".join(names))
    atomic_write(os.path.join(out, "data.yaml"),
                 f"path: {json.dumps(os.path.abspath(out))}\ntrain: images\nval: images\nnc: {len(names)}\n"
                 f"names: [{', '.join(json.dumps(n) for n in names)}]\n")


# =============================================================================
#  Object finding
# =============================================================================
def detect_mask(img, sensitivity=0.5, obj_pct=20.0, min_area_pct=0.05, max_area_pct=6.0):
    """
    Classical, no AI. Estimates the background with a big median filter (it removes anything smaller
    than `obj_pct` % of the image width), subtracts it, thresholds the difference and turns the
    blobs into boxes. Shadows (darker than the background) count less than brighter or coloured objects.
    Returns [(x1, y1, x2, y2, score)] in original image pixels.
    """
    H, W = img.shape[:2]
    s = min(1.0, 1600.0 / max(H, W))
    work = cv2.resize(img, (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    Hw, Ww = work.shape[:2]

    t = min(1.0, 300.0 / max(Hw, Ww))
    small = cv2.resize(work, (max(8, int(Ww * t)), max(8, int(Hw * t))), interpolation=cv2.INTER_AREA)
    k = int(2 * obj_pct / 100.0 * small.shape[1]) | 1
    k = max(5, min(k, (int(0.9 * min(small.shape[:2])) | 1)))
    bg_small = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB), k)
    bg = cv2.resize(bg_small, (Ww, Hw), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    d = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32) - bg
    dL = np.where(d[..., 0] > 0, d[..., 0], 0.35 * d[..., 0])
    dist = np.sqrt(dL ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    dist = cv2.GaussianBlur(dist, (0, 0), max(1.0, 0.002 * max(Hw, Ww)))
    d8 = np.clip(dist, 0, 255).astype(np.uint8)

    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    floor = 12 + (1.0 - float(np.clip(sensitivity, 0, 1))) * 30
    thr = max(floor, 0.6 * otsu)
    mask = (d8 > thr).astype(np.uint8)

    ks = max(3, int(0.003 * max(Hw, Ww)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((ks, ks), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((ks * 3, ks * 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, cnts, -1, 1, -1)                       # fill holes

    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        frac = area / float(Hw * Ww) * 100
        if frac < min_area_pct or frac > max_area_pct or min(w, h) < 6 or max(w, h) / max(1, min(w, h)) > 4:
            continue
        if area / float(w * h) < 0.4:          # ragged / merged blobs (e.g. tray edges) are not single objects
            continue
        score = float(dist[lab == i].mean() / 255.0)
        cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = None
        if cs:
            c = max(cs, key=cv2.contourArea)
            poly = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2).astype(float) / s
        out.append((x / s, y / s, (x + w) / s, (y + h) / s, min(1.0, score * 2), poly))
    return out


def search_by_samples(img, groups, n_cls, cfg, sam=None):
    """IMAGE SEARCH with your sample images: every sample is slid over the image (at several sizes) and the places
    that look like it become boxes. With SAM 3 available every found box is then tightened to the object's outline.
    Returns [(x1, y1, x2, y2, score, None)] in original image pixels."""
    H, W = img.shape[:2]
    s = min(1.0, 900.0 / max(H, W))
    work = cv2.resize(img, (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    g = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    Hw, Ww = g.shape
    thr = float(cfg.get("search_thr", 0.6))
    m = float(cfg.get("margin", 0.15))
    ins = m / (1.0 + 2.0 * m)                         # samples include a context margin: the object is the inner part
    seeds = []
    for ci in range(n_cls):
        samples = groups[ci] if ci < len(groups) else []
        if not samples:
            continue
        pick = np.unique(np.linspace(0, len(samples) - 1, min(len(samples), 4)).astype(int))
        for k in pick:
            t0 = cv2.cvtColor(samples[k], cv2.COLOR_BGR2GRAY)
            for f in (0.7, 0.85, 1.0, 1.2, 1.4):
                tw, th = int(round(t0.shape[1] * s * f)), int(round(t0.shape[0] * s * f))
                if tw < 12 or th < 12 or tw >= Ww or th >= Hw:
                    continue
                tpl = cv2.resize(t0, (tw, th), interpolation=cv2.INTER_AREA)
                if float(tpl.std()) < 2.0:
                    continue
                res = cv2.matchTemplate(g, tpl, cv2.TM_CCOEFF_NORMED)
                ker = np.ones((max(3, th // 2) | 1, max(3, tw // 2) | 1), np.uint8)
                peak = (res >= thr) & (res >= cv2.dilate(res, ker) - 1e-6)
                ys, xs = np.where(peak)
                order = np.argsort(-res[ys, xs])[:200]
                for j in order:
                    y, x = int(ys[j]), int(xs[j])
                    seeds.append(((x + ins * tw) / s, (y + ins * th) / s, (x + tw - ins * tw) / s,
                                  (y + th - ins * th) / s, float(res[y, x]), None))
    seeds = nms(seeds, 0.4)[:300]
    if sam is not None and seeds:
        try:
            ref = sam.refine_boxes(img, [c[:4] for c in seeds])
            seeds = [(r[0], r[1], r[2], r[3], c[4], None) if r is not None and iou(r, c[:4]) >= 0.3 else c
                     for c, r in zip(seeds, ref)]
        except Exception as e:
            if is_oom(e):
                raise
            log_error("SAM 3 could not tighten the searched boxes:\n" + traceback.format_exc())
    return seeds


def masks_to_instances(masks, scores, min_pixels=20):
    """masks: (N, H, W) mask probabilities or booleans. Returns [(x1, y1, x2, y2, score, polygon)] where the
    box is the tight bounding box of the mask's biggest connected part (speckles are ignored)."""
    out = []
    for m, sc in zip(masks, scores):
        m = (np.asarray(m) > 0.5).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < min_pixels:
            continue
        x, y, w, h = cv2.boundingRect(c)
        eps = 0.004 * cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float)
        out.append((float(x), float(y), float(x + w), float(y + h), float(sc), poly))
    return out


class Sam3Detector:
    """SAM 3 through Hugging Face transformers, with memory protection:
    inference mode, fp16 on CUDA, cache clearing, automatic retry on smaller images after an out-of-memory error,
    and a switch to the CPU as the last resort."""

    def __init__(self, device_pref="auto", max_side=1600, half=False, share=False):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch = torch
        self.device = pick_device(device_pref)
        self.max_side = max_side
        self.amp = self.device == "cuda"
        self.notice = ""
        self.want = self.device                      # where it should run when memory allows
        self.loc = {"model": self.device}
        self.share_ok = bool(share)                  # experimental: encode the image once for several prompts
        self.half = bool(half) and self.amp          # fp16 weights: half the GPU memory, slightly lower quality
        self.model = Sam3Model.from_pretrained("facebook/sam3").eval()
        if self.half:
            self.model = self.model.half()
        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            if not is_oom(e):
                raise
            free_memory(True)                              # not enough GPU memory even for SAM 3: start on the CPU
            self.device, self.amp, self.half = "cpu", False, False
            self.model = self.model.float().to("cpu")
            self.loc = {"model": "cpu"}
            self.notice = "GPU memory was not enough: SAM 3 runs on the CPU (slower)."
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    # ---- memory protection
    def _ctx(self):
        st = contextlib.ExitStack()
        st.enter_context(self.torch.inference_mode())
        if self.amp:
            st.enter_context(self.torch.autocast("cuda", dtype=self.torch.float16))
        return st

    def _forward(self, model, **inputs):
        try:
            with self._ctx():
                return model(**inputs)
        except Exception as e:
            if self.amp and not is_oom(e):                 # fp16 trouble: continue in full precision
                self.amp = False
                self.half = False
                self.model.float()
                if hasattr(self, "tracker"):
                    self.tracker.float()
                with self._ctx():
                    return model(**inputs)
            raise

    # ---- keeping the models where they fit
    def _net(self, which):
        return self.model if which == "model" else getattr(self, "tracker", None)

    def _place(self, which):
        """Make sure the model is on the working device (it may have been parked on the CPU to free GPU memory)."""
        m = self._net(which)
        if m is not None and self.loc.get(which) != self.device:
            m.to(self.device)
            self.loc[which] = self.device

    def _offload_other(self, which):
        """Out of memory: park the model that is NOT needed right now on the CPU. True if something was freed."""
        other = "tracker" if which == "model" else "model"
        m = self._net(other)
        if m is None or self.half or self.device == "cpu" or self.loc.get(other, self.device) == "cpu":
            return False
        m.to("cpu")
        self.loc[other] = "cpu"
        free_memory(True)
        return True

    def to_cpu(self):
        self.device = "cpu"
        self.amp = False
        self.half = False
        self.model.float().to("cpu")
        if hasattr(self, "tracker"):
            self.tracker.float().to("cpu")
        self.loc = {"model": "cpu", "tracker": "cpu"}
        free_memory(True)
        self.notice = "GPU memory was not enough: SAM 3 continues on the CPU (slower). It returns to the GPU when memory is free."

    def try_gpu(self):
        """Back from the CPU to the GPU as soon as enough GPU memory is free again."""
        if self.device != "cpu" or self.want != "cuda":
            return
        free = gpu_free_gb()
        if free is not None and free > 6.0:
            self.device, self.amp = "cuda", True
            self.loc = {"model": "cpu", "tracker": "cpu"}      # moved back on demand by _place
            self.notice = "GPU memory is free again: SAM 3 is back on the GPU."

    def _guarded(self, fn, img, which="model"):
        """fn(small_image, scale) -> result. When the memory runs out: free the cache, park the other model on the
        CPU, retry on smaller images, and only as a last resort continue on the CPU."""
        last = None
        sizes = [self.max_side, int(self.max_side * 0.7), int(self.max_side * 0.5)]
        i = 0
        while i < len(sizes):
            small, sc = shrink(img, sizes[i])
            try:
                self._place(which)
                return fn(small, sc)
            except Exception as e:
                if not is_oom(e):
                    raise
                last = e
                free_memory(True)
                if self._offload_other(which):
                    continue                                   # same size again, now with more free memory
                i += 1
        if self.device != "cpu":
            self.to_cpu()
            small, sc = shrink(img, int(self.max_side * 0.7))
            return fn(small, sc)
        raise last

    # ---- results
    def _instances(self, inputs, outputs, threshold):
        res = self.processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist())[0]

        def arr(x):
            return None if x is None else np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)
        scores, masks, boxes = arr(res.get("scores")), arr(res.get("masks")), arr(res.get("boxes"))
        scores = np.zeros(0) if scores is None else scores.ravel()
        out = []
        if masks is not None and len(masks) and masks.ndim >= 3:
            # segment first, then take the box FROM the mask (tighter than the predicted box)
            out += masks_to_instances(masks.reshape(len(scores), *masks.shape[-2:]), scores)
        elif boxes is not None:
            for bx, sc in zip(boxes.reshape(-1, 4), scores):
                out.append((float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]), float(sc), None))
        return out

    @staticmethod
    def _unscale(inst, sc):
        return [(c[0] / sc, c[1] / sc, c[2] / sc, c[3] / sc, c[4], None) for c in inst]

    def detect(self, img_bgr, prompts, threshold=0.5):
        def run(img, sc):
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            out = []
            if len(prompts) > 1 and self.share_ok:
                try:                                   # the image is encoded ONCE, every prompt reuses it
                    inputs = self.processor(images=pil, text=prompts[0], return_tensors="pt").to(self.device)
                    with self._ctx():
                        emb = self.model.get_vision_features(pixel_values=inputs.pixel_values)
                    for prompt in prompts:
                        ti = self.processor(text=prompt, return_tensors="pt").to(self.device)
                        outputs = self._forward(self.model, vision_embeds=emb, **ti)
                        out += self._instances(inputs, outputs, threshold)
                        del outputs
                    return self._unscale(out, sc)
                except Exception as e:
                    if is_oom(e):
                        raise
                    self.share_ok = False              # this transformers version cannot: one pass per prompt
                    log_error("Shared image encoding not available, using one pass per prompt:\n" + traceback.format_exc())
                    out = []
            for prompt in prompts:
                inputs = self.processor(images=pil, text=prompt, return_tensors="pt").to(self.device)
                outputs = self._forward(self.model, **inputs)
                out += self._instances(inputs, outputs, threshold)
                del inputs, outputs
            return self._unscale(out, sc)
        r = nms(self._guarded(run, img_bgr), 0.5)
        free_memory()
        return r

    def detect_exemplar(self, img_bgr, box, threshold=0.5):
        """Visual prompt: 'find everything that looks like the object inside this box' (x1, y1, x2, y2)."""
        def run(img, sc):
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=pil, input_boxes=[[[float(v) * sc for v in box]]], input_boxes_labels=[[1]],
                                    return_tensors="pt").to(self.device)
            outputs = self._forward(self.model, **inputs)
            return self._unscale(self._instances(inputs, outputs, threshold), sc)
        r = nms(self._guarded(run, img_bgr), 0.5)
        free_memory()
        return r

    def _load_tracker(self):
        if not hasattr(self, "tracker"):
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
            tr = Sam3TrackerModel.from_pretrained("facebook/sam3").eval()
            if self.half:
                tr = tr.half()
            try:
                self.tracker = tr.to(self.device)
            except Exception as e:
                if not is_oom(e):
                    raise
                free_memory(True)
                self.tracker = tr.to("cpu")
                self.loc["tracker"] = "cpu"
                self._offload_other("tracker")             # park SAM 3 on the CPU, then bring the tracker over
                if self.loc.get("model") == "cpu":
                    self.tracker = tr.to(self.device)
            self.loc.setdefault("tracker", self.device)
            if hasattr(self, "tracker"):
                self.loc["tracker"] = next(self.tracker.parameters()).device.type
            self.tracker_proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

    @staticmethod
    def _mask_box(m):
        m = (np.asarray(m) > 0).astype(np.uint8)
        cs = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c) < 12:
            return None
        x, y, w, h = cv2.boundingRect(c)
        return (float(x), float(y), float(x + w), float(y + h))

    def boxes_at_point(self, img_bgr, x, y):
        """Click & segment: [(box, score)] of the masks SAM 3 finds under the point (original image pixels)."""
        self._load_tracker()

        def run(img, sc):
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = self.tracker_proc(images=pil, input_points=[[[[float(x) * sc, float(y) * sc]]]], input_labels=[[[1]]],
                                       return_tensors="pt").to(self.device)
            out = self._forward(self.tracker, multimask_output=True, **inputs)
            masks = self.tracker_proc.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"])[0]
            scores = out.iou_scores.float().cpu().numpy().reshape(-1)
            masks = masks.reshape(-1, *masks.shape[-2:]).numpy() > 0
            res = []
            for m, scr in zip(masks, scores):
                b = self._mask_box(m)
                if b:
                    res.append(((b[0] / sc, b[1] / sc, b[2] / sc, b[3] / sc), float(scr)))
            return res
        r = self._guarded(run, img_bgr, "tracker")
        free_memory()
        return r

    def refine_boxes(self, img_bgr, boxes, chunk=6):
        """Base-model mode: SAM 3 box prompts -> mask -> tight box. Returns one box (or None) per input box."""
        self._load_tracker()
        out = [None] * len(boxes)
        for i in range(0, len(boxes), chunk):
            part = boxes[i:i + chunk]

            def run(img, sc, part=part):
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                inputs = self.tracker_proc(images=pil, input_boxes=[[[float(v) * sc for v in b] for b in part]],
                                           return_tensors="pt").to(self.device)
                o = self._forward(self.tracker, multimask_output=False, **inputs)
                masks = self.tracker_proc.post_process_masks(o.pred_masks.float().cpu(), inputs["original_sizes"])[0]
                masks = masks.reshape(len(part), -1, *masks.shape[-2:])[:, 0].numpy() > 0
                res = []
                for m in masks:
                    b = self._mask_box(m)
                    res.append(None if b is None else (b[0] / sc, b[1] / sc, b[2] / sc, b[3] / sc))
                return res
            out[i:i + chunk] = self._guarded(run, img_bgr, "tracker")
            free_memory()
        return out


def _norm_name(n):
    return " ".join(str(n).lower().replace("_", " ").replace("-", " ").split())


def map_model_classes(model_names, app_names):
    """{model class id: app class index or None}. By name first; by index when no name matches and the counts agree."""
    app = {_norm_name(n): i for i, n in enumerate(app_names)}
    m = {int(k): app.get(_norm_name(v)) for k, v in dict(model_names).items()}
    if all(v is None for v in m.values()) and len(m) == len(app_names):
        m = {k: k for k in m}
    return m


class BaseModel:
    """Your trained YOLO model (.pt, needs `pip install ultralytics`): predicts boxes + classes."""

    def __init__(self, path, conf, imgsz, device_pref, app_names):
        from ultralytics import YOLO
        self.model = YOLO(path)
        self.conf, self.imgsz = conf, imgsz
        self.device = pick_device(device_pref)
        self.map = map_model_classes(self.model.names, app_names)
        self.unmapped = sorted(str(self.model.names[k]) for k, v in self.map.items() if v is None)

    def predict(self, img):
        """-> [(x1, y1, x2, y2, conf, app_class_index)]"""
        attempts = [(self.device, self.imgsz), (self.device, 640), ("cpu", 640)]
        last = None
        for dev, sz in attempts:
            try:
                kw = dict(conf=self.conf, verbose=False, device=dev if dev != "cuda" else 0)
                if sz:
                    kw["imgsz"] = int(sz)
                r = self.model.predict(img, **kw)[0]
                bx = r.boxes
                xyxy, cf, cl = bx.xyxy.cpu().numpy(), bx.conf.cpu().numpy(), bx.cls.cpu().numpy().astype(int)
                out = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c), self.map.get(int(k)))
                       for b, c, k in zip(xyxy, cf, cl)]
                if dev != self.device:
                    self.device = dev
                return out
            except Exception as e:
                if not is_oom(e):
                    raise
                last = e
                free_memory()
        raise last


def click_to_box(img, x, y, prior_wh, sam=None):
    """Box of the object under the point (x, y). SAM 3 point prompt if available, else GrabCut around the click.
    prior_wh = typical (width, height) of an object in this image, used to reject silly results."""
    H, W = img.shape[:2]
    pw, ph = prior_wh
    prior_area = pw * ph

    def ok(b):
        a = (b[2] - b[0]) * (b[3] - b[1])
        return b[0] - 2 <= x <= b[2] + 2 and b[1] - 2 <= y <= b[3] + 2 and 0.2 * prior_area <= a <= 4.0 * prior_area

    if sam is not None:
        try:
            best = None
            for b, score in sam.boxes_at_point(img, x, y):
                if ok(b):
                    a_ = (b[2] - b[0]) * (b[3] - b[1])
                    q = float(score) - abs(np.log(a_ / prior_area)) * 0.3
                    if best is None or q > best[0]:
                        best = (q, b)
            if best is not None:
                return best[1]
        except Exception:
            log_error("SAM 3 click segmentation failed:\n" + traceback.format_exc())
    # GrabCut inside a window around the click
    rw, rh = pw * 1.25, ph * 1.25
    wx0, wy0 = int(max(0, x - rw)), int(max(0, y - rh))
    wx1, wy1 = int(min(W, x + rw)), int(min(H, y + rh))
    sub = img[wy0:wy1, wx0:wx1]
    rx0, ry0 = max(0, x - rw / 2 - wx0), max(0, y - rh / 2 - wy0)
    rect = (int(rx0), int(ry0), int(min(sub.shape[1] - rx0, rw)) - 1, int(min(sub.shape[0] - ry0, rh)) - 1)
    box = None
    if rect[2] > 8 and rect[3] > 8:
        try:
            gc = np.zeros(sub.shape[:2], np.uint8)
            cv2.grabCut(sub, gc, rect, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_RECT)
            fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
            if n > 1:
                cx, cy = int(x - wx0), int(y - wy0)
                k = lab[min(max(cy, 0), lab.shape[0] - 1), min(max(cx, 0), lab.shape[1] - 1)]
                if k == 0:                                   # click on background: nearest big component to the click
                    d = [(np.hypot(st[i, 0] + st[i, 2] / 2 - cx, st[i, 1] + st[i, 3] / 2 - cy), i) for i in range(1, n) if st[i, 4] > 30]
                    k = min(d)[1] if d else 0
                if k:
                    bx, by, bw, bh = st[k, :4]
                    cand = (float(bx + wx0), float(by + wy0), float(bx + bw + wx0), float(by + bh + wy0))
                    if ok(cand):
                        box = cand
        except Exception:
            box = None
    cx0, cy0 = x, y
    if box is not None:
        ratio = (box[2] - box[0]) * (box[3] - box[1]) / prior_area
        if 0.7 <= ratio <= 1.6:
            return box
        cx0, cy0 = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2    # partial outline: keep its centre, use the typical size
    x1 = min(max(0.0, cx0 - pw / 2), max(0.0, W - pw))
    y1 = min(max(0.0, cy0 - ph / 2), max(0.0, H - ph))
    return (x1, y1, min(float(W), x1 + pw), min(float(H), y1 + ph))


def find_similar(img, exemplar_crop, ex_box, cands, clf, cfg, thr):
    """Keep the candidates that look like the exemplar (embedding similarity) and have a similar size."""
    H, W = img.shape[:2]
    ex_area = max(1.0, (ex_box[2] - ex_box[0]) * (ex_box[3] - ex_box[1]))
    keep = []
    for c in cands:
        ar = (c[2] - c[0]) * (c[3] - c[1]) / ex_area
        if 0.35 <= ar <= 2.8:
            keep.append(c)
    keep = nms(keep, 0.5)
    if not keep:
        return []
    S = clf.embedder.embed(sample_variants(exemplar_crop, cfg.get("rotate", False)))
    X = clf.embedder.embed([crop_context(img, c, cfg["margin"]) for c in keep])
    sims = (X @ S.T).max(axis=1)
    return [(c[0], c[1], c[2], c[3], float(sm), c[5] if len(c) > 5 else None) for c, sm in zip(keep, sims) if sm >= thr]


# =============================================================================
#  Class assignment from samples (few-shot)
# =============================================================================
class SimpleEmbedder:
    """No downloads: colour histogram + tiny grayscale thumbnail + HOG shape descriptor."""
    name = "simple"

    @staticmethod
    def _unit(v):
        return v / (np.linalg.norm(v) + 1e-9)

    @staticmethod
    def _hog(g, cells=4, bins=9):
        """Small HOG: gradient-orientation histograms on a cells x cells grid (no cv2.HOGDescriptor needed)."""
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins
        b = np.minimum(ang.astype(int), bins - 1)
        h, w = g.shape
        out = []
        for i in range(cells):
            for j in range(cells):
                sl = (slice(i * h // cells, (i + 1) * h // cells), slice(j * w // cells, (j + 1) * w // cells))
                out.append(np.bincount(b[sl].ravel(), weights=mag[sl].ravel(), minlength=bins))
        v = np.concatenate(out).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def _feat(self, bgr):
        im = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        hist = np.sqrt(hist / (hist.sum() + 1e-9))
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        th = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).flatten()
        th = (th - th.mean()) / (th.std() + 1e-6)
        hog = self._hog(g)
        return self._unit(np.concatenate([self._unit(hist), 0.7 * self._unit(th), self._unit(hog)]))

    def embed(self, bgr_list):
        return np.stack([self._feat(b) for b in bgr_list]).astype(np.float32)


class DinoEmbedder:
    """DINOv2-small image embeddings (needs torch + transformers, downloads ~90 MB once). Memory-safe: the batch
    is halved after an out-of-memory error, and the model moves to the CPU as the last resort."""
    name = "dinov2"

    def __init__(self, device_pref="auto"):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.device = pick_device(device_pref)
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(self.device).eval()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def embed(self, bgr_list):
        outs, i, bs = [], 0, 16
        while i < len(bgr_list):
            try:
                batch = []
                for b in bgr_list[i:i + bs]:
                    im = cv2.cvtColor(cv2.resize(b, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                    batch.append(((im.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1))
                x = self.torch.from_numpy(np.stack(batch)).to(self.device)
                with self.torch.inference_mode():
                    o = self.model(pixel_values=x)
                f = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
                f = f.float().cpu().numpy()
                outs.append(f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9))
                i += bs
            except Exception as e:
                if not is_oom(e):
                    raise
                free_memory()
                if bs > 1:
                    bs = max(1, bs // 2)
                elif self.device != "cpu":
                    self.device = "cpu"
                    self.model.to("cpu")
                else:
                    raise
        return np.vstack(outs).astype(np.float32)


def make_embedder(backend, say, device="auto"):
    if backend in ("auto", "dinov2"):
        try:
            say("Loading DINOv2 classifier (first time it downloads about 90 MB)...")
            return DinoEmbedder(device)
        except Exception as e:
            if backend == "dinov2":
                raise
            say(f"DINOv2 not available ({type(e).__name__}); using the simple feature classifier")
    return SimpleEmbedder()


def sample_variants(s, rotate=False):
    """Mirror images of a sample (and, with rotate=True, 90-degree rotations too)."""
    out = []
    for k in (range(4) if rotate else range(1)):
        r = np.ascontiguousarray(np.rot90(s, k))
        out += [r, np.ascontiguousarray(r[:, ::-1])]
    return out


class Classifier:
    """Class assignment on image embeddings.
    Few samples  -> nearest neighbour (closest sample of each class).
    Enough samples (>= MIN_FIT per class) -> softmax regression trained on all samples. It can start from a
    classifier pre-trained on your labelled dataset (train_classifier.py) and keeps improving as you accept images:
    every verified box is added as a new sample, weighted more than the pre-trained ones."""
    MIN_FIT = 5
    MAX_PER_CLASS = 1500
    USER_WEIGHT = 3.0

    def __init__(self, samples_by_class, backend, say, rotate=False, base_path="", class_names=None, device="auto"):
        self.embedder = make_embedder(backend, say, device)
        self.rotate = rotate
        self.S = None
        self.labels = np.zeros(0, int)
        self.groups = np.zeros(0, int)
        self.rw = np.zeros(0)
        self.next_group = 0
        self.W = self.b = None
        self.dirty = False
        if base_path:
            try:
                self._load_base(base_path, class_names or [], say)
            except Exception as e:
                say(f"Pre-trained classifier not used: {e}")
        for ci, samples in enumerate(samples_by_class):
            for s in samples:
                self.add(ci, s)
        self.dirty = True

    def _load_base(self, path, class_names, say):
        d = np.load(path, allow_pickle=True)
        names = [str(n) for n in d["names"]]
        if str(d["embedder"]) != self.embedder.name:
            raise ValueError(f"it was trained with '{d['embedder']}' but '{self.embedder.name}' is running "
                             "(set Class assignment to match)")
        if [n.lower() for n in class_names[:len(names)]] != [n.lower() for n in names]:
            raise ValueError(f"class names differ (file: {names}; app: {class_names})")
        S = d["S"].astype(np.float32)
        self.S, self.labels = S, d["labels"].astype(int)
        self.groups = np.arange(len(S))
        self.next_group = len(S)
        self.rw = np.ones(len(S))
        say(f"Pre-trained classifier loaded ({len(S)} samples).")

    def add(self, ci, crop, weight=None):
        """Teach one more sample of class `ci` (index len(classes) = ignore)."""
        v = self.embedder.embed(sample_variants(crop, self.rotate))
        w = self.USER_WEIGHT if weight is None else weight
        self.S = v if self.S is None else np.vstack([self.S, v])
        self.labels = np.concatenate([self.labels, np.full(len(v), ci, int)])
        self.groups = np.concatenate([self.groups, np.full(len(v), self.next_group, int)])
        self.rw = np.concatenate([self.rw, np.full(len(v), w)])
        self.next_group += 1
        gs = np.unique(self.groups[self.labels == ci])
        if len(gs) > self.MAX_PER_CLASS:                          # forget the oldest sample of that class
            drop = (self.groups == gs[0])
            self.S, self.labels, self.groups, self.rw = self.S[~drop], self.labels[~drop], self.groups[~drop], self.rw[~drop]
        self.dirty = True

    def fit(self):
        self.W = self.b = None
        self.dirty = False
        if self.S is None:
            return
        ids = sorted(set(self.labels.tolist()))
        if len(ids) < 2 or any(len(np.unique(self.groups[self.labels == c])) < self.MIN_FIT for c in ids):
            return
        y = np.searchsorted(ids, self.labels)
        n, d = self.S.shape
        K = len(ids)
        Y = np.eye(K)[y]
        wsum = np.bincount(y, weights=self.rw, minlength=K)
        sw = (self.rw * (self.rw.sum() / (K * wsum))[y])[:, None]      # every class counts equally
        W, b = np.zeros((d, K)), np.zeros(K)
        X = self.S * 6.0
        for _ in range(400):
            z = X @ W + b
            z -= z.max(axis=1, keepdims=True)
            P = np.exp(z)
            P /= P.sum(axis=1, keepdims=True)
            G = (P - Y) * sw / n
            W -= 4.0 * (X.T @ G + 1e-3 * W)
            b -= 4.0 * G.sum(axis=0)
        self.W, self.b, self.ids = W, b, ids

    def predict(self, crops):
        """Returns [(class_index, confidence 0..1)] for every crop."""
        if self.S is None or not crops:
            return [(0, 0.0)] * len(crops)
        return self.predict_emb(self.embedder.embed(crops))

    def predict_emb(self, Xq):
        if self.S is None or len(Xq) == 0:
            return [(0, 0.0)] * len(Xq)
        if self.dirty:
            self.fit()
        if self.W is not None:
            z = Xq * 6.0 @ self.W + self.b
            z -= z.max(axis=1, keepdims=True)
            P = np.exp(z)
            P /= P.sum(axis=1, keepdims=True)
            return [(self.ids[int(np.argmax(p))], float(p.max())) for p in P]
        sims = Xq @ self.S.T
        ids = sorted(set(self.labels.tolist()))
        per = np.stack([sims[:, self.labels == c].max(axis=1) for c in ids], axis=1)
        if len(ids) == 1:
            return [(ids[0], float(np.clip(p[0], 0, 1))) for p in per]
        z = np.exp((per - per.max(axis=1, keepdims=True)) * 30.0)
        prob = z / z.sum(axis=1, keepdims=True)
        return [(ids[int(np.argmax(p))], float(p.max())) for p in prob]


# =============================================================================
#  Background worker (never touches tkinter)
# =============================================================================
def worker_main(gen, cfg, samples_by_class, task_q, res_q, learn_q):
    """Background thread: never touches tkinter. Every task is protected: an error in one image is reported
    and the worker simply continues with the next one."""
    def say(msg):
        res_q.put(("status", gen, msg))

    warned = set()

    def warn(key, text):
        if key not in warned:
            warned.add(key)
            res_q.put(("warn", gen, text))

    st = dict(clf=None, sam=None, base=None, sam_failed=False, base_failed=False, sam_errors=0)
    cand_cache = {}                                # path -> detected objects before classification (reused when you teach new samples)
    buffer = []                                    # learned samples that arrived before the classifier existed
    n_cls = cfg["n_classes"]

    def drain_learn():
        while True:
            try:
                ci, crop = learn_q.get_nowait()
            except queue.Empty:
                return
            if st["clf"] is not None:
                st["clf"].add(ci, crop)
            else:
                buffer.append((ci, crop))

    def get_clf():
        if st["clf"] is None:
            st["clf"] = Classifier(samples_by_class, cfg["backend"], say, cfg.get("rotate", False),
                                   cfg.get("base", ""), cfg.get("class_names", []), cfg.get("device", "auto"))
            say(f"Classifier ready ({st['clf'].embedder.name}).")
            for ci, crop in buffer:
                st["clf"].add(ci, crop)
            buffer.clear()
        return st["clf"]

    def get_sam():
        if st["sam"] is None and not st["sam_failed"]:
            try:
                say("Loading SAM 3...")
                dev = cfg.get("device", "auto")
                free_gb = gpu_free_gb()
                if free_gb is not None and pick_device(dev) == "cuda":
                    if free_gb < 2.0:
                        warn("gpu", f"Only {free_gb:.1f} GB of GPU memory is free (other programs are using it).\n\n"
                                    "SAM 3 will run on the CPU (slower). To use the GPU: close other programs that use it "
                                    "(check with 'nvidia-smi' in a terminal), then start this app again.")
                        dev = "cpu"
                    elif free_gb < 4.0:
                        warn("gpu", f"Only {free_gb:.1f} GB of GPU memory is free. SAM 3 may be slow. Closing other "
                                    "programs that use the GPU ('nvidia-smi' shows them) will help.")
                st["sam"] = Sam3Detector(dev, cfg.get("max_side", 1600), cfg.get("half", False), cfg.get("share", False))
                say(f"SAM 3 ready on {st['sam'].device}.")
            except Exception as e:
                st["sam_failed"] = True
                log_error("SAM 3 could not be loaded:\n" + traceback.format_exc())
                warn("sam", f"SAM 3 could not be loaded ({type(e).__name__}: {e}).\n\n"
                            "The classical mask method is used instead. Install torch + transformers and log in to "
                            "Hugging Face to use SAM 3.")
        return st["sam"]

    def get_base():
        if cfg.get("base_model") and st["base"] is None and not st["base_failed"]:
            try:
                say("Loading the base model...")
                st["base"] = BaseModel(cfg["base_model"], cfg.get("base_conf", 0.25), cfg.get("base_imgsz", 0),
                                       cfg.get("device", "auto"), cfg.get("class_names", []))
                if st["base"].unmapped:
                    warn("unmapped", "These classes of the base model do not exist in the app and are skipped:\n  - "
                                     + "\n  - ".join(st["base"].unmapped))
            except Exception as e:
                st["base_failed"] = True
                log_error("Base model could not be loaded:\n" + traceback.format_exc())
                warn("base", f"The base model could not be loaded ({type(e).__name__}: {e}).\n\n"
                             "Detection continues without it. (YOLO models need: pip install ultralytics)")
        return st["base"]

    def sam_notice():
        if st["sam"] is not None and st["sam"].notice:
            say(st["sam"].notice)
            st["sam"].notice = ""

    def sam_failed_once(e):
        """A SAM 3 error that is not about memory: after 3 in a row the classical method takes over."""
        st["sam_errors"] += 1
        log_error("SAM 3 error:\n" + traceback.format_exc())
        if st["sam_errors"] >= 3:
            st["sam"], st["sam_failed"] = None, True
            free_memory(True)
            warn("sam3x", f"SAM 3 failed several times ({type(e).__name__}: {e}).\nThe classical mask method is used "
                          "from now on. Details are in annotator_errors.log.")

    while True:
        item = task_q.get()
        if item is None:
            return
        sim = item if isinstance(item, tuple) else None
        path = sim[1] if sim else item
        reclass = bool(sim and sim[0] == "reclass")
        if reclass:
            sim = None
        try:
            img = read_image(path)
            if img is None:
                raise IOError(f"cannot read {path}")
            H, W = img.shape[:2]
            method = cfg["method"]
            if method == METHOD_SAM and get_sam() is None:
                method = METHOD_MASK
            if method == METHOD_SAMPLES:
                get_sam()                                  # optional: only used to tighten the found boxes
            sam = st["sam"] if method in (METHOD_SAM, METHOD_SAMPLES) else None
            if sam is not None:
                sam.try_gpu()
            drain_learn()

            if sim is not None and sim[0] == "recls":     # ("recls", path, [(i, x1, y1, x2, y2)])
                clf = get_clf()
                drain_learn()
                items = sim[2]
                preds = clf.predict([crop_context(img, it[1:5], cfg["margin"]) for it in items])
                res_q.put(("recls", gen, path, [(it[0], it[1], it[2], it[3], it[4], cl, cf)
                                                for it, (cl, cf) in zip(items, preds)]))
                continue

            if sim is not None and sim[0] == "click":     # ("click", path, x, y, prior_w, prior_h, active_cls)
                _, _, cx, cy, pw, ph, active = sim
                say("Segmenting the object under your click...")
                bx = click_to_box(img, cx, cy, (pw, ph), sam)
                sam_notice()
                clf = get_clf()
                drain_learn()
                cl, conf = clf.predict([crop_context(img, bx, cfg["margin"])])[0]
                if cl >= n_cls:
                    cl = active
                res_q.put(("click_result", gen, path, Box(bx[0], bx[1], bx[2], bx[3], cl, 1.0)))
                continue

            if sim is not None:                       # ("similar", path, exemplar_crop, ex_box, cls, thr)
                _, _, ex_crop, ex_box, ex_cls, thr = sim
                say("Searching for similar objects...")
                clf = get_clf()
                drain_learn()
                cands = []
                if sam is not None:
                    try:
                        cands += sam.detect_exemplar(img, ex_box, cfg["sam_thr"])
                    except Exception as e:
                        if is_oom(e):
                            raise
                        log_error("SAM 3 visual prompt failed:\n" + traceback.format_exc())
                        say(f"SAM 3 visual prompt failed ({type(e).__name__}); using the text prompt")
                    prompts = [p.strip() for p in cfg["prompt"].split(",") if p.strip()] or ["screw"]
                    cands += sam.detect(img, prompts, max(0.2, cfg["sam_thr"] - 0.2))
                    sam_notice()
                else:
                    cands += detect_mask(img, cfg["sens"], cfg["obj_pct"], 0.005, 30.0)
                found = find_similar(img, ex_crop, ex_box, cands, clf, cfg, thr)
                res_q.put(("similar", gen, path, [Box(c[0], c[1], c[2], c[3], ex_cls, c[4], c[5]) for c in found]))
                continue

            where = f" ({st['sam'].device})" if st["sam"] is not None else ""
            say(f"Re-checking classes: {os.path.basename(path)}" if reclass and path in cand_cache
                else f"Detecting{where}: {os.path.basename(path)}")
            base = get_base()
            if base is not None:
                # 1) your model predicts boxes + classes   2) SAM 3 tightens every box   3) the class stays as predicted
                dets = base.predict(img)
                kept = [d for d in dets if d[5] is not None]
                if len(kept) < len(dets):
                    warn("unmapped2", "Some predicted classes do not exist in the app and were skipped.")
                boxes_xyxy = [d[:4] for d in kept]
                refined = [None] * len(kept)
                if sam is not None and kept:
                    try:
                        refined = sam.refine_boxes(img, boxes_xyxy)
                        st["sam_errors"] = 0
                    except Exception as e:
                        if is_oom(e):
                            raise
                        sam_failed_once(e)
                    sam_notice()
                boxes = []
                for d, r in zip(kept, refined):
                    x1, y1, x2, y2 = d[:4]
                    if r is not None and iou(r, d[:4]) >= 0.3:
                        x1, y1, x2, y2 = r
                    boxes.append(Box(x1, y1, x2, y2, d[5], d[4]))
                if boxes and (cfg.get("n_ignore", 0) > 0 or any(ci >= n_cls for ci, _ in buffer) or
                              (st["clf"] is not None and st["clf"].S is not None and (st["clf"].labels >= n_cls).any())):
                    clf = get_clf()
                    drain_learn()
                    preds = clf.predict([crop_context(img, (b.x1, b.y1, b.x2, b.y2), cfg["margin"]) for b in boxes])
                    boxes = [b for b, (cl, cf) in zip(boxes, preds) if not (cl >= n_cls and cf > 0.6)]
                boxes.sort(key=lambda b: (b.y1, b.x1))
                res_q.put(("result", gen, path, boxes))
                continue

            import time as _t
            _t0 = _t.time()
            clf = get_clf()
            drain_learn()
            _t1 = _t.time()
            cands = cand_cache.get(path) if reclass else None
            if cands is None:
                if method == METHOD_SAMPLES:
                    say("Searching the image for your samples...")
                    cands = search_by_samples(img, samples_by_class, n_cls, cfg, sam)
                    sam_notice()
                elif sam is not None:
                    prompts = [p.strip() for p in cfg["prompt"].split(",") if p.strip()] or ["screw"]
                    try:
                        cands = sam.detect(img, prompts, cfg["sam_thr"])
                        st["sam_errors"] = 0
                    except Exception as e:
                        if is_oom(e):
                            raise
                        sam_failed_once(e)
                        cands = detect_mask(img, cfg["sens"], cfg["obj_pct"], cfg["min_pct"], cfg["max_pct"])
                    sam_notice()
                else:
                    cands = detect_mask(img, cfg["sens"], cfg["obj_pct"], cfg["min_pct"], cfg["max_pct"])
                cands = [c for c in cands
                         if cfg["min_pct"] <= (c[2] - c[0]) * (c[3] - c[1]) / float(W * H) * 100 <= cfg["max_pct"]]
                cands = nms(cands, 0.6)
                cand_cache[path] = cands
                while len(cand_cache) > 12:
                    cand_cache.pop(next(iter(cand_cache)))
            _t2 = _t.time()
            crops = [crop_context(img, c, cfg["margin"]) for c in cands]
            preds = clf.predict(crops)
            _t3 = _t.time()
            res_q.put(("timing", gen, path, f"load classifier {_t1 - _t0:.1f}s | find objects {_t2 - _t1:.1f}s "
                       f"({len(cands)} found, {'SAM 3 on ' + sam.device if sam is not None else 'mask method'}) | "
                       f"classes {_t3 - _t2:.1f}s | image {W}x{H}"))
            boxes = [Box(c[0], c[1], c[2], c[3], cl, conf, c[5] if len(c) > 5 else None)
                     for c, (cl, conf) in zip(cands, preds)
                     if cl < n_cls]                      # objects closest to an "ignore" sample are dropped
            boxes.sort(key=lambda b: (b.y1, b.x1))
            res_q.put(("result", gen, path, boxes))
        except Exception as e:
            text = traceback.format_exc()
            log_error(text)
            if is_oom(e):
                text = ("Out of memory. Lower 'Max image side' in Setup, choose Device = cpu, or close other programs.\n"
                        + text)
            res_q.put(("similar_error" if sim is not None else "error", gen, path, text))
        finally:
            free_memory()


# =============================================================================
#  Sample picker window (drag boxes on an example image)
# =============================================================================
class SamplePicker(tk.Toplevel):
    def __init__(self, app, img, class_idx):
        super().__init__(app)
        self.app, self.img = app, img
        self.title("Pick samples: drag a box around each object")
        self.geometry("1000x760")
        self.added = []          # (class_idx, rect_item)
        self.drag = None
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Label(bar, text="Class of the boxes you draw:").pack(side="left")
        self.var_cls = tk.StringVar()
        self.cb = ttk.Combobox(bar, textvariable=self.var_cls, state="readonly", width=28,
                               values=[c["name"] for c in app.classes] + ["(IGNORE - do not label)"])
        self.cb.current(min(class_idx, len(app.classes)))
        self.cb.pack(side="left", padx=6)
        self.lbl = ttk.Label(bar, text="0 samples added")
        self.lbl.pack(side="left", padx=12)
        ttk.Button(bar, text="Done", command=self.close).pack(side="right")
        ttk.Button(bar, text="Undo last", command=self.undo).pack(side="right", padx=6)
        self.canvas = tk.Canvas(self, bg="#222222", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.motion)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.photo = None
        self.after(100, self.redraw)

    def redraw(self):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            return
        H, W = self.img.shape[:2]
        self.scale = min(cw / W, ch / H)
        nw, nh = max(1, int(W * self.scale)), max(1, int(H * self.scale))
        self.ox, self.oy = (cw - nw) // 2, (ch - nh) // 2
        small = cv2.resize(self.img, (nw, nh), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(self.ox, self.oy, anchor="nw", image=self.photo)
        self.added = [(ci, None) for ci, _ in self.added]       # rectangles are redrawn only for new drags

    def to_img(self, x, y):
        H, W = self.img.shape[:2]
        return (min(max((x - self.ox) / self.scale, 0), W), min(max((y - self.oy) / self.scale, 0), H))

    def press(self, e):
        self.drag = (e.x, e.y)

    def motion(self, e):
        if self.drag:
            self.canvas.delete("tmp")
            self.canvas.create_rectangle(self.drag[0], self.drag[1], e.x, e.y, outline="white", dash=(4, 3), tags="tmp")

    def release(self, e):
        if not self.drag:
            return
        x0, y0 = self.drag
        self.drag = None
        self.canvas.delete("tmp")
        if abs(e.x - x0) < 8 or abs(e.y - y0) < 8:
            return
        ix0, iy0 = self.to_img(min(x0, e.x), min(y0, e.y))
        ix1, iy1 = self.to_img(max(x0, e.x), max(y0, e.y))
        ci = self.cb.current()
        crop = crop_context(self.img, (ix0, iy0, ix1, iy1), self.app.get_settings()["margin"])
        self.app.samples_of(ci).append(crop.copy())
        rect = self.canvas.create_rectangle(min(x0, e.x), min(y0, e.y), max(x0, e.x), max(y0, e.y),
                                            outline=("#888888" if ci >= len(self.app.classes) else PALETTE[ci % len(PALETTE)]), width=3)
        self.added.append((ci, rect))
        self.lbl.config(text=f"{len(self.added)} samples added")

    def undo(self):
        if not self.added:
            return
        ci, rect = self.added.pop()
        if self.app.samples_of(ci):
            self.app.samples_of(ci).pop()
        if rect is not None:
            self.canvas.delete(rect)
        self.lbl.config(text=f"{len(self.added)} samples added")

    def close(self):
        self.app.refresh_all_setup()
        self.destroy()


class DataPicker(SamplePicker):
    """Pick class samples straight from the images of the data folder.
    Left: list of the images. Choose an image, choose the class, drag a box around an object -> the crop is added
    to that class at once. Boxes are EDITABLE: click a box to select it, drag it to move, drag its handles to
    resize (the sample is re-cut), 1-9 or the class list change its class, Delete removes it (and the sample).
    Ctrl/Cmd+Z or 'Undo last' undoes the last action."""
    HANDLE = 7

    def __init__(self, app, folder, class_idx):
        self.files = list_images(folder)
        super().__init__(app, np.zeros((10, 10, 3), np.uint8), class_idx)
        self.title("Select samples from the data: choose an image, choose the class, draw a box")
        self.geometry("1250x780")
        self.cur = None
        self.recs = {}                  # path -> [record]; record = {ci, box (x0, y0, x1, y1), crop}
        self.history = []
        self.sel = None
        self.mode = None
        self.start = None
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=(8, 0), pady=4, before=self.canvas)
        ttk.Label(left, text=f"Images ({len(self.files)})").pack(anchor="w")
        fr = ttk.Frame(left)
        fr.pack(fill="y", expand=True)
        self.lst = tk.Listbox(fr, width=34, exportselection=False, activestyle="none")
        sb = ttk.Scrollbar(fr, orient="vertical", command=self.lst.yview)
        self.lst.configure(yscrollcommand=sb.set)
        self.lst.pack(side="left", fill="y", expand=True)
        sb.pack(side="left", fill="y")
        self.lst.bind("<<ListboxSelect>>", lambda e: self.on_pick())
        ttk.Label(left, text="Drag on empty space: new box.\nClick a box: select / move / resize.\n"
                             "1-9 or the class list: change class.\nDelete: remove.  Ctrl+Z: undo.\n"
                             "A tick marks images already used.", foreground=MUTED, justify="left").pack(anchor="w", pady=6)
        for f in self.files:
            self.lst.insert("end", "   " + os.path.basename(f))
        for k in ("<Control-z>", "<Command-z>"):
            self.bind(k, lambda e: self.undo())
        self.bind("<Key>", self.on_key)
        self.cb.bind("<<ComboboxSelected>>", lambda e: self.on_combo())
        self.update_label()
        if self.files:
            self.lst.selection_set(0)
            self.on_pick()

    # ---- helpers
    def margin(self):
        return self.app.get_settings()["margin"]

    def col(self, ci):
        return "#888888" if ci >= len(self.app.classes) else PALETTE[ci % len(PALETTE)]

    def make_crop(self, box):
        return crop_context(self.img, box, self.margin()).copy()

    def drop_sample(self, ci, crop):
        lst = self.app.samples_of(ci)
        for i, c in enumerate(lst):
            if c is crop:
                del lst[i]
                return

    def update_label(self):
        parts = [f"{c['name']}: {len(c['samples'])}" for c in self.app.classes]
        self.lbl.config(text="Samples  " + "   ".join(parts))

    def mark(self, path):
        i = self.files.index(path)
        used = bool(self.recs.get(path))
        self.lst.delete(i)
        self.lst.insert(i, ("\u2713  " if used else "   ") + os.path.basename(path))
        self.lst.selection_clear(0, "end")
        self.lst.selection_set(i)

    def on_pick(self):
        sel = self.lst.curselection()
        if not sel:
            return
        path = self.files[sel[0]]
        im = read_image(path)
        if im is None:
            messagebox.showerror("Error", f"Cannot read {os.path.basename(path)}", parent=self)
            return
        self.cur, self.img, self.sel = path, im, None
        self.redraw()

    def cur_recs(self):
        return self.recs.setdefault(self.cur, []) if self.cur else []

    # ---- drawing
    def redraw(self):
        super().redraw()
        if getattr(self, "cur", None) is None or not self.canvas.winfo_exists():
            return
        for k, r in enumerate(self.recs.get(self.cur, [])):
            x0, y0, x1, y1 = r["box"]
            c0, d0 = self.ox + x0 * self.scale, self.oy + y0 * self.scale
            c1, d1 = self.ox + x1 * self.scale, self.oy + y1 * self.scale
            self.canvas.create_rectangle(c0, d0, c1, d1, outline=self.col(r["ci"]), width=2)
            if k == self.sel:
                h = 4
                for hx, hy in ((c0, d0), ((c0 + c1) / 2, d0), (c1, d0), (c0, (d0 + d1) / 2), (c1, (d0 + d1) / 2),
                               (c0, d1), ((c0 + c1) / 2, d1), (c1, d1)):
                    self.canvas.create_rectangle(hx - h, hy - h, hx + h, hy + h, fill="white", outline="#222222")

    def hit(self, x, y):
        """-> (record index, mode) under the canvas point, or (None, None). mode: 'move' or a handle like 'nw'."""
        recs = self.recs.get(self.cur, [])
        order = ([self.sel] if self.sel is not None and self.sel < len(recs) else []) + \
                [k for k in reversed(range(len(recs))) if k != self.sel]
        t = self.HANDLE
        for k in order:
            x0, y0, x1, y1 = recs[k]["box"]
            c0, d0 = self.ox + x0 * self.scale, self.oy + y0 * self.scale
            c1, d1 = self.ox + x1 * self.scale, self.oy + y1 * self.scale
            if not (c0 - t <= x <= c1 + t and d0 - t <= y <= d1 + t):
                continue
            m = ("n" if abs(y - d0) <= t else "s" if abs(y - d1) <= t else "") + \
                ("w" if abs(x - c0) <= t else "e" if abs(x - c1) <= t else "")
            if k == self.sel and m:
                return k, m
            if c0 <= x <= c1 and d0 <= y <= d1:
                return k, "move"
        return None, None

    # ---- mouse
    def press(self, e):
        self.focus_set()
        k, mode = self.hit(e.x, e.y)
        if k is not None:
            self.sel, self.mode = k, mode
            r = self.recs[self.cur][k]
            self.start = (e.x, e.y, tuple(r["box"]))
            self.cb.current(min(r["ci"], len(self.app.classes)))
            self.redraw()
            return
        self.sel, self.mode = None, "new"
        self.drag = (e.x, e.y)
        self.redraw()

    def motion(self, e):
        if self.mode == "new" and self.drag:
            self.canvas.delete("tmp")
            self.canvas.create_rectangle(self.drag[0], self.drag[1], e.x, e.y, outline="white", dash=(4, 3), tags="tmp")
        elif self.mode and self.sel is not None and self.start:
            H, W = self.img.shape[:2]
            sx, sy, ob = self.start
            nb = apply_box_drag(ob, self.mode, (e.x - sx) / self.scale, (e.y - sy) / self.scale, W, H)
            self.recs[self.cur][self.sel]["box"] = tuple(nb)
            self.redraw()

    def release(self, e):
        mode, self.mode = self.mode, None
        if self.cur is None:
            return
        if mode == "new":
            x0, y0 = self.drag if self.drag else (e.x, e.y)
            self.drag = None
            self.canvas.delete("tmp")
            if abs(e.x - x0) < 8 or abs(e.y - y0) < 8:
                return
            ix0, iy0 = self.to_img(min(x0, e.x), min(y0, e.y))
            ix1, iy1 = self.to_img(max(x0, e.x), max(y0, e.y))
            ci = self.cb.current()
            try:
                crop = self.make_crop((ix0, iy0, ix1, iy1))
            except Exception as ex:
                log_error(f"DataPicker.release: {ex}")
                return
            self.app.samples_of(ci).append(crop)
            rec = {"ci": ci, "box": (ix0, iy0, ix1, iy1), "crop": crop}
            self.cur_recs().append(rec)
            self.history.append(("add", self.cur, rec))
            self.sel = len(self.recs[self.cur]) - 1
        elif mode and self.sel is not None and self.start:
            rec = self.recs[self.cur][self.sel]
            old_box = self.start[2]
            self.start = None
            if tuple(rec["box"]) == tuple(old_box):
                return
            old_crop = rec["crop"]
            try:
                new_crop = self.make_crop(rec["box"])
            except Exception as ex:
                log_error(f"DataPicker.release: {ex}")
                rec["box"] = old_box
                self.redraw()
                return
            self.replace_crop(rec, new_crop)
            self.history.append(("edit", self.cur, rec, old_box, old_crop))
        self.mark(self.cur)
        self.redraw()
        self.update_label()

    def replace_crop(self, rec, new_crop):
        lst = self.app.samples_of(rec["ci"])
        for i, c in enumerate(lst):
            if c is rec["crop"]:
                lst[i] = new_crop
                break
        else:
            lst.append(new_crop)
        rec["crop"] = new_crop

    # ---- class change / delete / undo
    def set_rec_class(self, rec, ci, record=True):
        old = rec["ci"]
        if ci == old:
            return
        self.drop_sample(old, rec["crop"])
        rec["ci"] = ci
        self.app.samples_of(ci).append(rec["crop"])
        if record:
            self.history.append(("cls", self.cur, rec, old))

    def sel_rec(self):
        recs = self.recs.get(self.cur, [])
        return recs[self.sel] if self.sel is not None and self.sel < len(recs) else None

    def on_combo(self):
        r = self.sel_rec()
        if r is not None:
            self.set_rec_class(r, self.cb.current())
            self.redraw()
            self.update_label()

    def on_key(self, e):
        r = self.sel_rec()
        if e.keysym in ("Delete", "BackSpace") and r is not None:
            self.drop_sample(r["ci"], r["crop"])
            self.recs[self.cur].remove(r)
            self.history.append(("del", self.cur, r))
            self.sel = None
            self.mark(self.cur)
        elif e.char and e.char in "123456789" and r is not None:
            k = int(e.char) - 1
            if k < len(self.app.classes):
                self.set_rec_class(r, k)
                self.cb.current(k)
        elif e.keysym == "Escape":
            self.sel = None
        else:
            return
        self.redraw()
        self.update_label()

    def undo(self):
        if not self.history:
            return
        h = self.history.pop()
        kind, path, rec = h[0], h[1], h[2]
        if kind == "add":
            self.drop_sample(rec["ci"], rec["crop"])
            if rec in self.recs.get(path, []):
                self.recs[path].remove(rec)
        elif kind == "del":
            self.recs.setdefault(path, []).append(rec)
            self.app.samples_of(rec["ci"]).append(rec["crop"])
        elif kind == "edit":
            _, _, _, old_box, old_crop = h
            self.replace_crop(rec, old_crop)
            rec["box"] = old_box
        elif kind == "cls":
            self.set_rec_class(rec, h[3], record=False)
        self.sel = None
        self.mark(path)
        if path == self.cur:
            self.redraw()
        self.update_label()


# =============================================================================
#  Main application
# =============================================================================
BG, PANEL, PANEL2, FG, MUTED = "#14161b", "#1d2027", "#272b34", "#e8eaf0", "#8b91a1"
ACCENT, GREEN, RED, BORDER = "#7c5cff", "#22a06b", "#d6455d", "#2f343e"


def apply_theme(root):
    """Dark, Roboflow-like look."""
    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except tk.TclError:
        pass
    root.configure(bg=BG)
    for k, v in (("*Listbox.background", PANEL2), ("*Listbox.foreground", FG), ("*Listbox.selectBackground", ACCENT),
                 ("*Listbox.selectForeground", "#ffffff"), ("*Listbox.highlightThickness", 0), ("*Listbox.borderWidth", 0),
                 ("*TCombobox*Listbox.background", PANEL2), ("*TCombobox*Listbox.foreground", FG),
                 ("*TCombobox*Listbox.selectBackground", ACCENT)):
        root.option_add(k, v)
    st.configure(".", background=PANEL, foreground=FG, fieldbackground=PANEL2, bordercolor=BORDER,
                 lightcolor=PANEL, darkcolor=PANEL, troughcolor=PANEL2, focuscolor=PANEL, insertcolor=FG)
    st.configure("TButton", background=PANEL2, foreground=FG, padding=(10, 5), borderwidth=1, relief="flat")
    st.map("TButton", background=[("active", "#353a46"), ("pressed", "#3f4552")])
    for name, col, hov in (("Accent", ACCENT, "#6a49f0"), ("Accept", GREEN, "#1c8a5b"), ("Reject", RED, "#bb3a50"),
                           ("ToolOn", ACCENT, "#6a49f0")):
        st.configure(f"{name}.TButton", background=col, foreground="#ffffff", padding=(10, 5))
        st.map(f"{name}.TButton", background=[("active", hov), ("pressed", hov)])
    st.configure("Tool.TButton", padding=(4, 8))
    st.configure("ToolOn.TButton", padding=(4, 8))
    st.configure("TLabelframe", background=PANEL, bordercolor=BORDER)
    st.configure("TLabelframe.Label", background=PANEL, foreground=MUTED)
    st.configure("Muted.TLabel", foreground=MUTED)
    st.configure("Title.TLabel", font=("TkDefaultFont", 10, "bold"))
    st.configure("TCheckbutton", background=PANEL, foreground=FG)
    st.map("TCheckbutton", background=[("active", PANEL)], indicatorcolor=[("selected", ACCENT), ("!selected", PANEL2)])
    st.configure("TEntry", fieldbackground=PANEL2, foreground=FG)
    st.configure("TCombobox", fieldbackground=PANEL2, foreground=FG, arrowcolor=FG, background=PANEL2)
    st.map("TCombobox", fieldbackground=[("readonly", PANEL2)], foreground=[("readonly", FG)])
    st.configure("TSpinbox", fieldbackground=PANEL2, foreground=FG, arrowcolor=FG, background=PANEL2)
    st.configure("Horizontal.TScale", background=PANEL, troughcolor=PANEL2)


def _num(var, default):
    try:
        return float(var.get())
    except (tk.TclError, ValueError):
        return default


class App(tk.Tk):
    def __init__(self, images=None, output=None):
        super().__init__()
        self._err_count = 0
        apply_theme(self)
        self.title("Screw annotation assistant")
        self.geometry("1200x820")
        self.minsize(1000, 700)
        self.classes = [{"name": n, "samples": []} for n in DEFAULT_CLASSES]
        self.ignore_samples = []      # crops of things that must NOT be labelled (e.g. screws on the metal plate)
        self._thumb_refs = []
        self.running = False
        self.gen = 0
        self.task_q = None
        self.res_q = None
        self.warned = False

        self.var_images = tk.StringVar(value=images or "")
        self.var_output = tk.StringVar(value=output or "")
        self.var_method = tk.StringVar(value=METHOD_SAM)
        self.var_prompt = tk.StringVar(value="screw")
        self.var_sam_thr = tk.DoubleVar(value=0.5)
        self.var_sens = tk.DoubleVar(value=0.5)
        self.var_obj = tk.DoubleVar(value=20.0)
        self.var_min = tk.DoubleVar(value=0.05)
        self.var_max = tk.DoubleVar(value=6.0)
        self.var_margin = tk.DoubleVar(value=0.15)
        self.var_backend = tk.StringVar(value="auto")
        self.var_sim = tk.DoubleVar(value=0.65)
        self.var_learn = tk.BooleanVar(value=True)
        self.var_rotate = tk.BooleanVar(value=False)
        self.var_neg = tk.BooleanVar(value=True)
        self.var_base = tk.StringVar(value="")
        self.var_base_model = tk.StringVar(value="")
        self.var_base_conf = tk.DoubleVar(value=0.25)
        self.var_base_imgsz = tk.IntVar(value=0)
        self.var_device = tk.StringVar(value="auto")
        self.var_maxside = tk.IntVar(value=1600)
        self.var_half = tk.BooleanVar(value=False)
        self.var_search = tk.DoubleVar(value=0.6)
        self.var_share = tk.BooleanVar(value=False)
        self.learned = set()
        self.var_bright = tk.DoubleVar(value=0.0)
        self.var_contrast = tk.DoubleVar(value=1.0)
        self.var_auto = tk.BooleanVar(value=False)

        self.setup_frame = ttk.Frame(self)
        self.review_frame = ttk.Frame(self)
        self._build_setup()
        self._build_review()
        self.show_setup()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ settings
    def report_callback_exception(self, exc, val, tb):
        """Errors inside buttons / key presses / timers are logged; the window keeps running."""
        text = "".join(traceback.format_exception(exc, val, tb))
        log_error(text)
        self._err_count += 1
        try:
            self.status_var.set(f"Error: {val}  (details in annotator_errors.log)")
        except Exception:
            pass
        if self._err_count in (1, 10, 50):
            try:
                messagebox.showwarning("Something went wrong", f"{type(val).__name__}: {val}\n\nThe program keeps running. "
                                       f"Details were written to annotator_errors.log.")
            except Exception:
                pass

    def get_settings(self):
        return dict(method=self.var_method.get(), prompt=self.var_prompt.get().strip() or "screw",
                    sam_thr=_num(self.var_sam_thr, 0.5), sens=_num(self.var_sens, 0.5),
                    obj_pct=_num(self.var_obj, 20.0), min_pct=_num(self.var_min, 0.05),
                    max_pct=_num(self.var_max, 6.0), margin=_num(self.var_margin, 0.15),
                    backend=self.var_backend.get(), rotate=bool(self.var_rotate.get()),
                    base=self.var_base.get().strip(), base_model=self.var_base_model.get().strip(),
                    base_conf=_num(self.var_base_conf, 0.25), base_imgsz=int(_num(self.var_base_imgsz, 0)),
                    device=self.var_device.get(), max_side=int(_num(self.var_maxside, 1600)),
                    search_thr=_num(self.var_search, 0.6),
                    half=bool(self.var_half.get()), share=bool(self.var_share.get()))

    def apply_settings(self, s):
        for var, key in ((self.var_method, "method"), (self.var_prompt, "prompt"), (self.var_sam_thr, "sam_thr"),
                         (self.var_sens, "sens"), (self.var_obj, "obj_pct"), (self.var_min, "min_pct"),
                         (self.var_max, "max_pct"), (self.var_margin, "margin"), (self.var_backend, "backend"),
                         (self.var_rotate, "rotate"), (self.var_base, "base"), (self.var_base_model, "base_model"),
                         (self.var_base_conf, "base_conf"), (self.var_base_imgsz, "base_imgsz"),
                         (self.var_device, "device"), (self.var_maxside, "max_side"), (self.var_search, "search_thr"),
                         (self.var_half, "half"), (self.var_share, "share")):
            if key in s:
                try:
                    var.set(s[key])
                except tk.TclError:
                    pass

    # ------------------------------------------------------------------ setup screen
    def _build_setup(self):
        f = self.setup_frame
        f.columnconfigure(0, weight=1)
        pad = dict(padx=8, pady=4)

        top = ttk.LabelFrame(f, text="1. Folders")
        top.grid(row=0, column=0, sticky="ew", **pad)
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Images folder").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.var_images).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(top, text="Browse", command=self.browse_images).grid(row=0, column=2, **pad)
        ttk.Label(top, text="Output folder").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.var_output).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(top, text="Browse", command=self.browse_output).grid(row=1, column=2, **pad)
        ttk.Button(top, text="Load saved setup", command=self.load_setup_clicked).grid(row=1, column=3, **pad)

        mid = ttk.Frame(f)
        mid.grid(row=1, column=0, sticky="nsew", **pad)
        mid.columnconfigure(1, weight=1)
        f.rowconfigure(1, weight=1)

        lf = ttk.LabelFrame(mid, text="2. Classes (order = class id)")
        lf.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self.lb = tk.Listbox(lf, height=9, width=30, exportselection=False)
        self.lb.grid(row=0, column=0, columnspan=3, padx=6, pady=6)
        self.lb.bind("<<ListboxSelect>>", lambda e: self.refresh_thumbs())
        ttk.Button(lf, text="Add", command=self.add_class).grid(row=1, column=0, padx=4, pady=4)
        ttk.Button(lf, text="Rename", command=self.rename_class).grid(row=1, column=1, padx=4)
        ttk.Button(lf, text="Remove", command=self.remove_class).grid(row=1, column=2, padx=4)

        sf = ttk.LabelFrame(mid, text="3. Samples of the selected class")
        sf.grid(row=0, column=1, sticky="nsew")
        sf.columnconfigure(0, weight=1)
        sf.rowconfigure(1, weight=1)
        bar = ttk.Frame(sf)
        bar.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        ttk.Button(bar, text="Add sample image files...", command=self.add_sample_files).pack(side="left")
        ttk.Button(bar, text="Pick samples from an image...", command=self.pick_samples).pack(side="left", padx=6)
        ttk.Button(bar, text="Select from data...", style="Accent.TButton", command=self.pick_from_data).pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="Clear all", command=self.clear_samples).pack(side="left")
        self.lbl_samples = ttk.Label(bar, text="")
        self.lbl_samples.pack(side="right")
        self.thumb_frame = ttk.Frame(sf)
        self.thumb_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        ttk.Label(sf, text="Click a thumbnail to remove it. A sample can be a crop of the object, or a box drawn on "
                           "any example image. 3-10 samples per class work well; include different poses.",
                  foreground="gray", wraplength=640).grid(row=2, column=0, sticky="w", padx=6, pady=4)

        ig = ttk.Frame(sf)
        ig.grid(row=3, column=0, sticky="ew", padx=6, pady=4)
        ttk.Label(ig, text="Things to IGNORE (e.g. screws on the metal plate):").pack(side="left")
        ttk.Button(ig, text="Draw on an image...", command=self.pick_ignore).pack(side="left", padx=6)
        ttk.Button(ig, text="Add files...", command=self.add_ignore_files).pack(side="left")
        ttk.Button(ig, text="Clear", command=self.clear_ignore).pack(side="left", padx=6)
        self.lbl_ignore = ttk.Label(ig, text="0 ignore samples")
        self.lbl_ignore.pack(side="left", padx=6)

        df = ttk.LabelFrame(f, text="4. Detection")
        df.grid(row=2, column=0, sticky="ew", **pad)
        for c in range(6):
            df.columnconfigure(c, weight=1 if c in (1, 3, 5) else 0)

        def row(r, c, label, widget):
            ttk.Label(df, text=label).grid(row=r, column=c, sticky="w", padx=6, pady=3)
            widget.grid(row=r, column=c + 1, sticky="ew", padx=6, pady=3)

        row(0, 0, "Find objects with", ttk.Combobox(df, textvariable=self.var_method, state="readonly",
                                                    values=[METHOD_SAM, METHOD_MASK, METHOD_SAMPLES]))
        row(0, 2, "SAM 3 text prompt", ttk.Entry(df, textvariable=self.var_prompt))
        row(0, 4, "SAM 3 threshold", ttk.Spinbox(df, textvariable=self.var_sam_thr, from_=0.05, to=0.95, increment=0.05))
        row(1, 0, "Mask sensitivity (0-1)", ttk.Spinbox(df, textvariable=self.var_sens, from_=0.0, to=1.0, increment=0.1))
        row(1, 2, "Largest object (% of width)", ttk.Spinbox(df, textvariable=self.var_obj, from_=3, to=80, increment=1))
        row(1, 4, "Context margin", ttk.Spinbox(df, textvariable=self.var_margin, from_=0.0, to=1.0, increment=0.05))
        row(2, 0, "Min object area (% of image)", ttk.Spinbox(df, textvariable=self.var_min, from_=0.001, to=20, increment=0.01))
        row(2, 2, "Max object area (% of image)", ttk.Spinbox(df, textvariable=self.var_max, from_=1, to=100, increment=1))
        row(2, 4, "Class assignment", ttk.Combobox(df, textvariable=self.var_backend, state="readonly",
                                                   values=["auto", "dinov2", "simple"]))

        ttk.Checkbutton(df, text="Rotate samples by 90 degrees (use only if objects appear in any orientation)",
                        variable=self.var_rotate).grid(row=3, column=0, columnspan=4, sticky="w", padx=6, pady=3)

        row(3, 4, "Search threshold (samples)", ttk.Spinbox(df, textvariable=self.var_search, from_=0.3, to=0.95, increment=0.05))
        ttk.Label(df, text="Pre-trained classifier (.npz)").grid(row=4, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(df, textvariable=self.var_base).grid(row=4, column=1, columnspan=3, sticky="ew", padx=6, pady=3)
        ttk.Button(df, text="Browse", command=lambda: self.var_base.set(
            filedialog.askopenfilename(title="classifier.npz", filetypes=[("Classifier", "*.npz")]) or self.var_base.get())
        ).grid(row=4, column=4, sticky="w", padx=6, pady=3)

        ttk.Label(df, text="Base model (YOLO .pt, optional)").grid(row=5, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(df, textvariable=self.var_base_model).grid(row=5, column=1, columnspan=3, sticky="ew", padx=6, pady=3)
        ttk.Button(df, text="Browse", command=lambda: self.var_base_model.set(
            filedialog.askopenfilename(title="Base model", filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")])
            or self.var_base_model.get())).grid(row=5, column=4, sticky="w", padx=6, pady=3)
        ttk.Spinbox(df, textvariable=self.var_base_conf, from_=0.05, to=0.95, increment=0.05, width=6).grid(
            row=5, column=5, sticky="w", padx=6, pady=3)
        ttk.Label(df, foreground=MUTED, wraplength=900, justify="left",
                  text="With a base model: the model predicts the boxes and classes first, then SAM 3 tightens each box "
                       "(choose SAM 3 above). The class stays as the model predicted. Without one: SAM 3 / mask finds the "
                       "objects and your samples decide the class. The number is the model confidence.").grid(
            row=6, column=0, columnspan=6, sticky="w", padx=6)
        ttk.Label(df, text="Device").grid(row=7, column=0, sticky="w", padx=6, pady=3)
        ttk.Combobox(df, textvariable=self.var_device, state="readonly", values=["auto", "cuda", "mps", "cpu"],
                     width=10).grid(row=7, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(df, text="Max image side for models (memory)").grid(row=7, column=2, sticky="w", padx=6, pady=3)
        ttk.Spinbox(df, textvariable=self.var_maxside, from_=640, to=4096, increment=128, width=8).grid(
            row=7, column=3, sticky="w", padx=6, pady=3)
        ttk.Label(df, text="Base model image size (0 = default)").grid(row=7, column=4, sticky="w", padx=6, pady=3)
        ttk.Spinbox(df, textvariable=self.var_base_imgsz, from_=0, to=4096, increment=32, width=8).grid(
            row=7, column=5, sticky="w", padx=6, pady=3)

        ttk.Checkbutton(df, text="SAM 3 in half precision (less GPU memory; use only if memory runs out - quality can drop)",
                        variable=self.var_half).grid(row=8, column=0, columnspan=4, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(df, text="Faster with several prompts (experimental)",
                        variable=self.var_share).grid(row=8, column=4, columnspan=2, sticky="w", padx=6, pady=2)

        bot = ttk.Frame(f)
        bot.grid(row=3, column=0, sticky="ew", **pad)
        ttk.Button(bot, text="Start annotating  >", style="Accent.TButton", command=self.start).pack(side="left", ipadx=14, ipady=6)
        self.lbl_resume = ttk.Label(bot, text="", foreground="gray")
        self.lbl_resume.pack(side="left", padx=14)
        self.refresh_all_setup()

    def browse_images(self):
        d = filedialog.askdirectory(title="Folder with the images")
        if d:
            self.var_images.set(d)
            if not self.var_output.get():
                self.var_output.set(default_output_for(d))
                self.maybe_load_setup(silent=False)

    def browse_output(self):
        d = filedialog.askdirectory(title="Output folder")
        if d:
            self.var_output.set(d)
            self.maybe_load_setup(silent=False)

    def refresh_class_list(self):
        sel = self.lb.curselection()
        self.lb.delete(0, "end")
        for i, c in enumerate(self.classes):
            self.lb.insert("end", f"{i}: {c['name']}   ({len(c['samples'])} samples)")
            self.lb.itemconfig(i, foreground=PALETTE[i % len(PALETTE)])
        if self.classes:
            self.lb.selection_set(sel[0] if sel and sel[0] < len(self.classes) else 0)

    def sel_class(self):
        s = self.lb.curselection()
        return s[0] if s else None

    def refresh_thumbs(self):
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self._thumb_refs = []
        ci = self.sel_class()
        if ci is None:
            return
        samples = self.classes[ci]["samples"]
        self.lbl_samples.config(text=f"{len(samples)} samples")
        for k, s in enumerate(samples[:28]):
            h, w = s.shape[:2]
            sc = 72.0 / max(h, w)
            th = cv2.resize(s, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
            ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(th, cv2.COLOR_BGR2RGB)))
            self._thumb_refs.append(ph)
            lab = tk.Label(self.thumb_frame, image=ph, relief="solid", bd=1, cursor="hand2")
            lab.grid(row=k // 9, column=k % 9, padx=3, pady=3)
            lab.bind("<Button-1>", lambda e, i=k: self.remove_sample(i))
        if len(samples) > 28:
            ttk.Label(self.thumb_frame, text=f"... and {len(samples) - 28} more").grid(row=4, column=0, columnspan=9)

    def refresh_all_setup(self):
        self.refresh_class_list()
        self.refresh_thumbs()
        self.refresh_ignore_label()

    def add_class(self):
        n = simpledialog.askstring("Add class", "Class name:", parent=self)
        if n and n.strip():
            self.classes.append({"name": n.strip(), "samples": []})
            self.refresh_class_list()
            self.lb.selection_clear(0, "end")
            self.lb.selection_set(len(self.classes) - 1)
            self.refresh_thumbs()

    def rename_class(self):
        ci = self.sel_class()
        if ci is None:
            return
        n = simpledialog.askstring("Rename class", "New name:", initialvalue=self.classes[ci]["name"], parent=self)
        if n and n.strip():
            self.classes[ci]["name"] = n.strip()
            self.refresh_class_list()

    def remove_class(self):
        ci = self.sel_class()
        if ci is None:
            return
        if self.accepted_count() > 0:
            messagebox.showwarning("Cannot remove", "Some images are already accepted with these class ids.\n"
                                   "Removing a class would change the ids in the saved labels.\n"
                                   "You can rename it, or add new classes at the end.")
            return
        if messagebox.askyesno("Remove class", f"Remove '{self.classes[ci]['name']}' and its samples?"):
            del self.classes[ci]
            self.refresh_all_setup()

    def add_sample_files(self):
        ci = self.sel_class()
        if ci is None:
            return
        paths = filedialog.askopenfilenames(title="Sample images", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")])
        n = 0
        for p in paths:
            im = read_image(p)
            if im is not None:
                self.classes[ci]["samples"].append(im)
                n += 1
        if n:
            self.refresh_all_setup()

    def pick_samples(self):
        ci = self.sel_class()
        if ci is None:
            return
        p = filedialog.askopenfilename(title="Example image to draw samples on",
                                       filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")])
        if not p:
            return
        im = read_image(p)
        if im is None:
            messagebox.showerror("Error", "Cannot read that image")
            return
        SamplePicker(self, im, ci)

    def pick_from_data(self):
        if self.sel_class() is None:
            messagebox.showinfo("Select from data", "Add or select a class first.")
            return
        d = self.var_images.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showinfo("Select from data", "Choose the images folder (step 1) first.")
            return
        if not list_images(d):
            messagebox.showinfo("Select from data", "No images found in that folder.")
            return
        DataPicker(self, d, self.sel_class())

    def remove_sample(self, i):
        ci = self.sel_class()
        if ci is not None and messagebox.askyesno("Remove sample", "Remove this sample?"):
            del self.classes[ci]["samples"][i]
            self.refresh_all_setup()

    def clear_samples(self):
        ci = self.sel_class()
        if ci is not None and self.classes[ci]["samples"] and messagebox.askyesno("Clear", "Remove all samples of this class?"):
            self.classes[ci]["samples"] = []
            self.refresh_all_setup()

    # ------------------------------------------------------------------ save / load setup
    def save_setup(self):
        out = self.var_output.get()
        d = os.path.join(out, SETUP_DIR)
        sd = os.path.join(d, "samples")
        shutil.rmtree(sd, ignore_errors=True)
        os.makedirs(sd, exist_ok=True)
        data = []
        for ci, c in enumerate(self.classes):
            files = []
            for si, im in enumerate(c["samples"]):
                fn = f"c{ci}_{si}.png"
                write_image(os.path.join(sd, fn), im)
                files.append(fn)
            data.append({"name": c["name"], "samples": files})
        ign = []
        for si, im in enumerate(self.ignore_samples):
            fn = f"ignore_{si}.png"
            write_image(os.path.join(sd, fn), im)
            ign.append(fn)
        atomic_write(os.path.join(d, "setup.json"),
                     json.dumps({"classes": data, "settings": self.get_settings(), "ignore": ign}, indent=2))

    def load_setup(self, out):
        d = os.path.join(out, SETUP_DIR)
        with open(os.path.join(d, "setup.json"), encoding="utf-8") as f:
            data = json.load(f)
        classes = []
        for c in data["classes"]:
            samples = [im for im in (read_image(os.path.join(d, "samples", fn)) for fn in c["samples"]) if im is not None]
            classes.append({"name": c["name"], "samples": samples})
        if classes:
            self.classes = classes
        self.ignore_samples = [im for im in (read_image(os.path.join(d, "samples", fn)) for fn in data.get("ignore", [])) if im is not None]
        self.apply_settings(data.get("settings", {}))
        self.refresh_all_setup()

    def maybe_load_setup(self, silent=True):
        out = self.var_output.get()
        if out and os.path.exists(os.path.join(out, SETUP_DIR, "setup.json")):
            if silent or messagebox.askyesno("Saved setup found", "This output folder has a saved setup (classes, samples, settings).\nLoad it?"):
                try:
                    self.load_setup(out)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not load the setup: {e}")
        self.update_resume_label()

    def load_setup_clicked(self):
        out = self.var_output.get()
        if not out or not os.path.exists(os.path.join(out, SETUP_DIR, "setup.json")):
            messagebox.showinfo("Load saved setup", "Choose an output folder that contains a saved setup first.")
            return
        self.load_setup(out)
        self.update_resume_label()

    def update_resume_label(self):
        dec = self.load_progress()
        a = sum(1 for v in dec.values() if v == "accepted")
        r = sum(1 for v in dec.values() if v == "rejected")
        self.lbl_resume.config(text=f"Already reviewed in this output folder: {a} accepted, {r} rejected (will continue after them)" if dec else "")

    # ------------------------------------------------------------------ progress
    def progress_path(self):
        return os.path.join(self.var_output.get(), PROGRESS_FILE)

    def load_progress(self):
        try:
            with open(self.progress_path(), encoding="utf-8") as f:
                return json.load(f).get("decisions", {})
        except Exception:
            return {}

    def save_progress(self):
        atomic_write(self.progress_path(), json.dumps({"decisions": self.decisions}, indent=1))

    def accepted_count(self):
        dec = getattr(self, "decisions", None) or self.load_progress()
        return sum(1 for v in dec.values() if v == "accepted")

    # ------------------------------------------------------------------ review screen
    def _build_review(self):
        f = self.review_frame
        f.columnconfigure(1, weight=1)
        f.rowconfigure(1, weight=1)
        self.tool = "select"

        # ---- top bar
        bar = ttk.Frame(f)
        bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.lbl_file = ttk.Label(bar, text="", style="Title.TLabel")
        self.lbl_file.pack(side="left", padx=12, pady=8)
        ttk.Button(bar, text="Setup", command=self.show_setup).pack(side="right", padx=10, pady=6)
        ttk.Button(bar, text="Export dataset", command=lambda: self.export_dataset(False)).pack(side="right", pady=6)
        self.lbl_prog = ttk.Label(bar, text="", style="Muted.TLabel")
        self.lbl_prog.pack(side="right", padx=12)
        nav = ttk.Frame(bar)
        nav.pack(side="left", padx=40)
        ttk.Button(nav, text="<", width=3, command=self.go_back).pack(side="left")
        self.lbl_pos = ttk.Label(nav, text="0 / 0", width=12, anchor="center")
        self.lbl_pos.pack(side="left", padx=6)
        ttk.Button(nav, text=">", width=3, command=self.skip).pack(side="left")

        # ---- left tool strip
        tools = ttk.Frame(f, width=76)
        tools.grid(row=1, column=0, sticky="ns")
        self.tool_btns = {}
        for key, text, cmd in (("select", "Select\n(V)", lambda: self.set_tool("select")),
                               ("segment", "Click &\nsegment", lambda: self.set_tool("segment")),
                               ("hand", "Hand\n(H)", lambda: self.set_tool("hand")),
                               (None, None, None),
                               ("zin", "Zoom +\n(+)", lambda: self.zoom_by(1.25)),
                               ("zout", "Zoom -\n(-)", lambda: self.zoom_by(1 / 1.25)),
                               ("fit", "Fit\n(0)", self.zoom_fit),
                               (None, None, None),
                               ("smart", "Find\nsimilar (F)", self.find_similar_selected),
                               ("ignore", "Not an\nobject (X)", self.ignore_selected),
                               ("delete", "Delete\n(Del)", self.delete_selected),
                               ("redo", "Detect\nagain (D)", self.redetect)):
            if key is None:
                ttk.Separator(tools, orient="horizontal").pack(fill="x", padx=8, pady=6)
                continue
            btn = ttk.Button(tools, text=text, width=9, style="Tool.TButton", command=cmd)
            btn.pack(padx=6, pady=2)
            self.tool_btns[key] = btn
        self.set_tool("select")

        # ---- canvas
        cf = ttk.Frame(f)
        cf.grid(row=1, column=1, sticky="nsew")
        cf.columnconfigure(0, weight=1)
        cf.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(cf, bg="#0d0f13", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb = ttk.Scrollbar(cf, orient="vertical", command=lambda *a: self.scroll_view("y", *a))
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb = ttk.Scrollbar(cf, orient="horizontal", command=lambda *a: self.scroll_view("x", *a))
        self.hsb.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.on_wheel(type("E", (), {"delta": 120, "state": e.state, "x": e.x, "y": e.y})()))
        self.canvas.bind("<Button-5>", lambda e: self.on_wheel(type("E", (), {"delta": -120, "state": e.state, "x": e.x, "y": e.y})()))
        for b in (2, 3):
            self.canvas.bind(f"<ButtonPress-{b}>", self.pan_start)
            self.canvas.bind(f"<B{b}-Motion>", self.pan_move)
            self.canvas.bind(f"<ButtonRelease-{b}>", self.pan_end)

        # ---- right panel
        side = ttk.Frame(f, width=300)
        side.grid(row=1, column=2, sticky="ns")
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)
        ttk.Label(side, text="CLASSES", style="Muted.TLabel").pack(anchor="w", padx=12, pady=(10, 2))
        self.legend = ttk.Frame(side)
        self.legend.pack(fill="x", padx=8)
        ttk.Label(side, text="ANNOTATIONS", style="Muted.TLabel").pack(anchor="w", padx=12, pady=(12, 2))
        lf = ttk.Frame(side)
        lf.pack(fill="both", expand=True, padx=8)
        self.ann_list = tk.Listbox(lf, height=8, exportselection=False, activestyle="none", font=("TkDefaultFont", 11))
        self.ann_list.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.ann_list.yview)
        sb.pack(side="right", fill="y")
        self.ann_list.config(yscrollcommand=sb.set)
        self.ann_list.bind("<<ListboxSelect>>", self.on_ann_select)
        self.ann_sig = None
        ttk.Label(side, text="SMART SELECT", style="Muted.TLabel").pack(anchor="w", padx=12, pady=(12, 2))
        sf2 = ttk.Frame(side)
        sf2.pack(fill="x", padx=12)
        ttk.Label(sf2, text="Similarity").pack(side="left")
        ttk.Spinbox(sf2, textvariable=self.var_sim, from_=0.2, to=0.98, increment=0.05, width=6).pack(side="left", padx=6)
        ttk.Checkbutton(side, text="Learn from it for next images", variable=self.var_learn).pack(anchor="w", padx=12, pady=2)
        ttk.Checkbutton(side, text="Deleted false boxes = negative examples", variable=self.var_neg).pack(anchor="w", padx=12, pady=2)
        ttk.Label(side, text="IMAGE ADJUSTMENTS (view only)", style="Muted.TLabel").pack(anchor="w", padx=12, pady=(12, 2))
        ttk.Label(side, text="Brightness").pack(anchor="w", padx=12)
        ttk.Scale(side, from_=-120, to=120, variable=self.var_bright, command=lambda v: self.view_changed()).pack(fill="x", padx=12)
        ttk.Label(side, text="Contrast").pack(anchor="w", padx=12)
        ttk.Scale(side, from_=0.4, to=3.0, variable=self.var_contrast, command=lambda v: self.view_changed()).pack(fill="x", padx=12)
        br = ttk.Frame(side)
        br.pack(fill="x", padx=12, pady=4)
        ttk.Checkbutton(br, text="Auto", variable=self.var_auto, command=self.view_changed).pack(side="left")
        ttk.Button(br, text="Reset", width=7, command=self.reset_view_adjust).pack(side="right")
        ttk.Label(side, style="Muted.TLabel", wraplength=275, justify="left",
                  text="Click a box to select it; drag it to move, drag the handles to resize. Drag on empty space "
                       "to draw a box. 'Click & segment' tool: click an object to get its box. Scroll / trackpad two-finger "
                       "scroll = zoom; Shift+scroll = pan sideways; Hand tool, scrollbars or right-button drag = pan.").pack(
            anchor="w", padx=12, pady=(6, 10))

        # ---- bottom bar
        bot = ttk.Frame(f)
        bot.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.status_var = tk.StringVar(value="")
        ttk.Label(bot, textvariable=self.status_var, style="Muted.TLabel").pack(side="left", padx=12)
        ttk.Button(bot, text="Reject  (R)", style="Reject.TButton", command=self.reject).pack(side="right", padx=(6, 12), pady=8)
        ttk.Button(bot, text="Accept  (A)", style="Accept.TButton", command=self.accept).pack(side="right", pady=8)

        self.images, self.idx, self.img = [], 0, None
        self.cache, self.pending, self.decisions = {}, set(), {}
        self.cur_boxes, self.sel, self.active_cls, self.busy = [], None, 0, False
        self.photo, self.photo_key, self.drag = None, None, None
        self.zoom, self.vx, self.vy, self.pan_drag = 1.0, None, None, None
        self.auto_lut = None
        self.photo_pos = (0, 0)
        self.edit = None
        self.edit_pushed = False
        self.undo_stack, self.redo_stack, self.clip = {}, {}, []
        self.scale, self.ox, self.oy = 1.0, 0, 0

    def set_tool(self, tool):
        self.tool = tool
        for k in ("select", "segment", "hand"):
            self.tool_btns[k].configure(style="ToolOn.TButton" if k == tool else "Tool.TButton")
        if hasattr(self, "canvas"):
            self.canvas.config(cursor={"hand": "fleur", "segment": "plus"}.get(tool, "crosshair"))
        if hasattr(self, "status_var") and tool == "segment":
            self.status_var.set("Click & segment: click on an object to get its box.")

    def on_ann_select(self, e):
        s = self.ann_list.curselection()
        if s and s[0] < len(self.cur_boxes):
            self.sel = s[0]
            self.redraw()

    def refresh_ann_list(self):
        sig = tuple(b.cls for b in self.cur_boxes)
        if sig != self.ann_sig:
            self.ann_sig = sig
            self.ann_list.delete(0, "end")
            for i, b in enumerate(self.cur_boxes):
                name = self.classes[b.cls]["name"] if b.cls < len(self.classes) else f"class {b.cls}"
                self.ann_list.insert("end", f" {i + 1:>2}   {name}")
                self.ann_list.itemconfig(i, foreground=PALETTE[b.cls % len(PALETTE)] if b.cls < len(self.classes) else FG)
        self.ann_list.selection_clear(0, "end")
        if self.sel is not None and self.sel < len(self.cur_boxes):
            self.ann_list.selection_set(self.sel)
            self.ann_list.see(self.sel)

    def build_legend(self):
        for w in self.legend.winfo_children():
            w.destroy()
        self.legend_btns, self.legend_counts = [], []
        for i, c in enumerate(self.classes):
            row = tk.Frame(self.legend, bg=PANEL, highlightthickness=0)
            row.pack(fill="x", pady=1)
            sw = tk.Frame(row, bg=PALETTE[i % len(PALETTE)], width=6)
            sw.pack(side="left", fill="y")
            key = f"{i + 1}" if i < 9 else "-"
            kl = tk.Label(row, text=key, bg=PANEL2, fg=MUTED, width=2)
            kl.pack(side="left", padx=(8, 8), pady=5)
            nl = tk.Label(row, text=c["name"], anchor="w", bg=PANEL, fg=FG, font=("TkDefaultFont", 11))
            nl.pack(side="left", fill="x", expand=True)
            cl = tk.Label(row, text="0", bg=PANEL, fg=MUTED, width=4)
            cl.pack(side="right", padx=4)
            for w in (row, kl, nl, cl, sw):
                w.bind("<Button-1>", lambda e, k=i: self.set_class(k))
            self.legend_btns.append((row, kl, nl, cl))
            self.legend_counts.append(cl)
        self.mark_active()

    def mark_active(self):
        for i, (row, kl, nl, cl) in enumerate(self.legend_btns):
            bg = "#2c3040" if i == self.active_cls else PANEL
            for w in (row, nl, cl):
                w.config(bg=bg)
            kl.config(bg=ACCENT if i == self.active_cls else PANEL2, fg="#ffffff" if i == self.active_cls else MUTED)

    # ---- starting / stopping
    def start(self):
        folder = self.var_images.get()
        if not os.path.isdir(folder):
            messagebox.showerror("Images folder", "Please choose a valid images folder.")
            return
        if not self.var_output.get():
            self.var_output.set(default_output_for(folder))
        images = list_images(folder)
        if not images:
            messagebox.showerror("Images folder", "No images found in that folder.")
            return
        if not self.classes:
            messagebox.showerror("Classes", "Add at least one class.")
            return
        bm = self.var_base_model.get().strip()
        if bm and not os.path.isfile(bm):
            messagebox.showerror("Base model", f"The base model file was not found:\n{bm}\n\nFix the path or clear it.")
            return
        if all(len(c["samples"]) == 0 for c in self.classes):
            if not messagebox.askyesno("No samples", "No samples were given, so every object will get the first class "
                                       f"('{self.classes[0]['name']}') and you must fix the classes by hand.\n\nContinue?"):
                return
        empty = [c["name"] for c in self.classes if not c["samples"]]
        if empty and len(empty) < len(self.classes):
            if not messagebox.askyesno("Classes without samples", "These classes have no samples and will never be "
                                       "assigned automatically:\n  - " + "\n  - ".join(empty) + "\n\nContinue?"):
                return
        try:
            os.makedirs(self.var_output.get(), exist_ok=True)
            self.save_setup()
            write_dataset_files(self.var_output.get(), [c["name"] for c in self.classes])
        except Exception as e:
            messagebox.showerror("Output folder", f"Cannot write to the output folder: {e}")
            return

        self.images = images
        self.decisions = self.load_progress()
        self.decisions = {k: v for k, v in self.decisions.items() if os.path.exists(os.path.join(folder, k))}
        self.cache, self.pending, self.warned = {}, set(), False
        self.timing_txt = {}
        self.active_cls = 0
        self.spawn_worker()
        self.running = True
        self.setup_frame.pack_forget()
        self.review_frame.pack(fill="both", expand=True)
        self.build_legend()
        self.bind("<Key>", self.on_key)
        self.bind_shortcuts()
        self.after(80, self.poll)
        first = next((i for i, p in enumerate(images) if os.path.basename(p) not in self.decisions), 0)
        self.show(first)

    def spawn_worker(self):
        self.gen += 1
        self.task_q, self.res_q, self.learn_q = queue.LifoQueue(), queue.Queue(), queue.Queue()
        groups = [[s.copy() for s in c["samples"]] for c in self.classes]
        if self.ignore_samples:
            groups.append([s.copy() for s in self.ignore_samples])      # last group = "ignore"
        cfg = dict(self.get_settings(), n_classes=len(self.classes), class_names=[c["name"] for c in self.classes],
                   n_ignore=len(self.ignore_samples))
        self.worker = threading.Thread(target=worker_main, daemon=True,
                                       args=(self.gen, cfg, groups, self.task_q, self.res_q, self.learn_q))
        self.worker.start()

    def samples_of(self, ci):
        """Sample list of a class; the index len(classes) means the IGNORE samples."""
        return self.ignore_samples if ci >= len(self.classes) else self.classes[ci]["samples"]

    def ignore_selected(self):
        """Selected box is NOT an object: delete it and learn it as an 'ignore' sample."""
        if self.sel is None or self.sel >= len(self.cur_boxes) or self.img is None:
            return
        b = self.cur_boxes[self.sel]
        self.ignore_samples.append(crop_context(self.img, (b.x1, b.y1, b.x2, b.y2), self.get_settings()["margin"]).copy())
        self.push_undo()
        del self.cur_boxes[self.sel]
        self.sel = None
        self.teach(len(self.classes), self.ignore_samples[-1])
        try:
            self.save_setup()
        except Exception:
            pass
        self.status_var.set(f"Learned as 'ignore' ({len(self.ignore_samples)} ignore samples). Next images use it.")
        self.refresh_ignore_label()
        self.redraw()

    def teach(self, ci, crop, refresh=True):
        """Give the running classifier a new sample (class ci, or ignore) and re-predict the upcoming images."""
        if self.task_q is None:
            return
        self.learn_q.put((ci, crop))
        if refresh:
            self.refresh_predictions()

    def refresh_predictions(self):
        keep = self.images[self.idx]
        base = bool(self.var_base_model.get().strip())
        if base:                                  # classes come from the base model: keep what was predicted
            return
        self.cache = {k: v for k, v in self.cache.items() if k == keep}
        for j in range(self.idx + 1, self.idx + 1 + PREFETCH):
            if j < len(self.images):
                p = self.images[j]
                if p not in self.pending:
                    self.pending.add(p)
                    self.task_q.put(("reclass", p))

    def recheck_current(self, skip=None):
        """After a correction (no base model): the other boxes of this image that you have not confirmed are
        classified again with the new knowledge, so the same mistake does not repeat in the same picture."""
        if self.task_q is None or self.var_base_model.get().strip() or self.img is None:
            return
        todo = [(i, b.x1, b.y1, b.x2, b.y2) for i, b in enumerate(self.cur_boxes) if i != skip and b.conf < 0.999]
        if todo:
            self.task_q.put(("recls", self.images[self.idx], todo))

    def learn_boxes(self, boxes, path, refresh=True):
        """Every box you confirmed becomes a training sample of its class (kept in the saved setup as well)."""
        n = 0
        margin = self.get_settings()["margin"]
        for b in boxes:
            key = (path, b.cls, round(b.x1), round(b.y1), round(b.x2), round(b.y2))
            if b.cls >= len(self.classes) or key in self.learned or self.img is None:
                continue
            self.learned.add(key)
            crop = crop_context(self.img, (b.x1, b.y1, b.x2, b.y2), margin).copy()
            self.classes[b.cls]["samples"].append(crop)
            self.teach(b.cls, crop, refresh=False)
            n += 1
        if n and refresh:
            self.refresh_predictions()
        return n

    def request_click(self, x, y):
        """Click on a missed object: ask the worker to find its outline and draw the box."""
        if self.task_q is None or self.busy:
            return
        H, W = self.img.shape[:2]
        sizes = [(b.x2 - b.x1, b.y2 - b.y1) for b in self.cur_boxes if b.x2 - b.x1 > 10 and b.y2 - b.y1 > 10]
        if sizes:
            pw, ph = float(np.median([a for a, _ in sizes])), float(np.median([b for _, b in sizes]))
        else:
            pw = ph = 0.07 * max(H, W)
        self.status_var.set("Segmenting the object under your click...")
        self.task_q.put(("click", self.images[self.idx], float(x), float(y), pw, ph, self.active_cls))

    def find_similar_selected(self):
        """Use the selected (or just drawn) box as an exemplar and add every similar object in this image."""
        if self.sel is None or self.sel >= len(self.cur_boxes) or self.img is None or self.task_q is None:
            self.status_var.set("Draw a box around one object (or click a box) first, then press F.")
            return
        b = self.cur_boxes[self.sel]
        crop = crop_context(self.img, (b.x1, b.y1, b.x2, b.y2), self.get_settings()["margin"]).copy()
        thr = _num(self.var_sim, 0.65)
        self.status_var.set("Searching for similar objects...")
        if self.var_learn.get() and b.cls < len(self.classes):      # the exemplar also teaches the classifier
            self.classes[b.cls]["samples"].append(crop)
            self.teach(b.cls, crop)
            try:
                self.save_setup()
            except Exception:
                pass
        self.task_q.put(("similar", self.images[self.idx], crop, (b.x1, b.y1, b.x2, b.y2), b.cls, thr))

    def refresh_ignore_label(self):
        if hasattr(self, "lbl_ignore"):
            self.lbl_ignore.config(text=f"{len(self.ignore_samples)} ignore samples")

    def add_ignore_files(self):
        paths = filedialog.askopenfilenames(title="Images of things to IGNORE",
                                            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")])
        for p in paths:
            im = read_image(p)
            if im is not None:
                self.ignore_samples.append(im)
        self.refresh_ignore_label()

    def pick_ignore(self):
        p = filedialog.askopenfilename(title="Example image to draw IGNORE boxes on",
                                       filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")])
        if not p:
            return
        im = read_image(p)
        if im is None:
            messagebox.showerror("Error", "Cannot read that image")
            return
        SamplePicker(self, im, len(self.classes))

    def clear_ignore(self):
        if self.ignore_samples and messagebox.askyesno("Clear", "Remove all ignore samples?"):
            self.ignore_samples = []
            self.refresh_ignore_label()

    def stop_worker(self):
        self.running = False
        if self.task_q is not None:
            self.task_q.put(None)
            self.task_q = None

    def show_setup(self):
        self.stop_worker()
        self.unbind("<Key>")
        self.unbind_shortcuts()
        self.review_frame.pack_forget()
        self.setup_frame.pack(fill="both", expand=True)
        self.refresh_all_setup()
        self.update_resume_label()

    def on_close(self):
        self.stop_worker()
        self.destroy()

    # ---- detection results
    def request(self, path):
        if path in self.cache or path in self.pending or self.task_q is None:
            return
        self.pending.add(path)
        self.task_q.put(path)

    def poll(self):
        if not self.running:
            return
        try:
            while True:
                msg = self.res_q.get_nowait()
                if msg[1] != self.gen:
                    continue
                kind = msg[0]
                if kind == "status":
                    self.det_msg = msg[2]
                    self.status_var.set(msg[2])
                elif kind == "timing":
                    self.timing_txt[msg[2]] = msg[3]
                elif kind == "similar":
                    path, new = msg[2], msg[3]
                    if self.images and path == self.images[self.idx]:
                        added = 0
                        self.push_undo()
                        for nb in new:
                            if all(iou((nb.x1, nb.y1, nb.x2, nb.y2), (o.x1, o.y1, o.x2, o.y2)) < 0.4 for o in self.cur_boxes):
                                self.cur_boxes.append(nb)
                                added += 1
                        self.cache[path] = self.cur_boxes
                        self.status_var.set(f"Found {added} more similar object(s)." if added else
                                            "No new similar objects found (try lowering the similarity).")
                        self.redraw()
                elif kind == "recls":
                    path, res = msg[2], msg[3]
                    if self.images and path == self.images[self.idx]:
                        changed = 0
                        for i, x1, y1, x2, y2, cl, cf in res:
                            if i < len(self.cur_boxes):
                                b = self.cur_boxes[i]
                                if (abs(b.x1 - x1) < 1 and abs(b.y1 - y1) < 1 and abs(b.x2 - x2) < 1 and
                                        abs(b.y2 - y2) < 1 and b.conf < 0.999 and cl < len(self.classes) and b.cls != cl):
                                    if not changed:
                                        self.push_undo()
                                    b.cls, b.conf, changed = cl, cf, changed + 1
                        if changed:
                            self.cache[path] = self.cur_boxes
                            self.status_var.set(f"Learned from your correction: {changed} other box(es) in this image "
                                                "got a new class.")
                            self.redraw()
                elif kind == "click_result":
                    path, nb = msg[2], msg[3]
                    if self.images and path == self.images[self.idx]:
                        self.push_undo()
                        self.cur_boxes.append(nb)
                        self.sel = len(self.cur_boxes) - 1
                        self.cache[path] = self.cur_boxes
                        self.status_var.set("Box added. Press 1-3 to change its class, or drag its handles to adjust.")
                        self.redraw()
                elif kind == "similar_error":
                    self.status_var.set("Similar search failed: " + msg[3].strip().splitlines()[-1])
                elif kind == "warn":
                    if not self.warned:
                        self.warned = True
                        messagebox.showwarning("Notice", msg[2], parent=self)
                elif kind in ("result", "error"):
                    path = msg[2]
                    self.pending.discard(path)
                    if kind == "error":
                        self.cache[path] = []
                        self.status_var.set(f"Error on {os.path.basename(path)}: {msg[3].strip().splitlines()[-1]}")
                    else:
                        self.cache[path] = msg[3]
                    if self.images and path == self.images[self.idx]:
                        self.busy = False
                        self.cur_boxes = self.cache[path]
                        self.sel = None
                        self.redraw()
                        self.status_var.set(("Ready.  " + self.timing_txt.pop(path, "")).strip() if kind == "result"
                                            else self.status_var.get())
                        for j in range(self.idx + 1, self.idx + 1 + PREFETCH):
                            if j < len(self.images):
                                self.request(self.images[j])
        except queue.Empty:
            pass
        except Exception:
            log_error("Error while handling a result:\n" + traceback.format_exc())
            self.status_var.set("An error occurred (details in annotator_errors.log). You can continue.")
        self.watchdog()
        self.tick_busy()
        self.after(80, self.poll)

    def tick_busy(self):
        """While waiting for a detection show what the worker is doing and for how long."""
        import time
        if self.busy:
            t0 = getattr(self, "_busy_since", None)
            if t0 is None:
                self._busy_since = t0 = time.time()
            el = int(time.time() - t0)
            if el >= 3 and el != getattr(self, "_busy_shown", -1):
                self._busy_shown = el
                msg = getattr(self, "det_msg", "Detecting...")
                hint = ""
                if el >= 60:
                    hint = "  - very slow: the model may be running on the CPU or the GPU is busy (see annotator_errors.log)"
                self.status_var.set(f"{msg}   [{el}s]{hint}")
        else:
            self._busy_since = None

    def watchdog(self):
        """If the background thread ever dies, start a new one and ask again for the current image."""
        w = getattr(self, "worker", None)
        if self.running and w is not None and not w.is_alive() and self.task_q is not None:
            log_error("The detection thread stopped unexpectedly and was restarted.")
            self.spawn_worker()
            self.pending = set()
            if self.images and self.cur_boxes == [] and self.busy:
                self.request(self.images[self.idx])
            self.status_var.set("The detection thread was restarted.")

    # ---- navigation
    def show(self, idx):
        idx = max(0, min(idx, len(self.images) - 1))
        self.idx = idx
        path = self.images[idx]
        name = os.path.basename(path)
        self.img = read_image(path)
        self.photo_key = None
        self.zoom, self.vx, self.vy, self.pan_drag = 1.0, None, None, None
        self.auto_lut = None
        if self.img is not None:
            mean = float(np.clip(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY).mean(), 5, 250)) / 255.0
            gamma = float(np.clip(np.log(0.5) / np.log(mean), 0.3, 3.0))
            self.auto_lut = (np.power(np.arange(256) / 255.0, gamma) * 255).astype(np.uint8)
        self.sel = None
        if self.img is None:
            self.cur_boxes, self.busy = [], False
            self.status_var.set(f"Cannot read {name}")
        elif path in self.cache:
            self.cur_boxes, self.busy = self.cache[path], False
            for j in range(idx + 1, idx + 1 + PREFETCH):
                if j < len(self.images):
                    self.request(self.images[j])
        elif self.decisions.get(name) == "accepted" and os.path.exists(os.path.join(self.var_output.get(), "labels", os.path.splitext(name)[0] + ".txt")):
            H, W = self.img.shape[:2]
            self.cache[path] = read_yolo_boxes(os.path.join(self.var_output.get(), "labels", os.path.splitext(name)[0] + ".txt"), W, H)
            self.cur_boxes, self.busy = self.cache[path], False
        else:
            self.cur_boxes, self.busy = [], True
            self.request(path)
        self.update_progress_label()
        self.redraw()

    def update_progress_label(self):
        a = sum(1 for v in self.decisions.values() if v == "accepted")
        r = sum(1 for v in self.decisions.values() if v == "rejected")
        name = os.path.basename(self.images[self.idx])
        st = self.decisions.get(name, "")
        self.lbl_file.config(text=f"{name}" + (f"   [{st}]" if st else ""))
        self.lbl_prog.config(text=f"accepted {a}   rejected {r}   left {len(self.images) - a - r}")
        self.lbl_pos.config(text=f"{self.idx + 1} / {len(self.images)}")

    def release_image(self, path):
        """A finished (accepted / rejected) image no longer needs its working data: free it and run the garbage
        collector / empty the GPU cache so memory does not grow while you annotate hundreds of images."""
        try:
            self.cache.pop(path, None)
            self.undo_stack.pop(path, None)
            self.redo_stack.pop(path, None)
            self.pending.discard(path)
            self.photo, self.photo_key = None, None
            self.cur_boxes = []
            free_memory(True)
        except Exception as e:
            log_error(f"release_image: {e}")

    def advance(self):
        for i in list(range(self.idx + 1, len(self.images))) + list(range(0, self.idx)):
            if os.path.basename(self.images[i]) not in self.decisions:
                self.show(i)
                return
        self.update_progress_label()
        a = sum(1 for v in self.decisions.values() if v == "accepted")
        r = sum(1 for v in self.decisions.values() if v == "rejected")
        res = self.export_dataset(True)
        extra = f"\n\nFinal dataset ({res[1]} images, no samples):\n{res[0]}" if res else ""
        messagebox.showinfo("All images reviewed", f"Done.\nAccepted: {a}\nRejected: {r}\n\nWorking folder: "
                                                   f"{os.path.abspath(self.var_output.get())}{extra}")

    def final_dataset_dir(self):
        """The finished dataset goes into a NEW folder inside the main folder (the folder that holds the images
        folder and the working output folder)."""
        imgs = os.path.abspath(self.var_images.get().rstrip("/\\") or self.var_images.get())
        dst = os.path.join(os.path.dirname(imgs), "final_dataset")
        if os.path.abspath(dst) in (imgs, os.path.abspath(self.var_output.get())):
            dst += "_export"
        return dst

    def export_dataset(self, quiet=False):
        """Copy ONLY the accepted images and their labels (+ classes.txt, data.yaml) into final_dataset/.
        No samples, no _setup, no progress file. Safe to run again: images/ and labels/ there are rebuilt."""
        try:
            out = self.var_output.get()
            dec = self.decisions if getattr(self, "decisions", None) else self.load_progress()
            acc = sorted(n for n, v in dec.items() if v == "accepted")
            if not acc:
                if not quiet:
                    messagebox.showinfo("Export dataset", "No accepted images yet.")
                return None
            dst = self.final_dataset_dir()
            di, dl = os.path.join(dst, "images"), os.path.join(dst, "labels")
            for d in (di, dl):
                shutil.rmtree(d, ignore_errors=True)
                os.makedirs(d, exist_ok=True)
            n = 0
            for name in acc:
                si = os.path.join(out, "images", name)
                sl = os.path.join(out, "labels", os.path.splitext(name)[0] + ".txt")
                if not os.path.exists(si):
                    continue
                shutil.copy2(si, os.path.join(di, name))
                if os.path.exists(sl):
                    shutil.copy2(sl, os.path.join(dl, os.path.basename(sl)))
                else:
                    atomic_write(os.path.join(dl, os.path.splitext(name)[0] + ".txt"), "")
                n += 1
            write_dataset_files(dst, [c["name"] for c in self.classes])
            self.status_var.set(f"Final dataset: {n} images in {dst}")
            if not quiet:
                messagebox.showinfo("Export dataset", f"{n} images and labels were copied to:\n{dst}\n\n"
                                                      "It contains images/, labels/, classes.txt and data.yaml only (no samples).")
            return dst, n
        except Exception as e:
            log_error("Export failed:\n" + traceback.format_exc())
            if not quiet:
                messagebox.showerror("Export dataset", f"Could not export the dataset:\n{e}")
            return None

    def skip(self):
        if self.idx + 1 < len(self.images):
            self.show(self.idx + 1)

    def go_back(self):
        if self.idx > 0:
            self.show(self.idx - 1)

    # ---- decisions
    def out_paths(self, name):
        out = self.var_output.get()
        return os.path.join(out, "images", name), os.path.join(out, "labels", os.path.splitext(name)[0] + ".txt")

    def accept(self):
        try:
            self._accept()
        except Exception as e:
            log_error("Accept failed:\n" + traceback.format_exc())
            messagebox.showerror("Could not save", f"The image could not be saved and was NOT accepted:\n{e}\n\n"
                                 "Check the output folder (free space, permissions) and try again.")

    def _accept(self):
        if self.busy or self.img is None:
            return
        path = self.images[self.idx]
        name = os.path.basename(path)
        img_out, lbl_out = self.out_paths(name)
        os.makedirs(os.path.dirname(img_out), exist_ok=True)
        os.makedirs(os.path.dirname(lbl_out), exist_ok=True)
        if os.path.abspath(path) != os.path.abspath(img_out):
            shutil.copy2(path, img_out)
        H, W = self.img.shape[:2]
        write_yolo_boxes(lbl_out, self.cur_boxes, W, H)
        write_dataset_files(self.var_output.get(), [c["name"] for c in self.classes])
        self.decisions[name] = "accepted"
        self.save_progress()
        learned = self.learn_boxes(self.cur_boxes, path, refresh=False)
        if learned:
            if self.var_base_model.get().strip():
                pass                             # classes come from the base model
            else:
                self.cache = {}                  # next images: classes re-checked with the new samples (fast)
                for j in range(self.idx + 1, self.idx + 1 + PREFETCH):
                    if j < len(self.images) and self.images[j] not in self.pending:
                        self.pending.add(self.images[j])
                        self.task_q.put(("reclass", self.images[j]))
            try:
                self.save_setup()
            except Exception:
                pass
        self.release_image(path)
        self.advance()
        if learned:
            self.status_var.set(f"Learned {learned} object(s) from the accepted image; next images use them.")

    def reject(self):
        try:
            self._reject()
        except Exception as e:
            log_error("Reject failed:\n" + traceback.format_exc())
            messagebox.showerror("Error", f"Could not update the output folder:\n{e}")

    def _reject(self):
        if self.img is None:
            return
        name = os.path.basename(self.images[self.idx])
        img_out, lbl_out = self.out_paths(name)
        for p in (lbl_out, img_out):
            if os.path.exists(p) and os.path.abspath(p) != os.path.abspath(self.images[self.idx]):
                os.remove(p)
        self.decisions[name] = "rejected"
        self.save_progress()
        self.release_image(self.images[self.idx])
        self.advance()

    def redetect(self):
        path = self.images[self.idx]
        self.cache.pop(path, None)
        self.push_undo()
        self.busy, self.cur_boxes, self.sel = True, [], None
        self.request(path)
        self.redraw()

    # ---- undo / redo / copy / paste
    @staticmethod
    def _snap(boxes):
        return [Box(b.x1, b.y1, b.x2, b.y2, b.cls, b.conf, None) for b in boxes]

    def push_undo(self):
        path = self.images[self.idx]
        st = self.undo_stack.setdefault(path, [])
        st.append(self._snap(self.cur_boxes))
        del st[:-60]
        self.redo_stack[path] = []

    def _restore(self, snap, path):
        self.cur_boxes = self._snap(snap)
        self.cache[path] = self.cur_boxes
        self.sel = None
        self.redraw()

    def undo(self):
        if not self.images or self.busy:
            return
        path = self.images[self.idx]
        st = self.undo_stack.get(path)
        if not st:
            self.status_var.set("Nothing to undo.")
            return
        self.redo_stack.setdefault(path, []).append(self._snap(self.cur_boxes))
        self._restore(st.pop(), path)
        self.status_var.set("Undone.")

    def redo(self):
        if not self.images or self.busy:
            return
        path = self.images[self.idx]
        st = self.redo_stack.get(path)
        if not st:
            self.status_var.set("Nothing to redo.")
            return
        self.undo_stack.setdefault(path, []).append(self._snap(self.cur_boxes))
        self._restore(st.pop(), path)
        self.status_var.set("Redone.")

    def copy_boxes(self):
        """Ctrl/Cmd+C: copy the selected box, or all boxes of the image when none is selected."""
        if self.sel is not None and self.sel < len(self.cur_boxes):
            self.clip = self._snap([self.cur_boxes[self.sel]])
        else:
            self.clip = self._snap(self.cur_boxes)
        self.status_var.set(f"Copied {len(self.clip)} box(es). Ctrl/Cmd+V pastes them (also on another image).")

    def paste_boxes(self):
        if not self.clip or self.img is None or self.busy:
            self.status_var.set("Nothing copied yet.")
            return
        H, W = self.img.shape[:2]
        self.push_undo()
        shift = 0.0
        for c in self.clip:                         # pasting on top of identical boxes: move the copy a little
            if any(iou((c.x1, c.y1, c.x2, c.y2), (o.x1, o.y1, o.x2, o.y2)) > 0.9 for o in self.cur_boxes):
                shift = 12.0
        for c in self.clip:
            w, h = c.x2 - c.x1, c.y2 - c.y1
            x1 = min(max(0.0, c.x1 + shift), max(0.0, W - w))
            y1 = min(max(0.0, c.y1 + shift), max(0.0, H - h))
            self.cur_boxes.append(Box(x1, y1, x1 + w, y1 + h, c.cls, 1.0, None))
        self.cache[self.images[self.idx]] = self.cur_boxes
        self.sel = len(self.cur_boxes) - 1
        self.status_var.set(f"Pasted {len(self.clip)} box(es).")
        self.redraw()

    def bind_shortcuts(self):
        self._shortcut_seqs = []
        for mod in ("Control", "Command"):
            for key, fn in (("c", self.copy_boxes), ("v", self.paste_boxes), ("z", self.undo), ("Z", self.redo), ("y", self.redo)):
                seq = f"<{mod}-{key}>"
                try:
                    self.bind(seq, lambda e, f=fn: (f(), "break")[1])
                    self._shortcut_seqs.append(seq)
                except tk.TclError:
                    pass

    def unbind_shortcuts(self):
        for seq in getattr(self, "_shortcut_seqs", []):
            try:
                self.unbind(seq)
            except tk.TclError:
                pass

    # ---- editing boxes
    def set_class(self, k):
        if k >= len(self.classes):
            return
        self.active_cls = k
        if self.sel is not None and self.sel < len(self.cur_boxes):
            self.push_undo()
            self.cur_boxes[self.sel].cls = k
            self.cur_boxes[self.sel].conf = 1.0
            if self.learn_boxes([self.cur_boxes[self.sel]], self.images[self.idx]):
                self.status_var.set("Learned from your correction.")
                self.recheck_current(skip=self.sel)
        self.mark_active()
        self.redraw()

    def delete_selected(self):
        if self.sel is not None and self.sel < len(self.cur_boxes):
            b = self.cur_boxes[self.sel]
            others = [o for k, o in enumerate(self.cur_boxes) if k != self.sel]
            # an automatically detected box that you delete and that does not overlap any box you keep is a false
            # detection (e.g. an empty hole) -> learn it as "not an object" (negative example)
            is_false = (self.var_neg.get() and b.conf < 0.999 and self.img is not None and
                        all(iou((b.x1, b.y1, b.x2, b.y2), (o.x1, o.y1, o.x2, o.y2)) < 0.2 for o in others))
            if is_false:
                self.ignore_selected()
                return
            self.push_undo()
            del self.cur_boxes[self.sel]
            self.sel = None
            self.redraw()

    def on_key(self, e):
        k = e.keysym
        if k in ("Return", "KP_Enter"):                   # Enter only deselects the box; it never accepts the image
            self.sel = None
            self.redraw()
        elif k in ("a", "A"):
            self.accept()
        elif k in ("r", "R"):
            self.reject()
        elif k in ("s", "S", "Right"):
            self.skip()
        elif k in ("b", "B", "Left"):
            self.go_back()
        elif k in ("Delete", "BackSpace"):
            self.delete_selected()
        elif k in ("v", "V"):
            self.set_tool("select")
        elif k in ("c", "C"):
            self.set_tool("segment")
        elif k in ("h", "H"):
            self.set_tool("hand")
        elif k in ("f", "F"):
            self.find_similar_selected()
        elif k in ("x", "X"):
            self.ignore_selected()
        elif k in ("d", "D"):
            self.redetect()
        elif k == "Escape":
            self.sel = None
            self.redraw()
        elif k in ("plus", "equal", "KP_Add"):
            self.zoom_by(1.25)
        elif k in ("minus", "KP_Subtract"):
            self.zoom_by(1 / 1.25)
        elif k == "0":
            self.zoom_fit()
        elif k.isdigit() and k != "0":
            self.set_class(int(k) - 1)

    # ---- canvas
    def to_img(self, x, y):
        H, W = self.img.shape[:2]
        return min(max((x - self.ox) / self.scale, 0), W), min(max((y - self.oy) / self.scale, 0), H)

    def hit(self, ix, iy):
        best, area = None, None
        for i, b in enumerate(self.cur_boxes):
            if b.x1 <= ix <= b.x2 and b.y1 <= iy <= b.y2:
                a = (b.x2 - b.x1) * (b.y2 - b.y1)
                if area is None or a < area:
                    best, area = i, a
        return best

    HANDLE_PX = 9

    # ---- zoom / pan / brightness (view only)
    def view_changed(self):
        self.photo_key = None
        self.redraw()

    def reset_view_adjust(self):
        self.var_bright.set(0.0)
        self.var_contrast.set(1.0)
        self.var_auto.set(False)
        self.view_changed()

    def zoom_fit(self):
        self.zoom, self.vx, self.vy = 1.0, None, None
        self.view_changed()

    def zoom_by(self, f, x=None, y=None):
        """Zoom in/out keeping the image point under (x, y) fixed (default: canvas centre)."""
        if self.img is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        x = cw / 2 if x is None else x
        y = ch / 2 if y is None else y
        H, W = self.img.shape[:2]
        ix, iy = (x - self.ox) / self.scale, (y - self.oy) / self.scale
        base = min(cw / W, ch / H)
        self.zoom = min(max(self.zoom * f, 1.0), 40.0)
        ns = base * self.zoom
        self.vx, self.vy = ix + (cw / 2 - x) / ns, iy + (ch / 2 - y) / ns     # keeps (ix, iy) under the cursor
        self.view_changed()

    def on_wheel(self, e):
        """Mouse wheel / trackpad two-finger scroll: zoom. Shift + scroll = pan left/right, Alt/Option + scroll = pan up/down."""
        if self.img is None:
            return
        d = e.delta
        step = d / 120.0 if abs(d) >= 100 else d / 5.0            # trackpads send many small deltas
        step = max(-3.0, min(3.0, step))
        if e.state & 0x0001:                                       # Shift: horizontal pan (also how macOS sends sideways swipes)
            self.pan_by(-step * 60, 0)
        elif e.state & 0x0008 or e.state & 0x0010 or e.state & 0x20000:   # Alt / Option: vertical pan
            self.pan_by(0, -step * 60)
        else:
            self.zoom_by(1.15 ** step, e.x, e.y)

    def pan_by(self, dx, dy):
        """Move the view by (dx, dy) screen pixels."""
        if self.img is None or self.vx is None:
            return
        self.vx, self.vy = self.vx + dx / self.scale, self.vy + dy / self.scale
        self.redraw()

    def scroll_view(self, axis, *args):
        if self.img is None or self.vx is None:
            return
        H, W = self.img.shape[:2]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if args[0] == "moveto":
            f = float(args[1])
            if axis == "x":
                self.vx = f * W + cw / 2 / self.scale
            else:
                self.vy = f * H + ch / 2 / self.scale
        elif args[0] == "scroll":
            n = int(args[1]) * (0.1 if args[2] == "units" else 0.9)
            if axis == "x":
                self.vx += n * cw / self.scale * 0.5
            else:
                self.vy += n * ch / self.scale * 0.5
        self.redraw()

    def update_scrollbars(self, W, H, cw, ch):
        hw, hh = cw / self.scale, ch / self.scale
        x0, y0 = self.vx - hw / 2, self.vy - hh / 2
        self.hsb.set(max(0.0, x0 / W), min(1.0, (x0 + hw) / W))
        self.vsb.set(max(0.0, y0 / H), min(1.0, (y0 + hh) / H))

    def pan_start(self, e):
        if self.img is not None:
            self.pan_drag = (e.x, e.y, self.vx, self.vy)

    def pan_move(self, e):
        if not self.pan_drag or self.img is None:
            return
        x0, y0, vx0, vy0 = self.pan_drag
        self.vx, self.vy = vx0 - (e.x - x0) / self.scale, vy0 - (e.y - y0) / self.scale
        self.redraw()

    def pan_end(self, e):
        self.pan_drag = None

    def adjust_view(self, crop):
        """Brightness / contrast / auto-gamma, applied to what is shown only."""
        if self.var_auto.get() and self.auto_lut is not None:
            crop = cv2.LUT(crop, self.auto_lut)
        b, c = self.var_bright.get(), self.var_contrast.get()
        if abs(b) > 0.5 or abs(c - 1.0) > 0.01:
            crop = cv2.convertScaleAbs(crop, alpha=c, beta=b)
        return crop

    def handle_at(self, e):
        """Which resize handle of the selected box is under the mouse? -> 'nw','n','ne','e','se','s','sw','w' or None."""
        if self.sel is None or self.sel >= len(self.cur_boxes):
            return None
        b = self.cur_boxes[self.sel]
        x1, y1 = self.ox + b.x1 * self.scale, self.oy + b.y1 * self.scale
        x2, y2 = self.ox + b.x2 * self.scale, self.oy + b.y2 * self.scale
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        pts = {"nw": (x1, y1), "n": (xm, y1), "ne": (x2, y1), "e": (x2, ym),
               "se": (x2, y2), "s": (xm, y2), "sw": (x1, y2), "w": (x1, ym)}
        best, bd = None, self.HANDLE_PX + 1
        for k, (hx, hy) in pts.items():
            d = max(abs(e.x - hx), abs(e.y - hy))
            if d < bd:
                best, bd = k, d
        return best

    def on_press(self, e):
        if self.img is None or self.busy:
            return
        self.drag = None
        self.edit = None
        self.edit_pushed = False
        if self.tool == "hand" or e.state & 0x0001:            # Hand tool or Shift + left button = pan
            self.pan_start(e)
            return
        h = self.handle_at(e)
        if h is not None:                                    # resize the selected box
            b = self.cur_boxes[self.sel]
            self.edit = (h, (b.x1, b.y1, b.x2, b.y2), self.to_img(e.x, e.y))
            return
        ix, iy = self.to_img(e.x, e.y)
        hit = self.hit(ix, iy)
        if hit is not None:                                  # select + move
            self.sel = hit
            b = self.cur_boxes[hit]
            self.edit = ("move", (b.x1, b.y1, b.x2, b.y2), (ix, iy))
            self.redraw()
        else:                                                # draw a new box
            self.sel = None
            self.drag = (e.x, e.y)
            self.redraw()

    def on_motion(self, e):
        if self.pan_drag:
            self.pan_move(e)
        elif self.edit and self.sel is not None and self.sel < len(self.cur_boxes):
            mode, orig, start = self.edit
            ix, iy = self.to_img(e.x, e.y)
            if not self.edit_pushed:                         # one undo step per move / resize
                self.push_undo()
                self.edit_pushed = True
            b = self.cur_boxes[self.sel]
            b.x1, b.y1, b.x2, b.y2 = apply_box_drag(orig, mode, ix - start[0], iy - start[1], *self.img.shape[1::-1])
            b.poly, b.conf = None, 1.0                       # the outline no longer matches an edited box
            self.redraw()
        elif self.drag:
            self.canvas.delete("tmp")
            self.canvas.create_rectangle(self.drag[0], self.drag[1], e.x, e.y, outline=PALETTE[self.active_cls % len(PALETTE)],
                                         width=2, dash=(5, 3), tags="tmp")

    def on_release(self, e):
        if self.pan_drag:
            self.pan_drag = None
            return
        if self.edit:
            self.edit_pushed = False
            self.edit = None
            self.redraw()
            return
        if not self.drag:
            return
        x0, y0 = self.drag
        self.drag = None
        self.canvas.delete("tmp")
        if self.tool == "segment" and self.img is not None and abs(e.x - x0) < 5 and abs(e.y - y0) < 5:
            self.request_click(*self.to_img(x0, y0))           # a plain click on a missed object (Click & segment tool only)
            return
        if abs(e.x - x0) < 8 or abs(e.y - y0) < 8 or self.img is None:
            return
        ax, ay = self.to_img(min(x0, e.x), min(y0, e.y))
        bx, by = self.to_img(max(x0, e.x), max(y0, e.y))
        self.push_undo()
        self.cur_boxes.append(Box(ax, ay, bx, by, self.active_cls, 1.0))
        self.sel = len(self.cur_boxes) - 1
        self.redraw()

    def redraw(self):
        c = self.canvas
        c.delete("all")
        if self.img is None:
            return
        cw, ch = c.winfo_width(), c.winfo_height()
        if cw < 20 or ch < 20:
            return
        H, W = self.img.shape[:2]
        self.scale = min(cw / W, ch / H) * self.zoom
        if self.vx is None:
            self.vx, self.vy = W / 2.0, H / 2.0
        self.vx, self.vy = min(max(self.vx, 0), W), min(max(self.vy, 0), H)
        self.ox, self.oy = cw / 2 - self.vx * self.scale, ch / 2 - self.vy * self.scale
        self.update_scrollbars(W, H, cw, ch)
        # only the visible part of the image is cut out and resized
        X0, Y0 = max(0, int(np.floor(-self.ox / self.scale))), max(0, int(np.floor(-self.oy / self.scale)))
        X1, Y1 = min(W, int(np.ceil((cw - self.ox) / self.scale))), min(H, int(np.ceil((ch - self.oy) / self.scale)))
        key = (self.idx, cw, ch, round(self.scale, 5), round(self.ox), round(self.oy), X0, Y0, X1, Y1,
               round(self.var_bright.get()), round(self.var_contrast.get(), 2), self.var_auto.get())
        if key != self.photo_key:
            self.photo = None
            if X1 - X0 >= 1 and Y1 - Y0 >= 1:
                crop = self.adjust_view(self.img[Y0:Y1, X0:X1])
                nw, nh = max(1, int(round((X1 - X0) * self.scale))), max(1, int(round((Y1 - Y0) * self.scale)))
                small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR)
                self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            self.photo_key = key
            self.photo_pos = (self.ox + X0 * self.scale, self.oy + Y0 * self.scale)
        if self.photo is not None:
            c.create_image(self.photo_pos[0], self.photo_pos[1], anchor="nw", image=self.photo)
        counts = {}
        for i, b in enumerate(self.cur_boxes):
            col = PALETTE[b.cls % len(PALETTE)] if b.cls < len(self.classes) else "#ffffff"
            x1, y1, x2, y2 = (self.ox + b.x1 * self.scale, self.oy + b.y1 * self.scale,
                              self.ox + b.x2 * self.scale, self.oy + b.y2 * self.scale)
            if i == self.sel:                       # selected: a thin bright outline + small handles, edges stay easy to see
                c.create_rectangle(x1, y1, x2, y2, outline="#ffffff", width=1)
                for hx, hy in ((x1, y1), ((x1 + x2) / 2, y1), (x2, y1), (x2, (y1 + y2) / 2),
                               (x2, y2), ((x1 + x2) / 2, y2), (x1, y2), (x1, (y1 + y2) / 2)):
                    c.create_rectangle(hx - 3, hy - 3, hx + 3, hy + 3, fill="#ffffff", outline=col)
                continue
            c.create_rectangle(x1, y1, x2, y2, outline=col, width=2)
        if self.busy:
            c.create_rectangle(cw // 2 - 130, ch // 2 - 24, cw // 2 + 130, ch // 2 + 24, fill="#000000", outline="white")
            c.create_text(cw // 2, ch // 2, text="Detecting...", fill="white", font=("TkDefaultFont", 14, "bold"))
        for i, cl in enumerate(self.legend_counts):
            cl.config(text=str(sum(1 for b in self.cur_boxes if b.cls == i)))
        self.refresh_ann_list()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="folder with the images (can also be chosen in the window)")
    ap.add_argument("--output", default=None, help="output folder (default: <images folder name>_annotated, next to the images folder)")
    a = ap.parse_args()
    sys.excepthook = lambda t, v, tb: log_error("".join(traceback.format_exception(t, v, tb)))
    try:
        threading.excepthook = lambda args: log_error("Thread error:\n" + "".join(
            traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))
    except Exception:
        pass
    app = App(a.images, a.output)
    if a.output:
        app.maybe_load_setup(silent=True)
    app.mainloop()


if __name__ == "__main__":
    main()