"""
Safety layer for the Type2 nut detector (RealSense, 3 classes incl. "occluded").

Three independent checks run on top of your YOLO model, so a single model miss
does not become a silent miss:

  A. YOLO            - your trained model (all classes, low confidence)
  B. SAM 3           - open-vocabulary "nut" masks; any mask YOLO did not explain
                       is cropped and re-classified (heavier, run on demand)
  C. Leftover search - model-free: compare the live frame with an EMPTY tray
                       (depth + color). Cover everything YOLO explained plus the
                       fixtures; any nut-sized blob left over is a MISS.

Status of a report:  BG_MISMATCH the scene does not match the stored empty-tray picture (old file, moved camera/tray,
                     changed light): the leftover search is NOT trusted, recapture the background (key b)
                       PASS        verified (empty-tray background present) and nothing to check
                     CHECK       at least one alert (miss / weak box / possible stack / large unexplained blob ...)
                     UNVERIFIED  no empty-tray background, so only YOLO looked: this is NOT a pass

Depth is used to:
  - measure every candidate in millimetres (size filter is camera-height independent,
    so small screws are rejected by real size, not pixels)
  - flag possible stacks: a box whose height is ~2 nut thicknesses but is not
    labelled "occluded"
  - return 3D points (camera frame, optional robot frame) for the arm
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np


# =============================================================================
# Config (override from config/config.py, see run_inspection.py)
# =============================================================================
@dataclass
class SafetyConfig:
    # --- YOLO --------------------------------------------------------------
    yolo_conf_min: float = 0.25        # detections kept for display / matching
    explain_conf: float = 0.50         # only boxes >= this "explain" a region in leftover search
    occluded_names: tuple = ("occluded",)   # class name(s) meaning stacked / overlapped
    box_margin_px: int = 6             # grow explained boxes a little (edges, shadows)

    # --- real nut size (millimetres) -----------------------------------------
    # Set these from YOUR nuts: run_inspection.py key 'm' prints the measured sizes of the boxes YOLO is sure about.
    nut_side_mm: tuple = (8.0, 45.0)   # equivalent side sqrt(area): below = small screw (ignored), above = ALERT (large blob)
    nut_thickness_mm: float | None = None   # None = learn it per class from the confident boxes of each frame
    max_aspect: float = 2.5            # elongated blobs (cables, tool edges): ALERT as large_blob, not ignored
    min_width_mm: float = 7.0          # a blob whose SHORT side is below this is a screw / cable / dust: ignored

    # --- leftover search -----------------------------------------------------
    fg_mode: str = "either"            # "depth" | "color" | "both" | "either"
    min_height_mm: float = 2.0         # depth: above empty-tray surface
    max_height_mm: float = 60.0        # ignore hands / arm passing over
    color_diff_thr: float = 28.0       # LAB distance vs empty-tray image
    holes_as_fg: bool = True           # shiny metal often gives depth = 0 -> suspicious
    min_fill: float = 0.35             # blob area / area of its minimum rectangle: ragged slivers fail this
    split_touching: bool = True        # split a blob of touching nuts into single nuts (distance transform + watershed)
    bg_mismatch_frac: float = 0.20     # more of the tray area than this is unexplained AND differs from the empty tray = the
                                       # background is old / the camera or tray moved / the light changed: status BG_MISMATCH
    brightness_compensation: bool = True   # remove a global brightness change vs the empty-tray picture (auto exposure)
    open_px: int = 3
    close_px: int = 5

    # --- stack check ----------------------------------------------------------
    stack_ratio: float = 1.6           # p90 height > ratio * thickness -> possible stack

    # --- SAM 3 -----------------------------------------------------------------
    sam_prompts: tuple = ("hex nut", "nut")
    sam_score_min: float = 0.30
    sam_explained_frac: float = 0.30   # mask counted as explained if >= this fraction is covered
    crop_conf_min: float = 0.25
    crop_min_side: int = 320

    # --- geometry --------------------------------------------------------------
    tray_roi: list | None = None       # polygon [[x,y],...] in color pixels; None = whole image
    ignore_polys: list = field(default_factory=list)   # fixtures, clamps, labels...
    cam_to_robot: list | None = None   # 4x4 hand-eye matrix (camera -> robot base), metres


@dataclass
class Item:
    box: list                  # [x1,y1,x2,y2] color pixels
    label: str                 # class name, "unknown", or "MISS"
    score: float
    source: str                # "yolo" | "sam3" | "leftover" | "leftover+sam3"
    status: str = "ok"         # ok | review | possible_stack | recovered_by_sam3 | miss
    size_mm: float | None = None
    height_mm: float | None = None
    xyz_cam: list | None = None
    xyz_robot: list | None = None
    contour: np.ndarray | None = None

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)


# =============================================================================
# Helpers
# =============================================================================
def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x)


def poly_mask(shape, polys, fill_if_none=True):
    m = np.zeros(shape[:2], np.uint8)
    if polys is None:
        if fill_if_none:
            m[:] = 1
        return m.astype(bool)
    for p in polys:
        cv2.fillPoly(m, [np.asarray(p, np.int32)], 1)
    return m.astype(bool)


class Intrinsics:
    """Pinhole intrinsics of the COLOR stream (depth is aligned to color)."""
    def __init__(self, fx, fy, ppx, ppy, rs_intr=None):
        self.fx, self.fy, self.ppx, self.ppy, self.rs = fx, fy, ppx, ppy, rs_intr

    @classmethod
    def from_rs(cls, rs_intr):
        return cls(rs_intr.fx, rs_intr.fy, rs_intr.ppx, rs_intr.ppy, rs_intr)

    def mm_per_px(self, z_m):
        return z_m * 1000.0 / self.fx, z_m * 1000.0 / self.fy

    def deproject(self, u, v, z_m):
        if self.rs is not None:
            import pyrealsense2 as rs
            return list(rs.rs2_deproject_pixel_to_point(self.rs, [float(u), float(v)], float(z_m)))
        return [(u - self.ppx) * z_m / self.fx, (v - self.ppy) * z_m / self.fy, z_m]


def robust_depth(depth_m, mask, fallback_depth=None):
    """Median valid depth inside mask; falls back to empty-tray depth (holes on shiny metal)."""
    vals = depth_m[mask & (depth_m > 0)]
    if vals.size >= 5:
        return float(np.median(vals))
    if fallback_depth is not None:
        vals = fallback_depth[mask & (fallback_depth > 0)]
        if vals.size:
            return float(np.median(vals))
    return 0.0


def _contour_of(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def size_mm_of(mask, z_m, intr: Intrinsics):
    """-> (equivalent side in mm = sqrt(area), aspect of the minimum-area rectangle, fill = area / rectangle area,
    short side of that rectangle in mm). The rectangle is rotated, so a thin DIAGONAL blob is recognised as thin."""
    if z_m <= 0 or not mask.any():
        return None, None, None, None
    mx, my = intr.mm_per_px(z_m)
    area_px = float(np.count_nonzero(mask))
    aspect, fill = 1.0, 1.0
    short_mm = float(np.sqrt(area_px) * (mx + my) / 2.0)
    c = _contour_of(mask)
    if c is not None and len(c) >= 3:
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        short_mm = float(min(rw, rh) * (mx + my) / 2.0)
        if min(rw, rh) > 0.5:
            aspect, fill = max(rw, rh) / min(rw, rh), min(1.0, area_px / (rw * rh))
        else:
            aspect = 99.0
    return float(np.sqrt(area_px * mx * my)), float(aspect), float(fill), short_mm


def nut_sized(side_mm, aspect, cfg: SafetyConfig, fill=None):
    if side_mm is None:
        return False
    return (cfg.nut_side_mm[0] <= side_mm <= cfg.nut_side_mm[1] and aspect <= cfg.max_aspect
            and (fill is None or fill >= cfg.min_fill))


# =============================================================================
# Background (empty tray)
# =============================================================================
def ensure_metres(depth):
    """RealSense wrappers differ: if the median valid depth is above 20 it cannot be metres, so it is millimetres."""
    v = depth[depth > 0]
    if v.size and float(np.median(v)) > 20.0:
        warnings.warn("depth looks like millimetres - converting to metres (check depth_scale in get_frames)")
        return depth.astype(np.float32) / 1000.0
    return depth


class Background:
    def __init__(self, color_bgr, depth_m):
        self.color = color_bgr
        self.depth = ensure_metres(np.asarray(depth_m, np.float32))
        self.lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    @classmethod
    def capture(cls, grab_fn, n=30):
        """grab_fn() -> (color_bgr, depth_m). Median over n frames, ignoring depth holes."""
        colors, depths, tries = [], [], 0
        while len(colors) < n:
            tries += 1
            if tries > n * 30:
                raise RuntimeError("Camera gave no frames while capturing the background")
            c, d = grab_fn()
            if c is None:
                continue
            colors.append(np.array(c, copy=True)); depths.append(np.array(d, copy=True))   # frames may be reused buffers
        D = np.stack(depths).astype(np.float32)
        D[D <= 0] = np.nan
        depth = np.nan_to_num(np.nanmedian(D, axis=0), nan=0.0)
        color = np.median(np.stack(colors), axis=0).astype(np.uint8)
        return cls(color, depth)

    def save(self, path):
        np.savez_compressed(path, color=self.color, depth=self.depth)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["color"], z["depth"])


# =============================================================================
# A. YOLO
# =============================================================================
def run_yolo(model, image_bgr, cfg: SafetyConfig, conf=None, imgsz=None) -> list[Item]:
    kw = dict(conf=conf or cfg.yolo_conf_min, iou=0.6, verbose=False)
    if imgsz:
        kw["imgsz"] = imgsz
    r = model.predict(source=image_bgr, **kw)[0]
    names = r.names
    items = []
    for b, s, c in zip(_np(r.boxes.xyxy), _np(r.boxes.conf), _np(r.boxes.cls)):
        items.append(Item(b.tolist(), names[int(c)], float(s), "yolo"))
    return items


def explained_mask(shape, yolo_items, cfg: SafetyConfig):
    m = poly_mask(shape, cfg.ignore_polys, fill_if_none=False) if cfg.ignore_polys else np.zeros(shape[:2], bool)
    H, W = shape[:2]
    g = cfg.box_margin_px
    for it in yolo_items:
        if it.score < cfg.explain_conf:       # weak boxes do NOT hide anything
            continue
        x1, y1, x2, y2 = map(int, it.box)
        m[max(0, y1 - g):min(H, y2 + g), max(0, x1 - g):min(W, x2 + g)] = True
    return m


# =============================================================================
# C. Leftover search
# =============================================================================
def foreground(color_bgr, depth_m, bg: Background, cfg: SafetyConfig):
    """Pixels that differ from the empty tray. Returns (fg_mask, height_mm)."""
    valid_bg = bg.depth > 0
    valid = (depth_m > 0) & valid_bg
    height_mm = np.zeros(depth_m.shape, np.float32)
    height_mm[valid] = (bg.depth[valid] - depth_m[valid]) * 1000.0   # camera looks down

    depth_fg = valid & (height_mm >= cfg.min_height_mm) & (height_mm <= cfg.max_height_mm)
    if cfg.holes_as_fg:
        depth_fg |= valid_bg & (depth_m <= 0)

    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if cfg.brightness_compensation:      # auto-exposure / light drift: most of the tray is unchanged, so the median
        lab[..., 0] -= float(np.median(lab[..., 0] - bg.lab[..., 0]))     # difference in L is the global change
    color_fg = np.linalg.norm(lab - bg.lab, axis=2) >= cfg.color_diff_thr

    fg = {"depth": depth_fg, "color": color_fg,
          "both": depth_fg & color_fg, "either": depth_fg | color_fg}[cfg.fg_mode]
    fg = fg.astype(np.uint8)
    if cfg.open_px:
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((cfg.open_px,) * 2, np.uint8))
    if cfg.close_px:
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((cfg.close_px,) * 2, np.uint8))
    return fg.astype(bool), height_mm


def _split_blob(m, frac=0.55):
    """Split a blob of touching objects: seeds = the peaks of the distance transform, then watershed.
    Returns a list of bool masks (the blob itself if it cannot be split)."""
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = m[y0:y1, x0:x1].astype(np.uint8)
    d = cv2.distanceTransform(np.pad(sub, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    if d.max() < 3:
        return [m]
    seeds = (d >= frac * d.max()).astype(np.uint8)
    n, lab = cv2.connectedComponents(seeds)
    if n <= 2:
        return [m]
    markers = lab.astype(np.int32)
    markers[sub == 0] = n                                   # outside = its own "background" label
    img = cv2.cvtColor((255 - 255 * d / d.max()).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(img, markers)
    out = []
    for k in range(1, n):
        piece = np.zeros(m.shape, bool)
        piece[y0:y1, x0:x1] = (markers == k) & (sub > 0)
        if piece.sum() >= 6:
            out.append(piece)
    return out or [m]


def _blob_item(m, depth_m, bg, intr, height_mm, cfg):
    """-> (kind, Item). kind: 'nut' | 'small' (ignored) | 'odd' (too big / elongated / ragged / unsized: ALERT)."""
    ys, xs = np.nonzero(m)
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    z = robust_depth(depth_m, m, bg.depth)
    side, aspect, fill, short_mm = size_mm_of(m, z, intr)
    item = Item(box, "MISS", 1.0, "leftover", "miss", size_mm=side,
                height_mm=float(np.percentile(height_mm[m], 90)) if m.any() else None)
    item.contour = _contour_of(m)
    if side is None:                                         # no depth anywhere: cannot size it, so do not trust it
        kind = "odd"
    elif side < cfg.nut_side_mm[0] or short_mm < cfg.min_width_mm:
        kind = "small"                                       # small screw, thin screw lying down, cable, dust
    elif nut_sized(side, aspect, cfg, fill):
        kind = "nut"
    else:
        kind = "odd"
    if kind == "odd":
        item.label, item.status = "UNKNOWN BLOB", "large_blob"
    return kind, item


def leftover_search(color_bgr, depth_m, bg, explained, intr, cfg: SafetyConfig):
    """Returns (alerts, rejected, residual, height_mm). `alerts` = nut-sized misses AND oversized / odd blobs
    (status 'miss' / 'large_blob'); only things smaller than a nut (small screws, dust) are ignored (`rejected`)."""
    fg, height_mm = foreground(color_bgr, depth_m, bg, cfg)
    roi = poly_mask(color_bgr.shape, [cfg.tray_roi] if cfg.tray_roi else None)
    residual = fg & roi & ~explained
    fg_frac = float(residual.sum()) / max(1, int(roi.sum()))        # share of the tray area that differs from the empty tray
                                                                    # and that YOLO does not explain

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(residual.astype(np.uint8), 8)
    alerts, rejected = [], []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 6:
            continue
        m = lbl == k
        kind, item = _blob_item(m, depth_m, bg, intr, height_mm, cfg)
        if kind in ("nut", "odd") and cfg.split_touching:
            pieces = _split_blob(m)
            if len(pieces) > 1:                              # touching nuts: judge every piece on its own
                parts = [_blob_item(p, depth_m, bg, intr, height_mm, cfg) for p in pieces]
                # a blob that already looks like ONE nut is only split if every piece is nut-sized again
                # (a nut with a hole or a ragged edge must not become several "nuts")
                if kind == "odd" or all(pk == "nut" for pk, _ in parts):
                    for pk, pi in parts:
                        (rejected if pk == "small" else alerts).append(pi)
                    continue
        (rejected if kind == "small" else alerts).append(item)
    return alerts, rejected, residual, height_mm, fg_frac


# =============================================================================
# Stack check (helps the "occluded" class)
# =============================================================================
def stack_check(yolo_items, height_mm, cfg: SafetyConfig):
    """Height of every YOLO box above the empty tray. A box much taller than its class normally is = possible stack.
    Normal height = cfg.nut_thickness_mm, or (None) the median of the confident boxes of the same class in this frame
    (at least 3 needed; otherwise the median of all confident boxes)."""
    for it in yolo_items:
        x1, y1, x2, y2 = map(int, it.box)
        dx, dy = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)         # inner 60% of the box: no edges / neighbours
        patch = height_mm[max(0, y1 + dy):max(0, y2 - dy), max(0, x1 + dx):max(0, x2 - dx)]
        patch = patch[patch > 0]
        if patch.size >= 10:
            it.height_mm = float(np.percentile(patch, 90))
    ref_cls, ref_all = {}, None
    if cfg.nut_thickness_mm is None:
        conf = [i for i in yolo_items if i.height_mm and i.score >= cfg.explain_conf and i.label not in cfg.occluded_names]
        for lab in {i.label for i in conf}:
            hs = [i.height_mm for i in conf if i.label == lab]
            if len(hs) >= 3:
                ref_cls[lab] = float(np.median(hs))
        if len(conf) >= 3:
            ref_all = float(np.median([i.height_mm for i in conf]))
    for it in yolo_items:
        if not it.height_mm or it.label in cfg.occluded_names:
            continue
        ref = cfg.nut_thickness_mm if cfg.nut_thickness_mm is not None else ref_cls.get(it.label, ref_all)
        if ref and it.height_mm > cfg.stack_ratio * ref:
            it.status = "possible_stack"
    return yolo_items


# =============================================================================
# B. SAM 3 (on demand)
# =============================================================================
def load_sam3():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    return Sam3Processor(build_sam3_image_model())


def sam3_masks(processor, color_bgr, cfg: SafetyConfig):
    """Returns list of (bool mask HxW, score). Check mask/box format for your sam3 version."""
    from PIL import Image
    H, W = color_bgr.shape[:2]
    try:
        import torch
        ctx = torch.inference_mode()
    except ImportError:
        torch, ctx = None, contextlib.nullcontext()
    out_masks = []
    with ctx:
        state = processor.set_image(Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)))
        for p in cfg.sam_prompts:
            if hasattr(processor, "reset_all_prompts"):
                processor.reset_all_prompts(state)
            out = processor.set_text_prompt(state=state, prompt=p)
            if not isinstance(out, dict):                  # some versions fill `state` in place and return nothing
                out = state
            if "masks" not in out or "scores" not in out:
                raise RuntimeError(f"unexpected SAM 3 output, keys: {list(out)[:10]} - check your sam3 version")
            masks, scores = _np(out["masks"]), _np(out["scores"]).reshape(-1)
            masks = masks.reshape(-1, *masks.shape[-2:])
            for m, s in zip(masks, scores):
                if s < cfg.sam_score_min:
                    continue
                m = m > 0.5 if m.dtype != bool else m
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                out_masks.append((m, float(s)))
        del state
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()                           # give the GPU memory back after every SAM 3 pass
    # dedupe by mask IoU
    kept = []
    for m, s in sorted(out_masks, key=lambda t: -t[1]):
        if all((m & k).sum() / max((m | k).sum(), 1) < 0.6 for k, _ in kept):
            kept.append((m, s))
    return kept


def classify_crop(model, color_bgr, box, cfg: SafetyConfig):
    H, W = color_bgr.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1), (y2 - y1)
    c = color_bgr[int(max(0, y1 - ph)):int(min(H, y2 + ph)), int(max(0, x1 - pw)):int(min(W, x2 + pw))]
    if c.size == 0:
        return "unknown", 0.0
    s = max(1.0, cfg.crop_min_side / min(c.shape[:2]))
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    dets = run_yolo(model, c, cfg, conf=cfg.crop_conf_min, imgsz=max(c.shape[:2]))
    if not dets:
        return "unknown", 0.0
    cx, cy = c.shape[1] / 2, c.shape[0] / 2
    best = min(dets, key=lambda d: np.hypot(d.center[0] - cx, d.center[1] - cy))
    return best.label, best.score


def sam3_recover(processor, model, color_bgr, depth_m, bg, explained, intr, cfg, misses):
    """SAM masks not explained by YOLO -> size check in mm -> crop classify.
    Leftover misses hit by SAM are upgraded to 'leftover+sam3'."""
    recovered, rejected = [], []
    for m, s in sam3_masks(processor, color_bgr, cfg):
        if (m & explained).sum() / max(m.sum(), 1) >= cfg.sam_explained_frac:
            continue
        ys, xs = np.nonzero(m)
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        z = robust_depth(depth_m, m, bg.depth if bg else None)
        side, aspect, fill, short_mm = size_mm_of(m, z, intr)
        if short_mm is not None and short_mm < cfg.min_width_mm or not nut_sized(side, aspect, cfg, fill):
            rejected.append(Item(box, "sam_rejected", s, "sam3", "rejected", size_mm=side))
            continue
        hit = next((mi for mi in misses if _box_iou(mi.box, box) > 0.2), None)
        label, score = classify_crop(model, color_bgr, box, cfg) if model else ("unknown", 0.0)
        if hit is not None:
            hit.source, hit.label, hit.score = "leftover+sam3", f"MISS ({label})", score or s
            continue
        recovered.append(Item(box, label, score or s, "sam3", "recovered_by_sam3", size_mm=side))
    return recovered, rejected


def _box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


# =============================================================================
# Full inspection
# =============================================================================
def add_3d(items, depth_m, bg, intr, cfg):
    """3D point + size of every item. The depth is taken from the INNER half of the box (the nut top, not the tray
    around it): the nearest quarter of the valid depth values. Holes on shiny metal fall back to the tray depth."""
    T = np.asarray(cfg.cam_to_robot, np.float64) if cfg.cam_to_robot is not None else None
    H, W = depth_m.shape
    for it in items:
        u, v = it.center
        x1, y1, x2, y2 = [int(round(c)) for c in it.box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        dx, dy = (x2 - x1) // 4, (y2 - y1) // 4
        patch = depth_m[y1 + dy:max(y1 + dy + 1, y2 - dy), x1 + dx:max(x1 + dx + 1, x2 - dx)]
        vals = patch[patch > 0]
        if vals.size >= 5:
            z = float(np.percentile(vals, 25))
        else:
            m = np.zeros((H, W), bool); m[y1:y2, x1:x2] = True
            z = robust_depth(depth_m, m, bg.depth if bg else None)
        if z <= 0:
            continue
        it.xyz_cam = [round(float(q), 4) for q in intr.deproject(u, v, z)]
        if it.size_mm is None:                              # YOLO boxes: equivalent side of the box in mm
            mx, my = intr.mm_per_px(z)
            it.size_mm = float(np.sqrt((x2 - x1) * mx * (y2 - y1) * my))
        if T is not None:
            p = T @ np.array([*it.xyz_cam, 1.0])
            it.xyz_robot = [round(float(q), 4) for q in p[:3]]


def inspect(color_bgr, depth_m, model, intr: Intrinsics, cfg: SafetyConfig,
            bg: Background | None = None, sam_processor=None):
    depth_m = ensure_metres(np.asarray(depth_m, np.float32))
    if bg is not None and bg.color.shape[:2] != color_bgr.shape[:2]:
        raise ValueError(f"The empty-tray background is {bg.color.shape[1]}x{bg.color.shape[0]} but the camera gives "
                         f"{color_bgr.shape[1]}x{color_bgr.shape[0]}: capture the background again (key 'b')")
    yolo = run_yolo(model, color_bgr, cfg)
    for it in yolo:
        if it.score < cfg.explain_conf:
            it.status = "review"
    explained = explained_mask(color_bgr.shape, yolo, cfg)

    misses, left_rej, residual, height_mm = [], [], None, np.zeros(depth_m.shape, np.float32)
    mismatch, fg_frac = False, 0.0
    if bg is not None:
        misses, left_rej, residual, height_mm, fg_frac = leftover_search(color_bgr, depth_m, bg, explained, intr, cfg)
        if fg_frac > cfg.bg_mismatch_frac:                   # nothing below can be trusted: do not flood with alerts
            mismatch, misses, left_rej = True, [], []
        else:
            stack_check(yolo, height_mm, cfg)
        # a leftover blob sitting on a weak YOLO box is the same object: keep one alert (the box)
        weak = [y for y in yolo if y.status == "review"]
        for mi in list(misses):
            w = next((y for y in weak if _box_iou(y.box, mi.box) > 0.3), None)
            if w is not None:
                w.source, w.size_mm = "yolo+leftover", mi.size_mm
                misses.remove(mi)

    recovered, sam_rej = [], []
    if sam_processor is not None:
        recovered, sam_rej = sam3_recover(sam_processor, model, color_bgr, depth_m, bg,
                                          explained, intr, cfg, misses)

    items = yolo + recovered + misses
    add_3d(items, depth_m, bg, intr, cfg)

    alerts = [i for i in items if i.status in ("miss", "large_blob", "review", "possible_stack", "recovered_by_sam3")]
    warns = []
    if bg is None:
        warns.append("No empty-tray background: the leftover search is OFF and only YOLO is checking.")
    if mismatch:
        warns.append(f"{fg_frac * 100:.0f}% of the tray area differs from the empty-tray picture (and YOLO does not explain it): the stored background is "
                     "old, the camera or tray moved, or the light changed. The leftover search is NOT trusted. "
                     "Capture the background again with an empty tray (key b) and set the tray area (key r).")
    status = ("BG_MISMATCH" if mismatch else "CHECK" if alerts else ("UNVERIFIED" if bg is None else "PASS"))
    report = {
        "status": status,
        "pass": status == "PASS",
        "warnings": warns,
        "tray_changed_pct": round(fg_frac * 100, 1),
        "counts": {
            "yolo": len(yolo),
            "by_class": {n: sum(i.label == n for i in yolo) for n in sorted({i.label for i in yolo})},
            "review_low_conf": sum(i.status == "review" for i in yolo),
            "possible_stack": sum(i.status == "possible_stack" for i in yolo),
            "recovered_by_sam3": len(recovered),
            "leftover_miss": sum(i.status == "miss" for i in misses),
            "large_blob": sum(i.status == "large_blob" for i in misses),
            "leftover_ignored_small": len(left_rej),
            "sam3_rejected_by_size": len(sam_rej),
        },
        "alerts": [_to_json(i) for i in alerts],
        "items": [_to_json(i) for i in items],
    }
    debug = {"explained": explained, "residual": residual, "height_mm": height_mm,
             "rejected": left_rej + sam_rej}
    return report, items, debug


def _to_json(i: Item):
    return {"label": i.label, "status": i.status, "source": i.source, "score": round(i.score, 3),
            "bbox_xyxy": [round(v, 1) for v in i.box],
            "center_px": [round(v, 1) for v in i.center],
            "size_mm": None if i.size_mm is None else round(i.size_mm, 1),
            "height_mm": None if i.height_mm is None else round(i.height_mm, 1),
            "xyz_cam_m": i.xyz_cam, "xyz_robot_m": i.xyz_robot}


# =============================================================================
# Drawing
# =============================================================================
PALETTE = [(60, 180, 60), (220, 160, 40), (40, 200, 220), (200, 80, 200), (100, 100, 240)]
STATUS_COLOR = {"miss": (255, 0, 255), "large_blob": (128, 0, 255), "review": (0, 165, 255),
                "possible_stack": (0, 100, 255), "recovered_by_sam3": (255, 255, 0)}


def draw(color_bgr, items, debug, cfg: SafetyConfig, report=None, show_rejected=False):
    img = color_bgr.copy()
    if cfg.tray_roi:
        cv2.polylines(img, [np.asarray(cfg.tray_roi, np.int32)], True, (200, 200, 200), 1)
    for p in cfg.ignore_polys:
        cv2.polylines(img, [np.asarray(p, np.int32)], True, (120, 120, 120), 1)
    if show_rejected:
        for r in debug.get("rejected", []):
            x1, y1, x2, y2 = map(int, r.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (130, 130, 130), 1)
    names = sorted({i.label for i in items if i.source == "yolo"})
    for i in items:
        x1, y1, x2, y2 = map(int, i.box)
        if i.status in STATUS_COLOR:
            col = STATUS_COLOR[i.status]
        else:
            col = PALETTE[names.index(i.label) % len(PALETTE)] if i.label in names else (255, 255, 255)
        if i.status in ("miss", "large_blob") and i.contour is not None:
            overlay = img.copy()
            cv2.drawContours(overlay, [i.contour], -1, col, -1)
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if i.status != "ok" else 2)
        tag = i.label if i.status == "ok" else f"{i.label} [{i.status}]"
        if i.size_mm:
            tag += f" {i.size_mm:.0f}mm"
        cv2.putText(img, tag, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    if report is not None:
        st = report["status"]
        txt = {"PASS": "PASS", "UNVERIFIED": "UNVERIFIED: no background (press b)",
               "BG_MISMATCH": "BACKGROUND DOES NOT MATCH: press b (empty tray)"}.get(st, f"CHECK: {len(report['alerts'])} alert(s)")
        col = {"PASS": (0, 140, 0), "UNVERIFIED": (0, 140, 230), "BG_MISMATCH": (0, 140, 230)}.get(st, (0, 0, 200))
        cv2.rectangle(img, (0, 0), (max(260, 11 * len(txt) + 16), 30), col, -1)
        cv2.putText(img, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img