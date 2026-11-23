"""
RealSense recorder + live nut detection with SAM 3 (text prompt).

Same recording as before (colour + depth video, intrinsics), plus a live detector:
  - SAM 3 finds everything that matches the TEXT PROMPT in the colour image (default "nut"),
  - the box of every object is taken from its outline (segmentation first, then box),
  - the distance to every object is read from the aligned depth image.

Two detectors vote: SAM 3 (text prompt) and a classical mask method (background subtraction). A box is drawn only
where BOTH found an object at the same place (green). The box itself comes from SAM 3's outline.

Keys (click on the "RGB" window first)
    d   detection ON / OFF
    m   also show what only ONE detector saw (yellow = SAM 3 only, red = mask only)
    s   save a snapshot (image with boxes + a .json with boxes and distances)
    q   quit and save the videos

Run
    python realsense_record_detect.py
    python realsense_record_detect.py --prompt "nut, screw" --threshold 0.4

Needs   pip install pyrealsense2 opencv-python numpy pillow torch transformers
SAM 3   weights "facebook/sam3": accept the licence on Hugging Face and log in once (huggingface-cli login).
Detection runs in a background thread, so the live video and the recording never wait for it. The boxes you see
belong to the latest finished detection (a fraction of a second to a few seconds old, depending on your GPU).
"""
import argparse
import json
import os
import queue
import threading
import time
import traceback
from datetime import datetime

import cv2
import numpy as np

# ==================== CONFIG ====================
WIDTH, HEIGHT, FPS = 1280, 720, 30
SAVE_RAW_DEPTH = False
OUTPUT_DIR = "Jinhap_Nuts_Type_4"
PROMPT = "nut"            # what SAM 3 should find; several words separated by commas: "nut, screw"
THRESHOLD = 0.35          # SAM 3 confidence: lower finds more (and more wrong) objects
MIN_AREA_PCT = 0.02       # ignore objects smaller than this % of the image
MAX_AREA_PCT = 10.0       # ignore objects larger than this % of the image
DETECT_AT_START = True
# ---- the second opinion: classical mask (background subtraction, no AI)
USE_MASK_AGREEMENT = True   # True: a box is drawn only where SAM 3 AND the mask method both see an object
AGREE_IOU = 0.25            # how much the two boxes must overlap to count as "the same nut" (0-1); a mask box whose
                            # centre lies inside the SAM 3 box (or the other way round) also counts as agreeing
MASK_FALLBACK = True        # if the mask method finds NOTHING in a frame (e.g. dark tray), SAM 3 boxes are drawn
                            # alone instead of drawing nothing; the status line says so
SHOW_SINGLE_AT_START = True # start with yellow (SAM 3 only) / red (mask only) boxes visible; key m switches them
MASK_SENSITIVITY = 0.5      # 0-1, higher finds fainter objects
MASK_OBJECT_PCT = 20.0      # largest object width as % of the image width (sizes the background estimate)


# ==================== CLASSICAL MASK DETECTOR (no AI) ====================
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


def inside(a, b):
    """Is the centre of box a inside box b?"""
    cx, cy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def agree(sam_boxes, mask_boxes, thr):
    """-> (agreed, sam_only, mask_only). A SAM 3 box is 'agreed' when a mask box overlaps it by at least `thr` (IoU);
    every mask box is used at most once."""
    agreed, sam_only, used = [], [], set()
    for sb in sorted(sam_boxes, key=lambda b: -b[4]):
        best, bi = 0.0, None
        for j, mb in enumerate(mask_boxes):
            if j in used:
                continue
            v = iou(sb, mb)
            if v < thr and (inside(mb, sb) or inside(sb, mb)):
                v = thr
            if v > best:
                best, bi = v, j
        if bi is not None and best >= thr:
            used.add(bi)
            agreed.append((sb[0], sb[1], sb[2], sb[3], sb[4]))
        else:
            sam_only.append((sb[0], sb[1], sb[2], sb[3], sb[4]))
    mask_only = [(m[0], m[1], m[2], m[3], m[4]) for j, m in enumerate(mask_boxes) if j not in used]
    return agreed, sam_only, mask_only


# ==================== SAM 3 DETECTOR ====================
class NutDetector:
    """SAM 3 text-prompt detector running in its own thread. submit() gives it the newest frame (older waiting
    frames are dropped), latest() returns the boxes of the last finished detection."""

    def __init__(self, prompt, threshold):
        self.prompts = [p.strip() for p in prompt.split(",") if p.strip()] or ["nut"]
        self.threshold = threshold
        self.enabled = DETECT_AT_START
        self.ready = False
        self.error = ""
        self.last_seconds = 0.0
        self._q = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._boxes = []                       # agreed boxes [(x1, y1, x2, y2, score)]
        self._sam_only, self._mask_only = [], []
        self.show_all = SHOW_SINGLE_AT_START
        self.counts = (0, 0, 0)                # (SAM 3 boxes, mask boxes, agreed boxes) of the last detection
        self.note = ""
        self._stamp = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---- public
    def submit(self, bgr):
        if not (self.enabled and self.ready):
            return
        try:
            self._q.put_nowait(bgr.copy())
        except queue.Full:
            pass                               # still busy: the newest frame is tried next time

    def latest(self):
        with self._lock:
            return list(self._boxes), self._stamp

    def latest_extra(self):
        with self._lock:
            return list(self._sam_only), list(self._mask_only)

    def clear(self):
        with self._lock:
            self._boxes, self._sam_only, self._mask_only, self._stamp = [], [], [], 0.0

    def stop(self):
        self._stop = True
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    # ---- worker
    def _load(self):
        import torch
        from PIL import Image
        from transformers import Sam3Model, Sam3Processor
        self.torch, self.Image = torch, Image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading SAM 3 on {self.device} ...")
        self.model = Sam3Model.from_pretrained("facebook/sam3").eval().to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.ready = True
        print("[INFO] SAM 3 ready. Press 'd' in the RGB window to switch detection on/off.")

    def _loop(self):
        try:
            self._load()
        except Exception as e:
            self.error = f"SAM 3 could not be loaded: {type(e).__name__}: {e}"
            print("[ERROR]", self.error)
            print("        (pip install torch transformers, accept the facebook/sam3 licence, huggingface-cli login)")
            return
        while not self._stop:
            frame = self._q.get()
            if frame is None:
                return
            t0 = time.time()
            try:
                sam_boxes = self._detect(frame)
                if USE_MASK_AGREEMENT:
                    H, W = frame.shape[:2]
                    mb = [(c[0], c[1], c[2], c[3], float(c[4])) for c in
                          detect_mask(frame, MASK_SENSITIVITY, MASK_OBJECT_PCT, MIN_AREA_PCT, MAX_AREA_PCT)]
                    boxes, s_only, m_only = agree(sam_boxes, mb, AGREE_IOU)
                    self.note = ""
                    if MASK_FALLBACK and not mb and sam_boxes:
                        boxes, s_only, self.note = list(sam_boxes), [], "mask found nothing: SAM 3 only"
                    self.counts = (len(sam_boxes), len(mb), len(boxes))
                else:
                    self.counts = (len(sam_boxes), 0, len(sam_boxes))
                    boxes, s_only, m_only = sam_boxes, [], []
                with self._lock:
                    self._boxes, self._sam_only, self._mask_only = boxes, s_only, m_only
                    self._stamp = time.time()
                self.last_seconds = time.time() - t0
                self.error = ""
                if self.counts != getattr(self, "_printed", None):
                    self._printed = self.counts
                    print(f"[DETECT] SAM 3: {self.counts[0]}  mask: {self.counts[1]}  drawn: {self.counts[2]}"
                          f"  ({self.last_seconds:.1f} s) {self.note}")
            except Exception as e:
                if "out of memory" in str(e).lower():
                    self.error = "GPU out of memory - close other programs"
                    try:
                        self.torch.cuda.empty_cache()
                    except Exception:
                        pass
                else:
                    self.error = f"{type(e).__name__}: {e}"
                    traceback.print_exc()
                time.sleep(0.5)

    def _detect(self, bgr):
        torch = self.torch
        H, W = bgr.shape[:2]
        pil = self.Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        found = []
        for prompt in self.prompts:
            inputs = self.processor(images=pil, text=prompt, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
            res = self.processor.post_process_instance_segmentation(
                outputs, threshold=self.threshold, mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist())[0]
            scores = res["scores"].detach().float().cpu().numpy().ravel()
            masks = res["masks"].detach().cpu().numpy()
            for m, sc in zip(masks.reshape(len(scores), *masks.shape[-2:]), scores):
                b = mask_to_box(m)                 # segmentation first, then the box from the outline
                if b is None:
                    continue
                area = (b[2] - b[0]) * (b[3] - b[1]) / float(W * H) * 100.0
                if MIN_AREA_PCT <= area <= MAX_AREA_PCT:
                    found.append((b[0], b[1], b[2], b[3], float(sc)))
            del inputs, outputs, res
        return nms(found, 0.5)


def mask_to_box(m):
    m = (np.asarray(m) > 0.5).astype(np.uint8)
    cs = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 20:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return float(x), float(y), float(x + w), float(y + h)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def nms(boxes, thr=0.5):
    keep = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if all(iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def distance_m(depth_raw, box, depth_scale):
    """Median distance (metres) in the middle part of the box; 0 when the depth is missing there."""
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    cx, cy, w, h = (x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1
    rx, ry = max(1, w // 4), max(1, h // 4)
    patch = depth_raw[max(0, cy - ry):cy + ry + 1, max(0, cx - rx):cx + rx + 1]
    patch = patch[patch > 0]
    return float(np.median(patch)) * depth_scale if patch.size else 0.0


def draw(image, boxes, depth_raw, depth_scale, det):
    out = image.copy()
    if det.show_all and det.enabled:
        s_only, m_only = det.latest_extra()
        for b in s_only:
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 220, 255), 1)       # yellow
        for b in m_only:
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 60, 255), 1)        # red
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        d = distance_m(depth_raw, b, depth_scale)
        label = f"{b[4]:.2f}" + (f"  {d * 100:.0f} cm" if d > 0 else "")
        cv2.putText(out, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    if not det.ready and not det.error:
        status = "SAM 3 loading..."
    elif det.error:
        status = det.error[:80]
    elif not det.enabled:
        status = "detection OFF (press d)"
    else:
        mode = "SAM 3 + mask agree" if USE_MASK_AGREEMENT else "SAM 3 only"
        c = det.counts
        status = (f"{mode} | '{', '.join(det.prompts)}' | SAM {c[0]}  mask {c[1]}  drawn {len(boxes)} | "
                  f"{det.last_seconds:.1f} s {det.note}")
    cv2.putText(out, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ==================== MAIN ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=PROMPT, help='what to find, e.g. "nut" or "nut, screw"')
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--no-detect", action="store_true", help="record only (detection can be started with 'd')")
    args = ap.parse_args()

    import pyrealsense2 as rs

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_RAW_DEPTH:
        os.makedirs(f"{OUTPUT_DIR}/depth_raw", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    print("[INFO] Starting RealSense...")
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] Depth scale: {depth_scale} meters/unit")

    # constant frame rate (otherwise auto-exposure can lower FPS in dim light)
    for sensor in profile.get_device().query_sensors():
        if sensor.supports(rs.option.auto_exposure_priority):
            sensor.set_option(rs.option.auto_exposure_priority, 0)

    # ---- intrinsics (saved once)
    color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    def intr(i):
        return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "cx": i.ppx, "cy": i.ppy,
                "distortion_model": str(i.model), "distortion_coeffs": list(i.coeffs)}

    intrinsics_path = f"{OUTPUT_DIR}/intrinsics_{timestamp}.json"
    with open(intrinsics_path, "w") as f:
        json.dump({"timestamp": timestamp, "color": intr(color_intr), "depth": intr(depth_intr),
                   "depth_scale": depth_scale}, f, indent=4)
    print(f"[INFO] Intrinsics saved to {intrinsics_path}")

    align = rs.align(rs.stream.color)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    color_writer = cv2.VideoWriter(f"{OUTPUT_DIR}/color_{timestamp}.mp4", fourcc, FPS, (WIDTH, HEIGHT))
    depth_writer = cv2.VideoWriter(f"{OUTPUT_DIR}/depth_{timestamp}.mp4", fourcc, FPS, (WIDTH, HEIGHT))
    colorizer = rs.colorizer()

    det = NutDetector(args.prompt, args.threshold)
    det.enabled = DETECT_AT_START and not args.no_detect
    print(f"[INFO] Prompt: {det.prompts}  threshold: {det.threshold}")
    print("[INFO] Green = both detectors agree. Yellow = SAM 3 only, red = mask only (key m hides/shows them).")

    frame_id, last_frame_number, dropped_frames, snap_id = 0, None, 0, 0
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            if SAVE_RAW_DEPTH:
                np.save(f"{OUTPUT_DIR}/depth_raw/depth_{frame_id:06d}.npy", depth_raw)
            depth_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # repeat a frame when the camera skipped some, so the video plays at real-time speed
            frame_number = color_frame.get_frame_number()
            repeats = 1
            if last_frame_number is not None:
                gap = frame_number - last_frame_number
                if 1 < gap < FPS * 2:
                    repeats = gap
                    dropped_frames += gap - 1
            last_frame_number = frame_number
            for _ in range(repeats):
                color_writer.write(color_image)          # the recording stays clean: no boxes drawn into it
                depth_writer.write(depth_color)

            # ---- live detection (background thread)
            det.submit(color_image)
            boxes, _ = det.latest()
            shown = draw(color_image, boxes if det.enabled else [], depth_raw, depth_scale, det)
            cv2.imshow("RGB", shown)
            cv2.imshow("Depth (Visualization)", depth_color)
            frame_id += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                det.enabled = not det.enabled
                if not det.enabled:
                    det.clear()
                print(f"[INFO] Detection {'ON' if det.enabled else 'OFF'}")
            elif key == ord("m"):
                det.show_all = not det.show_all
                print(f"[INFO] Show single-detector boxes: {'ON' if det.show_all else 'OFF'}")
            elif key == ord("s"):
                snap_id += 1
                base = f"{OUTPUT_DIR}/snapshot_{timestamp}_{snap_id:03d}"
                cv2.imwrite(base + ".png", shown)
                cv2.imwrite(base + "_clean.png", color_image)
                info = [{"box": [round(v, 1) for v in b[:4]], "score": round(b[4], 3),
                         "distance_m": round(distance_m(depth_raw, b, depth_scale), 4)} for b in boxes]
                with open(base + ".json", "w") as f:
                    json.dump({"prompt": det.prompts, "objects": info}, f, indent=2)
                print(f"[INFO] Snapshot saved: {base}.png  ({len(boxes)} objects)")
    except KeyboardInterrupt:
        pass

    print("[INFO] Stopping...")
    det.stop()
    pipeline.stop()
    color_writer.release()
    depth_writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captured {frame_id} frames, filled {dropped_frames} dropped frames.")
    print("[INFO] Recording saved successfully.")


if __name__ == "__main__":
    main()
