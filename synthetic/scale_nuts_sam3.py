"""
Make a "bigger nut" copy of a YOLO-labeled dataset (e.g. Product 1 -> Product 3 size):
    1. read every image and its YOLO labels (class cx cy w h, normalized)
    2. SAM 3 segments the nut inside each labeled box
    3. each nut is cut out, enlarged by SCALE_X, and pasted back at the same center
    4. the bounding box is recomputed from the enlarged mask and saved as the new label

Dataset layout (Roboflow YOLO export):
    DATASET_DIR/train/images, DATASET_DIR/train/labels   (same for valid/ and test/)
    or DATASET_DIR/images and DATASET_DIR/labels

Install:  pip install torch transformers opencv-python numpy pillow
          (SAM 3 weights "facebook/sam3" may need you to accept the license on Hugging Face
           and run `huggingface-cli login` once.)
Run:      1) set LIMIT = 20 and run it, then look at OUT_DIR/preview/ to check the quality
          2) set LIMIT = None and run it again for all images (finished files are skipped)
"""
import glob, os, random, shutil, traceback
import cv2
import numpy as np

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
DATASET_DIR = "dataset"
OUT_DIR = "dataset_scaled"
SCALE_X = 1.078            # enlargement factor = Product 3 size / Product 1 size
SCALE_JITTER = 0.02        # each nut uses SCALE_X * random(1-0.02 ... 1+0.02); 0 = exact
LIMIT = 20                 # test on the first N images; None = all images
SUFFIX = "_x"              # output names: image_x1.078.jpg
COPY_ORIGINALS = False     # also copy the original images and labels into OUT_DIR

MIN_MASK_FILL = 0.35       # mask must fill at least this share of its box, else skip the nut
MAX_NEIGHBOR_IOU = 0.15    # skip enlarging a nut if its new box overlaps another nut more than this
PREVIEW_COUNT = 20         # save before/after pictures for the first N images
SEED = 0
# ---------------------------------------------------------------

IMG_EXT = ("jpg", "jpeg", "png", "webp", "bmp")


# ---------- SAM 3 ----------
def load_sam():
    import torch
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device).eval()
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    return model, processor, device


def segment_boxes(sam, img_bgr, boxes):
    """boxes: list of [x1,y1,x2,y2] pixels -> list of 0/1 masks (full image size)."""
    import torch
    from PIL import Image
    model, processor, device = sam
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil, input_boxes=[[list(map(float, b)) for b in boxes]],
                       return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
    masks = masks.numpy()
    if masks.ndim == 4:            # (boxes, 1, H, W)
        masks = masks[:, 0]
    return [(m > 0).astype(np.uint8) for m in masks]


# ---------- labels ----------
def read_labels(path):
    """YOLO lines -> list of (class_id, x1, y1, x2, y2) later converted to pixels. Polygons become boxes."""
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path).read().strip().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        v = [float(x) for x in p[1:]]
        if len(v) == 4:
            cx, cy, w, h = v
            rows.append((c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        else:  # polygon label -> bounding box
            xs, ys = v[0::2], v[1::2]
            rows.append((c, min(xs), min(ys), max(xs), max(ys)))
    return rows


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ---------- compositing ----------
def largest_component(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    return (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)


def paste_scaled(canvas, src, mask, scale):
    """Cut the masked nut from `src`, scale it, paste on `canvas` at the same center. Returns new box."""
    H, W = mask.shape
    x, y, w, h = cv2.boundingRect(mask)
    pad = 4
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    patch = src[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    m = cv2.erode(m, np.ones((3, 3), np.uint8))                 # drop the halo
    nw, nh = max(2, int(round(patch.shape[1] * scale))), max(2, int(round(patch.shape[0] * scale)))
    interp = cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA
    patch_s = cv2.resize(patch, (nw, nh), interpolation=interp)
    m_s = cv2.resize(m, (nw, nh), interpolation=interp)
    m_s = cv2.GaussianBlur(m_s, (5, 5), 0).astype(np.float32) / 255.0
    cx, cy = x + w / 2.0, y + h / 2.0                           # center of the original nut
    px0, py0 = int(round(cx - (x - x0 + w / 2.0) * scale)), int(round(cy - (y - y0 + h / 2.0) * scale))
    # clip to the image
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


def process_image(sam, img, labels):
    """Returns (new_image, new_label_rows, stats).
    Overlapping nuts: the most complete nut (mask closest to the normal size of its class) is
    pasted LAST, so it stays on top of the nuts it covers, like in the real photo."""
    H, W = img.shape[:2]
    boxes = [(c, x1 * W, y1 * H, x2 * W, y2 * H) for c, x1, y1, x2, y2 in labels]
    stats = dict(scaled=0, skipped_mask=0, skipped_overlap=0)
    if not boxes:
        return img.copy(), [], stats
    raw = segment_boxes(sam, img, [b[1:] for b in boxes])

    # 1) clean the masks and measure how complete each nut is
    masks, areas = [], []
    for (c, x1, y1, x2, y2), mask in zip(boxes, raw):
        bm = np.zeros_like(mask)
        bm[max(0, int(y1)):int(y2) + 1, max(0, int(x1)):int(x2) + 1] = 1
        mask = largest_component(mask & bm)
        box_area = max(1.0, (x2 - x1) * (y2 - y1))
        ok = mask is not None and mask.sum() / box_area >= MIN_MASK_FILL
        masks.append(mask if ok else None)
        areas.append(float(mask.sum()) if ok else 0.0)
    med = {}
    for c in set(b[0] for b in boxes):
        v = [a for a, b in zip(areas, boxes) if b[0] == c and a > 0]
        med[c] = float(np.median(v)) if v else 1.0
    completeness = [a / med[b[0]] for a, b in zip(areas, boxes)]
    order = sorted(range(len(boxes)), key=lambda i: completeness[i])   # most hidden first

    # 2) paste in that order (complete nuts last = on top)
    out = img.copy()
    new_rows = {}
    for i in order:
        c, x1, y1, x2, y2 = boxes[i]
        mask = masks[i]
        if mask is None:
            stats["skipped_mask"] += 1
            new_rows[i] = (c, x1, y1, x2, y2)
            continue
        scale = SCALE_X * (1 + random.uniform(-SCALE_JITTER, SCALE_JITTER))
        bx, by, bw, bh = cv2.boundingRect(mask)
        ncx, ncy = bx + bw / 2, by + bh / 2
        pred = (ncx - bw * scale / 2, ncy - bh * scale / 2, ncx + bw * scale / 2, ncy + bh * scale / 2)
        # nuts that already overlap in the photo may overlap a bit more; others may not
        too_much = any(iou(pred, o[1:]) > max(MAX_NEIGHBOR_IOU, iou(boxes[i][1:], o[1:]) + 0.10)
                       for j, o in enumerate(boxes) if j != i)
        if too_much:
            stats["skipped_overlap"] += 1
            new_rows[i] = (c, x1, y1, x2, y2)
            continue
        if scale < 1.02:  # shrinking leaves a hole where the old nut was -> fill it first
            hole = cv2.dilate(mask, np.ones((7, 7), np.uint8)) * 255
            out = cv2.inpaint(out, hole, 5, cv2.INPAINT_TELEA)
        nb = paste_scaled(out, img, mask, scale)
        if nb is None:
            stats["skipped_mask"] += 1
            new_rows[i] = (c, x1, y1, x2, y2)
            continue
        stats["scaled"] += 1
        new_rows[i] = (c, *nb)
    return out, [new_rows[i] for i in range(len(boxes))], stats


def write_labels(path, rows, W, H):
    lines = []
    for c, x1, y1, x2, y2 in rows:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    open(path, "w").write("\n".join(lines))


def find_sets(root):
    sets = []
    for split in ("train", "valid", "val", "test"):
        im, lb = os.path.join(root, split, "images"), os.path.join(root, split, "labels")
        if os.path.isdir(im):
            sets.append((split, im, lb))
    if not sets and os.path.isdir(os.path.join(root, "images")):
        sets.append(("", os.path.join(root, "images"), os.path.join(root, "labels")))
    return sets


def main():
    random.seed(SEED)
    sets = find_sets(DATASET_DIR)
    if not sets:
        raise SystemExit(f"No images folder found under {DATASET_DIR}")
    sam = load_sam()
    tag = f"{SUFFIX}{SCALE_X:.3f}"
    os.makedirs(os.path.join(OUT_DIR, "preview"), exist_ok=True)
    done = total_scaled = total_skipped = 0
    errors = []
    for split, im_dir, lb_dir in sets:
        out_im, out_lb = os.path.join(OUT_DIR, split, "images"), os.path.join(OUT_DIR, split, "labels")
        os.makedirs(out_im, exist_ok=True); os.makedirs(out_lb, exist_ok=True)
        paths = sorted(p for e in IMG_EXT for p in glob.glob(os.path.join(im_dir, f"*.{e}")))
        for p in paths:
            if LIMIT is not None and done >= LIMIT:
                break
            name, ext = os.path.splitext(os.path.basename(p))
            out_path = os.path.join(out_im, f"{name}{tag}.jpg")
            lbl_out = os.path.join(out_lb, f"{name}{tag}.txt")
            if os.path.exists(out_path) and os.path.exists(lbl_out):
                continue                                              # resume support
            try:
                img = cv2.imread(p)
                labels = read_labels(os.path.join(lb_dir, f"{name}.txt"))
                new_img, rows, st = process_image(sam, img, labels)
                cv2.imwrite(out_path, new_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                write_labels(lbl_out, rows, img.shape[1], img.shape[0])
                if COPY_ORIGINALS:
                    shutil.copy(p, os.path.join(out_im, os.path.basename(p)))
                    if os.path.exists(os.path.join(lb_dir, f"{name}.txt")):
                        shutil.copy(os.path.join(lb_dir, f"{name}.txt"), os.path.join(out_lb, f"{name}.txt"))
                total_scaled += st["scaled"]; total_skipped += st["skipped_mask"] + st["skipped_overlap"]
                if done < PREVIEW_COUNT:
                    prev = np.hstack([cv2.resize(img, None, fx=0.5, fy=0.5), cv2.resize(new_img, None, fx=0.5, fy=0.5)])
                    cv2.imwrite(os.path.join(OUT_DIR, "preview", f"{name}.jpg"), prev)
                done += 1
                if done % 50 == 0:
                    print(f"{done} images done, {total_scaled} nuts enlarged, {total_skipped} skipped")
            except Exception:
                errors.append((p, traceback.format_exc()))
    for f in ("data.yaml",):
        if os.path.exists(os.path.join(DATASET_DIR, f)):
            shutil.copy(os.path.join(DATASET_DIR, f), os.path.join(OUT_DIR, f))
    if errors:
        with open(os.path.join(OUT_DIR, "errors.txt"), "w") as f:
            for p, tb in errors:
                f.write(f"{p}\n{tb}\n")
    print(f"Finished: {done} images, {total_scaled} nuts enlarged, {total_skipped} skipped (bad mask or overlap), {len(errors)} errors")
    print(f"Output: {os.path.abspath(OUT_DIR)}   (check preview/ first)")


if __name__ == "__main__":
    main()
