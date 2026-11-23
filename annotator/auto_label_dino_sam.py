"""
Auto-labeling with Grounding DINO (finds nuts from text) + SAM (draws exact outlines).

Install:
    pip install torch torchvision transformers pillow opencv-python numpy

Run:
    python auto_label_dino_sam.py --images "Pasted image.png" --out labels_out \
        --prompt "metal nut." --roi "340,210,1170,20,1250,350,420,550"
    (--images can be one image file or a folder of images)

Outputs (in --out):
    yolo_det/*.txt   YOLO boxes   (class cx cy w h, normalized)
    yolo_seg/*.txt   YOLO polygons (class x1 y1 x2 y2 ..., normalized)
    results/*_result.jpg   result images (green = auto-keep, red = needs review)
    results.csv      one row per nut: score, area, equivalent diameter, review flag
"""
import argparse, csv, glob, os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          SamModel, SamProcessor)

DINO_ID = "IDEA-Research/grounding-dino-base"
SAM_ID = "facebook/sam-vit-base"


def load_models(device):
    dino_proc = AutoProcessor.from_pretrained(DINO_ID)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(device).eval()
    sam_proc = SamProcessor.from_pretrained(SAM_ID)
    sam = SamModel.from_pretrained(SAM_ID).to(device).eval()
    return dino_proc, dino, sam_proc, sam


@torch.no_grad()
def detect(image, prompt, dino_proc, dino, device, box_thr, text_thr):
    """Grounding DINO: text prompt -> boxes (x1,y1,x2,y2 in pixels) + scores."""
    inputs = dino_proc(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = dino(**inputs)
    kwargs = dict(text_threshold=text_thr, target_sizes=[image.size[::-1]])
    try:  # newer transformers versions
        res = dino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_thr, **kwargs)[0]
    except TypeError:  # older versions
        res = dino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_thr, **kwargs)[0]
    return res["boxes"].cpu(), res["scores"].cpu()


@torch.no_grad()
def segment(image, boxes, sam_proc, sam, device):
    """SAM: boxes -> one binary mask per box."""
    if len(boxes) == 0:
        return np.zeros((0, image.size[1], image.size[0]), bool)
    inputs = sam_proc(image, input_boxes=[boxes.tolist()], return_tensors="pt").to(device)
    out = sam(**inputs, multimask_output=False)
    masks = sam_proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())[0]          # (N, 1, H, W)
    return masks[:, 0].numpy().astype(bool)


def inside_roi(box, roi):
    if roi is None:
        return True
    cx, cy = float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)
    return cv2.pointPolygonTest(roi, (cx, cy), False) >= 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="Pasted image.png", help="a single image file or a folder of images / video frames")
    ap.add_argument("--out", default="labels_out")
    ap.add_argument("--prompt", default="metal nut.", help="lowercase, each phrase ends with a period")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--review-below", type=float, default=0.5, help="scores below this go to human review")
    ap.add_argument("--max-area-frac", type=float, default=0.03, help="drop boxes bigger than this share of the image (trays, plates)")
    ap.add_argument("--roi", default=None, help="polygon x1,y1,x2,y2,... keep only nuts whose center is inside (e.g. the black tray)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_proc, dino, sam_proc, sam = load_models(device)
    roi = None
    if args.roi:
        v = [int(x) for x in args.roi.split(",")]
        roi = np.array(v, np.int32).reshape(-1, 1, 2)

    for d in ("yolo_det", "yolo_seg", "results"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    rows = []
    if os.path.isfile(args.images):
        paths = [args.images]
    else:
        paths = sorted(p for ext in ("jpg", "jpeg", "png", "webp", "bmp") for p in glob.glob(os.path.join(args.images, f"*.{ext}")))
    if not paths:
        raise SystemExit(f"No images found at: {args.images}")

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        image = Image.open(path).convert("RGB")
        W, H = image.size

        boxes, scores = detect(image, args.prompt, dino_proc, dino, device, args.box_thr, args.text_thr)
        # remove duplicates, oversized boxes (tray/plate) and boxes outside the ROI
        if len(boxes):
            keep = nms(boxes, scores, 0.5)
            boxes, scores = boxes[keep], scores[keep]
            area_frac = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) / (W * H)
            ok = [(af <= args.max_area_frac) and inside_roi(b, roi) for af, b in zip(area_frac, boxes)]
            boxes, scores = boxes[ok], scores[ok]

        masks = segment(image, boxes, sam_proc, sam, device)
        overlay = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        det_lines, seg_lines = [], []

        for i, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
            x1, y1, x2, y2 = [float(v) for v in box]
            area = int(mask.sum())
            eq_diam = float(np.sqrt(4 * area / np.pi))
            review = float(score) < args.review_below
            color = (0, 0, 255) if review else (0, 200, 0)

            det_lines.append(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
                seg_lines.append("0 " + " ".join(f"{px/W:.6f} {py/H:.6f}" for px, py in c))
                cv2.drawContours(overlay, [c.reshape(-1, 1, 2)], -1, color, 2)
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(overlay, f"{float(score):.2f}", (int(x1), max(12, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            rows.append([name, i, f"{float(score):.3f}", area, f"{eq_diam:.1f}", "REVIEW" if review else "auto"])

        open(os.path.join(args.out, "yolo_det", f"{name}.txt"), "w").write("\n".join(det_lines))
        open(os.path.join(args.out, "yolo_seg", f"{name}.txt"), "w").write("\n".join(seg_lines))
        result_path = os.path.abspath(os.path.join(args.out, "results", f"{name}_result.jpg"))
        cv2.imwrite(result_path, overlay)
        print(f"{name}: {len(boxes)} nuts -> result image saved to {result_path}")

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "nut_id", "score", "mask_area_px", "equiv_diameter_px", "status"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
