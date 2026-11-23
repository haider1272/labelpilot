"""
Pick images from a Roboflow YOLO export and prepare everything for annotator_gui.py

    python prepare_annotation_set.py --dataset model1_main-22 --out annotate_100 --n 100 --samples 8

--dataset  the folder Roboflow downloaded (contains data.yaml and train/ valid/ test/)
Creates:
    annotate_100/images/                 the N images you will annotate (unlabelled copies)
    annotate_100/samples/<class>/*.png   class samples, for checking or for "Add sample image files"
    annotate_100/annotated/_setup/       the same samples already in the annotator's setup format

Then start the tool:  python annotator_gui.py --images annotate_100/images --output annotate_100/annotated
(the classes and samples load automatically)
"""
import argparse, json, os, random, re, shutil, sys
import cv2, numpy as np

# ---- Roboflow download (used when no --dataset is given and the folder is not there yet) ----
API_KEY = "PIaaOVp8LLo7pEXXxLTf"      # or set the environment variable ROBOFLOW_API_KEY
WORKSPACE, PROJECT, VERSION = "thaising", "model1_main-o4gum", 17
FORMAT = "yolo26"                      # if your roboflow version rejects this, use "yolov8"

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_names(yaml_path):
    txt = open(yaml_path, encoding="utf-8").read()
    m = re.search(r"^names:\s*\[(.*?)\]", txt, re.M | re.S)
    if m:
        return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
    names = {}
    blk = re.search(r"^names:\s*\n((?:[ \t]+.*\n?)+)", txt, re.M)
    if blk:
        for line in blk.group(1).splitlines():
            k = re.match(r"\s*(\d+)\s*:\s*(.+)", line)
            if k:
                names[int(k.group(1))] = k.group(2).strip().strip("'\"")
        return [names[i] for i in sorted(names)]
    names, lines = [], txt.splitlines()
    for i, l in enumerate(lines):
        if l.startswith("names:"):
            for l2 in lines[i + 1:]:
                m2 = re.match(r"\s*-\s*(.+)", l2)
                if not m2:
                    break
                names.append(m2.group(1).strip().strip("'\""))
    return names


def read_img(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="Roboflow export folder (default: the first sub-folder here that has a data.yaml)")
    ap.add_argument("--out", default=None, help="output folder (default: annotate_100 next to this script)")
    ap.add_argument("--n", type=int, default=100, help="images to extract")
    ap.add_argument("--samples", type=int, default=30, help="samples per class")
    ap.add_argument("--margin", type=float, default=0.15, help="context around each sample (same as the tool)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    here = os.path.dirname(os.path.abspath(__file__))
    if a.out is None:
        a.out = os.path.join(here, "annotate_100")        # always created inside this script's folder
    if a.dataset is None:
        target = os.path.join(here, f"{PROJECT}-{VERSION}")
        if not os.path.isfile(os.path.join(target, "data.yaml")):
            print(f"Downloading {WORKSPACE}/{PROJECT} v{VERSION} from Roboflow into {target} ...")
            try:
                from roboflow import Roboflow
            except ImportError:
                raise SystemExit("Run first:  pip install roboflow")
            key = os.environ.get("ROBOFLOW_API_KEY", API_KEY)
            Roboflow(api_key=key).workspace(WORKSPACE).project(PROJECT).version(VERSION).download(FORMAT, location=target)
        a.dataset = target
        print("using:", a.dataset)
    yaml_path = os.path.join(a.dataset, "data.yaml")
    names = read_names(yaml_path)
    if not names:
        raise SystemExit(f"Could not read class names from {yaml_path}")
    print("classes:", names)

    items = []   # (image path, [(cls, cx, cy, w, h)])
    for split in ("train", "valid", "val", "test"):
        idir, ldir = os.path.join(a.dataset, split, "images"), os.path.join(a.dataset, split, "labels")
        if not os.path.isdir(idir):
            continue
        for f in sorted(os.listdir(idir)):
            if f.lower().endswith(EXTS):
                lp = os.path.join(ldir, os.path.splitext(f)[0] + ".txt")
                boxes = []
                if os.path.exists(lp):
                    for line in open(lp).read().splitlines():
                        p = line.split()
                        if len(p) >= 5:
                            v = list(map(float, p[1:]))
                            if len(v) > 4:   # polygon -> box
                                xs, ys = v[0::2], v[1::2]
                                v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
                            boxes.append((int(float(p[0])), *v[:4]))
                items.append((os.path.join(idir, f), boxes))
    if not items:
        raise SystemExit("No images found. Expected <dataset>/train/images etc.")
    print(f"{len(items)} images in the dataset")

    # ---- choose images: make sure every class is present, then fill randomly
    chosen, seen = [], set()
    by_class = {c: [it for it in items if any(b[0] == c for b in it[1])] for c in range(len(names))}
    per = max(1, a.n // (2 * len(names)))
    for c, lst in by_class.items():
        random.shuffle(lst)
        for it in lst[:per]:
            if it[0] not in seen:
                seen.add(it[0]); chosen.append(it)
    rest = [it for it in items if it[0] not in seen]
    random.shuffle(rest)
    chosen += rest[:max(0, a.n - len(chosen))]
    chosen = chosen[:a.n]

    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    for p, _ in chosen:
        shutil.copy2(p, os.path.join(img_dir, os.path.basename(p)))
    print(f"copied {len(chosen)} images to {img_dir}")

    # ---- samples per class: clean boxes (not tiny, not touching the border), from different images
    pool = {c: [] for c in range(len(names))}
    for p, boxes in items:
        for c, cx, cy, w, h in boxes:
            if c in pool and 0.02 < w < 0.5 and 0.02 < h < 0.5 and cx - w / 2 > 0.01 and cx + w / 2 < 0.99 \
                    and cy - h / 2 > 0.01 and cy + h / 2 < 0.99:
                pool[c].append((p, cx, cy, w, h))
    setup_dir = os.path.join(a.out, "annotated", "_setup")
    sdir = os.path.join(setup_dir, "samples")
    shutil.rmtree(sdir, ignore_errors=True)
    os.makedirs(sdir)
    classes_json = []
    try:                                  # cheap colour/shape features, used to pick DIFFERENT-looking samples
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from annotator_gui import SimpleEmbedder
        feat = SimpleEmbedder()
    except Exception:
        feat = None

    def crop_of(item):
        p, cx, cy, w, h = item
        im = read_img(p)
        H, W = im.shape[:2]
        x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
        mx, my = (x2 - x1) * a.margin, (y2 - y1) * a.margin
        return im[max(0, int(y1 - my)):min(H, int(y2 + my)), max(0, int(x1 - mx)):min(W, int(x2 + mx))]

    classes_json = []
    for c, name in enumerate(names):
        cand = pool[c]
        random.shuffle(cand)
        used, firsts = set(), []
        for it in cand:                       # at most one candidate per image -> variety
            if it[0] not in used:
                used.add(it[0]); firsts.append(it)
            if len(firsts) == 300:
                break
        crops = [crop_of(it) for it in firsts]
        pick_idx = list(range(min(a.samples, len(firsts))))
        if feat is not None and len(firsts) > a.samples:
            # k-means on the features; the sample closest to each cluster centre = a typical example of each look
            F = feat.embed(crops).astype(np.float32)
            cv2.setRNGSeed(a.seed)
            _, lab, cen = cv2.kmeans(F, a.samples, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3),
                                     3, cv2.KMEANS_PP_CENTERS)
            pick_idx = []
            for k in range(a.samples):
                members = np.where(lab.ravel() == k)[0]
                if len(members):
                    pick_idx.append(int(members[np.argmin(((F[members] - cen[k]) ** 2).sum(1))]))
        folder = os.path.join(a.out, "samples", re.sub(r"[^\w\-]+", "_", name))
        os.makedirs(folder, exist_ok=True)
        files = []
        for k, i in enumerate(pick_idx):
            fn = f"c{c}_{k}.png"
            for d in (folder, sdir):
                cv2.imencode(".png", crops[i])[1].tofile(os.path.join(d, fn))
            files.append(fn)
        classes_json.append({"name": name, "samples": files})
        print(f"  {name}: {len(files)} samples (of {len(cand)} candidate boxes)")
        if len(files) < 3:
            print(f"  WARNING: few samples for '{name}'. Add some by hand in the tool.")
    json.dump({"classes": classes_json, "settings": {}}, open(os.path.join(setup_dir, "setup.json"), "w"), indent=2)

    print(f"\nDone. Start the tool with:\n  python annotator_gui.py --images \"{img_dir}\" --output \"{os.path.join(a.out, 'annotated')}\"")


if __name__ == "__main__":
    main()
