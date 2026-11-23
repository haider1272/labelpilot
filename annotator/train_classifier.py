"""
Train the class classifier (inserted screw / loose screw / occluded ...) from your LABELLED Roboflow dataset.
The annotator then starts with a classifier that has seen thousands of real examples instead of 8 samples.

    python train_classifier.py --dataset model1_main-o4gum-17

Needs annotator_gui.py in the same folder, plus:  pip install torch transformers opencv-python numpy pillow
Output: classifier.npz   -> in annotator_gui.py (Setup > Pre-trained classifier) choose this file.
It also prints the accuracy on the validation images and a confusion matrix, so you can see which classes get mixed up.
"""
import argparse, os, random, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotator_gui as A

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
    out, lines = [], txt.splitlines()
    for i, l in enumerate(lines):
        if l.startswith("names:"):
            for l2 in lines[i + 1:]:
                m2 = re.match(r"\s*-\s*(.+)", l2)
                if not m2:
                    break
                out.append(m2.group(1).strip().strip("'\""))
    return out


def collect(dataset, splits):
    """-> {image path: [(class, cx, cy, w, h)]}"""
    items = {}
    for split in splits:
        idir, ldir = os.path.join(dataset, split, "images"), os.path.join(dataset, split, "labels")
        if not os.path.isdir(idir):
            continue
        for f in sorted(os.listdir(idir)):
            if not f.lower().endswith(EXTS):
                continue
            lp = os.path.join(ldir, os.path.splitext(f)[0] + ".txt")
            if not os.path.exists(lp):
                continue
            boxes = []
            for line in open(lp).read().splitlines():
                p = line.split()
                if len(p) >= 5:
                    v = list(map(float, p[1:]))
                    if len(v) > 4:
                        xs, ys = v[0::2], v[1::2]
                        v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
                    boxes.append((int(float(p[0])), *v[:4]))
            if boxes:
                items[os.path.join(idir, f)] = boxes
    return items


def sample_crops(items, n_classes, per_class, margin, min_px, rng):
    """Pick up to per_class boxes per class (spread over many images), crop them with the same margin as the tool."""
    pool = {c: [] for c in range(n_classes)}
    for p, boxes in items.items():
        for c, cx, cy, w, h in boxes:
            if c in pool:
                pool[c].append((p, cx, cy, w, h))
    chosen = {}
    for c, lst in pool.items():
        rng.shuffle(lst)
        for it in lst[:per_class]:
            chosen.setdefault(it[0], []).append((c, *it[1:]))
    crops, labels = [], []
    for k, (p, lst) in enumerate(chosen.items()):
        img = A.read_image(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        for c, cx, cy, w, h in lst:
            x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
            if x2 - x1 < min_px or y2 - y1 < min_px:
                continue
            crops.append(A.crop_context(img, (x1, y1, x2, y2), margin))
            labels.append(c)
        if k % 200 == 0:
            print(f"   cropped {k}/{len(chosen)} images", flush=True)
    return crops, np.array(labels, int)


def embed_all(emb, crops, batch=64):
    out = []
    for i in range(0, len(crops), batch):
        out.append(emb.embed(crops[i:i + batch]))
        if (i // batch) % 10 == 0:
            print(f"   embedded {min(i + batch, len(crops))}/{len(crops)}", flush=True)
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="Roboflow export folder (default: first sub-folder with data.yaml)")
    ap.add_argument("--out", default="classifier.npz")
    ap.add_argument("--per-class", type=int, default=1200, help="training crops per class")
    ap.add_argument("--val-per-class", type=int, default=300)
    ap.add_argument("--keep", type=int, default=500, help="samples per class stored in the file (the annotator keeps learning on top)")
    ap.add_argument("--margin", type=float, default=0.15, help="must equal 'Context margin' in the annotator")
    ap.add_argument("--min-px", type=float, default=10)
    ap.add_argument("--backend", default="auto", choices=["auto", "dinov2", "simple"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    if a.dataset is None:
        for d in sorted(os.listdir(".")):
            if os.path.isfile(os.path.join(d, "data.yaml")):
                a.dataset = d
                break
        if a.dataset is None:
            raise SystemExit("No dataset folder found. Use --dataset <folder>")
    names = read_names(os.path.join(a.dataset, "data.yaml"))
    print("dataset:", a.dataset, "| classes:", names)

    train = collect(a.dataset, ("train",))
    val = collect(a.dataset, ("valid", "val", "test"))
    if not val:                                           # no validation split: hold out 15% of the images
        keys = sorted(train)
        rng.shuffle(keys)
        cut = max(1, int(0.15 * len(keys)))
        val = {k: train[k] for k in keys[:cut]}
        train = {k: train[k] for k in keys[cut:]}
    print(f"{len(train)} training images, {len(val)} validation images")

    emb = A.make_embedder(a.backend, print)
    print("embedder:", emb.name)
    print("cropping + embedding training boxes ...")
    crops, y = sample_crops(train, len(names), a.per_class, a.margin, a.min_px, rng)
    X = embed_all(emb, crops)
    print("per class:", {names[c]: int((y == c).sum()) for c in range(len(names))})

    clf = A.Classifier([], "simple", print)               # only used as a container for the maths below
    clf.embedder = emb
    clf.S, clf.labels, clf.groups, clf.rw = X, y, np.arange(len(X)), np.ones(len(X))
    clf.dirty = True
    clf.fit()
    if clf.W is None:
        raise SystemExit("Not enough samples per class to train (need at least 5 of every class).")

    print("evaluating on validation images ...")
    vc, vy = sample_crops(val, len(names), a.val_per_class, a.margin, a.min_px, rng)
    pred = np.array([p[0] for p in clf.predict_emb(embed_all(emb, vc))])
    acc = float((pred == vy).mean())
    print(f"\nvalidation accuracy: {acc * 100:.1f}%  ({len(vy)} boxes)")
    cm = np.zeros((len(names), len(names)), int)
    for t, p in zip(vy, pred):
        cm[t, p] += 1
    w = max(len(n) for n in names) + 2
    print("\nconfusion matrix (rows = true class, columns = predicted):")
    print(" " * w + "".join(f"{n[:12]:>14}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<{w}}" + "".join(f"{v:>14}" for v in cm[i]))

    keep_idx = []
    for c in range(len(names)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx := list(idx))
        keep_idx += idx[:a.keep]
    np.savez_compressed(a.out, S=X[keep_idx].astype(np.float16), labels=y[keep_idx], names=np.array(names),
                        embedder=emb.name, margin=a.margin)
    print(f"\nSaved {a.out}. In annotator_gui.py choose it under Setup > Pre-trained classifier "
          f"and keep 'Context margin' = {a.margin}.")


if __name__ == "__main__":
    main()
