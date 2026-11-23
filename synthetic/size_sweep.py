"""
Size sweep: which nut size should the model be trained on so that it detects ALL the other sizes?

For every size S (e.g. 20, 22, ... 30 mm) the script
  1. makes a dataset where the nuts are enlarged by X = S / BASE_SIZE_MM   (uses screw_resizer_yaml.py)
  2. trains one YOLO model on that size
  3. tests EVERY model on EVERY size's test images  -> a "trained on  x  tested on" table
  4. picks the ideal training size: best average score on the OTHER sizes, with the best worst case

Put screw_resizer_yaml.py in the same folder.
Install:  pip install ultralytics opencv-python numpy pillow pyyaml matplotlib
Run:      python size_sweep.py             (edit the settings below first)
          python size_sweep.py --dry-run   (fake numbers, to check the table / ranking / heatmap logic)

Notes
  * BASE_SIZE_MM must be the size of the nuts in your ORIGINAL dataset, and it should be the SMALLEST size,
    so every X is >= 1 (the resizer enlarges, it does not shrink).
  * Training uses the same settings for every size, and the same seed, so the models are comparable.
  * Test on REAL photos of every real size you own (add them under REAL_TEST_YAMLS); synthetic-only results
    show how well the model handles scaling, not how well it handles the real nuts.
"""
import argparse, csv, itertools, json, os, sys
import numpy as np

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
DATA_YAML = "dataset/data.yaml"      # the ORIGINAL (smallest size) dataset
BASE_SIZE_MM = 20.0                  # size of the nuts in that dataset
SIZES_MM = [20, 22, 24, 26, 28, 30]  # sizes to generate / train on / test on
TRAIN_GROUPS = None                  # None = one model per size; or e.g. [[20], [24], [30], [20, 30], [20, 24, 30]]
CLASSES_TO_SCALE = None              # None = every class in data.yaml; or e.g. [0, 1]
MASK_METHOD = "auto"                 # "auto" (SAM 3 -> GrabCut -> rectangle), "grabcut", "rect"
MARGIN, FEATHER = 0.1, 0.6           # only used by the rectangle fallback

OUT_ROOT = "sweep"
MODEL = "yolov8n.pt"                 # starting weights; use the model you normally train
EPOCHS, IMGSZ, BATCH = 100, 640, 16
DEVICE = None                        # None = automatic, or 0, "cpu"
SEED = 0
SCALE_AUG = 0.0                      # YOLO's own random scaling. 0 = off, so only OUR size changes are tested
                                     # (set e.g. 0.5 to see how much built-in scale augmentation already helps)
METRIC = "map50"                     # "map50", "map50_95", "recall" or "precision"
COVER_FRACTION = 0.90                # a size counts as "covered" if the score >= this x the score of a model trained on it
REAL_TEST_YAMLS = {}                 # e.g. {"real_24mm": "real24/data.yaml"}  extra test sets of real nuts
# ---------------------------------------------------------------


def tag(s):
    return f"{float(s):g}mm"


def make_size_dataset(size, out_root, classes, seed):
    """Enlarged copy of the dataset for one size; returns the path to its data.yaml."""
    if abs(size - BASE_SIZE_MM) < 1e-9:
        return os.path.abspath(DATA_YAML)
    out = os.path.join(out_root, f"data_{tag(size)}")
    yml = os.path.join(out, "data.yaml")
    if os.path.exists(yml):
        return os.path.abspath(yml)                      # already made
    import screw_resizer_yaml as r
    ds = r.load_dataset(DATA_YAML)
    cls = classes if classes is not None else list(ds["names"])
    cfg = r.make_cfg(size / BASE_SIZE_MM, MARGIN, FEATHER, 0.0, cls, MASK_METHOD)
    r.run_dataset(ds, list(ds["splits"]), out, cfg, "", seed)
    return os.path.abspath(yml)


def group_yaml(group, yamls, out_root):
    """data.yaml for a model trained on several sizes at once (train folders are combined)."""
    if len(group) == 1:
        return yamls[group[0]]
    import yaml
    base = yaml.safe_load(open(yamls[group[0]]))
    trains = []
    for s in group:
        y = yaml.safe_load(open(yamls[s]))
        root = y.get("path") or os.path.dirname(yamls[s])
        trains.append(os.path.join(root, y["train"]) if not os.path.isabs(y["train"]) else y["train"])
    first = yaml.safe_load(open(yamls[group[0]]))
    root0 = first.get("path") or os.path.dirname(yamls[group[0]])
    out = dict(base)
    out["path"] = os.path.abspath(root0)
    out["train"] = trains
    p = os.path.join(out_root, f"group_{'_'.join(tag(s) for s in group)}.yaml")
    yaml.safe_dump(out, open(p, "w"), sort_keys=False)
    return os.path.abspath(p)


def train_model(name, data_yaml, out_root):
    from ultralytics import YOLO
    best = os.path.join(out_root, "runs", name, "weights", "best.pt")
    if os.path.exists(best):
        return best                                       # already trained
    YOLO(MODEL).train(data=data_yaml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE, seed=SEED,
                      scale=SCALE_AUG, project=os.path.abspath(os.path.join(out_root, "runs")),
                      name=name, exist_ok=True, plots=False, verbose=False)
    return best


def evaluate(weights, data_yaml):
    import yaml
    from ultralytics import YOLO
    y = yaml.safe_load(open(data_yaml))
    split = "test" if y.get("test") else "val"
    r = YOLO(weights).val(data=data_yaml, split=split, imgsz=IMGSZ, device=DEVICE, plots=False, verbose=False)
    return {"map50": float(r.box.map50), "map50_95": float(r.box.map),
            "recall": float(r.box.mr), "precision": float(r.box.mp)}


def fake_score(train_sizes, test_size):
    """dry-run only: best at the trained size, falls off with the size ratio."""
    d = min(abs(np.log(test_size / t)) for t in train_sizes)
    return float(np.clip(0.95 * np.exp(-4.0 * d) + 0.02, 0, 1))


def rank(groups, tests, M):
    """M[g][t] = score. Returns a list of dicts, best first."""
    ref = {t: max(M[g][t] for g in groups if len(g) == 1 and g[0] == t) if any(len(g) == 1 and g[0] == t for g in groups) else None
           for t in tests}
    rows = []
    for g in groups:
        others = [t for t in tests if t not in g]
        vals = [M[g][t] for t in others]
        if not vals:
            continue
        covered = sum(1 for t in others if ref[t] is not None and M[g][t] >= COVER_FRACTION * ref[t])
        rows.append({"trained_on": g, "mean_other": float(np.mean(vals)), "worst_other": float(np.min(vals)),
                     "covered": covered, "of": len(others)})
    rows.sort(key=lambda r: (-r["mean_other"], -r["worst_other"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="fake scores; no training")
    ap.add_argument("--out", default=OUT_ROOT)
    a = ap.parse_args()
    out_root = a.out
    os.makedirs(out_root, exist_ok=True)
    if min(SIZES_MM) < BASE_SIZE_MM:
        sys.exit("BASE_SIZE_MM must be the smallest size (the resizer only enlarges).")
    groups = TRAIN_GROUPS or [[s] for s in SIZES_MM]
    groups = [tuple(g) for g in groups]
    tests = list(SIZES_MM)

    M = {g: {} for g in groups}
    if a.dry_run:
        for g in groups:
            for t in tests:
                M[g][t] = fake_score(g, t)
    else:
        yamls = {s: make_size_dataset(s, out_root, CLASSES_TO_SCALE, SEED) for s in SIZES_MM}
        extra = {k: os.path.abspath(v) for k, v in REAL_TEST_YAMLS.items()}
        for g in groups:
            name = "train_" + "_".join(tag(s) for s in g)
            weights = train_model(name, group_yaml(g, yamls, out_root), out_root)
            for t in tests:
                M[g][t] = evaluate(weights, yamls[t])[METRIC]
                print(f"trained on {[tag(s) for s in g]}, tested on {tag(t)}: {METRIC} = {M[g][t]:.3f}")
            for k, y in extra.items():
                print(f"trained on {[tag(s) for s in g]}, tested on {k}: {METRIC} = {evaluate(weights, y)[METRIC]:.3f}")

    # ---- table ----
    names = ["+".join(tag(s) for s in g) for g in groups]
    csv_path = os.path.join(out_root, "size_matrix.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trained_on \\ tested_on"] + [tag(t) for t in tests])
        for g, n in zip(groups, names):
            w.writerow([n] + [f"{M[g][t]:.4f}" for t in tests])
    print("\n" + f"{'trained on':<14}" + "".join(f"{tag(t):>9}" for t in tests))
    for g, n in zip(groups, names):
        print(f"{n:<14}" + "".join(f"{M[g][t]:>9.3f}" for t in tests))

    # ---- ideal size ----
    rows = rank(groups, tests, M)
    print(f"\nRanking by the average {METRIC} on the sizes the model was NOT trained on:")
    for i, r in enumerate(rows, 1):
        n = "+".join(tag(s) for s in r["trained_on"])
        print(f"{i}. trained on {n:<14} mean {r['mean_other']:.3f}   worst {r['worst_other']:.3f}   "
              f"covers {r['covered']}/{r['of']} other sizes (>= {COVER_FRACTION:.0%} of their own model)")
    best = rows[0]
    print(f"\nIDEAL TRAINING SIZE: {'+'.join(tag(s) for s in best['trained_on'])}")
    json.dump({"matrix": {n: {tag(t): M[g][t] for t in tests} for g, n in zip(groups, names)},
               "ranking": [{**r, "trained_on": list(r["trained_on"])} for r in rows]},
              open(os.path.join(out_root, "size_results.json"), "w"), indent=1)

    # ---- heatmap ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        arr = np.array([[M[g][t] for t in tests] for g in groups])
        fig, ax = plt.subplots(figsize=(1.2 * len(tests) + 3, 0.7 * len(groups) + 2))
        im = ax.imshow(arr, vmin=0, vmax=1, cmap="YlGn")
        ax.set_xticks(range(len(tests)), [tag(t) for t in tests])
        ax.set_yticks(range(len(groups)), names)
        ax.set_xlabel("tested on size"); ax.set_ylabel("trained on size")
        for i in range(len(groups)):
            for j in range(len(tests)):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, label=METRIC)
        fig.tight_layout()
        fig.savefig(os.path.join(out_root, "size_heatmap.png"), dpi=150)
        print(f"Heatmap: {os.path.join(out_root, 'size_heatmap.png')}")
    except ImportError:
        print("(matplotlib not installed: no heatmap, the table is in size_matrix.csv)")
    print(f"Table: {csv_path}")


if __name__ == "__main__":
    main()
