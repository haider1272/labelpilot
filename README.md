# LabelPilot

SAM 3-assisted data annotation automation: you review, LabelPilot labels. It learns from your corrections, builds synthetic training data, and ships with a depth-based safety layer as a worked example.

Built by Msis Robotics. First use case: nut/screw inspection (classes `inserted`, `loose`, `occluded`), but classes are user-defined.

## Layout
| Folder | Purpose |
|---|---|
| `annotator/` | **Annotation** – SAM 3 assisted annotation GUI (`annotator_gui.py`), annotation-set prep, few-shot classifier, DINO+SAM auto-label |
| `synthetic/` | **Synthetic data** – enlarge nuts to build synthetic bigger-nut data (`screw_resizer_yaml.py`), size sweep |
| `safety/` | **Example app** – RealSense high-stand camera "missed nut" safety layer (YOLO + SAM 3 + leftover search + depth, robot coordinates) |
| `legacy/` | Earlier classical / experimental detectors |
| `docs/` | Workflow decks and comparison images |
| `data/` | Local datasets (git-ignored) |

## Quick start
```bash
pip install -r requirements.txt

# Annotate
python annotator/annotator_gui.py

# Synthetic bigger-nut dataset (measured sizes: small 33.0x23.9 mm, large 38.55x32.79 mm)
python synthetic/screw_resizer_yaml.py --yaml final_dataset/data.yaml --out big_dataset \
    --splits train --class-scale 0:1.37 1:1.17 2:1.37      # 0 inserted, 1 loose, 2 occluded

# Safety example inspection (run from safety/, needs config/config.py and src/librealsense.py)
cd safety && python run_inspection.py
```

## Safety example keys (`run_inspection.py`)
`b` capture empty-tray background · `r` tray area · `i` ignore area · `c` clear areas · `m` measure sizes ·
`s` full inspection (SAM 3) · `d` debug view · `q` quit.
Status: `PASS` / `CHECK` / `UNVERIFIED` / `BG_MISMATCH`.

## Notes
- `safety/` expects your own `src/librealsense.py` (`initialize_realsense`, `get_frames`) and `config/config.py` (see `config.example.py`).
- `size_sweep.py` imports `screw_resizer_yaml.py` – both live in `synthetic/`.
- Weights, datasets, backgrounds are git-ignored.
