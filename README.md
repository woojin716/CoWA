# CoWA

Co-occurrence-aware Weight Adaptation for Test-Time Adaptation on Chest X-ray Classification.

> MICCAI 2026 submission codebase. Work in progress — repository is being cleaned up from the research workspace.

## Overview

TODO: brief method description, key idea, headline result.

## Repository structure

```
CoWA/
├── src/
│   ├── load_dataset.py                          # Dataset loaders (CheXpert / MIMIC / NIH / VinDr)
│   ├── pretrained_inference.py                  # Source-model loading and inference utilities
│   ├── tta_baselines_offline.py                 # Offline baseline TTA methods (BaseAdapter)
│   ├── tta_ours_weight_modified.py              # CoWA: co-occurrence-aware weight adaptation
│   ├── run_tta_experiments_ours.py              # Entry: run CoWA across source–target pairs
│   └── run_tta_experiments_baseline_cached.py   # Entry: run baselines across source–target pairs
├── scripts/
│   ├── run_ours_all.sh                          # Sweep CoWA over source/target combos
│   └── run_baseline_all.sh                      # Sweep baselines over source/target combos
├── configs/
│   ├── tta_ours_configs.json
│   └── tta_baseline_configs.json
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.x, single NVIDIA GPU.

## Data

### Quick path: pre-processed test tensors (recommended for reviewers)

To skip raw data download and torchxrayvision preprocessing, we provide cached
test tensors (one `.pt` per dataset, xrayvision-preprocessed and ready to consume).

**Download**: <TODO: Google Drive link>

After download, place files so the layout is:

```
CoWA/
└── help/
    └── test/
        ├── testset_chexpert.pt
        ├── testset_mimic.pt
        ├── testset_nih.pt
        └── testset_vindr.pt
```

This matches the default `--cache-dir ./help/test` in the runners. The `help/`
directory is gitignored — these files live outside version control.

### Source-domain classifiers

Source models are loaded via [torchxrayvision](https://github.com/mlmed/torchxrayvision)
and **download automatically on first run** (no manual setup needed):

| Source key | torchxrayvision weights              |
|------------|--------------------------------------|
| `chexpert` | `densenet121-res224-chex`            |
| `mimic_ch` | `densenet121-res224-mimic_ch`        |
| `nih`      | `densenet121-res224-nih`             |

### Full reproduction from raw data (optional)

If you prefer to rebuild test tensors yourself, download the original datasets:

- **CheXpert** — https://stanfordmlgroup.github.io/competitions/chexpert/
- **MIMIC-CXR** — https://physionet.org/content/mimic-cxr/
- **NIH ChestX-ray14** — https://nihcc.app.box.com/v/ChestXray-NIHCC
- **VinDr-CXR** — https://physionet.org/content/vindr-cxr/

Loaders in [src/load_dataset.py](src/load_dataset.py) build a deterministic
stratified split with `random_seed=42`. MIMIC-CXR uses the official PhysioNet
split (`mimic-cxr-2.0.0-split.csv.gz`).

## Reproducing experiments

TODO: minimal command examples for the main table and ablations.

All shell scripts assume the project root as the working directory:

```bash
# Baselines (sweeps source–target combos, all baseline methods)
bash scripts/run_baseline_all.sh <exp_id> <batch_size> <device> <seeds>

# CoWA (sweeps source–target combos)
bash scripts/run_ours_all.sh <exp_id> <batch_size>
```

Or invoke a single run directly:

```bash
python src/run_tta_experiments_ours.py \
    --source mimic_ch --target chexpert \
    --batch-size 32 --device cuda --exp_id demo
```

## Citation

```
TODO: BibTeX once accepted.
```

## License

TODO (MIT recommended for code; data licenses follow original sources).
