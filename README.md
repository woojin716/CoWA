# CoWA

Co-occurrence-aware Weight Adaptation for Test-Time Adaptation on Chest X-ray Classification.

> MICCAI 2026 submission codebase. Work in progress — repository is being cleaned up from the research workspace.

## Overview

TODO: brief method description, key idea, headline result.

## Repository structure

```
CoWA/
├── src/              # Core source: dataset loading, TTA methods, baselines
├── scripts/          # Shell entry points for experiments
├── configs/          # JSON configs for runs / sweeps
├── analysis/         # Post-hoc analysis and plotting scripts
├── notebooks/        # Selected analysis notebooks
├── requirements.txt
└── README.md
```

(Layout will be finalized as files are migrated from the research workspace.)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.x, single NVIDIA GPU.

## Data

This repository does **not** include datasets. Download from the original sources:

- **CheXpert** — https://stanfordmlgroup.github.io/competitions/chexpert/
- **MIMIC-CXR** — https://physionet.org/content/mimic-cxr/
- **NIH ChestX-ray14** — https://nihcc.app.box.com/v/ChestXray-NIHCC
- **VinDr-CXR** — https://physionet.org/content/vindr-cxr/
- **PadChest** — https://bimcv.cipf.es/bimcv-projects/padchest/

Expected layout under a user-configured `DATA_ROOT`:

```
$DATA_ROOT/
├── chexpert/
├── mimic/
├── nih/
├── vindr/
└── padchest/
```

## Pretrained source models

Source-domain classifiers are released as GitHub Release assets (see Releases tab — TODO). Place them under `pretrained/` (gitignored).

## Reproducing experiments

TODO: minimal command examples for the main table and ablations.

```bash
# Baseline (no adaptation)
bash scripts/run_baseline_all.sh

# Our method
bash scripts/run_ours_all.sh
```

## Citation

```
TODO: BibTeX once accepted.
```

## License

TODO (MIT recommended for code; data licenses follow original sources).
