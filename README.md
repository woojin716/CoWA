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

### Quick path: pre-processed test tensors

To skip raw data download and torchxrayvision preprocessing, we provide cached
test tensors (one `.pt` per dataset, xrayvision-preprocessed and ready to consume).

**Download**: https://drive.google.com/drive/folders/1Gtdlwx-TKqgvJYOSlcSj3jgxaQSYzyBp?usp=drive_link

After download, place files so the layout is:

```
CoWA/
└── data/
    ├── testset_chexpert.pt
    ├── testset_mimic.pt
    ├── testset_nih.pt
    └── testset_vindr.pt
```

This matches the default `--cache-dir ./data` in the runners. The `data/`
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
    --batch-size 32 --device cuda
```

Each invocation auto-creates a fresh run directory at
`results/<YYYYMMDD>/<NNN>/` (NNN = next 3-digit index). The shell scripts
above bundle a whole sweep into a single run directory.

---

## Development notes (internal)

> The section below is for the maintainer's own workflow during rebuttal.

A persistent Docker container `miccai_rbt` is configured on the dev server
with the codebase + cached test tensors mounted in. Do **not** `docker rm`
it — keep it around for quick rebuttal iteration. The container runs as
`uid=1000` (tako), so any files it writes (e.g. `results/`) are owned by
tako and can be deleted from the host without `sudo`.

```bash
# Start a stopped container
docker start miccai_rbt

# Open an interactive shell
docker exec -it miccai_rbt bash

# Stop when done (state preserved; restart with `docker start`)
docker stop miccai_rbt
```

If the container is ever accidentally removed, recreate it with:

```bash
docker run -d --name miccai_rbt \
    --gpus all --shm-size=8g \
    --user 1000:1000 -e HOME=/home/cowa \
    -v miccai_rbt-home:/home/cowa \
    -v /home/tako/disk/sdc/jwj_/CoWA-clean:/workspace/CoWA \
    -v /home/tako/disk/sdc/jwj_/DiffCE/miccai26/help/test:/workspace/CoWA/data:ro \
    -w /workspace/CoWA \
    pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime sleep infinity
```

The `miccai_rbt-home` volume holds the pip `--user` site, so deps survive
container removal. Only if that volume is also gone, run this first to
re-create it and install deps:

```bash
docker volume create miccai_rbt-home
docker run --rm -v miccai_rbt-home:/h \
    pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime chown 1000:1000 /h
# (then run the `docker run -d ...` above, then:)
docker exec miccai_rbt pip install --user --no-cache-dir -r requirements.txt
```

The `data/` mount points at the original `help/test/` location — no
duplicate copy of the 7 GB testset cache is made. Results written to
`/workspace/CoWA/results/` inside the container appear at
`/home/tako/disk/sdc/jwj_/CoWA-clean/results/` on the host as tako, so
`rm -rf results` works from the host directly.

## Citation

```
TODO: BibTeX once accepted.
```

## License

TODO (MIT recommended for code; data licenses follow original sources).
