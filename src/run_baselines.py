"""
TTA baseline experiment runner (cached version).

Loads pre-saved testset_*.pt files (created by save_test_dataset.py with
xrayvision preprocessing applied) to skip data preprocessing time.

Usage:
    python src/run_baselines.py --source chexpert --target mimic --method tent --exp_id test1
    python src/run_baselines.py --source chexpert --target all --method all --exp_id test1
    python src/run_baselines.py --source chexpert --target all --method all --exp_id test1 --cache-dir ./data
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import torch
torch.set_num_threads(2)
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List
import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, average_precision_score
)

from models import load_model, AVAILABLE_MODELS
from data import CANONICAL_LABELS
from baselines import get_adapter, OFFLINE_BASELINES

TTA_METHODS = OFFLINE_BASELINES
TARGET_DATASETS = ['chexpert', 'mimic', 'vindr', 'nih']
DEFAULT_CACHE_DIR = './data'


class CachedTestDataset(torch.utils.data.Dataset):
    """Tensor-backed dataset loaded from a pre-saved .pt file."""

    def __init__(self, images, labels, paths):
        self.images = images  # (N, 1, 224, 224)
        self.labels = labels  # (N, 6)
        self.paths = paths    # List[str]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],   # (1, 224, 224)
            'labels': self.labels[idx],  # (6,)
            'path': self.paths[idx],
        }


def load_cached_dataset(target_dataset: str, cache_dir: str = DEFAULT_CACHE_DIR):
    """Load a dataset from a testset_*.pt file."""
    cache_path = Path(cache_dir) / f"testset_{target_dataset}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached dataset not found: {cache_path}\n"
                                f"Run 'python save_test_dataset.py --datasets {target_dataset}' first.")

    print(f"  Loading cached dataset: {cache_path}")
    data = torch.load(cache_path, map_location='cpu', weights_only=False)
    print(f"  Loaded: images={data['images'].shape}, labels={data['labels'].shape}, "
          f"samples={data['num_samples']}")

    return CachedTestDataset(data['images'], data['labels'], data['paths'])


def create_dataloader_from_cached(dataset: CachedTestDataset, batch_size: int = 32, shuffle: bool = False, seed: int = 42):
    """Build a DataLoader from a CachedTestDataset (no preprocessing needed)."""
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        images = torch.stack([s['image'] for s in batch])    # (B, 1, 224, 224)
        labels = torch.stack([s['labels'] for s in batch])   # (B, 6)
        paths = [s['path'] for s in batch]
        return {'image': images, 'labels': labels, 'paths': paths}

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        import random as _random
        import numpy as _np
        _random.seed(worker_seed)
        _np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=g
    )


def compute_metrics_from_tensors(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    model: nn.Module = None
) -> Dict:
    """Compute metrics from prediction and label tensors (raw logits, with label mapping)."""
    outputs = predictions.numpy()
    labels_np = labels.numpy()

    label_mapping = {}
    if model is not None and hasattr(model, 'pathologies'):
        model_pathologies = model.pathologies
        for canon_label in CANONICAL_LABELS:
            for model_idx, model_label in enumerate(model_pathologies):
                if canon_label.lower().replace(' ', '_') == model_label.lower().replace(' ', '_'):
                    label_mapping[canon_label] = model_idx
                    break
                elif canon_label.lower() in model_label.lower() or model_label.lower() in canon_label.lower():
                    if canon_label == 'Pleural Effusion' and model_label == 'Effusion':
                        label_mapping[canon_label] = model_idx
                        break

    metrics = {}

    for idx, label in enumerate(CANONICAL_LABELS):
        y_true = labels_np[:, idx]

        if label in label_mapping:
            model_idx = label_mapping[label]
            y_pred = outputs[:, model_idx]
        else:
            y_pred = outputs[:, idx]

        y_pred_binary = (y_pred > threshold).astype(int)

        if len(np.unique(y_true)) < 2:
            continue

        try:
            auroc = roc_auc_score(y_true, y_pred)
            auprc = average_precision_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred_binary)

            metrics[label] = {
                'AUROC': auroc,
                'AUPRC': auprc,
                'F1': f1,
                'Precision': precision,
                'Recall': recall,
                'Accuracy': accuracy,
                'Positive_samples': int(y_true.sum()),
                'Total_samples': len(y_true)
            }
        except Exception as e:
            print(f"Warning: Could not calculate metrics for {label}: {e}")
            continue

    if metrics:
        avg_metrics = {}
        for metric_name in ['AUROC', 'AUPRC', 'F1', 'Precision', 'Recall', 'Accuracy']:
            values = [m[metric_name] for m in metrics.values() if not np.isnan(m[metric_name])]
            avg_metrics[metric_name] = np.mean(values) if values else np.nan

        metrics['AVERAGE'] = avg_metrics

    return metrics


def run_tta_experiment(
    source_model: str,
    target_dataset: str,
    tta_method: str,
    batch_size: int = 32,
    device: str = 'cuda',
    threshold: float = 0.5,
    cache_dir: str = DEFAULT_CACHE_DIR,
    seed: int = 42,
) -> Dict:
    """Run a single TTA experiment (online evaluation, cached version)."""
    print(f"\n{source_model.upper()} -> {target_dataset.upper()} ({tta_method.upper()}, ONLINE, CACHED, seed={seed})")

    if tta_method == 'oracle':
        actual_model = 'all'
    else:
        actual_model = source_model

    model_wrapper = load_model(model_name=actual_model, device=device)
    model = model_wrapper.model

    dataset = load_cached_dataset(target_dataset, cache_dir=cache_dir)

    dataloader = create_dataloader_from_cached(dataset, batch_size=batch_size, shuffle=False, seed=seed)

    adapter = get_adapter(tta_method, model, device=device)

    # Online evaluation: collect predictions while adapting
    _, predictions, labels = adapter.adapt(dataloader, online_eval=True, target_dataset=target_dataset)

    if predictions is not None and labels is not None:
        metrics = compute_metrics_from_tensors(predictions, labels, threshold=threshold, model=model)
    else:
        print("Warning: No predictions collected during online eval")
        metrics = {}

    result = {
        'source_model': source_model,
        'target_dataset': target_dataset,
        'tta_method': tta_method,
        'split': 'test',
        'batch_size': batch_size,
        'threshold': threshold,
        'online_eval': True,
        'metrics': metrics
    }

    if 'AVERAGE' in metrics:
        print(f"\n{'='*70}")
        print(f"Results: {tta_method.upper()} on {target_dataset.upper()} (ONLINE, CACHED)")
        print(f"{'='*70}")
        for metric_name, value in metrics['AVERAGE'].items():
            print(f"  {metric_name:15s}: {value:.6f}")
        print(f"{'='*70}\n")

    return result


def append_result_to_csv(result: Dict, output_dir: Path):
    """Append a single experiment result to the CSV files."""
    summary_csv = output_dir / 'tta_summary.csv'
    detail_csv = output_dir / 'tta_detail.csv'

    source = result['source_model']
    target = result['target_dataset']
    method = result['tta_method']
    metrics = result['metrics']

    # Append average metrics to the summary CSV
    if 'AVERAGE' in metrics:
        if not summary_csv.exists():
            with open(summary_csv, 'w') as f:
                f.write('Source,Target,Method,AUROC,AUPRC,F1,Precision,Recall,Accuracy\n')

        avg = metrics['AVERAGE']
        with open(summary_csv, 'a') as f:
            f.write(f'{source},{target},{method},'
                   f'{avg["AUROC"]:.6f},{avg["AUPRC"]:.6f},{avg["F1"]:.6f},'
                   f'{avg["Precision"]:.6f},{avg["Recall"]:.6f},{avg["Accuracy"]:.6f}\n')

    # Append per-pathology metrics to the detail CSV
    if not detail_csv.exists():
        with open(detail_csv, 'w') as f:
            f.write('Source,Target,Method,Pathology,AUROC,AUPRC,F1,Precision,Recall,Accuracy,Positive_samples,Total_samples\n')

    with open(detail_csv, 'a') as f:
        for pathology, path_metrics in metrics.items():
            if pathology == 'AVERAGE':
                continue
            f.write(f'{source},{target},{method},"{pathology}",'
                   f'{path_metrics["AUROC"]:.6f},{path_metrics["AUPRC"]:.6f},{path_metrics["F1"]:.6f},'
                   f'{path_metrics["Precision"]:.6f},{path_metrics["Recall"]:.6f},{path_metrics["Accuracy"]:.6f},'
                   f'{path_metrics["Positive_samples"]},{path_metrics["Total_samples"]}\n')

    print(f"✓ Results appended to: {summary_csv} and {detail_csv}")


def run_all_experiments(
    source_model: str = 'mimic_ch',
    target_datasets: List[str] = None,
    tta_methods: List[str] = None,
    batch_size: int = 32,
    device: str = 'cuda',
    output_dir: str = 'results',
    run_dir: str = None,
    exp_id: str = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    seed: int = 42,
):
    """Run all TTA experiments (online evaluation, cached version)."""
    if target_datasets is None:
        target_datasets = TARGET_DATASETS
    if tta_methods is None:
        tta_methods = TTA_METHODS

    if run_dir is not None:
        output_path = Path(run_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        from utils import make_run_dir
        output_path = make_run_dir(output_dir)

    summary_csv = output_path / 'tta_summary.csv'
    detail_csv = output_path / 'tta_detail.csv'

    # Setup log file
    log_file = output_path / 'exp.log'
    import sys
    import datetime

    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'a')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)

    print(f"\n{'='*60}")
    print(f"Experiment started: {datetime.datetime.now()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Seed: {seed}")
    print(f"Cache dir: {cache_dir}")
    print(f"{'='*60}\n")

    print(f"Output Directory: {output_path}")
    print(f"  Summary CSV: {summary_csv}")
    print(f"  Detail CSV: {detail_csv}\n")

    all_results = []
    for target_dataset in target_datasets:
        for tta_method in tta_methods:
            try:
                result = run_tta_experiment(
                    source_model=source_model,
                    target_dataset=target_dataset,
                    tta_method=tta_method,
                    batch_size=batch_size,
                    device=device,
                    cache_dir=cache_dir,
                    seed=seed,
                )
                all_results.append(result)
                append_result_to_csv(result, output_path)

                import gc
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"Error in experiment {source_model}->{target_dataset} ({tta_method}): {e}")
                import traceback
                traceback.print_exc()
                import gc
                torch.cuda.empty_cache()
                gc.collect()
                continue

    print_summary(all_results)

    print(f"\n{'='*60}")
    print(f"Experiment completed: {datetime.datetime.now()}")
    print(f"{'='*60}\n")

    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal


def print_summary(results: List[Dict]):
    """Print a summary of experiment results."""
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY (Cached)")
    print(f"{'='*80}\n")

    summary_data = []
    for result in results:
        if 'AVERAGE' in result['metrics']:
            row = {
                'Target': result['target_dataset'],
                'Method': result['tta_method'],
                'AUROC': result['metrics']['AVERAGE']['AUROC'],
                'AUPRC': result['metrics']['AVERAGE']['AUPRC'],
                'F1': result['metrics']['AVERAGE']['F1']
            }
            summary_data.append(row)

    if summary_data:
        df = pd.DataFrame(summary_data)

        print("AUROC by Method and Target Dataset:")
        print("-" * 80)
        pivot = df.pivot(index='Method', columns='Target', values='AUROC')
        print(pivot.to_string())
        print()

        print("\nBest Method per Target Dataset (by AUROC):")
        print("-" * 80)
        for target in df['Target'].unique():
            target_df = df[df['Target'] == target]
            best = target_df.loc[target_df['AUROC'].idxmax()]
            print(f"  {target:15s}: {best['Method']:15s} (AUROC: {best['AUROC']:.6f})")

        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='TTA Baseline Experiments - Online Evaluation (Cached)')

    parser.add_argument('--source', type=str, default='chexpert',
                        choices=list(AVAILABLE_MODELS.keys()),
                        help='Source model (default: chexpert)')
    parser.add_argument('--target', type=str, default='all',
                        help='Target dataset (comma-separated or "all")')
    parser.add_argument('--method', type=str, default='all',
                        help='TTA method (comma-separated or "all")')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base output directory (default: results). '
                             'Auto-creates <output_dir>/<YYYYMMDD>/<NNN>/ unless --run-dir is given.')
    parser.add_argument('--run-dir', type=str, default=None, dest='run_dir',
                        help='Specific run directory (overrides auto-creation under --output-dir).')
    parser.add_argument('--exp_id', type=str, default=None, dest='exp_id',
                        help='Optional experiment label (logged to exp.log).')
    parser.add_argument('--cache-dir', type=str, default=DEFAULT_CACHE_DIR,
                        help=f'Directory containing testset_*.pt files (default: {DEFAULT_CACHE_DIR})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    if args.target == 'all':
        target_datasets = TARGET_DATASETS
    else:
        target_datasets = [t.strip() for t in args.target.split(',')]

    if args.method == 'all':
        tta_methods = TTA_METHODS
    else:
        tta_methods = [m.strip() for m in args.method.split(',')]

    print(f"\nRunning TTA Baseline Experiments (Online Evaluation - CACHED)")
    print(f"  Experiment ID: {args.exp_id}")
    print(f"  Source model: {args.source}")
    print(f"  Target datasets: {target_datasets}")
    print(f"  TTA methods: {tta_methods}")
    print(f"  Cache dir: {args.cache_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output_dir}")
    print()

    run_all_experiments(
        source_model=args.source,
        target_datasets=target_datasets,
        tta_methods=tta_methods,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        exp_id=args.exp_id,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )

    print("\n✓ All experiments completed!")


if __name__ == '__main__':
    main()
