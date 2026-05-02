"""
TTA 실험 실행 스크립트 (Ours - Offline Evaluation) - Cached Version

사전 저장된 testset_*.pt 파일을 로드하여 데이터 전처리 시간을 절약.
(.pt 파일은 save_test_dataset.py로 생성, xrayvision 전처리 완료 상태)

사용법:
    python run_tta_experiments_ours_cached.py --source chexpert --target mimic --exp_id ours_test1
    python run_tta_experiments_ours_cached.py --source chexpert --target all --exp_id ours_test1
    python run_tta_experiments_ours_cached.py --source chexpert --target all --exp_id ours_test1 --cache-dir ./data
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')

import torch
torch.set_num_threads(8)
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

from pretrained_inference import load_model, AVAILABLE_MODELS
from load_dataset import CANONICAL_LABELS
#from tta_ours_cooccur_offline import get_ours_adapter
#from tta_ours_cooccur_noadapt import get_ours_adapter
from tta_ours_weight_modified import get_ours_adapter

# 실험 설정
TTA_METHODS = ['ours']
TARGET_DATASETS = ['chexpert', 'mimic', 'vindr', 'nih']
DEFAULT_CACHE_DIR = './data'


class CachedTestDataset(torch.utils.data.Dataset):
    """사전 저장된 .pt 파일에서 로드한 텐서 기반 Dataset"""

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
    """testset_*.pt 파일에서 데이터셋 로드"""
    cache_path = Path(cache_dir) / f"testset_{target_dataset}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached dataset not found: {cache_path}\n"
                                f"Run 'python save_test_dataset.py --datasets {target_dataset}' first.")

    print(f"  Loading cached dataset: {cache_path}")
    data = torch.load(cache_path, map_location='cpu', weights_only=False)
    print(f"  Loaded: images={data['images'].shape}, labels={data['labels'].shape}, "
          f"samples={data['num_samples']}")

    return CachedTestDataset(data['images'], data['labels'], data['paths'])


def create_dataloader_from_cached(dataset: CachedTestDataset, batch_size: int = 32, shuffle: bool = False):
    """CachedTestDataset으로부터 DataLoader 생성 (전처리 불필요)"""
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        images = torch.stack([s['image'] for s in batch])    # (B, 1, 224, 224)
        labels = torch.stack([s['labels'] for s in batch])   # (B, 6)
        paths = [s['path'] for s in batch]
        return {'image': images, 'labels': labels, 'paths': paths}

    def worker_init_fn(worker_id):
        worker_seed = 42 + worker_id
        import random as _random
        import numpy as _np
        _random.seed(worker_seed)
        _np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)

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
    """
    예측 텐서와 레이블 텐서로부터 메트릭 계산
    (raw logits 사용, label mapping 적용)
    """
    outputs = predictions.numpy()
    labels_np = labels.numpy()

    # Label 매핑 생성
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
    batch_size: int = 32,
    device: str = 'cuda',
    threshold: float = 0.5,
    output_dir: Path = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    **tta_kwargs
) -> Dict:
    """단일 Ours TTA 실험 실행 (Offline Evaluation)"""
    print(f"\n{source_model.upper()} -> {target_dataset.upper()} (OURS, OFFLINE)")

    # 1. Source 모델 로드
    model_wrapper = load_model(model_name=source_model, device=device)
    model = model_wrapper.model

    # 2. Cached 데이터셋 로드 (전처리 완료 상태)
    dataset = load_cached_dataset(target_dataset, cache_dir=cache_dir)

    # 3. DataLoader 생성
    dataloader = create_dataloader_from_cached(dataset, batch_size=batch_size, shuffle=False)

    # 4. Ours Adapter 생성
    adapter = get_ours_adapter(model, device=device,
                               source_name=source_model, target_name=target_dataset,
                               **tta_kwargs)

    # 5. Offline evaluation: Adapt → Predict
    _, predictions, labels = adapter.adapt(dataloader)

    if predictions is not None and labels is not None:
        metrics = compute_metrics_from_tensors(predictions, labels, threshold=threshold, model=model)
    else:
        print("Warning: No predictions collected during offline eval")
        metrics = {}

    # 6. 결과 정리
    result = {
        'source_model': source_model,
        'target_dataset': target_dataset,
        'tta_method': 'ours',
        'split': 'test',
        'batch_size': batch_size,
        'threshold': threshold,
        'online_eval': False,
        'metrics': metrics
    }

    if 'AVERAGE' in metrics:
        print(f"\n{'='*70}")
        print(f"Results: OURS on {target_dataset.upper()} (OFFLINE, CACHED)")
        print(f"{'='*70}")
        for metric_name, value in metrics['AVERAGE'].items():
            print(f"  {metric_name:15s}: {value:.6f}")
        print(f"{'='*70}\n")

    return result


def append_result_to_csv(result: Dict, output_dir: Path):
    """단일 실험 결과를 CSV 파일에 추가"""
    summary_csv = output_dir / 'tta_summary.csv'
    detail_csv = output_dir / 'tta_detail.csv'

    source = result['source_model']
    target = result['target_dataset']
    method = result['tta_method']
    metrics = result['metrics']

    # Summary CSV에 평균 메트릭 추가
    if 'AVERAGE' in metrics:
        if not summary_csv.exists():
            with open(summary_csv, 'w') as f:
                f.write('Source,Target,Method,AUROC,AUPRC,F1,Precision,Recall,Accuracy\n')

        avg = metrics['AVERAGE']
        with open(summary_csv, 'a') as f:
            f.write(f'{source},{target},{method},'
                   f'{avg["AUROC"]:.6f},{avg["AUPRC"]:.6f},{avg["F1"]:.6f},'
                   f'{avg["Precision"]:.6f},{avg["Recall"]:.6f},{avg["Accuracy"]:.6f}\n')

    # Detail CSV에 pathology별 메트릭 추가
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
    batch_size: int = 32,
    device: str = 'cuda',
    output_dir: str = 'results',
    run_dir: str = None,
    exp_id: str = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    **tta_kwargs
):
    """모든 Ours 실험 실행 (Offline Evaluation) - Cached Version"""
    if target_datasets is None:
        target_datasets = TARGET_DATASETS

    if run_dir is not None:
        output_path = Path(run_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        from run_utils import make_run_dir
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
    print(f"Ours config: {tta_kwargs}")
    print(f"Cache dir: {cache_dir}")
    print(f"{'='*60}\n")

    print(f"Output Directory: {output_path}")
    print(f"  Summary CSV: {summary_csv}")
    print(f"  Detail CSV: {detail_csv}\n")

    all_results = []
    for target_dataset in target_datasets:
        try:
            result = run_tta_experiment(
                source_model=source_model,
                target_dataset=target_dataset,
                batch_size=batch_size,
                device=device,
                output_dir=output_path,
                cache_dir=cache_dir,
                **tta_kwargs
            )
            all_results.append(result)
            append_result_to_csv(result, output_path)

            import gc
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error in experiment {source_model}->{target_dataset} (ours): {e}")
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
    """실험 결과 요약 출력"""
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY (Ours)")
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

        print("AUROC by Target Dataset:")
        print("-" * 80)
        for _, row in df.iterrows():
            print(f"  {row['Target']:15s}: AUROC={row['AUROC']:.6f}  AUPRC={row['AUPRC']:.6f}  F1={row['F1']:.6f}")

        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Ours TTA Experiments - Offline Evaluation (Cached)')

    parser.add_argument('--source', type=str, default='chexpert',
                        choices=list(AVAILABLE_MODELS.keys()),
                        help='Source model (default: chexpert)')
    parser.add_argument('--target', type=str, default='all',
                        help='Target dataset (comma-separated or "all")')
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
    # Ours 하이퍼파라미터 (default=None → 미지정 시 config 파일 값 사용)
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--steps', type=int, default=None,
                        help='Optimization steps per batch (default: from config)')
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--centroid-momentum', type=float, default=None)
    parser.add_argument('--proj-dim', type=int, default=None)
    parser.add_argument('--lambda-align', type=float, default=None)
    parser.add_argument('--lambda-repel', type=float, default=None)
    parser.add_argument('--pattern-temp', type=float, default=None)
    parser.add_argument('--min-weight', type=float, default=None)
    parser.add_argument('--cooccur-threshold', type=float, default=None)

    args = parser.parse_args()

    if args.target == 'all':
        target_datasets = TARGET_DATASETS
    else:
        target_datasets = [t.strip() for t in args.target.split(',')]

    # Ours 하이퍼파라미터: None이 아닌 값만 전달 → config 파일 기본값을 덮어씀
    _all_kwargs = {
        'lr': args.lr,
        'steps': args.steps,
        'top_k': args.top_k,
        'centroid_momentum': args.centroid_momentum,
        'proj_dim': args.proj_dim,
        'lambda_align': args.lambda_align,
        'lambda_repel': args.lambda_repel,
        'pattern_temp': args.pattern_temp,
        'min_weight': args.min_weight,
        'cooccur_threshold': args.cooccur_threshold,
    }
    tta_kwargs = {k: v for k, v in _all_kwargs.items() if v is not None}

    print(f"\nRunning Ours TTA Experiments (Offline Evaluation - CACHED)")
    print(f"  Experiment ID: {args.exp_id}")
    print(f"  Source model: {args.source}")
    print(f"  Target datasets: {target_datasets}")
    print(f"  Cache dir: {args.cache_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output_dir}")
    print(f"  Ours config: {tta_kwargs}")
    print()

    run_all_experiments(
        source_model=args.source,
        target_datasets=target_datasets,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        exp_id=args.exp_id,
        cache_dir=args.cache_dir,
        **tta_kwargs
    )

    print("\n✓ All experiments completed!")


if __name__ == '__main__':
    main()
