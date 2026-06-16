"""
Co-occurrence Weighted Entropy TTA (offline evaluation).

Test-time adaptation for multi-label chest X-rays that uses the disease
co-occurrence pattern as a per-sample weight for entropy minimization:
samples whose predicted co-occurrence matches the dataset statistics are
trusted more (stronger entropy), noisy ones less. Pass 1 accumulates the
co-occurrence matrix and adapts BN params; Pass 2 re-evaluates the dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple
import copy
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path

from data import CANONICAL_LABELS
from baselines import BaseAdapter


_FINAL_CONFIG_PATH = Path(__file__).parent.parent / 'configs' / 'tta_ours_configs_final.json'


def _load_final_configs():
    """Per-pair hyperparameters from configs/tta_ours_configs_final.json."""
    import json
    if _FINAL_CONFIG_PATH.exists():
        with open(_FINAL_CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}


OURS_FINAL_CONFIGS = _load_final_configs()


def _pair_key(source_name: str, target_name: str) -> str:
    src = "mimic" if source_name == "mimic_ch" else source_name
    return f"{src}_{target_name}"


def get_cowa_adapter(model: nn.Module, device: str = 'cuda',
                                  source_name: str = None, target_name: str = None,
                                  **kwargs) -> 'CooccurWeightedAdapter':
    """
    Build a co-occurrence weighted entropy adapter (offline).

    Hyperparameter priority (high -> low):
      1) **kwargs (CLI override)
      2) configs/tta_ours_configs_final.json::<source>_<target>
      3) CooccurWeightedAdapter constructor defaults
    """
    pair_cfg = {}
    if source_name and target_name:
        key = _pair_key(source_name, target_name)
        pair_cfg = dict(OURS_FINAL_CONFIGS.get(key, {}))
        if pair_cfg:
            print(f"  [cooccur_weighted-offline] pair config: {key}")
        else:
            print(f"  [cooccur_weighted-offline] WARNING: no entry for {key} in "
                  f"{_FINAL_CONFIG_PATH.name}; falling back to constructor defaults")

    valid_keys = {'lr', 'steps', 'cooccur_threshold', 'pattern_temp',
                  'min_weight', 'seed'}
    config = {**pair_cfg, **kwargs}
    config = {k: v for k, v in config.items() if k in valid_keys}
    print(f"  [cooccur_weighted-offline] config: {config}")
    return CooccurWeightedAdapter(model, device,
                                  source_name=source_name, target_name=target_name,
                                  **config)


class CooccurWeightedAdapter(BaseAdapter):
    """
    Co-occurrence Weighted Entropy Minimization (offline).

    Uses the co-occurrence matrix for per-sample weighting rather than
    as a regularizer:
    1. Incrementally update the co-occurrence matrix per batch.
    2. Derive a per-sample weight from how well its predicted
       co-occurrence pattern matches the accumulated matrix.
    3. Adapt BN affine params via weighted entropy minimization.

    Loss = (entropy * weight).mean()
    """

    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda',
                 lr: float = 1e-3,
                 steps: int = 1,
                 cooccur_threshold: float = 0.5,
                 pattern_temp: float = 0.1,
                 min_weight: float = 0.1,
                 seed: int = 42,
                 source_name: str = None,
                 target_name: str = None,
                 **kwargs):
        super().__init__(model, device)

        self.source_name = source_name
        self.target_name = target_name

        self.seed = seed
        self.lr = lr
        self.steps = steps

        self.cooccur_threshold = cooccur_threshold
        self.pattern_temp = pattern_temp
        self.min_weight = min_weight

        # Fix seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Map canonical labels to model output indices
        self.class_indices = []
        if hasattr(model, 'pathologies'):
            for canon in CANONICAL_LABELS:
                canon_norm = canon.lower().replace(' ', '_')
                for model_idx, model_label in enumerate(model.pathologies):
                    if not model_label:
                        continue
                    ml_norm = model_label.lower().replace(' ', '_')
                    if canon_norm == ml_norm or canon_norm in ml_norm or ml_norm in canon_norm:
                        self.class_indices.append(model_idx)
                        break
            print(f"  Canonical label mapping: {dict(zip(CANONICAL_LABELS, self.class_indices))}")
        else:
            self.class_indices = list(range(len(CANONICAL_LABELS)))
            print(f"  No pathologies attr, using indices 0~{len(CANONICAL_LABELS)-1}")
        self.num_classes = len(self.class_indices)

        self.model_state = copy.deepcopy(model.state_dict())

        # Train BN affine params only
        self.params = []
        for name, param in model.named_parameters():
            if 'bn' in name.lower() or 'norm' in name.lower():
                param.requires_grad = True
                self.params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler()

        # Co-occurrence incremental accumulation
        self.cooccur_matrix = None
        self._cooccur_sum = None   # (C, C) running sum of binary^T @ binary
        self._sample_count = 0

        print(f"CooccurWeighted (Offline): Initialized")
        print(f"  BN params: {len(self.params)}")
        print(f"  AMP: enabled")
        print(f"  lr={lr}, steps={steps}")
        print(f"  cooccur_threshold={self.cooccur_threshold}, pattern_temp={self.pattern_temp}, min_weight={self.min_weight}")

    def _compute_pattern_consistency(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Score how well each sample's predicted disease co-occurrence
        pattern matches the accumulated dataset matrix. Higher score means
        more reliable (entropy applied more strongly); lower score suggests
        noise (entropy applied more weakly).

        Args:
            probs: (N, C) predictions

        Returns:
            consistency: (N,) pattern matching scores in [0, 1]
        """
        N, C = probs.shape

        # Pairwise prediction products per sample
        pred_cooccur = probs.unsqueeze(2) * probs.unsqueeze(1)  # (N, C, C)

        # Expected co-occurrence pattern
        expected = self.cooccur_matrix.unsqueeze(0)  # (1, C, C)

        # Upper triangle mask (symmetric)
        mask = torch.triu(torch.ones(C, C, device=self.device), diagonal=1)

        # Pattern matching error (MSE)
        diff_sq = ((pred_cooccur - expected) ** 2) * mask.unsqueeze(0)
        distances = diff_sq.sum(dim=(1, 2)) / (mask.sum() + 1e-8)

        # Convert distance to consistency score
        consistency = torch.exp(-distances / self.pattern_temp)

        return consistency

    def _entropy_per_sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Binary entropy per sample (averaged over classes)
        Args:
            probs: (N, C) already sigmoid-activated probabilities
        """
        entropy = -probs * torch.log(probs + 1e-10) - \
                  (1 - probs) * torch.log(1 - probs + 1e-10)
        return entropy.mean(dim=1)  # (N,)

    def _update_cooccurrence(self, probs: torch.Tensor):
        """
        Incrementally update the co-occurrence matrix from a batch of
        predictions.

        Args:
            probs: (N, C) sigmoid probabilities (canonical labels only)
        """

        # Hard co-occurrence: binarize predictions at the threshold
        binary = (probs > self.cooccur_threshold).float()
        batch_cooccur = torch.matmul(binary.t(), binary)  # (C, C)

        if self._cooccur_sum is None:
            self._cooccur_sum = batch_cooccur
        else:
            self._cooccur_sum += batch_cooccur
        self._sample_count += probs.size(0)

        # Normalize: P(i,j) / sqrt(P(i) * P(j))
        cooccur = self._cooccur_sum / self._sample_count
        diag = torch.diag(cooccur)
        diag_sqrt = torch.sqrt(diag.unsqueeze(0) * diag.unsqueeze(1))
        cooccur = cooccur / (diag_sqrt + 1e-8)
        cooccur.fill_diagonal_(1.0)

        self.cooccur_matrix = cooccur

    def adapt(self, dataloader: DataLoader, **kwargs) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        """
        Offline adaptation:
        - Pass 1: accumulate co-occurrence per batch + weighted entropy adaptation
        - Pass 2: evaluate the full dataset with the adapted model
        """
        print(f"CooccurWeighted (Offline): Starting adaptation")

        # Fix seeds
        self._seed_everything(self.seed)

        # Reset incremental state
        self._cooccur_sum = None
        self._sample_count = 0
        self.cooccur_matrix = None

        # Pass 1: per-batch co-occurrence accumulation + weighted entropy adaptation
        print("  Pass 1: Adapting with per-batch co-occurrence weighted entropy...")

        total_loss = 0.0
        total_weight = 0.0
        total_consistency = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Adaptation", leave=False, ncols=120)
        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)

            # 1) Update co-occurrence accumulation (no grad)
            self.model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(images)
                canon_probs = outputs[:, self.class_indices]  # already sigmoid
                self._update_cooccurrence(canon_probs)

            # 2) Weighted entropy adaptation
            self.model.train()
            for _ in range(self.steps):
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)

                    # Canonical labels only (outputs already sigmoid)
                    canon_probs = outputs[:, self.class_indices]

                    # Per-sample entropy
                    entropy = self._entropy_per_sample(canon_probs)  # (N,)

                    # Co-occurrence pattern consistency → weights
                    consistency = self._compute_pattern_consistency(canon_probs)  # (N,)
                    weights = torch.clamp(consistency, min=self.min_weight)

                    # Weighted entropy loss
                    loss = (entropy * weights).mean()

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # Stats
            ent_val = loss.item()
            w_val = weights.mean().item()
            cons_val = consistency.mean().item()
            total_loss += ent_val
            total_weight += w_val
            total_consistency += cons_val
            num_batches += 1

            avg_loss = total_loss / num_batches
            avg_w = total_weight / num_batches
            pbar.set_postfix(
                loss=f"{ent_val:.4f}",
                w=f"{w_val:.2f}",
                cons=f"{cons_val:.2f}",
                avg_loss=f"{avg_loss:.4f}",
                avg_w=f"{avg_w:.2f}"
            )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_weight = total_weight / num_batches if num_batches > 0 else 0
        avg_consistency = total_consistency / num_batches if num_batches > 0 else 0
        print(f"  Adaptation completed.")
        print(f"    Avg weighted entropy: {avg_loss:.4f}")
        print(f"    Avg sample weight: {avg_weight:.3f}")
        print(f"    Avg consistency: {avg_consistency:.3f}")
        print(f"    Final co-occurrence mean off-diagonal: {self._get_mean_cooccur():.4f}")

        # Pass 2: evaluate the full dataset with the adapted model
        predictions, labels = self._evaluate(dataloader)
        return self.model, predictions, labels

    def _get_mean_cooccur(self) -> float:
        """Mean of the off-diagonal co-occurrence entries."""
        if self.cooccur_matrix is None:
            return 0.0
        mask = 1 - torch.eye(self.cooccur_matrix.size(0)).to(self.device)
        off_diag = self.cooccur_matrix * mask
        return off_diag.sum().item() / (mask.sum().item() + 1e-8)

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.cooccur_matrix = None
        print("CooccurWeighted (Offline): Model reset to initial state")
