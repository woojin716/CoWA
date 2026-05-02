"""
Co-occurrence Weighted Entropy TTA - Offline Evaluation Version

Multi-label 흉부 X-ray에서 co-occurrence 패턴을 sample weighting에 활용한 TTA.

핵심 아이디어:
- Co-occurrence matrix를 regularization이 아닌 sample weighting에 활용
- Sample의 disease co-occurrence 패턴이 전체 데이터와 일치하는 정도로 신뢰도 측정
- 높은 일치도 → entropy 강하게 적용, 낮은 일치도 → entropy 약하게 적용

Offline 평가:
- Pass 1: 배치마다 co-occurrence matrix 누적 갱신 + weighted entropy 적응
- Pass 2: 적응된 모델로 전체 데이터 재평가
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

from load_dataset import CANONICAL_LABELS
from tta_baselines_offline import BaseAdapter


def _load_ours_configs():
    """tta_ours_configs.json에서 기본 하이퍼파라미터 로드"""
    import json
    config_path = Path(__file__).parent.parent / 'configs' / 'tta_ours_configs.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

OURS_CONFIGS = _load_ours_configs()


def get_ours_adapter(model: nn.Module, device: str = 'cuda',
                                  source_name: str = None, target_name: str = None,
                                  **kwargs) -> 'CooccurWeightedAdapter':
    """
    Co-occurrence weighted entropy adapter 생성 (Offline)
    Config 파일의 기본값을 적용하고, kwargs로 override 가능
    """
    config = {**OURS_CONFIGS.get('cooccur_weighted', {}), **kwargs}
    valid_keys = {'lr', 'steps', 'cooccur_threshold', 'pattern_temp', 'min_weight', 'seed'}
    filtered = {k: v for k, v in config.items() if k in valid_keys}
    print(f"  [cooccur_weighted-offline] config: {filtered}")
    return CooccurWeightedAdapter(model, device,
                                  source_name=source_name, target_name=target_name,
                                  **config)


class CooccurWeightedAdapter(BaseAdapter):
    """
    Co-occurrence Weighted Entropy Minimization (Offline)

    Co-occurrence matrix를 regularization이 아닌 sample weighting에 활용:
    1. 배치가 들어올 때마다 co-occurrence matrix 누적 갱신
    2. Sample의 co-occurrence 패턴 일치도 → weight 계산
    3. Weighted entropy minimization으로 BN params 적응

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

        # 다른 adapter용 kwargs는 무시

        self.source_name = source_name
        self.target_name = target_name

        self.seed = seed
        self.lr = lr
        self.steps = steps

        self.cooccur_threshold = cooccur_threshold
        self.pattern_temp = pattern_temp
        self.min_weight = min_weight

        # Seed 고정
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Canonical label → 모델 출력 인덱스 매핑
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

        # BN affine params 학습
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
        **NOVEL: Co-occurrence를 sample 신뢰도 측정에 활용**

        Sample의 disease co-occurrence 패턴이 전체 데이터와 일치하는 정도
        → 높으면 신뢰도 높음 → entropy 강하게 적용
        → 낮으면 노이즈 가능성 → entropy 약하게 적용

        Args:
            probs: (N, C) predictions

        Returns:
            consistency: (N,) pattern matching scores in [0, 1]
        """
        N, C = probs.shape

        # Sample의 pairwise prediction products
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
        배치 예측을 누적하여 co-occurrence matrix 갱신 (incremental, soft)
        threshold 없이 raw probability를 그대로 사용

        Args:
            probs: (N, C) sigmoid probabilities (canonical labels only)
        """
        # ── Soft co-occurrence: threshold 없이 확률값 그대로 사용 ──
        batch_cooccur = torch.matmul(probs.t(), probs)  # (C, C)

        # ── Hard co-occurrence (original) ──
        # binary = (probs > self.cooccur_threshold).float()
        # batch_cooccur = torch.matmul(binary.t(), binary)  # (C, C)

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
        - Pass 1: 배치마다 co-occurrence 누적 갱신 + weighted entropy 적응
        - Pass 2: 적응된 모델로 전체 데이터 평가
        """
        print(f"CooccurWeighted (Offline): Starting adaptation")

        # Seed 고정
        self._seed_everything(self.seed)

        # Reset incremental state
        self._cooccur_sum = None
        self._sample_count = 0
        self.cooccur_matrix = None

        # ── Pass 1: 배치마다 co-occurrence 누적 + weighted entropy adaptation ──
        print("  Pass 1: Adapting with per-batch co-occurrence weighted entropy...")

        total_loss = 0.0
        total_weight = 0.0
        total_consistency = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Adaptation", leave=False, ncols=120)
        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)

            # 1) Co-occurrence 누적 갱신 (no grad)
            self.model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(images)
                canon_probs = outputs[:, self.class_indices]  # already sigmoid
                self._update_cooccurrence(canon_probs)

            # 2) Weighted entropy adaptation
            self.model.train()
            # for m in self.model.modules():
            #     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            #         m.eval()  # running stats 고정, affine만 gradient로 업데이트
            for _ in range(self.steps):
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)

                    # Canonical labels만 사용 (outputs already sigmoid)
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

            # tqdm에 실시간 loss 표시
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

        # ── Pass 2: 적응된 모델로 전체 데이터 평가 ──
        predictions, labels = self._evaluate(dataloader)
        return self.model, predictions, labels

    def _get_mean_cooccur(self) -> float:
        """Off-diagonal co-occurrence 평균"""
        if self.cooccur_matrix is None:
            return 0.0
        mask = 1 - torch.eye(self.cooccur_matrix.size(0)).to(self.device)
        off_diag = self.cooccur_matrix * mask
        return off_diag.sum().item() / (mask.sum().item() + 1e-8)

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.cooccur_matrix = None
        print("CooccurWeighted (Offline): Model reset to initial state")
