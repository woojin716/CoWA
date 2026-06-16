"""
Test-Time Adaptation (TTA) baselines for offline evaluation.

Offline protocol: adapt the model over the full dataset, then re-evaluate the
adapted model on the full dataset. BN is omitted since it is stateless and
equivalent to AdaBN offline.

Supported methods:
- Source-only: no adaptation (baseline)
- AdaBN: Adaptive Batch Normalization (Li et al., ICLR 2017)
- TENT: Test-Time Entropy Minimization (Wang et al., ICLR 2021)
- EATA: Efficient Anti-forgetting Test-Time Adaptation (Niu et al., ICML 2022)
- SAR: Sharpness-Aware and Reliable entropy minimization (Niu et al., ICLR 2023)
- CoTTA: Continual Test-Time Adaptation (Wang et al., CVPR 2022)
- MEMO: Test Time Robustness via Adaptation and Augmentation (Zhang et al., NeurIPS 2022)
- RoTTA: Robust Test-Time Adaptation (Yuan et al., CVPR 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from pathlib import Path
import torchvision.transforms as transforms


class BaseAdapter:
    """Base class for offline TTA adapters."""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    @staticmethod
    def _seed_everything(seed: int = 42):
        """Fix seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def adapt(self, dataloader: DataLoader, **kwargs) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        """
        Run offline adaptation.
        Pass 1: adapt the model over the full dataset.
        Pass 2: predict the full dataset with the adapted model.

        Returns:
            model: Adapted model
            predictions: All predictions (N, num_classes)
            labels: All labels (N, num_classes) or None
        """
        raise NotImplementedError

    def _evaluate(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the full dataset with the adapted model (Pass 2)."""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad(), torch.cuda.amp.autocast():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False, ncols=80):
                images = batch['image'].to(self.device, non_blocking=True)
                pred_logits = self.model(images)
                all_predictions.append(pred_logits.cpu())
                if 'labels' in batch:
                    all_labels.append(batch['labels'].cpu())

        all_predictions = torch.cat(all_predictions, dim=0) if all_predictions else None
        all_labels = torch.cat(all_labels, dim=0) if all_labels else None
        return all_predictions, all_labels

    def reset(self):
        raise NotImplementedError


class SourceOnlyAdapter(BaseAdapter):
    """Source-only baseline: use the source model without adaptation."""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        super().__init__(model, device)
        self.model.eval()

    def adapt(self, dataloader: DataLoader, **kwargs) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        self._seed_everything()
        print("Source-only: No adaptation performed")
        self.model.eval()
        predictions, labels = self._evaluate(dataloader)
        return self.model, predictions, labels

    def reset(self):
        pass


class AdaBNAdapter(BaseAdapter):
    """
    AdaBN: Adaptive Batch Normalization (Offline)
    Li et al., ICLR 2017 Workshop

    Pass 1: accumulate BN running stats over the full target data.
    Pass 2: predict the full dataset with the accumulated stats.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda', momentum=None, **kwargs):
        super().__init__(model, device)
        self.momentum = momentum
        self.model_state = copy.deepcopy(model.state_dict())

        if kwargs:
            print(f"  Note: AdaBN ignoring unused parameters: {list(kwargs.keys())}")

        print(f"AdaBN (Offline): Initialized (momentum={momentum})")

    def adapt(self, dataloader: DataLoader, **kwargs) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        self._seed_everything()
        print(f"AdaBN (Offline): Starting adaptation")

        # Reset BN statistics
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.reset_running_stats()
                module.momentum = self.momentum

        # Pass 1: accumulate BN stats over the full dataset
        self.model.train()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="AdaBN stats collection", leave=False, ncols=80):
                images = batch['image'].to(self.device)
                _ = self.model(images)

        print("AdaBN (Offline): Stats collection completed")

        # Pass 2: predict the full dataset with the adapted model
        predictions, labels = self._evaluate(dataloader)
        return self.model, predictions, labels

    def reset(self):
        self.model.load_state_dict(self.model_state)
        print("AdaBN (Offline): Model reset to initial state")


class TENTAdapter(BaseAdapter):
    """
    TENT: Test-Time Entropy Minimization (Offline)
    Wang et al., ICLR 2021

    Pass 1: entropy minimization over the full dataset (updates BN params).
    Pass 2: predict the full dataset with the adapted model.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 lr: float = 1e-3, steps: int = 1, **kwargs):
        super().__init__(model, device)

        if kwargs:
            print(f"  Note: TENT ignoring unused parameters: {list(kwargs.keys())}")

        self.lr = lr
        self.steps = steps
        self.model_state = copy.deepcopy(model.state_dict())

        self.params = []
        for name, param in model.named_parameters():
            if 'bn' in name.lower() or 'norm' in name.lower():
                param.requires_grad = True
                self.params.append(param)
            else:
                param.requires_grad = False

        print(f"TENT (Offline): Adapting {len(self.params)} BN parameters (lr={lr}, steps={steps})")
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def adapt(self, dataloader: DataLoader, **kwargs) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        self._seed_everything()
        print(f"TENT (Offline): Starting adaptation")

        total_loss = 0.0
        num_batches = 0

        # Pass 1: adaptation over the full dataset
        for batch in tqdm(dataloader, desc="TENT adaptation", leave=False, ncols=80):
            images = batch['image'].to(self.device)

            self.model.train()
            for _ in range(self.steps):
                outputs = self.model(images)
                loss = self._entropy_loss(outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"TENT (Offline): Adaptation completed. Average loss: {avg_loss:.4f}")

        # Pass 2: predict the full dataset with the adapted model
        predictions, labels = self._evaluate(dataloader)
        return self.model, predictions, labels

    def _entropy_loss(self, probs: torch.Tensor) -> torch.Tensor:
        # model output is already sigmoid-applied
        entropy = -probs * torch.log(probs + 1e-10) - (1 - probs) * torch.log(1 - probs + 1e-10)
        return entropy.mean()

    def reset(self):
        self.model.load_state_dict(self.model_state)
        print("TENT (Offline): Model reset to initial state")


class EATAAdapter(BaseAdapter):
    """
    EATA: Efficient Anti-forgetting Test-Time Adaptation (Offline)
    Niu et al., ICML 2022

    Pass 1: Fisher estimation + reliable-sample entropy minimization.
    Pass 2: predict the full dataset with the adapted model.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 lr: float = 1e-3, steps: int = 1,
                 e_margin: float = 0.4, fisher_alpha: float = 50.0, **kwargs):
        super().__init__(model, device)

        if kwargs:
            print(f"  Note: EATA ignoring: {list(kwargs.keys())}")

        self.lr = lr
        self.steps = steps
        self.e_margin = e_margin
        self.fisher_alpha = fisher_alpha
        self.model_state = copy.deepcopy(model.state_dict())
        self.log2 = np.log(2)

        self.params = []
        for name, param in model.named_parameters():
            if 'bn' in name.lower() or 'norm' in name.lower():
                param.requires_grad = True
                self.params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

        self.anchor = {n: p.clone().detach()
                       for n, p in model.named_parameters()
                       if 'bn' in n.lower() or 'norm' in n.lower()}
        self.fishers = {n: torch.ones_like(p).to(self.device) * 1e-4
                        for n, p in model.named_parameters()
                        if 'bn' in n.lower() or 'norm' in n.lower()}

        print(f"EATA (Offline): Initialized ({len(self.params)} BN params, e_margin={e_margin}, fisher_alpha={fisher_alpha})")

    def _compute_fishers(self, dataloader):
        """Estimate Fisher information."""
        self.model.train()
        for batch in dataloader:
            images = batch['image'].to(self.device)
            probs = self.model(images)  # already sigmoid-applied
            ent = (-probs * torch.log(probs + 1e-10)
                   - (1 - probs) * torch.log(1 - probs + 1e-10)).mean()
            ent.backward()
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n in self.fishers and p.grad is not None:
                        self.fishers[n] += p.grad.pow(2)
                        p.grad.zero_()
        num_batches = len(dataloader)
        if num_batches > 0:
            for n in self.fishers:
                self.fishers[n] /= num_batches

    def adapt(self, dataloader: DataLoader, **kwargs) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        self._seed_everything()
        print(f"EATA (Offline): Starting adaptation")

        print("  EATA: Estimating Fisher...")
        self._compute_fishers(dataloader)

        total_loss = 0.0
        num_batches = 0
        num_reliable = 0
        total_samples = 0

        # Pass 1: adaptation
        for batch in tqdm(dataloader, desc="EATA adaptation", leave=False, ncols=80):
            images = batch['image'].to(self.device)
            self.model.train()
            total_samples += images.size(0)

            for _ in range(self.steps):
                probs = self.model(images)  # already sigmoid-applied
                ent = (-probs * torch.log(probs + 1e-10)
                       - (1 - probs) * torch.log(1 - probs + 1e-10)).mean(dim=1) / self.log2

                reliable_mask = ent < self.e_margin
                if reliable_mask.sum() == 0:
                    continue

                num_reliable += reliable_mask.sum().item()
                ent_loss = ent[reliable_mask].mean()

                fisher_loss = torch.tensor(0.0, device=self.device)
                for n, p in self.model.named_parameters():
                    if n in self.fishers:
                        fisher_loss = fisher_loss + (self.fishers[n] * (p - self.anchor[n]).pow(2)).sum()

                loss = ent_loss + self.fisher_alpha * fisher_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() if reliable_mask.sum() > 0 else 0
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"EATA (Offline): Completed. Avg loss: {avg_loss:.4f}, reliable: {num_reliable}/{total_samples}")

        # Pass 2: evaluation
        predictions, labels = self._evaluate(dataloader)
        return self.model, predictions, labels

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.fishers = {n: torch.ones_like(p).to(self.device) * 1e-4
                        for n, p in self.model.named_parameters()
                        if 'bn' in n.lower() or 'norm' in n.lower()}
        print("EATA (Offline): Model reset to initial state")


class SARAdapter(BaseAdapter):
    """
    SAR: Sharpness-Aware and Reliable entropy minimization (Offline)
    Niu et al., ICLR 2023

    Pass 1: reliable entropy + SAM over the full dataset.
    Pass 2: predict the full dataset with the adapted model.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 lr: float = 1e-3, steps: int = 1,
                 e_margin: float = 0.4, rho: float = 0.05, **kwargs):
        super().__init__(model, device)

        if kwargs:
            print(f"  Note: SAR ignoring: {list(kwargs.keys())}")

        self.lr = lr
        self.steps = steps
        self.e_margin = e_margin
        self.rho = rho
        self.model_state = copy.deepcopy(model.state_dict())
        self.log2 = np.log(2)

        self.params = []
        for name, param in model.named_parameters():
            if 'bn' in name.lower() or 'norm' in name.lower():
                param.requires_grad = True
                self.params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9)
        print(f"SAR (Offline): Initialized ({len(self.params)} BN params, e_margin={e_margin}, rho={rho})")

    def adapt(self, dataloader: DataLoader, **kwargs) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        self._seed_everything()
        print(f"SAR (Offline): Starting adaptation")

        total_loss = 0.0
        num_batches = 0
        num_reliable = 0
        total_samples = 0

        # Pass 1: adaptation with SAM
        for batch in tqdm(dataloader, desc="SAR adaptation", leave=False, ncols=80):
            images = batch['image'].to(self.device)
            self.model.train()
            total_samples += images.size(0)

            for _ in range(self.steps):
                probs = self.model(images)  # already sigmoid-applied
                ent = (-probs * torch.log(probs + 1e-10)
                       - (1 - probs) * torch.log(1 - probs + 1e-10)).mean(dim=1) / self.log2

                reliable_mask = ent < self.e_margin
                if reliable_mask.sum() == 0:
                    continue

                num_reliable += reliable_mask.sum().item()
                loss = ent[reliable_mask].mean()

                # SAM: first step (ascent)
                self.optimizer.zero_grad()
                loss.backward()

                global_grad_norm = torch.norm(
                    torch.cat([p.grad.flatten() for p in self.params if p.grad is not None]), p=2
                )

                old_params = []
                with torch.no_grad():
                    scale = self.rho / (global_grad_norm + 1e-12)
                    for p in self.params:
                        old_params.append(p.data.clone())
                        if p.grad is not None:
                            p.data.add_(p.grad * scale)

                # SAM: second forward (at perturbed point)
                probs2 = self.model(images)  # already sigmoid-applied
                ent2 = (-probs2 * torch.log(probs2 + 1e-10)
                        - (1 - probs2) * torch.log(1 - probs2 + 1e-10)).mean(dim=1) / self.log2
                loss2 = ent2[reliable_mask].mean()

                # SAM: descent from original point
                self.optimizer.zero_grad()
                loss2.backward()

                with torch.no_grad():
                    for p, old_p in zip(self.params, old_params):
                        p.data.copy_(old_p)

                self.optimizer.step()

            total_loss += loss.item() if reliable_mask.sum() > 0 else 0
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"SAR (Offline): Completed. Avg loss: {avg_loss:.4f}, reliable: {num_reliable}/{total_samples}")

        # Pass 2: evaluation
        predictions, labels = self._evaluate(dataloader)
        return self.model, predictions, labels

    def reset(self):
        self.model.load_state_dict(self.model_state)
        print("SAR (Offline): Model reset to initial state")


class CoTTAAdapter(BaseAdapter):
    """
    CoTTA: Continual Test-Time Adaptation (Offline)
    Wang et al., CVPR 2022

    Pass 1: teacher-student adaptation over the full dataset.
    Pass 2: predict the full dataset with the teacher model.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 lr: float = 1e-3, ema_momentum: float = 0.999,
                 restoration_prob: float = 0.01, **kwargs):
        super().__init__(model, device)

        if kwargs:
            print(f"  Note: CoTTA ignoring unused parameters: {list(kwargs.keys())}")

        self.lr = lr
        self.ema_momentum = ema_momentum
        self.restoration_prob = restoration_prob
        self.model_state = copy.deepcopy(model.state_dict())

        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.eval()

        self.params = []
        for name, param in model.named_parameters():
            if 'bn' in name.lower() or 'norm' in name.lower():
                param.requires_grad = True
                self.params.append(param)
            else:
                param.requires_grad = False

        print(f"CoTTA (Offline): Adapting {len(self.params)} BN parameters")
        print(f"  EMA momentum: {self.ema_momentum}, Restoration prob: {self.restoration_prob}")

        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def adapt(self, dataloader: DataLoader, **kwargs) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        self._seed_everything()
        print(f"CoTTA (Offline): Starting adaptation")

        total_loss = 0.0
        num_batches = 0

        # Pass 1: adaptation
        for batch in tqdm(dataloader, desc="CoTTA adaptation", leave=False, ncols=80):
            images = batch['image'].to(self.device)

            self.model.train()
            outputs = self.model(images)
            loss = self._entropy_loss(outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._update_teacher()

            if torch.rand(1).item() < self.restoration_prob:
                self._restore_parameters()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"CoTTA (Offline): Adaptation completed. Average loss: {avg_loss:.4f}")

        # Pass 2: evaluate with the teacher model
        self.teacher_model.eval()
        self.model = self.teacher_model  # _evaluate uses self.model
        predictions, labels = self._evaluate(dataloader)
        return self.teacher_model, predictions, labels

    def _entropy_loss(self, probs: torch.Tensor) -> torch.Tensor:
        # model output is already sigmoid-applied
        entropy = -probs * torch.log(probs + 1e-10) - (1 - probs) * torch.log(1 - probs + 1e-10)
        return entropy.mean()

    def _update_teacher(self):
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(), self.model.parameters()
            ):
                teacher_param.data = (
                    self.ema_momentum * teacher_param.data +
                    (1 - self.ema_momentum) * student_param.data
                )

    def _restore_parameters(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.model_state:
                    param.data = self.model_state[name].to(self.device)

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.teacher_model.load_state_dict(self.model_state)
        print("CoTTA (Offline): Model reset to initial state")


class MEMOAdapter(BaseAdapter):
    """
    MEMO: Test Time Robustness via Adaptation and Augmentation
    Zhang et al., NeurIPS 2022

    Per-sample adaptation (single pass: adapt -> predict -> reset):
    - generate augmented views for each test sample
    - update BN params by minimizing marginal entropy
    - predict, then reset the model before the next sample
    """

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 lr: float = 1e-3, steps: int = 1,
                 n_augmentations: int = 8, **kwargs):
        super().__init__(model, device)

        if kwargs:
            print(f"  Note: MEMO ignoring unused parameters: {list(kwargs.keys())}")

        self.lr = lr
        self.steps = steps
        self.n_augmentations = n_augmentations
        self.model_state = copy.deepcopy(model.state_dict())

        self.params = []
        for name, param in model.named_parameters():
            if 'bn' in name.lower() or 'norm' in name.lower():
                param.requires_grad = True
                self.params.append(param)
            else:
                param.requires_grad = False

        print(f"MEMO: Adapting {len(self.params)} BN parameters")
        print(f"  n_augmentations: {self.n_augmentations}, steps: {self.steps}")

        self.augmentations = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    def adapt(self, dataloader: DataLoader, **kwargs) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        """MEMO: per-sample adapt → predict → reset (1 pass)"""
        self._seed_everything()
        print(f"MEMO: Starting adaptation")

        total_loss = 0.0
        num_samples = 0
        all_predictions = []
        all_labels = []

        for batch in tqdm(dataloader, desc="MEMO adaptation", leave=False, ncols=80):
            images = batch['image'].to(self.device)

            for i in range(images.size(0)):
                single_image = images[i:i+1]

                # Save state before per-sample adaptation
                sample_state = copy.deepcopy(self.model.state_dict())
                optimizer = torch.optim.Adam(self.params, lr=self.lr)

                for _ in range(self.steps):
                    self.model.train()

                    aug_images = [single_image]
                    for _ in range(self.n_augmentations - 1):
                        aug_images.append(self.augmentations(single_image))
                    aug_batch = torch.cat(aug_images, dim=0)

                    outputs = self.model(aug_batch)
                    loss = self._marginal_entropy(outputs)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                num_samples += 1

                # Predict AFTER per-sample adaptation
                self.model.eval()
                with torch.no_grad():
                    pred_logits = self.model(single_image)
                    all_predictions.append(pred_logits.cpu())

                # Reset model for next sample
                self.model.load_state_dict(sample_state)

            if 'labels' in batch:
                all_labels.append(batch['labels'].cpu())

        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        print(f"MEMO: Adaptation completed. Average loss: {avg_loss:.4f}")

        self.model.eval()

        all_predictions = torch.cat(all_predictions, dim=0) if all_predictions else None
        all_labels = torch.cat(all_labels, dim=0) if all_labels else None
        return self.model, all_predictions, all_labels

    def _marginal_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Marginal entropy: entropy of mean prediction (model output already sigmoid-applied)"""
        avg_probs = probs.mean(dim=0, keepdim=True)
        entropy = -avg_probs * torch.log(avg_probs + 1e-10) \
                  - (1 - avg_probs) * torch.log(1 - avg_probs + 1e-10)
        return entropy.mean()

    def reset(self):
        self.model.load_state_dict(self.model_state)
        print("MEMO: Model reset to initial state")


class RoTTAAdapter(BaseAdapter):
    """
    RoTTA: Robust Test-Time Adaptation (Offline)
    Yuan et al., CVPR 2023

    Pass 1: robust BN + memory bank + teacher-student adaptation over the full dataset.
    Pass 2: predict the full dataset with the teacher model.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 lr: float = 1e-3, steps: int = 1,
                 ema_momentum: float = 0.999, memory_size: int = 64,
                 nu: float = 0.001, **kwargs):
        super().__init__(model, device)

        if kwargs:
            print(f"  Note: RoTTA ignoring unused parameters: {list(kwargs.keys())}")

        self.lr = lr
        self.steps = steps
        self.ema_momentum = ema_momentum
        self.memory_size = memory_size
        self.nu = nu
        self.model_state = copy.deepcopy(model.state_dict())

        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.eval()

        self.global_bn_stats = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self.global_bn_stats[name] = {
                    'mean': module.running_mean.clone(),
                    'var': module.running_var.clone(),
                }

        self.params = []
        for name, param in model.named_parameters():
            if 'bn' in name.lower() or 'norm' in name.lower():
                param.requires_grad = True
                self.params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

        self.memory_images = []
        self.memory_ages = []
        self.time_step = 0

        print(f"RoTTA (Offline): Adapting {len(self.params)} BN parameters")
        print(f"  EMA: {ema_momentum}, memory_size: {memory_size}, nu: {nu}")

    def adapt(self, dataloader: DataLoader, **kwargs) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        self._seed_everything()
        print(f"RoTTA (Offline): Starting adaptation")

        total_loss = 0.0
        num_batches = 0

        # Pass 1: adaptation
        for batch in tqdm(dataloader, desc="RoTTA adaptation", leave=False, ncols=80):
            images = batch['image'].to(self.device)

            self._update_memory(images)
            self.time_step += 1
            self._update_robust_bn()

            self.model.train()
            for _ in range(self.steps):
                mem_images = self._sample_memory()
                if mem_images is None:
                    mem_images = images

                self.teacher_model.eval()
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(mem_images)

                student_outputs = self.model(mem_images)
                loss = self._cross_entropy_loss(student_outputs, teacher_outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self._update_teacher()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"RoTTA (Offline): Adaptation completed. Average loss: {avg_loss:.4f}")

        # Pass 2: evaluate with the teacher model
        self.teacher_model.eval()
        self.model = self.teacher_model
        predictions, labels = self._evaluate(dataloader)
        return self.teacher_model, predictions, labels

    def _update_memory(self, images: torch.Tensor):
        for i in range(images.size(0)):
            if len(self.memory_images) < self.memory_size:
                self.memory_images.append(images[i].detach().clone())
                self.memory_ages.append(self.time_step)
            else:
                oldest_idx = self.memory_ages.index(min(self.memory_ages))
                self.memory_images[oldest_idx] = images[i].detach().clone()
                self.memory_ages[oldest_idx] = self.time_step

    def _sample_memory(self) -> Optional[torch.Tensor]:
        if len(self.memory_images) == 0:
            return None
        n = min(len(self.memory_images), 16)
        indices = random.sample(range(len(self.memory_images)), n)
        return torch.stack([self.memory_images[i] for i in indices]).to(self.device)

    def _update_robust_bn(self):
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name in self.global_bn_stats:
                    self.global_bn_stats[name]['mean'] = (
                        (1 - self.nu) * self.global_bn_stats[name]['mean'] +
                        self.nu * module.running_mean
                    )
                    self.global_bn_stats[name]['var'] = (
                        (1 - self.nu) * self.global_bn_stats[name]['var'] +
                        self.nu * module.running_var
                    )
                    module.running_mean.copy_(self.global_bn_stats[name]['mean'])
                    module.running_var.copy_(self.global_bn_stats[name]['var'])

    def _update_teacher(self):
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(), self.model.parameters()
            ):
                teacher_param.data = (
                    self.ema_momentum * teacher_param.data +
                    (1 - self.ema_momentum) * student_param.data
                )

    def _cross_entropy_loss(self, student_probs, teacher_probs):
        # model output is already sigmoid-applied; no re-application of sigmoid
        teacher_probs = teacher_probs.detach()
        loss = -teacher_probs * torch.log(student_probs + 1e-10) \
               - (1 - teacher_probs) * torch.log(1 - student_probs + 1e-10)
        return loss.mean()

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.teacher_model.load_state_dict(self.model_state)
        self.memory_images.clear()
        self.memory_ages.clear()
        self.time_step = 0
        for name, module in self.model.named_modules():
            if name in self.global_bn_stats:
                self.global_bn_stats[name]['mean'] = module.running_mean.clone()
                self.global_bn_stats[name]['var'] = module.running_var.clone()
        print("RoTTA (Offline): Model reset to initial state")


def _load_baseline_configs():
    """Load default hyperparameters from tta_baseline_configs.json."""
    import json
    config_path = Path(__file__).parent.parent / 'configs' / 'tta_baseline_configs.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

BASELINE_CONFIGS = _load_baseline_configs()

ADAPTERS = {
    'source_only': SourceOnlyAdapter,
    'adabn': AdaBNAdapter,
    'tent': TENTAdapter,
    'eata': EATAAdapter,
    'sar': SARAdapter,
    'cotta': CoTTAAdapter,
    'memo': MEMOAdapter,
    'rotta': RoTTAAdapter,
}


def get_adapter(method: str, model: nn.Module, device: str = 'cuda', **kwargs) -> BaseAdapter:
    """
    Get a TTA adapter for offline evaluation.
    Applies defaults from the config file; kwargs override them.
    """
    if method not in ADAPTERS:
        raise ValueError(f"Unknown TTA method: {method}. Available: {list(ADAPTERS.keys())}")

    config = {**BASELINE_CONFIGS.get(method, {}), **kwargs}
    print(f"  [{method}] config: {config}")

    if method == 'source_only':
        return ADAPTERS[method](model, device)
    return ADAPTERS[method](model, device, **config)


# Offline TTA baselines (BN excluded)
OFFLINE_BASELINES = [
    'source_only',  # Lower bound
    'tent',         # Offline TENT (full pass adapt → eval)
    'cotta',        # + Continual adaptation
    'adabn',        # Offline AdaBN (full stats collection)
    'eata',         # + Anti-forgetting
    #'sar',          # + Sharpness-aware
    #'memo',         # Per-sample adaptation
    'rotta',        # + Robust adaptation
]
