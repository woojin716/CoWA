"""
TorchXRayVision Pretrained Models
데이터셋별 pretrained classifier 로딩 및 inference

지원하는 모델:
- PadChest (University of Alicante): densenet121-res224-pc
- CheXpert (Stanford): densenet121-res224-chex
- MIMIC-CXR (MIT): densenet121-res224-mimic_nb, densenet121-res224-mimic_ch
- 기타: densenet121-res224-nih, densenet121-res224-all
"""

import torch
import torch.nn as nn
import torchxrayvision as xrv
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import Union, List, Dict, Tuple
from torchvision import transforms


# 사용 가능한 pretrained 모델
AVAILABLE_MODELS = {
    'padchest': 'densenet121-res224-pc',
    'chexpert': 'densenet121-res224-chex',
    'chexpert_cosine': 'densenet121-res224-chex',  # CheXpert with custom cosine classifier
    #'mimic_nb': 'densenet121-res224-mimic_nb',
    'mimic_ch': 'densenet121-res224-mimic_ch',
    'nih': 'densenet121-res224-nih',
    'all': 'densenet121-res224-all',  # reference (upper bound)
}


class XRayPretrainedModel:
    """
    TorchXRayVision pretrained 모델 래퍼 클래스
    """

    def __init__(self, model_name: str = 'all', device: str = None):
        """
        Args:
            model_name: 모델 이름 ('padchest', 'chexpert', 'mimic_nb', 'mimic_ch', 'nih', 'all')
            device: 디바이스 ('cuda' 또는 'cpu')
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")

        self.model_name = model_name
        self.weights = AVAILABLE_MODELS[model_name]

        # 디바이스 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 모델 로드
        print(f"Loading {model_name} model ({self.weights})...")
        self.model = xrv.models.DenseNet(weights=self.weights)
        self.model = self.model.to(self.device)

        # Custom classifier 로드 (chexpert_cosine인 경우)
        if model_name == 'chexpert_cosine':
            print("  Loading custom IndependentCosineClassifier...")
            from pathlib import Path
            import sys
            sys.path.insert(0, str(Path(__file__).parent / 'pretrained'))
            from chexpert_ind_cosine_model import IndependentCosineClassifier

            # Custom classifier 로드
            custom_classifier = IndependentCosineClassifier(
                input_dim=1024,
                num_classes=4,
                embed_dim=3,
                scale=10.0
            )
            classifier_path = Path(__file__).parent / 'pretrained' / 'chexpert_ind_cosine_scale10.pt'
            custom_classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
            custom_classifier = custom_classifier.to(self.device)

            # 원본 classifier 교체
            self.model.classifier = custom_classifier

            # Custom labels (4 classes only)
            self.model.pathologies = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Consolidation']

            # Disable op_threshs (dimension mismatch: 4 vs 18)
            self.model.op_threshs = None

            print(f"  ✓ Custom classifier loaded from: {classifier_path.name}")

        self.model.eval()

        # 라벨 정보
        self.labels = self.model.pathologies
        print(f"Model loaded. Predicting {len(self.labels)} pathologies:")
        print(f"  {', '.join(self.labels)}")

        # Backbone과 Classifier 분리
        self._setup_feature_extractor()

        # TorchXRayVision의 기본 transform
        self.transform = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])

    def _setup_feature_extractor(self):
        """
        Backbone과 Classifier head를 분리
        DenseNet의 features 레이어를 직접 사용하여 feature extractor를 생성
        """
        # 원본 모델의 classifier 저장
        self.classifier = self.model.classifier

        # Feature dimension 계산
        # TorchXRayVision의 DenseNet은 features2 메서드를 사용
        if hasattr(self.model, 'features2'):
            self._features_fn = self.model.features2
        elif hasattr(self.model, 'features'):
            # fallback: features 레이어를 직접 사용
            self._features_fn = None
        else:
            raise AttributeError("Model does not have 'features' or 'features2' attribute")

        # Use random input instead of zeros to avoid normalization warning
        dummy_input = torch.randn(1, 1, 224, 224).to(self.device)
        with torch.no_grad():
            feat = self._extract_backbone_features(dummy_input)
            self._feature_dim = feat.shape[1]

        print(f"Feature extractor setup complete. Feature dimension: {self._feature_dim}")

    def _extract_backbone_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Backbone features 추출 (internal helper)

        Args:
            img: 전처리된 이미지 텐서

        Returns:
            Feature tensor
        """
        if self._features_fn is not None:
            # features2 메서드 사용 (TorchXRayVision DenseNet)
            features = self._features_fn(img)
        else:
            # features 레이어 직접 사용
            features = self.model.features(img)
            features = nn.functional.relu(features, inplace=True)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)

        return features

    def get_feature_dim(self) -> int:
        """
        Feature (latent) 차원 반환

        Returns:
            Feature 차원 크기
        """
        if hasattr(self, '_feature_dim') and self._feature_dim is not None:
            return self._feature_dim

        # Fallback: 실제로 forward pass를 수행
        dummy_input = torch.randn(1, 1, 224, 224).to(self.device)
        with torch.no_grad():
            features = self._extract_backbone_features(dummy_input)
        self._feature_dim = features.shape[1]
        return self._feature_dim

    def extract_features(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        단일 이미지에서 backbone feature (latent) 추출

        Args:
            image_path: 이미지 경로

        Returns:
            Feature vector (latent representation)
        """
        with torch.no_grad():
            # 이미지 전처리
            img = self.preprocess_image(image_path)
            img = img.to(self.device)

            # Backbone features 추출
            features = self._extract_backbone_features(img)
            features = features.cpu().numpy()[0]  # (feature_dim,)

            return features

    def extract_features_batch(self, image_paths: List[Union[str, Path]],
                               batch_size: int = 8) -> np.ndarray:
        """
        배치 이미지에서 backbone features 추출

        Args:
            image_paths: 이미지 경로 리스트
            batch_size: 배치 크기

        Returns:
            Features array (num_images, feature_dim)
        """
        all_features = []

        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]

                # 배치 전처리
                batch_images = []
                for path in batch_paths:
                    img = self.preprocess_image(path)
                    batch_images.append(img)

                batch_images = torch.cat(batch_images, dim=0).to(self.device)

                # Backbone features 추출
                features = self._extract_backbone_features(batch_images)
                features = features.cpu().numpy()

                all_features.append(features)

        return np.vstack(all_features)

    def predict_from_features(self, features: Union[np.ndarray, torch.Tensor],
                              apply_op_norm: bool = True) -> np.ndarray:
        """
        추출된 features에서 직접 예측 수행

        Args:
            features: Feature vectors (batch_size, feature_dim) 또는 (feature_dim,)
            apply_op_norm: op_norm을 적용할지 여부 (기본: True)

        Returns:
            Predictions (batch_size, num_classes) 또는 (num_classes,)
        """
        with torch.no_grad():
            # numpy array를 tensor로 변환
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()

            # 단일 샘플인 경우 batch 차원 추가
            if features.dim() == 1:
                features = features.unsqueeze(0)
                single_sample = True
            else:
                single_sample = False

            features = features.to(self.device)

            # Classifier 적용
            outputs = self.classifier(features)

            # Sigmoid 적용 (TorchXRayVision은 classifier 후 sigmoid 적용)
            # op_norm 적용 시에도 sigmoid가 먼저 필요
            if apply_op_norm and hasattr(self.model, 'op_threshs'):
                from torchxrayvision.models import op_norm
                outputs = torch.sigmoid(outputs)
                outputs = op_norm(outputs, self.model.op_threshs)
            elif hasattr(self.model, 'apply_sigmoid') and self.model.apply_sigmoid:
                outputs = torch.sigmoid(outputs)

            outputs = outputs.cpu().numpy()

            # 단일 샘플인 경우 batch 차원 제거
            if single_sample:
                outputs = outputs[0]

            return outputs

    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        이미지 전처리

        Args:
            image_path: 이미지 경로

        Returns:
            전처리된 이미지 텐서 (1, 1, 224, 224)
        """
        try:
            # 이미지 로드
            img = Image.open(image_path).convert('L')  # Grayscale
            img = np.array(img, dtype=np.float32)            
            img = np.clip(img, 0.0, 255.0)
            img = img[np.newaxis, :, :]  # (1, H, W)
            img = self.transform(img)  # Still (1, H, W) after crop and resize

            #img = (img / 255.0) * 2.0 - 1.0
            img = xrv.datasets.normalize(img, maxval=255, reshape=False)

            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img)

            img = img.unsqueeze(0)  # (1, 1, 224, 224)

            # # Final NaN check
            # if torch.isnan(img).any():
            #     print(f"Warning: NaN detected in preprocessed image {image_path}, replacing with zeros")
            #     img = torch.nan_to_num(img, nan=0.0)

            return img

        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            # Return None to skip this image
            return None



    @torch.no_grad()
    def predict(self, image_path: Union[str, Path], top_k: int = None) -> Dict:
        """
        단일 이미지에 대한 예측

        Args:
            image_path: 이미지 경로
            top_k: 상위 k개만 반환 (None이면 모두 반환)

        Returns:
            예측 결과 딕셔너리
        """
        # 이미지 전처리
        img = self.preprocess_image(image_path)
        if img is None:
            return None
        img = img.to(self.device)

        # 예측
        outputs = self.model(img)
        outputs = outputs.cpu().numpy()[0]  # (num_classes,)

        # Check for NaN in outputs
        if np.isnan(outputs).any():
            print(f"Warning: NaN in model output for image {image_path}, replacing with zeros")
            outputs = np.nan_to_num(outputs, nan=0.0)

        # 결과 정리
        predictions = {
            label: float(score)
            for label, score in zip(self.labels, outputs)
        }

        # top_k 필터링
        if top_k is not None:
            predictions = dict(
                sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
            )
        else:
            predictions = dict(
                sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            )

        return predictions

    @torch.no_grad()
    def predict_batch(self, image_paths: List[Union[str, Path]], batch_size: int = 8) -> List[Dict]:
        """
        배치 이미지에 대한 예측

        Args:
            image_paths: 이미지 경로 리스트
            batch_size: 배치 크기

        Returns:
            예측 결과 리스트
        """
        all_predictions = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]

            # 배치 전처리
            batch_images = []
            valid_paths = []
            for path in batch_paths:
                img = self.preprocess_image(path)
                if img is not None:
                    batch_images.append(img)
                    valid_paths.append(path)

            # 유효한 이미지가 없으면 스킵
            if len(batch_images) == 0:
                continue

            batch_images = torch.cat(batch_images, dim=0).to(self.device)

            x = batch_images

            # 예측
            outputs = self.model(batch_images)
            outputs = outputs.cpu().numpy()

            # 결과 정리
            for j, output in enumerate(outputs):
                # # Check for NaN in outputs
                # if np.isnan(output).any():
                #     print(f"Warning: NaN in model output for image {valid_paths[j]}, replacing with zeros")
                #     output = np.nan_to_num(output, nan=0.0)

                predictions = {
                    label: float(score)
                    for label, score in zip(self.labels, output)
                }
                predictions = dict(
                    sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                )
                all_predictions.append({
                    'image_path': str(valid_paths[j]),
                    'predictions': predictions
                })

        return all_predictions


def load_model(model_name: str = 'all', device: str = None) -> XRayPretrainedModel:
    """
    Pretrained 모델 로드

    Args:
        model_name: 모델 이름 ('padchest', 'chexpert', 'mimic_nb', 'mimic_ch', 'nih', 'all')
        device: 디바이스

    Returns:
        XRayPretrainedModel 인스턴스
    """
    return XRayPretrainedModel(model_name=model_name, device=device)


def evaluate_model(source_model: str, target_dataset: str, target_split: str = 'test',
                   device: str = None, batch_size: int = 8, threshold: float = 1.0) -> Dict:
    """
    Source 모델을 Target 데이터셋에서 평가
    Returns:
        성능 메트릭 딕셔너리
    """
    try:
        from sklearn.metrics import (
            roc_auc_score, f1_score, precision_score,
            recall_score, accuracy_score, average_precision_score
        )
        from load_dataset import get_dataset, CANONICAL_LABELS
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError(f"Required package not installed: {e}. Install with: pip install scikit-learn tqdm")

    print(f"\n{'='*70}")
    print(f"Evaluating {source_model.upper()} on {target_dataset.upper()} {target_split} set")
    print(f"{'='*70}\n")

    # 모델 로드
    print(f"Loading {source_model} model...")
    model = load_model(model_name=source_model, device=device)

    # ===== DEBUG: 모델 설정 확인 =====
    print(f"\n[DEBUG] Model configuration:")
    print(f"  op_threshs: {getattr(model.model, 'op_threshs', 'N/A')}")
    print(f"  apply_sigmoid: {getattr(model.model, 'apply_sigmoid', 'N/A')}")
    print(f"  op_threshs is None: {model.model.op_threshs is None if hasattr(model.model, 'op_threshs') else 'N/A'}")
    # ================================

    # 데이터셋 로드
    print(f"Loading {target_dataset} {target_split} dataset...")
    dataset = get_dataset(
        target_dataset,
        split=target_split,
        transform=None,
        use_canonical_labels=True,
        frontal_only=True
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Model pathologies: {model.labels}")
    print(f"Canonical labels: {CANONICAL_LABELS}\n")

    # Label 매핑 생성 (모델 pathology -> canonical label)
    label_mapping = {}
    for canon_label in CANONICAL_LABELS:
        # 모델의 pathology와 매칭
        for model_label in model.labels:
            if canon_label.lower().replace(' ', '_') == model_label.lower().replace(' ', '_'):
                label_mapping[canon_label] = model_label
                break
            elif canon_label.lower() in model_label.lower() or model_label.lower() in canon_label.lower():
                # 부분 매칭 (예: "Pleural Effusion" <-> "Effusion")
                if canon_label == 'Pleural Effusion' and model_label == 'Effusion':
                    label_mapping[canon_label] = model_label
                    break

    print(f"Label mapping: {label_mapping}\n")

    # 이미지 경로와 ground truth 추출
    image_paths = []
    y_true_all = {label: [] for label in CANONICAL_LABELS}

    for i in range(len(dataset)):
        sample = dataset[i]
        image_paths.append(sample['path'])

        # Ground truth labels (canonical labels)
        labels = sample['labels']  # tensor of size [5]
        for idx, label_name in enumerate(CANONICAL_LABELS):
            y_true_all[label_name].append(int(labels[idx].item()))

    # 배치 예측
    print(f"Running predictions on {len(image_paths)} images...")
    y_pred_all = {label: [] for label in CANONICAL_LABELS}
    y_pred_binary_all = {label: [] for label in CANONICAL_LABELS}

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
        batch_paths = image_paths[i:i+batch_size]
        batch_results = model.predict_batch(batch_paths, batch_size=batch_size)

        for result in batch_results:
            predictions = result['predictions']

            # Canonical labels에 대한 예측 점수 추출
            for canon_label in CANONICAL_LABELS:
                if canon_label in label_mapping:
                    model_label = label_mapping[canon_label]
                    score = predictions.get(model_label, 0.0)
                    y_pred_all[canon_label].append(score)
                    y_pred_binary_all[canon_label].append(1 if score > threshold else 0)
                else:
                    # 매핑되지 않은 라벨은 0으로 처리
                    y_pred_all[canon_label].append(0.0)
                    y_pred_binary_all[canon_label].append(0)

    # 메트릭 계산
    print("\nCalculating metrics...")
    print(f"Debug: y_true_all lengths: {[len(y_true_all[l]) for l in CANONICAL_LABELS]}")
    print(f"Debug: y_pred_all lengths: {[len(y_pred_all[l]) for l in CANONICAL_LABELS]}")

    metrics = {}

    for label in CANONICAL_LABELS:
        y_true = np.array(y_true_all[label])
        y_pred = np.array(y_pred_all[label])
        y_pred_binary = np.array(y_pred_binary_all[label])

        # 길이 체크
        if len(y_true) != len(y_pred):
            print(f"Warning: Length mismatch for {label}: y_true={len(y_true)}, y_pred={len(y_pred)}, skipping...")
            continue

        # NaN 체크
        if np.isnan(y_true).any():
            print(f"Warning: y_true contains {np.isnan(y_true).sum()} NaN values for {label}, skipping...")
            continue
        if np.isnan(y_pred).any():
            print(f"Warning: y_pred contains {np.isnan(y_pred).sum()} NaN values for {label}, skipping...")
            continue

        # 해당 라벨이 데이터셋에 존재하는지 확인
        if y_true.sum() == 0 and (1 - y_true).sum() == 0:
            # 모든 샘플이 동일한 클래스면 스킵
            continue

        try:
            # AUROC (최소 2개 클래스 필요)
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_pred)
                auprc = average_precision_score(y_true, y_pred)
            else:
                auroc = np.nan
                auprc = np.nan

            # F1, Precision, Recall, Accuracy
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

    # 평균 계산 (NaN 제외)
    if metrics:
        avg_metrics = {}
        for metric_name in ['AUROC', 'AUPRC', 'F1', 'Precision', 'Recall', 'Accuracy']:
            values = [m[metric_name] for m in metrics.values() if not np.isnan(m[metric_name])]
            avg_metrics[metric_name] = np.mean(values) if values else np.nan

        metrics['AVERAGE'] = avg_metrics

    return metrics


def print_evaluation_results(metrics: Dict):
    """
    평가 결과 출력

    Args:
        metrics: evaluate_model에서 반환된 메트릭 딕셔너리
    """
    import pandas as pd

    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70 + "\n")

    # DataFrame 생성
    df_data = []
    for label, label_metrics in metrics.items():
        if label == 'AVERAGE':
            continue
        row = {'Pathology': label}
        row.update(label_metrics)
        df_data.append(row)

    if df_data:
        df = pd.DataFrame(df_data)

        # 수치 열만 소수점 4자리로 포맷
        numeric_cols = ['AUROC', 'AUPRC', 'F1', 'Precision', 'Recall', 'Accuracy']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")

        print(df.to_string(index=False))
        print("\n" + "="*70)

        # 평균 출력
        if 'AVERAGE' in metrics:
            print("\nAverage Metrics:")
            print("-"*70)
            for metric_name, value in metrics['AVERAGE'].items():
                if not np.isnan(value):
                    print(f"  {metric_name:15s}: {value:.4f}")
                else:
                    print(f"  {metric_name:15s}: N/A")
            print("="*70)


def print_predictions(predictions: Dict, threshold: float = 0.5):
    """
    예측 결과 출력

    Args:
        predictions: 예측 결과 딕셔너리
        threshold: positive로 간주할 임계값
    """
    print("\n" + "="*60)
    print("Predictions:")
    print("="*60)

    for label, score in predictions.items():
        indicator = "✓" if score > threshold else " "
        print(f"[{indicator}] {label:30s}: {score:.4f}")

    print("="*60)

    # Positive 결과만 추출
    positive = {k: v for k, v in predictions.items() if v > threshold}
    if positive:
        print(f"\nPositive findings (> {threshold}):")
        for label, score in positive.items():
            print(f"  - {label}: {score:.4f}")
    else:
        print(f"\nNo positive findings (threshold: {threshold})")


def main():
    parser = argparse.ArgumentParser(description='TorchXRayVision Pretrained Model Inference and Evaluation')

    # Mode selection
    parser.add_argument('--mode', type=str, default='predict',
                        choices=['predict', 'evaluate', 'extract_features'],
                        help='Mode: predict (classification), evaluate (on dataset), or extract_features (backbone features)')

    # Model selection
    parser.add_argument('--model', type=str, default='all',
                        choices=list(AVAILABLE_MODELS.keys()),
                        help='Pretrained model to use (source model for evaluation)')

    # Predict mode arguments
    parser.add_argument('--image', type=str, default=None,
                        help='Path to X-ray image (for predict and extract_features mode)')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Show top-k predictions only (default: all)')
    parser.add_argument('--save-features', type=str, default=None,
                        help='Path to save extracted features as .npy file')

    # Evaluate mode arguments
    parser.add_argument('--target', type=str, default=None,
                        choices=['chexpert', 'mimic', 'padchest', 'vindr'],
                        help='Target dataset for evaluation (for evaluate mode)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test', 'valid'],
                        help='Dataset split (default: test)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation (default: 8)')

    # Common arguments
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: auto)')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Threshold for positive predictions (default: 1.0 for op_norm, 0.5 for sigmoid only)')

    args = parser.parse_args()

    if args.mode == 'predict':
        # Predict mode: single image inference
        if args.image is None:
            print("Error: --image is required for predict mode")
            return

        # 이미지 존재 확인
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return

        # 모델 로드
        model = load_model(model_name=args.model, device=args.device)

        # 예측
        print(f"\nProcessing image: {image_path}")
        predictions = model.predict(image_path, top_k=args.top_k)

        # 결과 출력
        print_predictions(predictions, threshold=args.threshold)

    elif args.mode == 'evaluate':
        # Evaluate mode: evaluate model on dataset
        if args.target is None:
            print("Error: --target is required for evaluate mode")
            return

        # 평가 실행
        metrics = evaluate_model(
            source_model=args.model,
            target_dataset=args.target,
            target_split=args.split,
            device=args.device,
            batch_size=args.batch_size,
            threshold=args.threshold
        )

        # 결과 출력
        print_evaluation_results(metrics)

    elif args.mode == 'extract_features':
        # Extract features mode: extract backbone features (latent)
        if args.image is None:
            print("Error: --image is required for extract_features mode")
            return

        # 이미지 존재 확인
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return

        # 모델 로드
        model = load_model(model_name=args.model, device=args.device)

        # Feature 추출
        print(f"\nExtracting features from: {image_path}")
        features = model.extract_features(image_path)

        print(f"\nExtracted feature vector:")
        print(f"  Shape: {features.shape}")
        print(f"  Mean: {features.mean():.4f}")
        print(f"  Std: {features.std():.4f}")
        print(f"  Min: {features.min():.4f}")
        print(f"  Max: {features.max():.4f}")

        # Feature 저장
        if args.save_features:
            np.save(args.save_features, features)
            print(f"\nFeatures saved to: {args.save_features}")

        # (Optional) 추출된 feature로 예측 검증
        print("\nVerifying prediction from extracted features:")
        predictions = model.predict_from_features(features)
        predictions_dict = {
            label: float(score)
            for label, score in zip(model.labels, predictions)
        }
        predictions_dict = dict(
            sorted(predictions_dict.items(), key=lambda x: x[1], reverse=True)
        )

        if args.top_k:
            predictions_dict = dict(list(predictions_dict.items())[:args.top_k])

        print_predictions(predictions_dict, threshold=args.threshold)


if __name__ == "__main__":
    main()
