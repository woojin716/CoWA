"""
데이터셋 로더 모듈
CheXpert, PadChest, MIMIC-CXR, VinDR-CXR 데이터셋을 위한 PyTorch Dataset 클래스들
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import ast
import gzip
import pickle
import hashlib



# 공통 라벨 정의 (Canonical Labels)
CANONICAL_LABELS = [
    'Atelectasis',
    'Cardiomegaly',
    'Pleural Effusion',
    'Consolidation',
    'Pneumothorax',
    'Edema',
]


class CheXpertDataset(Dataset):
    """
    CheXpert 데이터셋 로더

    Args:
        csv_path: CSV 파일 경로 (chexpert_train_info.csv or chexpert_test_info.csv)
        img_root_dir: 이미지 루트 디렉토리
        transform: 이미지 변환 (torchvision transforms)
        return_labels: 라벨을 반환할지 여부
        use_canonical_labels: 공통 라벨 사용 여부
    """

    LABEL_COLUMNS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

    # CheXpert 라벨 → Canonical 라벨 매핑
    LABEL_MAPPING = {
        'Atelectasis': 'Atelectasis',
        'Cardiomegaly': 'Cardiomegaly',
        'Pleural Effusion': 'Pleural Effusion',
        'Consolidation': 'Consolidation',
        'Pneumothorax': 'Pneumothorax',
        'Edema': 'Edema',
    }

    def __init__(self, csv_path, img_root_dir=None, transform=None, return_labels=True, use_canonical_labels=False, frontal_only=False):
        self.df = pd.read_csv(csv_path)
        self.img_root_dir = Path(img_root_dir) if img_root_dir else None
        self.transform = transform
        self.return_labels = return_labels
        self.use_canonical_labels = use_canonical_labels

        # 정면 샘플만 필터링
        if frontal_only:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)

        # 이미지 경로 처리
        self._process_image_paths()

        # print 제거 - get_dataset에서 통합 출력

    def _process_image_paths(self):
        """이미지 경로를 실제 경로로 변환"""
        if self.img_root_dir:
            # CSV의 경로에서 파일명 추출하여 실제 경로로 변환
            self.df['image_path'] = self.df['resized_img_path'].apply(
                lambda x: self._convert_path(x)
            )
        else:
            self.df['image_path'] = self.df['resized_img_path']

    def _convert_path(self, original_path):
        """CSV의 원본 경로를 실제 경로로 변환"""
        # patient/study/view 구조 추출
        parts = Path(original_path).parts
        # train 또는 valid 이후의 경로 추출
        try:
            idx = parts.index('train') if 'train' in parts else parts.index('valid')
            relative_path = Path(*parts[idx:])
            return self.img_root_dir / relative_path
        except:
            # 실패시 파일명만 사용
            return self.img_root_dir / Path(original_path).name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 이미지 로드
        img_path = row['image_path']
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        if self.return_labels:
            if self.use_canonical_labels:
                # Canonical 라벨 사용
                labels = torch.zeros(len(CANONICAL_LABELS), dtype=torch.float32)
                for orig_label, canon_label in self.LABEL_MAPPING.items():
                    if canon_label not in CANONICAL_LABELS:
                        continue
                    if orig_label in self.LABEL_COLUMNS:
                        val = row[orig_label]
                        if not pd.isna(val) and float(val) > 0:  # -1이나 1을 모두 positive로 처리
                            canon_idx = CANONICAL_LABELS.index(canon_label)
                            labels[canon_idx] = max(labels[canon_idx], float(val))
            else:
                # 원본 라벨 사용
                labels = []
                for col in self.LABEL_COLUMNS:
                    val = row[col]
                    if pd.isna(val):
                        labels.append(0.0)
                    else:
                        labels.append(float(val))
                labels = torch.tensor(labels, dtype=torch.float32)

            return {
                'image': image,
                'labels': labels,
                'path': str(img_path),
                'sex': row['Sex'],
                'age': row['Age'],
                'view': row['Frontal/Lateral'],
                'projection': row['AP/PA']
            }
        else:
            return {
                'image': image,
                'path': str(img_path)
            }


class PadChestDataset(Dataset):
    """
    PadChest 데이터셋 로더

    Args:
        csv_path: CSV 파일 경로
        img_dir: 이미지 디렉토리
        transform: 이미지 변환
        filter_labels: 특정 라벨만 필터링할 리스트 (None이면 모든 라벨 사용)
        use_canonical_labels: 공통 라벨 사용 여부
    """

    # PadChest 라벨 → Canonical 라벨 매핑
    LABEL_MAPPING = {
        'atelectasis': 'Atelectasis',
        'cardiomegaly': 'Cardiomegaly',
        'pleural effusion': 'Pleural Effusion',
        'consolidation': 'Consolidation',
        'pneumothorax': 'Pneumothorax',
        'pulmonary edema': 'Edema',
    }

    def __init__(self, csv_path, img_dir, transform=None, filter_labels=None, use_canonical_labels=False, frontal_only=False, max_samples=None, skip_samples=0, stratified=False, random_seed=42):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.filter_labels = filter_labels
        self.use_canonical_labels = use_canonical_labels

        # 캐시 키 생성 (csv_path, frontal_only를 기반으로)
        cache_key = f"{csv_path}_{frontal_only}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_dir = Path(csv_path).parent / '.cache'
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"padchest_{cache_hash}.pkl"

        # 캐시 파일이 있고 CSV보다 최신이면 로드
        csv_mtime = Path(csv_path).stat().st_mtime
        if cache_file.exists() and cache_file.stat().st_mtime > csv_mtime:
            print(f"Loading cached PadChest data from {cache_file.name}...")
            with open(cache_file, 'rb') as f:
                self.df = pickle.load(f)
        else:
            print(f"Processing PadChest CSV (this may take a minute on first run)...")
            self.df = pd.read_csv(csv_path, index_col=0)

            # 정면 샘플만 필터링 (PA, AP 등)
            if frontal_only:
                # Projection 컬럼에서 정면 뷰만 선택 (PA, AP, AP_horizontal, COSTAL)
                frontal_projections = ['PA', 'AP', 'AP_horizontal', 'COSTAL']
                self.df = self.df[self.df['Projection'].isin(frontal_projections)].reset_index(drop=True)

            # Labels 컬럼을 리스트로 변환
            self.df['Labels_list'] = self.df['Labels'].apply(self._parse_labels)

            # 이미지가 실제로 존재하는 행만 필터링
            print(f"Filtering images that exist on disk...")
            original_count = len(self.df)
            self.df['exists'] = self.df['ImageID'].apply(lambda x: (self.img_dir / x).exists())
            self.df = self.df[self.df['exists']].drop(columns=['exists']).reset_index(drop=True)
            filtered_count = original_count - len(self.df)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} images that don't exist on disk ({len(self.df)} remaining)")

            # 캐시에 저장
            with open(cache_file, 'wb') as f:
                pickle.dump(self.df, f)
            print(f"Cached processed data to {cache_file.name}")

        # Stratified sampling (canonical labels 기준 positive/negative 비율 유지)
        if stratified and max_samples is not None:
            self.df = self._stratified_sample(max_samples, skip_samples, random_seed)
        else:
            # 기존 방식: skip_samples 적용 후 max_samples 적용
            if skip_samples > 0 and len(self.df) > skip_samples:
                self.df = self.df.iloc[skip_samples:].reset_index(drop=True)

            if max_samples is not None and len(self.df) > max_samples:
                self.df = self.df.iloc[:max_samples].reset_index(drop=True)

        # 모든 고유 라벨 추출
        if use_canonical_labels:
            self.all_labels = CANONICAL_LABELS
        elif filter_labels is None:
            all_labels = set()
            for labels in self.df['Labels_list']:
                all_labels.update(labels)
            self.all_labels = sorted(list(all_labels))
        else:
            self.all_labels = filter_labels

        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}

        # print 제거 - get_dataset에서 통합 출력

    def _parse_labels(self, label_str):
        """라벨 문자열을 리스트로 변환하고 정제"""
        try:
            labels = ast.literal_eval(label_str) if isinstance(label_str, str) else []
            # 공백 제거 및 빈 문자열 필터링
            labels = [label.strip() for label in labels if isinstance(label, str)]
            labels = [label for label in labels if label]  # 빈 문자열 제거
            return labels
        except:
            return []

    def _has_canonical_positive(self, labels_list):
        """canonical labels 중 하나라도 있으면 True"""
        for label in labels_list:
            if label.lower() in self.LABEL_MAPPING:
                return True
        return False

    def _stratified_sample(self, max_samples, skip_samples, random_seed):
        """Canonical labels 기준으로 stratified sampling"""
        np.random.seed(random_seed)

        # Positive/Negative 분리
        self.df['_is_positive'] = self.df['Labels_list'].apply(self._has_canonical_positive)

        positive_df = self.df[self.df['_is_positive']].copy()
        negative_df = self.df[~self.df['_is_positive']].copy()

        total_positive = len(positive_df)
        total_negative = len(negative_df)
        total = total_positive + total_negative

        # 원본 비율 계산
        positive_ratio = total_positive / total if total > 0 else 0

        print(f"Original data: {total_positive} positive ({positive_ratio*100:.1f}%), {total_negative} negative")

        # skip_samples가 있으면 train/test 분리를 위해 적용
        # stratified 모드에서는 skip_samples를 비율 기반으로 적용
        if skip_samples > 0:
            skip_positive = int(skip_samples * positive_ratio)
            skip_negative = skip_samples - skip_positive

            positive_df = positive_df.iloc[skip_positive:].reset_index(drop=True)
            negative_df = negative_df.iloc[skip_negative:].reset_index(drop=True)

        # 샘플링할 개수 계산 (비율 유지)
        n_positive = int(max_samples * positive_ratio)
        n_negative = max_samples - n_positive

        # 실제 가능한 수로 조정
        n_positive = min(n_positive, len(positive_df))
        n_negative = min(n_negative, len(negative_df))

        # 랜덤 샘플링
        if len(positive_df) > n_positive:
            positive_indices = np.random.choice(len(positive_df), n_positive, replace=False)
            positive_sampled = positive_df.iloc[positive_indices]
        else:
            positive_sampled = positive_df

        if len(negative_df) > n_negative:
            negative_indices = np.random.choice(len(negative_df), n_negative, replace=False)
            negative_sampled = negative_df.iloc[negative_indices]
        else:
            negative_sampled = negative_df

        # 합치고 셔플
        result_df = pd.concat([positive_sampled, negative_sampled], ignore_index=True)
        result_df = result_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # 임시 컬럼 제거
        result_df = result_df.drop(columns=['_is_positive'])

        actual_positive = len(positive_sampled)
        actual_negative = len(negative_sampled)
        print(f"Sampled: {actual_positive} positive ({actual_positive/(actual_positive+actual_negative)*100:.1f}%), {actual_negative} negative")

        return result_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 이미지 로드
        img_name = row['ImageID']
        img_path = self.img_dir / img_name

        image = Image.open(img_path)

        # 16-bit grayscale 이미지 처리
        if image.mode == 'I;16':
            # 16-bit를 8-bit로 변환
            image = np.array(image, dtype=np.float32)
            # 정규화 (0-65535 -> 0-255)
            image = (image / 256).astype(np.uint8)
            image = Image.fromarray(image, mode='L')

        # RGB로 변환
        image = image.convert('L')

        if self.transform:
            image = self.transform(image)

        # 멀티-라벨 원-핫 인코딩
        labels = torch.zeros(len(self.all_labels), dtype=torch.float32)

        if self.use_canonical_labels:
            # Canonical 라벨로 변환
            for label in row['Labels_list']:
                label_lower = label.lower()
                if label_lower in self.LABEL_MAPPING:
                    canon_label = self.LABEL_MAPPING[label_lower]
                    if canon_label not in CANONICAL_LABELS:
                        continue
                    canon_idx = CANONICAL_LABELS.index(canon_label)
                    labels[canon_idx] = 1.0
        else:
            # 원본 라벨 사용
            for label in row['Labels_list']:
                if label in self.label_to_idx:
                    labels[self.label_to_idx[label]] = 1.0

        return {
            'image': image,
            'labels': labels,
            'labels_text': row['Labels_list'],
            'path': str(img_path),
            'patient_id': row['PatientID'],
            'sex': row['PatientSex_DICOM'],
            'age': row['PatientBirth'],
            'view': row['Projection']
        }


class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR 데이터셋 로더

    Args:
        label_csv_path: 라벨 CSV 파일 경로 (mimic-cxr-2.0.0-chexpert.csv.gz)
        metadata_csv_path: 메타데이터 CSV 파일 경로 (mimic-cxr-2.0.0-metadata.csv.gz)
        split_csv_path: train/val/test split CSV 파일 경로 (mimic-cxr-2.0.0-split.csv.gz)
        img_root_dir: 이미지 루트 디렉토리 (files/)
        transform: 이미지 변환
        split: 'train', 'validate', 'test' 중 하나 (None이면 전체)
        use_canonical_labels: 공통 라벨 사용 여부
    """

    LABEL_COLUMNS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

    # MIMIC-CXR 라벨 → Canonical 라벨 매핑
    LABEL_MAPPING = {
        'Atelectasis': 'Atelectasis',
        'Cardiomegaly': 'Cardiomegaly',
        'Pleural Effusion': 'Pleural Effusion',
        'Consolidation': 'Consolidation',
        'Pneumothorax': 'Pneumothorax',
        'Edema': 'Edema',
    }

    def __init__(self, label_csv_path, metadata_csv_path, split_csv_path,
                 img_root_dir, transform=None, split=None, use_canonical_labels=False, frontal_only=False):
        self.img_root_dir = Path(img_root_dir)
        self.transform = transform
        self.use_canonical_labels = use_canonical_labels

        # 캐시 키 생성 (split, frontal_only를 기반으로)
        cache_key = f"{split_csv_path}_{split}_{frontal_only}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_dir = Path(split_csv_path).parent / '.cache'
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"mimic_{cache_hash}.pkl"

        # 캐시 파일이 있고 CSV들보다 최신이면 로드
        csv_mtimes = [
            Path(label_csv_path).stat().st_mtime,
            Path(metadata_csv_path).stat().st_mtime,
            Path(split_csv_path).stat().st_mtime
        ]
        latest_csv_mtime = max(csv_mtimes)

        if cache_file.exists() and cache_file.stat().st_mtime > latest_csv_mtime:
            print(f"Loading cached MIMIC-CXR data from {cache_file.name}...")
            with open(cache_file, 'rb') as f:
                self.df = pickle.load(f)
        else:
            print(f"Processing MIMIC-CXR CSV files (this may take a minute on first run)...")
            # CSV 파일 로드 (gzip 압축 파일 지원)
            self.labels_df = self._read_csv(label_csv_path)
            self.metadata_df = self._read_csv(metadata_csv_path)
            self.split_df = self._read_csv(split_csv_path)

            # split_df에 이미 dicom_id, study_id, subject_id, split이 있음
            # metadata에서 ViewPosition만 추가로 병합
            self.df = self.split_df.merge(
                self.metadata_df[['dicom_id', 'ViewPosition']],
                on='dicom_id', how='left'
            )

            # subject_id와 study_id 기준으로 라벨 병합
            self.df = self.df.merge(
                self.labels_df,
                on=['subject_id', 'study_id'], how='left'
            )

            # split 필터링
            if split is not None:
                self.df = self.df[self.df['split'] == split].reset_index(drop=True)

            # 정면 샘플만 필터링 (PA 또는 AP)
            if frontal_only:
                self.df = self.df[self.df['ViewPosition'].isin(['PA', 'AP'])].reset_index(drop=True)

            # 캐시에 저장
            with open(cache_file, 'wb') as f:
                pickle.dump(self.df, f)
            print(f"Cached processed data to {cache_file.name}")

        # print 제거 - get_dataset에서 통합 출력

    def _read_csv(self, path):
        """CSV 파일 읽기 (gzip 지원)"""
        if str(path).endswith('.gz'):
            with gzip.open(path, 'rt') as f:
                return pd.read_csv(f)
        else:
            return pd.read_csv(path)

    def _get_image_path(self, dicom_id, subject_id, study_id):
        """dicom_id, subject_id, study_id로 이미지 경로 생성 (exists 체크 없음)"""
        p_dir = f"p{str(subject_id)[:2]}"
        patient_dir = f"p{subject_id}"
        study_dir = f"s{study_id}"
        img_name = f"{dicom_id}.jpg"
        return self.img_root_dir / p_dir / patient_dir / study_dir / img_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 이미지 경로 생성 (exists 체크 없이 직접 로드)
        img_path = self._get_image_path(row['dicom_id'], row['subject_id'], row['study_id'])

        try:
            image = Image.open(img_path).convert('L')
        except Exception:
            image = Image.new('L', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        # 라벨 추출
        if self.use_canonical_labels:
            # Canonical 라벨 사용
            labels = torch.zeros(len(CANONICAL_LABELS), dtype=torch.float32)
            for orig_label, canon_label in self.LABEL_MAPPING.items():
                if canon_label not in CANONICAL_LABELS:
                    continue
                if orig_label in self.LABEL_COLUMNS:
                    val = row[orig_label]
                    if not pd.isna(val) and float(val) > 0:  # -1이나 1을 모두 positive로 처리
                        canon_idx = CANONICAL_LABELS.index(canon_label)
                        labels[canon_idx] = max(labels[canon_idx], float(val))
        else:
            # 원본 라벨 사용
            labels = []
            for col in self.LABEL_COLUMNS:
                val = row[col]
                if pd.isna(val):
                    labels.append(0.0)
                else:
                    labels.append(float(val))
            labels = torch.tensor(labels, dtype=torch.float32)

        return {
            'image': image,
            'labels': labels,
            'path': str(img_path) if img_path else 'N/A',
            'dicom_id': row['dicom_id'],
            'subject_id': row['subject_id'],
            'study_id': row['study_id'],
            'view': row['ViewPosition'],
            'split': row['split']
        }


class VinDRCXRDataset(Dataset):
    """
    VinDR-CXR 데이터셋 로더

    Args:
        csv_path: 라벨 CSV 파일 경로 (image_labels_test.csv or image_labels_train.csv)
        img_dir: DICOM 이미지 디렉토리
        transform: 이미지 변환
        use_canonical_labels: 공통 라벨 사용 여부
        aggregate_train_labels: train set에서 여러 radiologist 평가를 집계할지 여부 (majority voting)
    """

    # VinDR-CXR의 모든 라벨
    LABEL_COLUMNS = [
        'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
        'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema',
        'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity',
        'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass',
        'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
        'Pulmonary fibrosis', 'Rib fracture', 'Other lesion',
        'COPD', 'Lung tumor', 'Pneumonia', 'Tuberculosis'
    ]

    # VinDR-CXR 라벨 → Canonical 라벨 매핑
    LABEL_MAPPING = {
        'Atelectasis': 'Atelectasis',
        'Cardiomegaly': 'Cardiomegaly',
        'Pleural effusion': 'Pleural Effusion',
        'Consolidation': 'Consolidation',
        'Pneumothorax': 'Pneumothorax',
        'Edema': 'Edema',
    }

    def __init__(self, csv_path, img_dir, transform=None, use_canonical_labels=False, aggregate_train_labels=True, split='test'):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.use_canonical_labels = use_canonical_labels
        self.aggregate_train_labels = aggregate_train_labels

        # 캐시 키 생성 (csv_path, aggregate_train_labels, filter_missing를 기반으로)
        cache_key = f"{csv_path}_{aggregate_train_labels}_v2"  # v2: filter missing files
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_dir = Path(csv_path).parent / '.cache'
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"vindr_{split}_{cache_hash}.pkl"

        # 캐시 파일이 있고 CSV보다 최신이면 로드
        csv_mtime = Path(csv_path).stat().st_mtime
        if cache_file.exists() and cache_file.stat().st_mtime > csv_mtime:
            print(f"Loading cached VinDR-CXR data from {cache_file.name}...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.df = cached_data['df']
                self.is_train = cached_data['is_train']
        else:
            print(f"Processing VinDR-CXR CSV (this may take a minute on first run)...")
            self.df = pd.read_csv(csv_path)

            # train set인지 확인 (rad_id 컬럼이 있으면 train)
            self.is_train = 'rad_id' in self.df.columns

            # train set이고 aggregate가 True면 majority voting으로 집계
            if self.is_train and self.aggregate_train_labels:
                self._aggregate_labels()

            # 이미지가 실제로 존재하는 행만 필터링
            print(f"Filtering images that exist on disk...")
            original_count = len(self.df)
            self.df['exists'] = self.df['image_id'].apply(
                lambda x: (self.img_dir / f"{x}.jpg").exists()
            )
            self.df = self.df[self.df['exists']].drop(columns=['exists']).reset_index(drop=True)
            filtered_count = original_count - len(self.df)
            if filtered_count > 0:
                print(f"  → Filtered out {filtered_count} missing images ({len(self.df)} remaining)")

            # 캐시에 저장
            with open(cache_file, 'wb') as f:
                pickle.dump({'df': self.df, 'is_train': self.is_train}, f)
            print(f"Cached processed data to {cache_file.name}")

        # 사용할 라벨 결정
        if use_canonical_labels:
            self.all_labels = CANONICAL_LABELS
        else:
            self.all_labels = self.LABEL_COLUMNS

    def _aggregate_labels(self):
        """train set의 여러 radiologist 평가를 majority voting으로 집계"""
        # 각 image_id에 대해 라벨별로 평균 계산 (0.5 이상이면 1, 아니면 0)
        label_cols = [col for col in self.LABEL_COLUMNS if col in self.df.columns]

        # 'Other diseases'가 train에는 있고 test에는 'Other disease'로 되어있음
        if 'Other diseases' in self.df.columns and 'Other disease' not in self.df.columns:
            label_cols.append('Other diseases')
        elif 'Other disease' in self.df.columns and 'Other diseases' not in self.df.columns:
            label_cols.append('Other disease')

        # No finding 추가
        if 'No finding' in self.df.columns:
            label_cols.append('No finding')

        agg_dict = {col: 'mean' for col in label_cols}
        self.df = self.df.groupby('image_id').agg(agg_dict).reset_index()

        # 0.5 threshold로 이진화
        for col in label_cols:
            self.df[col] = (self.df[col] >= 0.5).astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 이미지 로드 (JPG만 사용, exists 체크 없음)
        image_id = row['image_id']
        img_path = self.img_dir / f"{image_id}.jpg"

        try:
            image = Image.open(img_path).convert('L')
        except Exception:
            image = Image.new('L', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        # 라벨 추출
        if self.use_canonical_labels:
            # Canonical 라벨 사용
            labels = torch.zeros(len(CANONICAL_LABELS), dtype=torch.float32)
            for orig_label, canon_label in self.LABEL_MAPPING.items():
                if canon_label not in CANONICAL_LABELS:
                    continue
                if orig_label in self.df.columns:
                    val = row[orig_label]
                    if not pd.isna(val) and float(val) > 0:
                        canon_idx = CANONICAL_LABELS.index(canon_label)
                        labels[canon_idx] = 1.0
        else:
            # 원본 라벨 사용
            labels = []
            for col in self.LABEL_COLUMNS:
                # train과 test에서 컬럼명이 다를 수 있음
                if col in self.df.columns:
                    val = row[col]
                elif col == 'Other disease' and 'Other diseases' in self.df.columns:
                    val = row['Other diseases']
                else:
                    val = 0

                if pd.isna(val):
                    labels.append(0.0)
                else:
                    labels.append(float(val))
            labels = torch.tensor(labels, dtype=torch.float32)

        result = {
            'image': image,
            'labels': labels,
            'path': str(img_path),
            'image_id': row['image_id']
        }

        # rad_id가 있으면 추가 (aggregation 안한 경우)
        if 'rad_id' in row:
            result['rad_id'] = row['rad_id']

        return result


class NIHDataset(Dataset):
    """
    NIH ChestX-ray14 데이터셋 로더

    Args:
        csv_path: 라벨 CSV 파일 경로 (nih_test_labels.csv)
        img_dir: 이미지 디렉토리
        transform: 이미지 변환
        use_canonical_labels: 공통 라벨 사용 여부
    """

    # NIH ChestX-ray14의 모든 라벨
    LABEL_COLUMNS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
        'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening',
        'Pneumonia', 'Pneumothorax'
    ]

    # NIH 라벨 → Canonical 라벨 매핑
    LABEL_MAPPING = {
        'Atelectasis': 'Atelectasis',
        'Cardiomegaly': 'Cardiomegaly',
        'Effusion': 'Pleural Effusion',  # NIH의 'Effusion'을 'Pleural Effusion'으로 매핑
        'Consolidation': 'Consolidation',
        'Pneumothorax': 'Pneumothorax',
        'Edema': 'Edema',
    }

    def __init__(self, csv_path, img_dir, transform=None, use_canonical_labels=False):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.use_canonical_labels = use_canonical_labels

        # CSV 파일 로드
        self.df = pd.read_csv(csv_path)

        # 사용할 라벨 결정
        if use_canonical_labels:
            self.all_labels = CANONICAL_LABELS
        else:
            self.all_labels = self.LABEL_COLUMNS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 이미지 로드
        image_name = row['Image Index']
        img_path = self.img_dir / image_name

        try:
            image = Image.open(img_path).convert('L')
        except Exception:
            image = Image.new('L', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        # 라벨 파싱 ("|"로 구분된 다중 라벨)
        finding_labels = row['Finding Labels']
        label_list = []
        if pd.notna(finding_labels) and finding_labels != '':
            label_list = [label.strip() for label in str(finding_labels).split('|')]

        # 라벨 인코딩
        if self.use_canonical_labels:
            # Canonical 라벨 사용
            labels = torch.zeros(len(CANONICAL_LABELS), dtype=torch.float32)
            for label in label_list:
                if label in self.LABEL_MAPPING:
                    canon_label = self.LABEL_MAPPING[label]
                    if canon_label not in CANONICAL_LABELS:
                        continue
                    canon_idx = CANONICAL_LABELS.index(canon_label)
                    labels[canon_idx] = 1.0
        else:
            # 원본 라벨 사용 (one-hot 인코딩)
            labels = torch.zeros(len(self.LABEL_COLUMNS), dtype=torch.float32)
            for label in label_list:
                if label in self.LABEL_COLUMNS:
                    label_idx = self.LABEL_COLUMNS.index(label)
                    labels[label_idx] = 1.0

        result = {
            'image': image,
            'labels': labels,
            'labels_text': label_list,
            'path': str(img_path),
            'image_id': row['Image Index'],
            'patient_id': row['Patient ID'],
        }

        # 추가 메타데이터
        if 'Patient Age' in row and pd.notna(row['Patient Age']):
            result['age'] = int(row['Patient Age'])
        if 'Patient Gender' in row and pd.notna(row['Patient Gender']):
            result['sex'] = row['Patient Gender']
        if 'View Position' in row and pd.notna(row['View Position']):
            result['view'] = row['View Position']

        return result


def get_dataset(dataset_name, split='test', transform=None, **kwargs):
    """
        **kwargs: 데이터셋별 추가 인자
            - use_canonical_labels: 공통 라벨 사용 여부 (모든 데이터셋)
            - frontal_only: 정면 샘플만 사용 (CheXpert, PadChest, MIMIC-CXR)
            - aggregate_train_labels: VinDR-CXR train set에서 여러 radiologist 평가 집계 여부 (기본값: True)

    Returns:
        Dataset 객체
    """

    # 기본 데이터 경로
    DATA_ROOT = Path('data')

    if dataset_name.lower() == 'chexpert':
        csv_name = f'chexpert_{split}_info.csv' if split in ['train', 'test'] else 'chexpert_train_info.csv'
        csv_path = DATA_ROOT / 'chexpert' / 'images' / csv_name
        img_root_dir = DATA_ROOT / 'chexpert' / 'images'

        dataset = CheXpertDataset(
            csv_path=csv_path,
            img_root_dir=img_root_dir,
            transform=transform,
            **kwargs
        )
        frontal_msg = " (frontal only)" if kwargs.get('frontal_only', False) else ""
        print(f"CheXpert Dataset loaded: {len(dataset)} samples (split: {split}){frontal_msg}")
        return dataset

    elif dataset_name.lower() == 'padchest':
        csv_path = DATA_ROOT / 'padchest' / 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv'
        img_dir = DATA_ROOT / 'padchest' / 'images-224' / 'images-224'

        # PadChest는 공식 train/test split이 없음
        # Stratified sampling으로 positive/negative 비율 유지
        if split == 'test':
            if 'max_samples' not in kwargs:
                kwargs['max_samples'] = 5000
            kwargs['skip_samples'] = 0  # 처음부터 시작
            if 'stratified' not in kwargs:
                kwargs['stratified'] = True  # 기본적으로 stratified sampling 사용
        elif split == 'train':
            kwargs['skip_samples'] = 5000  # test 다음부터 시작
            if 'stratified' not in kwargs:
                kwargs['stratified'] = True  # train도 stratified sampling 사용
            # train은 max_samples 제한 없음 (사용자가 명시하지 않는 한)
        # 'valid'나 다른 split은 전체 데이터 사용

        dataset = PadChestDataset(
            csv_path=csv_path,
            img_dir=img_dir,
            transform=transform,
            **kwargs
        )
        frontal_msg = " (frontal only)" if kwargs.get('frontal_only', False) else ""
        skip_samples = kwargs.get('skip_samples', 0)
        if split == 'test':
            split_msg = f" (split: {split}, samples 0-{len(dataset)})"
        elif split == 'train':
            split_msg = f" (split: {split}, samples {skip_samples}+)"
        else:
            split_msg = f" (split: {split})"
        print(f"PadChest Dataset loaded: {len(dataset)} samples{frontal_msg}{split_msg}")
        print(f"Number of unique labels: {len(dataset.all_labels)}")
        return dataset

    elif dataset_name.lower() == 'mimic':
        mimic_root = DATA_ROOT / 'physionet.org' / 'files' / 'mimic-cxr-jpg' / '2.1.0'

        dataset = MIMICCXRDataset(
            label_csv_path=mimic_root / 'mimic-cxr-2.0.0-chexpert.csv.gz',
            metadata_csv_path=mimic_root / 'mimic-cxr-2.0.0-metadata.csv.gz',
            split_csv_path=mimic_root / 'mimic-cxr-2.0.0-split.csv.gz',
            img_root_dir=mimic_root / 'files',
            transform=transform,
            split=split,
            **kwargs
        )
        frontal_msg = " (frontal only)" if kwargs.get('frontal_only', False) else ""
        print(f"MIMIC-CXR Dataset loaded: {len(dataset)} samples (split: {split}){frontal_msg}")
        return dataset

    elif dataset_name.lower() == 'vindr' or dataset_name.lower() == 'vindr-cxr':
        vindr_root = DATA_ROOT / 'physionet.org' / 'files' / 'vindr-cxr' / '1.0.0'

        # split에 따라 적절한 CSV와 이미지 디렉토리 선택
        if split == 'test':
            csv_path = vindr_root / 'annotations' / 'image_labels_test.csv'
            img_dir = vindr_root / 'test'
        else:  # train 또는 기타
            csv_path = vindr_root / 'annotations' / 'image_labels_train.csv'
            img_dir = vindr_root / 'train'

        # VinDR-CXR은 모든 이미지가 frontal이므로 frontal_only 제거
        vindr_kwargs = {k: v for k, v in kwargs.items() if k != 'frontal_only'}

        dataset = VinDRCXRDataset(
            csv_path=csv_path,
            img_dir=img_dir,
            transform=transform,
            split=split,
            **vindr_kwargs
        )
        print(f"VinDR-CXR Dataset loaded: {len(dataset)} samples (split: {split}) - All frontal views")
        if hasattr(dataset, 'is_train') and dataset.is_train:
            if kwargs.get('aggregate_train_labels', True):
                print(f"Train labels aggregated using majority voting")
            else:
                print(f"Train labels NOT aggregated (multiple radiologist annotations per image)")
        return dataset

    elif dataset_name.lower() == 'nih' or dataset_name.lower() == 'nih-cxr14':
        nih_root = DATA_ROOT / 'nih'

        # NIH는 공식 split이 없으므로 모든 데이터를 test로 간주
        csv_path = nih_root / 'nih_test_labels.csv'
        img_dir = nih_root / 'images'

        # NIH는 모든 이미지가 frontal이므로 frontal_only 제거
        nih_kwargs = {k: v for k, v in kwargs.items() if k != 'frontal_only'}

        dataset = NIHDataset(
            csv_path=csv_path,
            img_dir=img_dir,
            transform=transform,
            **nih_kwargs
        )
        print(f"NIH ChestX-ray14 Dataset loaded: {len(dataset)} samples")
        print(f"All images are frontal view (PA/AP)")
        return dataset

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # 테스트 코드
    from torchvision import transforms

    # 기본 transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    print("=" * 80)
    print("Testing CheXpert Dataset")
    print("=" * 80)
    chexpert_test = get_dataset('chexpert', split='test', transform=transform)
    print(f"Dataset size: {len(chexpert_test)}")
    print(f"Label columns: {CheXpertDataset.LABEL_COLUMNS}")
    sample = chexpert_test[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print()

    print("Testing CheXpert with Canonical Labels")
    chexpert_canon = get_dataset('chexpert', split='test', transform=transform, use_canonical_labels=True)
    sample = chexpert_canon[0]
    print(f"Labels shape: {sample['labels'].shape} (expected: {len(CANONICAL_LABELS)})")
    print()

    print("=" * 80)
    print("Testing PadChest Dataset")
    print("=" * 80)
    padchest = get_dataset('padchest', split='test', transform=transform)
    print(f"Dataset size: {len(padchest)}")
    print(f"Number of labels: {len(padchest.all_labels)}")
    print(f"Sample labels: {padchest.all_labels[:10]}")
    sample = padchest[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print()

    print("Testing PadChest with Canonical Labels")
    padchest_canon = get_dataset('padchest', split='test', transform=transform, use_canonical_labels=True)
    sample = padchest_canon[0]
    print(f"Labels shape: {sample['labels'].shape} (expected: {len(CANONICAL_LABELS)})")
    print()

    print("=" * 80)
    print("Testing MIMIC-CXR Dataset")
    print("=" * 80)
    mimic_test = get_dataset('mimic', split='test', transform=transform)
    print(f"Dataset size: {len(mimic_test)}")
    print(f"Label columns: {MIMICCXRDataset.LABEL_COLUMNS}")
    sample = mimic_test[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print()

    print("Testing MIMIC-CXR with Canonical Labels")
    mimic_canon = get_dataset('mimic', split='test', transform=transform, use_canonical_labels=True)
    sample = mimic_canon[0]
    print(f"Labels shape: {sample['labels'].shape} (expected: {len(CANONICAL_LABELS)})")
    print()

    print("=" * 80)
    print("Testing VinDR-CXR Dataset")
    print("=" * 80)
    try:
        vindr_test = get_dataset('vindr-cxr', split='test', transform=transform)
        print(f"Dataset size: {len(vindr_test)}")
        print(f"Label columns: {VinDRCXRDataset.LABEL_COLUMNS}")
        sample = vindr_test[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        print()

        print("Testing VinDR-CXR with Canonical Labels")
        vindr_canon = get_dataset('vindr-cxr', split='test', transform=transform, use_canonical_labels=True)
        sample = vindr_canon[0]
        print(f"Labels shape: {sample['labels'].shape} (expected: {len(CANONICAL_LABELS)})")
    except ImportError as e:
        print(f"Skipping VinDR-CXR test: {e}")
    except Exception as e:
        print(f"Error testing VinDR-CXR: {e}")
