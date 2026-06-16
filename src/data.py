"""
Dataset loaders: PyTorch Dataset classes for CheXpert, PadChest, MIMIC-CXR, VinDR-CXR and NIH.
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



# Canonical labels shared across all datasets
CANONICAL_LABELS = [
    'Atelectasis',
    'Cardiomegaly',
    'Pleural Effusion',
    'Consolidation',
    'Pneumothorax',
    'Edema',
]


class CheXpertDataset(Dataset):
    """CheXpert dataset loader."""

    LABEL_COLUMNS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

    # CheXpert label -> canonical label mapping
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

        if frontal_only:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)

        self._process_image_paths()

    def _process_image_paths(self):
        if self.img_root_dir:
            self.df['image_path'] = self.df['resized_img_path'].apply(
                lambda x: self._convert_path(x)
            )
        else:
            self.df['image_path'] = self.df['resized_img_path']

    def _convert_path(self, original_path):
        """Map a CSV path to an actual path under img_root_dir."""
        parts = Path(original_path).parts
        # Take the path relative to the 'train'/'valid' segment
        try:
            idx = parts.index('train') if 'train' in parts else parts.index('valid')
            relative_path = Path(*parts[idx:])
            return self.img_root_dir / relative_path
        except:
            # Fall back to filename only
            return self.img_root_dir / Path(original_path).name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row['image_path']
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        if self.return_labels:
            if self.use_canonical_labels:
                labels = torch.zeros(len(CANONICAL_LABELS), dtype=torch.float32)
                for orig_label, canon_label in self.LABEL_MAPPING.items():
                    if canon_label not in CANONICAL_LABELS:
                        continue
                    if orig_label in self.LABEL_COLUMNS:
                        val = row[orig_label]
                        if not pd.isna(val) and float(val) > 0:  # treat both -1 and 1 as positive
                            canon_idx = CANONICAL_LABELS.index(canon_label)
                            labels[canon_idx] = max(labels[canon_idx], float(val))
            else:
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
    """PadChest dataset loader."""

    # PadChest label -> canonical label mapping
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

        # Cache key derived from csv_path and frontal_only
        cache_key = f"{csv_path}_{frontal_only}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_dir = Path(csv_path).parent / '.cache'
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"padchest_{cache_hash}.pkl"

        # Use the cache only if it is newer than the CSV
        csv_mtime = Path(csv_path).stat().st_mtime
        if cache_file.exists() and cache_file.stat().st_mtime > csv_mtime:
            print(f"Loading cached PadChest data from {cache_file.name}...")
            with open(cache_file, 'rb') as f:
                self.df = pickle.load(f)
        else:
            print(f"Processing PadChest CSV (this may take a minute on first run)...")
            self.df = pd.read_csv(csv_path, index_col=0)

            # Keep only frontal projections
            if frontal_only:
                frontal_projections = ['PA', 'AP', 'AP_horizontal', 'COSTAL']
                self.df = self.df[self.df['Projection'].isin(frontal_projections)].reset_index(drop=True)

            self.df['Labels_list'] = self.df['Labels'].apply(self._parse_labels)

            # Keep only rows whose image exists on disk
            print(f"Filtering images that exist on disk...")
            original_count = len(self.df)
            self.df['exists'] = self.df['ImageID'].apply(lambda x: (self.img_dir / x).exists())
            self.df = self.df[self.df['exists']].drop(columns=['exists']).reset_index(drop=True)
            filtered_count = original_count - len(self.df)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} images that don't exist on disk ({len(self.df)} remaining)")

            with open(cache_file, 'wb') as f:
                pickle.dump(self.df, f)
            print(f"Cached processed data to {cache_file.name}")

        # Stratified sampling keeps the positive/negative ratio of canonical labels
        if stratified and max_samples is not None:
            self.df = self._stratified_sample(max_samples, skip_samples, random_seed)
        else:
            # Plain slicing: apply skip_samples, then max_samples
            if skip_samples > 0 and len(self.df) > skip_samples:
                self.df = self.df.iloc[skip_samples:].reset_index(drop=True)

            if max_samples is not None and len(self.df) > max_samples:
                self.df = self.df.iloc[:max_samples].reset_index(drop=True)

        # Determine the set of labels to use
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

    def _parse_labels(self, label_str):
        """Parse a label string into a cleaned list of labels."""
        try:
            labels = ast.literal_eval(label_str) if isinstance(label_str, str) else []
            labels = [label.strip() for label in labels if isinstance(label, str)]
            labels = [label for label in labels if label]  # drop empty strings
            return labels
        except:
            return []

    def _has_canonical_positive(self, labels_list):
        """True if any label maps to a canonical label."""
        for label in labels_list:
            if label.lower() in self.LABEL_MAPPING:
                return True
        return False

    def _stratified_sample(self, max_samples, skip_samples, random_seed):
        """Stratified sampling that preserves the canonical positive/negative ratio."""
        np.random.seed(random_seed)

        self.df['_is_positive'] = self.df['Labels_list'].apply(self._has_canonical_positive)

        positive_df = self.df[self.df['_is_positive']].copy()
        negative_df = self.df[~self.df['_is_positive']].copy()

        total_positive = len(positive_df)
        total_negative = len(negative_df)
        total = total_positive + total_negative

        positive_ratio = total_positive / total if total > 0 else 0

        print(f"Original data: {total_positive} positive ({positive_ratio*100:.1f}%), {total_negative} negative")

        # Apply skip_samples per class so the train/test split keeps the ratio
        if skip_samples > 0:
            skip_positive = int(skip_samples * positive_ratio)
            skip_negative = skip_samples - skip_positive

            positive_df = positive_df.iloc[skip_positive:].reset_index(drop=True)
            negative_df = negative_df.iloc[skip_negative:].reset_index(drop=True)

        # Number to sample per class, preserving the ratio
        n_positive = int(max_samples * positive_ratio)
        n_negative = max_samples - n_positive

        # Clamp to what is actually available
        n_positive = min(n_positive, len(positive_df))
        n_negative = min(n_negative, len(negative_df))

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

        # Combine and shuffle
        result_df = pd.concat([positive_sampled, negative_sampled], ignore_index=True)
        result_df = result_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        result_df = result_df.drop(columns=['_is_positive'])

        actual_positive = len(positive_sampled)
        actual_negative = len(negative_sampled)
        print(f"Sampled: {actual_positive} positive ({actual_positive/(actual_positive+actual_negative)*100:.1f}%), {actual_negative} negative")

        return result_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row['ImageID']
        img_path = self.img_dir / img_name

        image = Image.open(img_path)

        # Convert 16-bit grayscale (0-65535) down to 8-bit (0-255)
        if image.mode == 'I;16':
            image = np.array(image, dtype=np.float32)
            image = (image / 256).astype(np.uint8)
            image = Image.fromarray(image, mode='L')

        image = image.convert('L')

        if self.transform:
            image = self.transform(image)

        # Multi-label one-hot encoding
        labels = torch.zeros(len(self.all_labels), dtype=torch.float32)

        if self.use_canonical_labels:
            for label in row['Labels_list']:
                label_lower = label.lower()
                if label_lower in self.LABEL_MAPPING:
                    canon_label = self.LABEL_MAPPING[label_lower]
                    if canon_label not in CANONICAL_LABELS:
                        continue
                    canon_idx = CANONICAL_LABELS.index(canon_label)
                    labels[canon_idx] = 1.0
        else:
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
    """MIMIC-CXR dataset loader. split is one of 'train', 'validate', 'test' (None loads all)."""

    LABEL_COLUMNS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

    # MIMIC-CXR label -> canonical label mapping
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

        # Cache key derived from split and frontal_only
        cache_key = f"{split_csv_path}_{split}_{frontal_only}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_dir = Path(split_csv_path).parent / '.cache'
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"mimic_{cache_hash}.pkl"

        # Use the cache only if it is newer than all source CSVs
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
            self.labels_df = self._read_csv(label_csv_path)
            self.metadata_df = self._read_csv(metadata_csv_path)
            self.split_df = self._read_csv(split_csv_path)

            # split_df already has dicom_id/study_id/subject_id/split; merge ViewPosition from metadata
            self.df = self.split_df.merge(
                self.metadata_df[['dicom_id', 'ViewPosition']],
                on='dicom_id', how='left'
            )

            # Merge labels on subject_id and study_id
            self.df = self.df.merge(
                self.labels_df,
                on=['subject_id', 'study_id'], how='left'
            )

            if split is not None:
                self.df = self.df[self.df['split'] == split].reset_index(drop=True)

            # Keep only frontal views (PA or AP)
            if frontal_only:
                self.df = self.df[self.df['ViewPosition'].isin(['PA', 'AP'])].reset_index(drop=True)

            with open(cache_file, 'wb') as f:
                pickle.dump(self.df, f)
            print(f"Cached processed data to {cache_file.name}")

    def _read_csv(self, path):
        """Read a CSV, transparently handling gzip-compressed files."""
        if str(path).endswith('.gz'):
            with gzip.open(path, 'rt') as f:
                return pd.read_csv(f)
        else:
            return pd.read_csv(path)

    def _get_image_path(self, dicom_id, subject_id, study_id):
        """Build the image path from dicom_id/subject_id/study_id (no existence check)."""
        p_dir = f"p{str(subject_id)[:2]}"
        patient_dir = f"p{subject_id}"
        study_dir = f"s{study_id}"
        img_name = f"{dicom_id}.jpg"
        return self.img_root_dir / p_dir / patient_dir / study_dir / img_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self._get_image_path(row['dicom_id'], row['subject_id'], row['study_id'])

        try:
            image = Image.open(img_path).convert('L')
        except Exception:
            image = Image.new('L', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        if self.use_canonical_labels:
            labels = torch.zeros(len(CANONICAL_LABELS), dtype=torch.float32)
            for orig_label, canon_label in self.LABEL_MAPPING.items():
                if canon_label not in CANONICAL_LABELS:
                    continue
                if orig_label in self.LABEL_COLUMNS:
                    val = row[orig_label]
                    if not pd.isna(val) and float(val) > 0:  # treat both -1 and 1 as positive
                        canon_idx = CANONICAL_LABELS.index(canon_label)
                        labels[canon_idx] = max(labels[canon_idx], float(val))
        else:
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
    """VinDR-CXR dataset loader. aggregate_train_labels merges multiple radiologist annotations by majority vote."""

    # All VinDR-CXR labels
    LABEL_COLUMNS = [
        'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
        'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema',
        'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity',
        'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass',
        'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
        'Pulmonary fibrosis', 'Rib fracture', 'Other lesion',
        'COPD', 'Lung tumor', 'Pneumonia', 'Tuberculosis'
    ]

    # VinDR-CXR label -> canonical label mapping
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

        # Cache key derived from csv_path and aggregate_train_labels
        cache_key = f"{csv_path}_{aggregate_train_labels}_v2"  # v2: filter missing files
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_dir = Path(csv_path).parent / '.cache'
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"vindr_{split}_{cache_hash}.pkl"

        # Use the cache only if it is newer than the CSV
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

            # The train set has a rad_id column; the test set does not
            self.is_train = 'rad_id' in self.df.columns

            if self.is_train and self.aggregate_train_labels:
                self._aggregate_labels()

            # Keep only rows whose image exists on disk
            print(f"Filtering images that exist on disk...")
            original_count = len(self.df)
            self.df['exists'] = self.df['image_id'].apply(
                lambda x: (self.img_dir / f"{x}.jpg").exists()
            )
            self.df = self.df[self.df['exists']].drop(columns=['exists']).reset_index(drop=True)
            filtered_count = original_count - len(self.df)
            if filtered_count > 0:
                print(f"  → Filtered out {filtered_count} missing images ({len(self.df)} remaining)")

            with open(cache_file, 'wb') as f:
                pickle.dump({'df': self.df, 'is_train': self.is_train}, f)
            print(f"Cached processed data to {cache_file.name}")

        if use_canonical_labels:
            self.all_labels = CANONICAL_LABELS
        else:
            self.all_labels = self.LABEL_COLUMNS

    def _aggregate_labels(self):
        """Aggregate multiple radiologist annotations per image by majority vote (mean >= 0.5)."""
        label_cols = [col for col in self.LABEL_COLUMNS if col in self.df.columns]

        # 'Other diseases' (train) vs 'Other disease' (test) naming differs
        if 'Other diseases' in self.df.columns and 'Other disease' not in self.df.columns:
            label_cols.append('Other diseases')
        elif 'Other disease' in self.df.columns and 'Other diseases' not in self.df.columns:
            label_cols.append('Other disease')

        if 'No finding' in self.df.columns:
            label_cols.append('No finding')

        agg_dict = {col: 'mean' for col in label_cols}
        self.df = self.df.groupby('image_id').agg(agg_dict).reset_index()

        # Binarize at a 0.5 threshold
        for col in label_cols:
            self.df[col] = (self.df[col] >= 0.5).astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = row['image_id']
        img_path = self.img_dir / f"{image_id}.jpg"

        try:
            image = Image.open(img_path).convert('L')
        except Exception:
            image = Image.new('L', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        if self.use_canonical_labels:
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
            labels = []
            for col in self.LABEL_COLUMNS:
                # Column names may differ between train and test
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

        # Include rad_id when present (i.e. annotations were not aggregated)
        if 'rad_id' in row:
            result['rad_id'] = row['rad_id']

        return result


class NIHDataset(Dataset):
    """NIH ChestX-ray14 dataset loader."""

    # All NIH ChestX-ray14 labels
    LABEL_COLUMNS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
        'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening',
        'Pneumonia', 'Pneumothorax'
    ]

    # NIH label -> canonical label mapping
    LABEL_MAPPING = {
        'Atelectasis': 'Atelectasis',
        'Cardiomegaly': 'Cardiomegaly',
        'Effusion': 'Pleural Effusion',  # NIH 'Effusion' maps to 'Pleural Effusion'
        'Consolidation': 'Consolidation',
        'Pneumothorax': 'Pneumothorax',
        'Edema': 'Edema',
    }

    def __init__(self, csv_path, img_dir, transform=None, use_canonical_labels=False):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.use_canonical_labels = use_canonical_labels

        self.df = pd.read_csv(csv_path)

        if use_canonical_labels:
            self.all_labels = CANONICAL_LABELS
        else:
            self.all_labels = self.LABEL_COLUMNS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name = row['Image Index']
        img_path = self.img_dir / image_name

        try:
            image = Image.open(img_path).convert('L')
        except Exception:
            image = Image.new('L', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        # Finding Labels is a "|"-separated multi-label field
        finding_labels = row['Finding Labels']
        label_list = []
        if pd.notna(finding_labels) and finding_labels != '':
            label_list = [label.strip() for label in str(finding_labels).split('|')]

        if self.use_canonical_labels:
            labels = torch.zeros(len(CANONICAL_LABELS), dtype=torch.float32)
            for label in label_list:
                if label in self.LABEL_MAPPING:
                    canon_label = self.LABEL_MAPPING[label]
                    if canon_label not in CANONICAL_LABELS:
                        continue
                    canon_idx = CANONICAL_LABELS.index(canon_label)
                    labels[canon_idx] = 1.0
        else:
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

        # Optional metadata
        if 'Patient Age' in row and pd.notna(row['Patient Age']):
            result['age'] = int(row['Patient Age'])
        if 'Patient Gender' in row and pd.notna(row['Patient Gender']):
            result['sex'] = row['Patient Gender']
        if 'View Position' in row and pd.notna(row['View Position']):
            result['view'] = row['View Position']

        return result


def get_dataset(dataset_name, split='test', transform=None, **kwargs):
    """Build a dataset by name.

    Common kwargs:
        - use_canonical_labels: use the shared canonical labels (all datasets)
        - frontal_only: keep frontal views only (CheXpert, PadChest, MIMIC-CXR)
        - aggregate_train_labels: majority-vote radiologist annotations on the VinDR-CXR train set (default: True)
    """

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

        # PadChest has no official train/test split; use stratified sampling to
        # preserve the positive/negative ratio. Test takes the first 5000 samples,
        # train starts after them. Other splits use the full dataset.
        if split == 'test':
            if 'max_samples' not in kwargs:
                kwargs['max_samples'] = 5000
            kwargs['skip_samples'] = 0
            if 'stratified' not in kwargs:
                kwargs['stratified'] = True
        elif split == 'train':
            kwargs['skip_samples'] = 5000
            if 'stratified' not in kwargs:
                kwargs['stratified'] = True

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

        # Pick the CSV and image directory for the split
        if split == 'test':
            csv_path = vindr_root / 'annotations' / 'image_labels_test.csv'
            img_dir = vindr_root / 'test'
        else:
            csv_path = vindr_root / 'annotations' / 'image_labels_train.csv'
            img_dir = vindr_root / 'train'

        # All VinDR-CXR images are frontal, so drop frontal_only
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

        # NIH has no official split, so all data is treated as test
        csv_path = nih_root / 'nih_test_labels.csv'
        img_dir = nih_root / 'images'

        # All NIH images are frontal, so drop frontal_only
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
    from torchvision import transforms

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
