import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


class IBDHKUC_Dataset(Dataset):
    """PyTorch Dataset for the IBD-HKUC ulcerative-colitis endoscopy collection.

    Images are loaded from JPEG files under ``ulcerative-colitis-all/``.
    Binary target: Mayo Endoscopic Score (MES) == 3 → class 1, otherwise 0.

    Args:
        csv (DataFrame): Sample metadata; must contain ``'filepath'``,
            ``'Video file'``, and ``'target'`` columns.
        mode (str): ``'train'``, ``'valid'``, or ``'test'``.  In ``'test'``
            mode the ``'Video file'`` identifier is returned alongside data
            and label.
        transform (callable | None): An ``albumentations`` transform applied
            to the raw RGB image array.
    """

    def __init__(self, csv: pd.DataFrame, mode: str, transform=None):
        self.csv       = csv.reset_index(drop=True)
        self.mode      = mode
        self.transform = transform

    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(self, index: int):
        """Load and return one sample.

        Returns:
            train / valid: ``(image_tensor, label)``
            test:          ``(video_file_id, image_tensor, label)``
        """
        row   = self.csv.iloc[index]
        image = cv2.cvtColor(cv2.imread(row.filepath), cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image  = torch.tensor(image.transpose(2, 0, 1)).float()
        target = torch.tensor(row.target).long()

        if self.mode == 'test':
            return row['Video file'], image, target
        return image, target


# ─────────────────────────────────────────────
# DataFrame loaders
# ─────────────────────────────────────────────

def _load_split(data_dir: str, csv_name: str) -> pd.DataFrame:
    """Load a split CSV and build the ``target`` and ``filepath`` columns.

    The ``MES`` column is binarised: MES == 3 → 1 (severe), otherwise 0.

    Args:
        data_dir (str): Dataset root directory.
        csv_name (str): CSV filename (e.g. ``'train_set.csv'``).

    Returns:
        DataFrame with ``'target'`` and ``'filepath'`` columns.
    """
    df = pd.read_csv(os.path.join(data_dir, csv_name))
    df = df.rename(columns={'MES': 'target'})
    df['target']   = (df['target'] == 3).astype(int)
    df['filepath'] = df['Video file'].apply(
        lambda x: os.path.join(data_dir, f'ulcerative-colitis-all/{x}.jpg')
    )
    print(df.groupby('target').size().to_string())
    return df


def get_df(data_dir: str):
    """Load the IBD-HKUC train/validation split.

    Args:
        data_dir (str): Dataset root directory.

    Returns:
        tuple: ``(df_train, df_valid, mel_idx)``
            where ``mel_idx=1`` denotes the positive (severe) class.
    """
    df_train = _load_split(data_dir, 'train_set.csv')
    df_valid = _load_split(data_dir, 'valid_set.csv')
    return df_train, df_valid, 1


def get_test_df(data_dir: str):
    """Load the IBD-HKUC train/test split.

    Args:
        data_dir (str): Dataset root directory.

    Returns:
        tuple: ``(df_train, df_test, mel_idx)``
    """
    df_train = _load_split(data_dir, 'train_set.csv')
    df_test  = _load_split(data_dir, 'test_set.csv')
    return df_train, df_test, 1
