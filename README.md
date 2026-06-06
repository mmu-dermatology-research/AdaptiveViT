# AdaptiveViT
Adaptive Distribution-aware Vision Transformer, **AdaptiveViT** is a hybrid CNN-Vision Transformer model designed for medical image classification under severe class imbalance and low-resolution image data.

## Installation

```bash
# Create and activate environment
conda create -n adaptivevit python=3.11 -y
conda activate adaptivevit

# Install PyTorch with CUDA 12.1 (adjust cu121 → cu118 if on CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all remaining dependencies
pip install -r requirements.txt
```
---

## Repository Structure

```
AdaptiveViT/
├── models/                     # model.py, imagenet.py
│   ├── adaptivevit.py
│   └── imagenet.py
│
├── datasets/                   # all dataset_*.py files + data_module.py
│   ├── data_module.py
│   ├── dataset_isic2017.py
│   ├── dataset_isic2018.py
│   ├── dataset_isic2024.py
│   ├── dataset_cbd4905.py
│   ├── dataset_derm7pt.py
│   ├── dataset_ibd_hkuc.py
│   └── README.md                # dataset modules details
│
├── config.yaml
│
├── train.py
├── predict.py
├── early_stopping.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --save-name adaptivevit_dataset_tag \
    --data-dir /path/to/dataset \
    --dataset dataset_tag \
    --image-size 224 \
    --enet-type efficientnet_b0 \
    --out-dim 2 \
    --batch-size 32 \
    --num-workers 8 \
    --init-lr 3e-5 \
    --n-epochs 30 \
    --seed 0 \
    --rho-strategy per_class_avg \
    --model-dir ./checkpoints/weights \
    --log-dir ./checkpoints/logs
```

> **Note:** `imbalance_ratio` is passed as a keyword argument and is therefore *not* automatically scattered by DataParallel. The model's `forward()` slices it to the correct sub-batch size on each device.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
    --kernel-type adaptivevit_dataset_tag \
    --data-dir /path/to/dataset \
    --dataset dataset_tag \
    --image-size 224 \
    --enet-type efficientnet_b0 \
    --out-dim 2 \
    --n-test 8 \
    --seed 0 \
    --rho-strategy per_class_avg \
    --model-dir ./checkpoints/weights \
    --sub-dir ./checkpoints/subs
```

> The script reports: **Accuracy**, **Precision**, **Recall**, **F1**, **ROC-AUC**, **Confusion Matrix**, and **Per-class Accuracy**.

### Key training arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name: `ISIC2017`, `ISIC2024`, `CBD4905`, `IMBD9810`, `Derm7pt`, `Derm7ptClinic`, `IBDHKUC` | required |
| `--out-dim` | Number of output classes (binary/multi-class) | `2` |
| `--rho-strategy` | Strategy for computing ρ: `per_class_avg`, `per_class`, `minmax`, `tail_head` | `per_class_avg` |
| `--batch-size` | Training batch size | `32` |
| `--init-lr` | Initial learning rate (Adam + cosine warmup) | `3e-5` |
| `--n-epochs` | Maximum training epochs (early stopping applies) | `20` |
| `--seed` | Random seed for reproducibility | `0` |

### Imbalance ratio strategies (ρ)

ρ is computed once from the full training set and used to condition both the model's distribution embedding and the DALoss gamma:

| Strategy | Formula | Use case |
|---|---|---|
| `per_class_avg` | Mean of per-class log(n_c / n_rest) | **Default.** Good balance of precision and simplicity |
| `per_class` | Per-class log(n_c / n_rest) → list[C] | Most precise for multi-class; each class gets its own γ |
| `minmax` | log(n_min / n_max) | Global severity signal |
| `tail_head` | log(n_tail / n_head) | Maximum contrast; most aggressive |


## Datasets

AdaptiveViT is evaluated on publicly available medical imaging benchmarks, primarily for skin lesion classification, with an additional HyperKvasirUC endoscopy dataset used to assess its generalisation capability across imaging domains.

| Dataset| Tag | Task (Binary or Multi-class) | Classes | Imbalance |
|---|---|---|---|---|
| **ISIC 2017** | `ISIC2017` | Skin lesion (binary / 3-class) | 2 or 3 | ~4.4:1 |
| **ISIC 2024** | `ISIC2024` | Skin lesion (binary) | 2 | ~9.7:1 |
| **ISIC Balanced** | `CBD4905` | Melanoma detection | 2 | ~1:1 (balanced) |
| **ISIC Imbalanced** | `IMBD9810` | A subset of ISIC-DICM-17K Melanoma detection | 2 | ~1.7:1 |
| **Derm7pt** | `Derm7pt`, `Derm7ptClinic` | Dermoscopy / clinical (binary / 5-class) | 2 or 5 | ~3.6:1 |
| **HyperKvasir UC** | `IBDHKUC` | Ulcerative colitis (Mayo score) | 2 | ~5.4:1 |

---
> **Note:** `datasets/README.md` for details description about dataset modules and common dataset module interface.


### Common dataset module interface

Every dataset module follows the same contract so that `train.py` and
`predict.py` can work with any of them interchangeably.

**`get_df(data_dir)` / `get_test_df(data_dir)`**

Each module exposes one or both of these functions:

```python
df_train, df_valid, mel_idx = dataset_<name>.get_df(data_dir)
df_train, df_test,  mel_idx = dataset_<name>.get_test_df(data_dir)
```

- `df_train` / `df_valid` / `df_test` — pandas DataFrames with at minimum
  two columns: one for the image file path and one for the integer class label.
- `mel_idx` — integer index of the melanoma (positive) class used for
  AUC computation (typically `1`).

**Dataset class**

Each module also provides a `torch.utils.data.Dataset` subclass:

```python
dataset = <Name>_Dataset(df, split, transform=transforms)
# split: one of 'train', 'valid', 'test'
```

`__getitem__` returns `(image_id, image_tensor, label)` so that prediction
outputs can be traced back to individual samples.

---

### Expected directory structure

Place each dataset under its own root directory and pass that path as
`--data-dir`.  The layout expected by each module is shown below.

```
dataset/
├── images/
│   └── *.jpg
├── train_set.csv
├── valid_set.csv
└── test_set.csv
```

CSV columns required:

| Column | Description |
|---|---|
| `derm` | Dermoscopy image filename (with extension) |
| `diagnosis` | String diagnosis label (mapped to integer internally) |
| `target` | Binary label — `1` = melanoma, `0` = non-melanoma |

The dataset module constructs the full image path as:
`<data_dir>/images/<derm>`

---

### Adapting to a new dataset

To add a new dataset, create `datasets/dataset_<name>.py` following this
template:

```python
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def get_df(data_dir: str):
    """
    Load train/validation splits.

    Returns:
        df_train (DataFrame): Must contain 'filepath' and 'target' columns.
        df_valid (DataFrame): Same schema as df_train.
        mel_idx  (int):       Class index of the positive (melanoma) class.
    """
    df_train = pd.read_csv(os.path.join(data_dir, 'train_set.csv'))
    df_valid = pd.read_csv(os.path.join(data_dir, 'valid_set.csv'))

    # --- Adjust column names to match your CSV ----------------------------
    # 'image_id' → the image filename stem or full filename
    # 'target'   → integer binary label (1 = positive class)
    df_train['filepath'] = df_train['image_id'].apply(
        lambda x: os.path.join(data_dir, 'images', f'{x}.jpg')
    )
    df_valid['filepath'] = df_valid['image_id'].apply(
        lambda x: os.path.join(data_dir, 'images', f'{x}.jpg')
    )

    mel_idx = 1   # index of the melanoma / positive class
    return df_train, df_valid, mel_idx


def get_test_df(data_dir: str):
    """Load the held-out test split.  Same schema as get_df."""
    df_train = pd.read_csv(os.path.join(data_dir, 'train_set.csv'))
    df_test  = pd.read_csv(os.path.join(data_dir, 'test_set.csv'))

    df_test['filepath'] = df_test['image_id'].apply(
        lambda x: os.path.join(data_dir, 'images', f'{x}.jpg')
    )

    mel_idx = 1
    return df_train, df_test, mel_idx


class MyDataset(Dataset):
    """
    Args:
        df:        DataFrame with 'filepath' and 'target' columns.
        split:     One of 'train', 'valid', 'test'.
        transform: Albumentations or torchvision transform pipeline.
    """

    def __init__(self, df: pd.DataFrame, split: str, transform=None):
        self.df        = df.reset_index(drop=True)
        self.split     = split
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row      = self.df.iloc[index]
        image_id = row['image_id']
        label    = int(row['target'])

        image = cv2.imread(row['filepath'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            result = self.transform(image=image)
            image  = result['image']

        return image_id, torch.tensor(image).float(), torch.tensor(label)
```

The only two things that must match your CSV are:

- **Image path column** — the column whose value, after any joining with
  `data_dir`, resolves to a readable image file.
- **Label column** — an integer column named `target` where `1` is the
  minority / positive class and `0` is the majority / negative class.
