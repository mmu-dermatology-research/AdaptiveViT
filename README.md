# AdaptiveViT
---
AdaptiveViT is a hybrid CNN-Vision Transformer model designed for medical image classification under severe class imbalance. It introduces three tightly coupled components that together allow the model to *learn how imbalanced its training data is* and continuously adapt its feature extraction and loss computation accordingly — without any per-dataset manual tuning.

---

## Datasets

AdaptiveViT is evaluated across seven publicly available medical imaging benchmarks spanning skin lesion classification and endoscopy:

| Dataset | Task (Binary or Multi-class) | Classes | Imbalance |
|---|---|---|---|
| **ISIC 2017** | Skin lesion (binary / 3-class) | 2 or 3 | ~4.4:1 |
| **ISIC 2024** | Skin lesion (binary) | 2 | ~9.7:1 |
| **CBD-4905** | Melanoma detection | 2 | ~1:1 (balanced) |
| **IMBD-9810** | Melanoma detection | 2 | ~1.7:1 |
| **Derm7pt** | Dermoscopy / clinical (binary / 5-class) | 2 | ~3.6:1 |
| **IBD-HKUC** | Ulcerative colitis (Mayo score) | 2 | ~5.4:1 |

---

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
│   ├── model.py
│   └── imagenet.py
│
├── datasets/                   # all dataset_*.py files + data_module.py
│   ├── data_module.py
│   ├── dataset_isic2017.py
│   ├── dataset_isic2018.py
│   ├── dataset_isic2024.py
│   ├── dataset_cbd4905.py
│   ├── dataset_derm7pt.py
│   └── dataset_ibd_hkuc.py
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
    --save-name adaptivevit_dataset_name \
    --data-dir /path/to/dataset \
    --dataset dataset_name \
    --image-size 224 \
    --enet-type efficientnet_b0 \
    --out-dim 2 \
    --batch-size 32 \
    --num-workers 8 \
    --init-lr 3e-5 \
    --n-epochs 30 \
    --seed 0 \
    --rho-strategy per_class_avg \
    --model-dir ./weights \
    --log-dir ./logs
```

> **Note:** `imbalance_ratio` is passed as a keyword argument and is therefore *not* automatically scattered by DataParallel. The model's `forward()` slices it to the correct sub-batch size on each device.
> 
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

### DALoss gamma

γ is derived automatically from ρ via `calculate_gamma()` in `data_module.py` — no per-dataset manual tuning is required:

```
γ₋ = γ_min + (γ_max − γ_min) · |tanh(ρ̄)|

γ_min = 0.5   (prevents collapse to standard cross-entropy when balanced)
γ_max = 5.0   (prevents precision collapse under extreme imbalance)
```

---

## Evaluation

```bash
python predict_hipervit_v2.py \
    --kernel-type adaptivevit_isic2017 \
    --data-dir /path/to/isic2017 \
    --dataset ISIC2017 \
    --image-size 224 \
    --enet-type efficientnet_b0 \
    --out-dim 2 \
    --n-test 8 \
    --seed 0 \
    --rho-strategy per_class_avg \
    --config configs/architecture.yaml \
    --model-dir ./weights \
    --sub-dir ./subs
```

The script reports: **Accuracy**, **Precision**, **Recall**, **F1**, **ROC-AUC**, **Confusion Matrix**, and **Per-class Accuracy**.
