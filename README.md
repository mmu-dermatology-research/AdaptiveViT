# AdaptiveViT: Distribution-Aware Hybrid CNN-ViT for Imbalanced Medical Image Classification

<p align="center">
  <img src="assets/architecture.png" alt="AdaptiveViT Architecture" width="800"/>
</p>

> Official PyTorch implementation of **AdaptiveViT**, accepted at *[Conference / Journal Name]*.
>
> **Paper:** [Title of Paper](#) · **Authors:** [Author Names] · **Institution:** [Institution]

---

## Overview

AdaptiveViT is a hybrid CNN-Vision Transformer model designed for medical image classification under severe class imbalance. It introduces three tightly coupled components that together allow the model to *learn how imbalanced its training data is* and continuously adapt its feature extraction and loss computation accordingly — without any per-dataset manual tuning.

**Key contributions:**

- **ADViT-Fuse** — a multi-scale fusion encoder that injects a learned *distribution embedding* (derived from the batch imbalance ratio ρ) into every transformer attention block, conditioning token interactions on dataset statistics.
- **Token Confidence Gate (TCG)** — a per-token sigmoid gate on the MLP branch of each transformer block that suppresses uninformative token updates, improving robustness to noisy minority-class patches.
- **Distribution-Aware Adaptive Loss (DALoss)** — a focal-style loss whose focusing exponent γ varies *continuously* with ρ rather than being set statically, so the model concentrates harder on difficult minority examples exactly when the imbalance is most severe.

---

## Architecture

```
Input Image [B, 3, H, W]
       │
       ▼
EfficientNet-B0 Backbone (frozen / pretrained)
  ├── Stage 2 → Feature Map [B, 40,  H/8,  W/8 ]
  ├── Stage 3 → Feature Map [B, 112, H/16, W/16]
  └── Stage 4 → Feature Map [B, 320, H/32, W/32]
       │
       ▼  (patchify + view → shared token count N₀)
Scale-specific Linear Projections
  patch_to_embedding_112  [B, N₀, 1960] → [N₀, B, D]
  patch_to_embedding_56   [B, N₀, 1372] → [N₀, B, D]
  patch_to_embedding_28   [B, N₀,  980] → [N₀, B, D]
       │
       ▼
ADViT-Fuse Encoder
  ┌─────────────────────────────────────────────────┐
  │  Imbalance Ratio ρ ──► Distribution Embedding   │
  │         [B]                    [B, D]            │
  │                                  │               │
  │  [Fuse Tokens | Scale Tokens] ◄──┤ (modulation)  │
  │        [F+3N₀, B, D]             │               │
  │                                  ▼               │
  │  ┌── ResidualAttentionBlock ─────────────────┐   │
  │  │  + dist_embed modulation (per token)      │   │
  │  │  + Self-Attention (pre-norm)              │   │
  │  │  + Token Confidence Gate × MLP            │   │
  │  └───────────────────────────────────────────┘   │
  │        × depth layers                            │
  │                                                  │
  │  Fuse Token outputs [F, B, D]                    │
  │       → concat → LayerNorm → Linear              │
  └─────────────────────────────────────────────────┘
       │
       ▼
Class Logits [B, num_classes]
```

The distribution embedding is adaptively blended with a learned identity residual via a modulation gate: when the dataset is balanced (ρ ≈ 0), the gate suppresses the distribution signal and the model behaves as a standard ViT; under extreme imbalance, the distribution signal dominates.

---

## Datasets

AdaptiveViT is evaluated across seven publicly available medical imaging benchmarks spanning skin lesion classification and endoscopy:

| Dataset | Task | Classes | Imbalance |
|---|---|---|---|
| **ISIC 2017** | Skin lesion (binary / 3-class) | 2 or 3 | ~4.4:1 |
| **ISIC 2024** | Skin lesion (binary) | 2 | ~9.7:1 |
| **CBD-4905** | Melanoma detection | 2 | ~1:1 (balanced) |
| **IMBD-9810** | Melanoma detection | 2 | ~1.7:1 |
| **IMBD-26k** | Melanoma detection | 2 | ~4.4:1 |
| **IMBD-56k** | Melanoma detection | 2 | ~10.6:1 |
| **Derm7pt** | Dermoscopy / clinical (binary) | 2 | ~3.6:1 |
| **IBD-HKUC** | Ulcerative colitis (Mayo score) | 2 | ~5.4:1 |

---

## Requirements

```bash
python >= 3.10
torch >= 2.0
torchvision
timm
einops
albumentations
warmup_scheduler
scikit-learn
pandas
numpy
tqdm
pyyaml
tensorboard
umap-learn        # for embedding analysis in predict script
scipy
matplotlib
seaborn
h5py              # for ISIC 2024 HDF5 image archive
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```
AdaptiveViT/
├── model.py                    # AdaptiveViT, ADViTFuse, DALoss, TCG
├── imagenet.py                 # Baseline timm wrapper (ViT / EfficientNet)
├── data_module.py              # ρ calculation, gamma derivation, weighted sampler
├── early_stopping.py           # Early stopping with best-weight restoration
│
├── dataset_isic2017.py         # ISIC 2017 dataset loader
├── dataset_isic2018.py         # ISIC 2018 dataset loader
├── dataset_isic2024.py         # ISIC 2024 dataset loader (HDF5 + CSV)
├── dataset_cbd4905.py          # CBD-4905 / IMBD variants loader
├── dataset_derm7pt.py          # Derm7pt dermoscopy + clinical loader
├── dataset_ibd_hkuc.py         # IBD-HKUC endoscopy loader
│
├── train_hipervit_v2.py        # Training script
├── predict_hipervit_v2.py      # Evaluation + embedding analysis script
│
├── configs/
│   └── architecture.yaml       # Model hyperparameter config
├── weights/                    # Saved checkpoints (created at runtime)
├── logs/                       # Training logs (created at runtime)
└── subs/                       # Prediction outputs (created at runtime)
```

---

## Configuration

Model hyperparameters are set in `configs/architecture.yaml`:

```yaml
model:
  image-size: 224
  patch-size: 7
  dim: 1024
  depth: 6
  heads: 8
  mlp-dim: 2048
  emb-dim: 64
  dim-head: 64
  dropout: 0.1
  emb-dropout: 0.1
```

---

## Training

```bash
python train_hipervit_v2.py \
    --save-name adaptivevit_isic2017 \
    --data-dir /path/to/isic2017 \
    --dataset ISIC2017 \
    --image-size 224 \
    --enet-type efficientnet_b0 \
    --out-dim 2 \
    --batch-size 32 \
    --num-workers 8 \
    --init-lr 3e-5 \
    --n-epochs 30 \
    --seed 0 \
    --rho-strategy per_class_avg \
    --config configs/architecture.yaml \
    --model-dir ./weights \
    --log-dir ./logs
```

### Key training arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name | required |
| `--out-dim` | Number of output classes | `2` |
| `--rho-strategy` | Strategy for computing ρ: `per_class_avg`, `per_class`, `minmax`, `tail_head` | `per_class_avg` |
| `--batch-size` | Training batch size | `32` |
| `--init-lr` | Initial learning rate (Adam + cosine warmup) | `3e-5` |
| `--n-epochs` | Maximum training epochs (early stopping applies) | `20` |
| `--seed` | Random seed for reproducibility | `0` |
| `--DEBUG` | Run with 150 samples / 2 epochs for quick sanity check | `False` |

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

The script reports: **Accuracy**, **Precision**, **Recall**, **F1**, **ROC-AUC**, **PR-AUC**, **ECE** (Expected Calibration Error), **Confusion Matrix**, and **Per-class Accuracy**.

### Embedding analysis

The predict script also supports interpretability analysis of the ADViT-Fuse token representations:

```python
# In predict_hipervit_v2.py — call directly after model loading:
fuse_tokens_representation(model, test_loader, imb_ratio)
```

This produces two figures in `--sub-dir`:

- **`seed_N_token_shift_magnitude.png`** — violin plot of per-sample embedding shift magnitude (adaptive vs. uniform modulation) for minority vs. majority class, with a Mann-Whitney U test.
- **`seed_N_umap_fuse_tokens.png`** — two-panel UMAP of fuse-token embedding space under adaptive vs. uniform (ρ=0) modulation.

---

## Pretrained Weights

Pretrained checkpoints for all datasets will be released upon publication. Each checkpoint is named:

```
{save_name}_seed_{seed}_best.pth
```

Loading a checkpoint:

```python
import torch, yaml
from model import AdaptiveViT

with open('configs/architecture.yaml') as f:
    config = yaml.safe_load(f)

model = AdaptiveViT(config=config, out_dim=2)
state_dict = torch.load('weights/adaptivevit_isic2017_seed_0_best.pth', map_location='cpu')

# Handles both single-GPU and DataParallel checkpoints
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=True)
model.eval()
```

---

## Reproducibility

All experiments use:

```bash
--seed 0   # or 1, 2 for multi-seed evaluation
```

The `set_seed()` function fixes Python, NumPy, and PyTorch (CPU + all GPUs) random states and enables `cudnn.deterministic`. Results may vary slightly across GPU architectures due to non-deterministic CUDA kernels in multi-head attention.

---

## Multi-GPU Training

AdaptiveViT supports `nn.DataParallel` out of the box. Set `CUDA_VISIBLE_DEVICES` before launching:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_hipervit_v2.py ...
```

> **Note:** `imbalance_ratio` is passed as a keyword argument and is therefore *not* automatically scattered by DataParallel. The model's `forward()` slices it to the correct sub-batch size on each device.

---

## Citation

If you use AdaptiveViT in your research, please cite:

```bibtex
@inproceedings{adaptivevit2025,
  title     = {[Paper Title]},
  author    = {[Authors]},
  booktitle = {[Conference / Journal]},
  year      = {2025}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

- EfficientNet backbone via [timm](https://github.com/huggingface/pytorch-image-models)
- Patch rearrangement via [einops](https://github.com/arogozhnikov/einops)
- ISIC datasets: [ISIC Archive](https://www.isic-archive.com)
- Derm7pt: [Kawahara et al., 2018](https://github.com/jeremykawahara/derm7pt)
- Gradual warmup scheduler via [ildoonet/pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr)
