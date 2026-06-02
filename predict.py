"""Prediction / evaluation script for AdaptiveViT.

Loads a trained checkpoint, runs inference with optional test-time augmentation
(TTA), and reports accuracy, AUC, F1, precision, recall, PR-AUC, ECE, and
confusion matrix.  Also supports UMAP and token-shift analysis for
interpretability.
"""

import os
import random
import time
import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
from umap import UMAP
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

from utils import get_trans, get_transforms

import datasets.dataset_isic2024  as DatasetISIC2024
import datasets.dataset_isic2017  as DatasetISIC2017
import datasets.dataset_isic2018  as DatasetISIC2018
import datasets.dataset_derm7pt   as DatasetDerm7pt
import datasets.dataset_ibd_hkuc  as DatasetIBDHKUC

from datasets.dataset_isic2024 import ISIC2024_Dataset
from datasets.dataset_isic2017 import ISIC2017_Dataset
from datasets.dataset_isic2018 import ISIC2018_Dataset
from datasets.dataset_derm7pt  import Derm7pt_Dataset
from datasets.dataset_ibd_hkuc import IBDHKUC_Dataset

from models.adaptivevit import AdaptiveViT
from datasets.data_module import calculate_imbalance_ratio_rho

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate AdaptiveViT checkpoint")

    parser.add_argument('--kernel-type',  type=str, required=True,
                        help="Checkpoint base name (without seed/ext suffix).")
    parser.add_argument('--data-dir',     type=str, default='/dataset/')
    parser.add_argument('--train-data-dir', type=str, default='./')
    parser.add_argument('--image-size',   type=int, required=True)
    parser.add_argument('--enet-type',    type=str, required=True)
    parser.add_argument('--pretrained',   action='store_true')
    parser.add_argument('--batch-size',   type=int, default=64)
    parser.add_argument('--num-workers',  type=int, default=12)
    parser.add_argument('--out-dim',      type=int, default=2)
    parser.add_argument('--DEBUG',        action='store_true')
    parser.add_argument('--model-dir',    type=str, default='./weights')
    parser.add_argument('--log-dir',      type=str, default='./logs')
    parser.add_argument('--sub-dir',      type=str, default='./subs')
    parser.add_argument('--eval',         type=str, default='best',
                        choices=['best', 'final'])
    parser.add_argument('--n-test',       type=int, default=8,
                        help="Number of TTA passes (1 = no augmentation).")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--dataset',      type=str, required=True,
                        choices=[
                            'ISIC2017', 'ISIC2018', 'ISIC2024',
                            'Derm7pt', 'Derm7ptClinic', 'IBDHKUC',
                        ])
    parser.add_argument('--config',       type=str,
                        default='./config.yaml')
    parser.add_argument('--seed',         type=int, default=0)
    parser.add_argument('--rho-strategy', type=str, default='per_class_avg',
                        choices=['per_class_avg', 'per_class', 'minmax', 'tail_head'],
                        help="Strategy for computing the imbalance ratio ρ.")

    args, _ = parser.parse_known_args()
    return args


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 0):
    """Fix all relevant random seeds for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────
# Calibration metric
# ─────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    mel_idx: int = 1,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    For binary classification the minority-class probability is used as
    confidence.  For multi-class, the maximum class probability is used.

    ``n_bins=15`` is recommended for severely imbalanced datasets (e.g. ISIC
    2024 at 1:9.74): fine enough to detect miscalibration near the decision
    boundary, but not so fine that minority-class bins become empty.

    Args:
        y_true (ndarray): Ground-truth labels, shape ``[N]``.
        probs (ndarray): Softmax probabilities, shape ``[N, C]``.
        mel_idx (int): Minority class column index (binary only).
        n_bins (int): Number of confidence bins.

    Returns:
        float: ECE ∈ [0, 1].
    """
    y_true = np.asarray(y_true)
    probs  = np.asarray(probs)
    assert probs.ndim == 2, "probs must be 2-D of shape (N, C)"

    preds = probs.argmax(axis=1)
    conf  = probs[:, mel_idx] if probs.shape[1] == 2 else probs.max(axis=1)

    ece        = 0.0
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_bounds[i], bin_bounds[i + 1]
        mask   = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if mask.any():
            acc = (preds[mask] == y_true[mask]).mean()
            ece += mask.mean() * abs(acc - conf[mask].mean())

    return float(ece)


# ─────────────────────────────────────────────
# Embedding analysis utilities
# ─────────────────────────────────────────────

@torch.no_grad()
def collect_embeddings(model, loader, imb_ratio_value: float):
    """Collect fuse-token embeddings from the model under a fixed ρ.

    Args:
        model (nn.Module): Loaded AdaptiveViT in eval mode.
        loader (DataLoader): Test loader (yields ``(id, data, target)``).
        imb_ratio_value (float): ρ value broadcast to every sample.  Pass
            ``0.0`` to disable adaptive modulation (uniform-bias baseline).

    Returns:
        tuple[ndarray, ndarray, ndarray]:
            ``(fuse_embeddings [N, D], targets [N], mod_gates [N])``
    """
    all_fuse, all_targets, all_gates = [], [], []
    for (_, data, target) in tqdm(loader, desc="embed"):
        B = data.shape[0]
        imbalance_ratio = torch.full((B,), imb_ratio_value, dtype=torch.float32, device=device)
        data, target    = data.to(device), target.to(device)
        _, fuse_out, mod_gate, _ = model(data, imbalance_ratio, return_embeddings=True)
        all_fuse.append(fuse_out.cpu())
        all_targets.append(target.cpu())
        all_gates.append(mod_gate.cpu().squeeze(-1))
    return (
        torch.cat(all_fuse).numpy(),
        torch.cat(all_targets).numpy(),
        torch.cat(all_gates).numpy(),
    )


def plot_shift_magnitude(
    fuse_adaptive: np.ndarray,
    fuse_uniform: np.ndarray,
    targets: np.ndarray,
    save_path: str,
):
    """Violin plot of per-sample token-shift magnitude: MEL vs NON-MEL.

    The shift is ``||fuse_adaptive - fuse_uniform||`` per sample.  A
    Mann-Whitney U test checks whether MEL shifts are significantly larger.

    Args:
        fuse_adaptive (ndarray): Fuse embeddings with adaptive ρ, shape ``[N, D]``.
        fuse_uniform (ndarray): Fuse embeddings with ρ=0, shape ``[N, D]``.
        targets (ndarray): Class labels, shape ``[N]``.
        save_path (str): Output PNG path.
    """
    delta        = np.linalg.norm(fuse_adaptive - fuse_uniform, axis=1)
    mel_mask     = targets == 1
    nonmel_mask  = targets == 0
    delta_mel    = delta[mel_mask]
    delta_nonmel = delta[nonmel_mask]

    stat, p = mannwhitneyu(delta_mel, delta_nonmel, alternative='greater')

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.violinplot([delta_nonmel, delta_mel], positions=[0, 1],
                  showmedians=True, showextrema=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Majority\n(NON-MEL)', 'Minority\n(MEL)'])
    ax.set_ylabel('Token Shift Magnitude (Δ)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Saved] {save_path}   (p={p:.4f})")
    plt.close()


def plot_umap(
    fuse_adaptive: np.ndarray,
    fuse_uniform: np.ndarray,
    targets: np.ndarray,
    save_path: str,
):
    """Two-panel UMAP of fuse-token embeddings: adaptive vs uniform modulation.

    Args:
        fuse_adaptive (ndarray): Fuse embeddings with adaptive ρ.
        fuse_uniform (ndarray): Fuse embeddings with ρ=0.
        targets (ndarray): Class labels.
        save_path (str): Output PNG path.
    """
    reducer       = UMAP(n_components=2, random_state=42, n_neighbors=30)
    proj_adaptive = reducer.fit_transform(fuse_adaptive)
    proj_uniform  = reducer.transform(fuse_uniform)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, proj, title in zip(
        axes,
        [proj_adaptive, proj_uniform],
        ['Adaptive Modulation', 'Uniform Bias'],
    ):
        for cls, col, lbl in [(1, '#e05252', 'Minority (MEL)'),
                               (0, '#5282e0', 'Majority (NON-MEL)')]:
            m = targets == cls
            ax.scatter(proj[m, 0], proj[m, 1], c=col, label=lbl,
                       alpha=0.90, s=16, linewidths=0)
        ax.set_title(title, fontsize=16)
        ax.legend(fontsize=12, markerscale=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] {save_path}")
    plt.close()


def fuse_tokens_representation(model, loader, imb_ratio):
    """Run the full fuse-token interpretability analysis.

    Collects embeddings under adaptive and uniform (ρ=0) modulation, then
    saves a shift-magnitude violin plot and a UMAP scatter plot.

    Args:
        model (nn.Module): Loaded AdaptiveViT in eval mode.
        loader (DataLoader): Test loader.
        imb_ratio (float | list[float]): Dataset-level ρ (scalar used here).
    """
    scalar = float(np.mean(imb_ratio)) if isinstance(imb_ratio, list) else float(imb_ratio)

    print("Collecting adaptive embeddings …")
    fuse_adap, targets, _ = collect_embeddings(model, loader, scalar)

    print("Collecting uniform-bias embeddings (ρ=0) …")
    fuse_unif, _, _ = collect_embeddings(model, loader, 0.0)

    plot_shift_magnitude(
        fuse_adap, fuse_unif, targets,
        save_path=os.path.join(args.sub_dir, f'seed_{args.seed}_token_shift_magnitude.png'),
    )
    plot_umap(
        fuse_adap, fuse_unif, targets,
        save_path=os.path.join(args.sub_dir, f'seed_{args.seed}_umap_fuse_tokens.png'),
    )
    print("Embedding analysis complete.")


# ─────────────────────────────────────────────
# Main evaluation routine
# ─────────────────────────────────────────────

def main():
    """Load data, model, run inference, and print evaluation metrics."""
    d = args.dataset
    if d == 'Derm7ptClinic':
        df_train, df_test, mel_idx = DatasetDerm7pt.get_clinic_test_df(args.data_dir, args.out_dim)
    elif d == 'Derm7pt':
        df_train, df_test, mel_idx = DatasetDerm7pt.get_derm_test_df(args.data_dir, args.out_dim)
    elif d == 'ISIC2017':
        df_train, df_test, mel_idx = DatasetISIC2017.get_test_df(args.data_dir, args.out_dim)
    elif d == 'IBDHKUC':
        df_train, df_test, mel_idx = DatasetIBDHKUC.get_test_df(args.data_dir)
    elif d == 'ISIC2018':
        df_train, df_test, mel_idx = DatasetISIC2018.get_test_df(args.data_dir)
    elif d == 'ISIC2024':
        df_train, df_test, mel_idx = DatasetISIC2024.get_test_df(args.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {d}")

    if args.DEBUG:
        df_test = pd.concat([
            df_test[df_test['target'] == 0].sample(25),
            df_test[df_test['target'] == 1].sample(5),
        ], ignore_index=True)

    _, transforms_val = get_transforms(args.image_size)

    # ── Dataset ───────────────────────────────────────────────────────────────
    if d in ('Derm7pt', 'Derm7ptClinic'):
        dataset_test = Derm7pt_Dataset(df_test, 'test', d, transform=transforms_val)
    elif d == 'ISIC2017':
        dataset_test = ISIC2017_Dataset(df_test, 'test', transform=transforms_val)
    elif d == 'IBDHKUC':
        dataset_test = IBDHKUC_Dataset(df_test, 'test', transform=transforms_val)
    elif d == 'ISIC2018':
        dataset_test = ISIC2018_Dataset(df_test, 'test', transform=transforms_val)
    elif d == 'ISIC2024':
        dataset_test = ISIC2024_Dataset(
            df_test, 'test', transform=transforms_val,
            hdf5_path=os.path.join(args.data_dir, 'train-image.hdf5'),
        )

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=RandomSampler(dataset_test), num_workers=args.num_workers,
    )

    # ρ for the test run is computed from the training split.
    imb_ratio = calculate_imbalance_ratio_rho(
        df_train, num_classes=args.out_dim, strategy=args.rho_strategy
    )
    print(f"{d} imb_ratio: {imb_ratio}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_file = os.path.join(
        args.model_dir, f'{args.kernel_type}_seed_{args.seed}_best.pth'
    )
    print(f"Loading checkpoint: {model_file}")

    model = AdaptiveViT(config=config, out_dim=args.out_dim)
    model = model.to(device)

    try:
        model.load_state_dict(torch.load(model_file, map_location=device), strict=True)
    except RuntimeError:
        state_dict = torch.load(model_file, map_location=device)
        state_dict = {
            k[7:] if k.startswith('module.') else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)

    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    scalar_rho = float(np.mean(imb_ratio)) if isinstance(imb_ratio, list) else float(imb_ratio)

    PROBS, TARGETS = [], []
    with torch.no_grad():
        for (_, data, target) in tqdm(test_loader, desc="predict"):
            B               = data.shape[0]
            imbalance_ratio = torch.full((B,), scalar_rho, dtype=torch.float32, device=device)
            data, target    = data.to(device), target.to(device)

            probs = torch.zeros((B, args.out_dim), device=device)
            for I in range(args.n_test):
                probs += model(get_trans(data, I), imbalance_ratio).softmax(1)
            probs /= args.n_test

            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

    PROBS   = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    # ── Metrics ───────────────────────────────────────────────────────────────
    preds = PROBS.argmax(1)
    if args.out_dim > 2:
        acc       = accuracy_score(TARGETS, preds)
        f1        = f1_score(TARGETS, preds, average='macro')
        precision = precision_score(TARGETS, preds, average='macro')
        recall    = recall_score(TARGETS, preds, average='macro')
        roc_auc   = roc_auc_score(TARGETS, PROBS, average='macro', multi_class='ovr')
        cm        = confusion_matrix(TARGETS, preds)
    else:
        acc       = accuracy_score(TARGETS, preds)
        f1        = f1_score(TARGETS, preds)
        precision = precision_score(TARGETS, preds)
        recall    = recall_score(TARGETS, preds)
        roc_auc   = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
        cm        = confusion_matrix(TARGETS, preds).ravel()

    conf_matrix     = confusion_matrix(TARGETS, preds)
    per_class_acc   = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    class_report    = classification_report(
        TARGETS, preds,
        target_names=['NON-MEL', 'MEL'] if args.out_dim == 2 else None,
    )

    print(f"accuracy:        {acc:.4f}")
    print(f"precision:       {precision:.4f}")
    print(f"recall:          {recall:.4f}")
    print(f"f1_score:        {f1:.4f}")
    print(f"roc_auc:         {roc_auc:.4f}")
    print(f"confusion_matrix:{cm}")
    print(f"per_class_acc:   {per_class_acc}")
    print(f"mean_acc:        {np.mean(per_class_acc):.4f}")

    content = (
        f"{time.ctime()} | {args.eval} {args.kernel_type} seed: {args.seed} "
        f"dataset: {args.dataset} | "
        f"accuracy: {acc:.4f} | precision: {precision:.4f} | recall: {recall:.4f} | "
        f"f1: {f1:.4f} | roc_auc: {roc_auc:.4f} | cm: {cm}\n"
        f"per_class_acc: {per_class_acc}\n"
        f"Classification Report:\n{class_report}\n"
    )
    with open(
        os.path.join(args.sub_dir, f'pred_seed_{args.seed}_{args.kernel_type}.txt'), 'a'
    ) as f:
        f.write(content)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.sub_dir, exist_ok=True)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"seed: {args.seed}")
    set_seed(args.seed)

    device_cuda = torch.cuda.is_available()
    if device_cuda:
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    device = torch.device("cuda" if device_cuda else "cpu")

    main()
