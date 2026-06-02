"""Training script for AdaptiveViT with DALoss.
"""

import gc
import os
import random
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data.sampler import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

from utils import get_trans, get_transforms

import datasets.dataset_isic2017  as DatasetISIC2017
import datasets.dataset_isic2024  as DatasetISIC2024
import datasets.dataset_cbd4905   as DatasetCBD4905
import datasets.dataset_derm7pt   as DatasetDerm7pt
import datasets.dataset_ibd_hkuc  as DatasetIBDHKUC

from datasets.dataset_isic2024 import ISIC2024_Dataset
from datasets.dataset_isic2017 import ISIC2017_Dataset
from datasets.dataset_cbd4905  import ISICCBD4905_Dataset
from datasets.dataset_derm7pt  import Derm7pt_Dataset
from datasets.dataset_ibd_hkuc import IBDHKUC_Dataset

from models.adaptivevit import AdaptiveViT, DALoss
from datasets.data_module import calculate_imbalance_ratio_rho, calculate_gamma
from early_stopping import EarlyStopping


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train AdaptiveViT with DALoss")

    parser.add_argument('--save-name',   type=str, required=True,
                        help="Base name used for weight and log files.")
    parser.add_argument('--data-dir',    type=str, default='.',
                        help="Root directory of the dataset.")
    parser.add_argument('--image-size',  type=int, required=True)
    parser.add_argument('--enet-type',   type=str, required=True,
                        help="EfficientNet variant (e.g. efficientnet_b0).")
    parser.add_argument('--pretrained',  action='store_true')
    parser.add_argument('--batch-size',  type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--init-lr',     type=float, default=3e-5)
    parser.add_argument('--out-dim',     type=int, default=2,
                        help="Number of output classes.")
    parser.add_argument('--n-epochs',    type=int, default=20)
    parser.add_argument('--gpu-gc',      action='store_true')
    parser.add_argument('--DEBUG',       action='store_true')
    parser.add_argument('--model-dir',   type=str, default='./weights')
    parser.add_argument('--log-dir',     type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--seed',        type=int, default=0)
    parser.add_argument('--dataset',     type=str, required=True,
                        choices=[
                            'ISIC2017', 'ISIC2024', 'CBD4905',
                            'IMBD9810', 'IMBD26k', 'IMBD56k',
                            'Derm7pt', 'Derm7ptClinic', 'IBDHKUC',
                        ])
    parser.add_argument('--config',      type=str,
                        default='./config.yaml',
                        help="Path to the model architecture YAML config.")
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
    torch.backends.cudnn.benchmark     = True


# ─────────────────────────────────────────────
# LR scheduler
# ─────────────────────────────────────────────

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """Patched ``GradualWarmupScheduler`` that avoids an off-by-one LR error.

    After the warm-up phase the underlying ``after_scheduler`` base LRs are
    scaled by ``multiplier`` exactly once, preventing the first post-warmup
    step from using an un-scaled rate.
    """

    def get_lr(self) -> list:
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        return [
            base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
            for base_lr in self.base_lrs
        ]


# ─────────────────────────────────────────────
# Training / validation loops
# ─────────────────────────────────────────────

def _build_imbalance_tensors(imb_ratio, batch_size: int):
    """Build per-sample imbalance tensors from the dataset-level ρ.

    When ``imb_ratio`` is a list (``'per_class'`` strategy), the forward-pass
    tensor uses the mean ρ while ``loss_imbalance_ratio`` maps each sample's
    ground-truth class to its class-specific ρ at loss time.

    Args:
        imb_ratio (float | list[float]): Dataset-level ρ.
        batch_size (int): Number of samples in the current batch.

    Returns:
        tuple[Tensor, Tensor]: ``(imbalance_ratio, loss_imbalance_ratio)``
            both of shape ``[B]``.
    """
    if isinstance(imb_ratio, list):
        scalar = float(np.mean(imb_ratio))
        imbalance_ratio      = torch.full((batch_size,), scalar,           dtype=torch.float32)
        # loss_imbalance_ratio is filled per-sample in train_epoch using target
        loss_imbalance_ratio = torch.full((batch_size,), scalar,           dtype=torch.float32)
    else:
        imbalance_ratio      = torch.full((batch_size,), imb_ratio,        dtype=torch.float32)
        loss_imbalance_ratio = torch.full((batch_size,), imb_ratio,        dtype=torch.float32)
    return imbalance_ratio, loss_imbalance_ratio


def train_epoch(model, loader, optimizer, imb_ratio) -> float:
    """Run one training epoch.

    Args:
        model (nn.Module): The AdaptiveViT model.
        loader (DataLoader): Training data loader (yields ``(data, target)``).
        optimizer (Optimizer): Parameter optimiser.
        imb_ratio (float | list[float]): Dataset-level ρ computed once before
            the epoch from the training-set statistics.

    Returns:
        float: Mean training loss over the epoch.
    """
    model.train()
    train_losses = []

    for data, target in tqdm(loader, desc="train"):
        B = data.shape[0]
        imbalance_ratio, loss_imbalance_ratio = _build_imbalance_tensors(imb_ratio, B)

        # Per-class strategy: map each sample to its class-specific ρ for loss.
        if isinstance(imb_ratio, list):
            rho_tensor           = torch.tensor(imb_ratio, dtype=torch.float32)
            loss_imbalance_ratio = rho_tensor[target]  # [B]

        data, target             = data.to(device), target.to(device)
        imbalance_ratio          = imbalance_ratio.to(device)
        loss_imbalance_ratio     = loss_imbalance_ratio.to(device)

        optimizer.zero_grad()
        logits = model(data, imbalance_ratio=imbalance_ratio)
        loss   = criterion(logits, target, loss_imbalance_ratio)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.detach().cpu().item())

    return float(np.mean(train_losses))


def val_epoch(model, loader, mel_idx: int, imb_ratio, n_test: int = 1):
    """Run one validation epoch with optional test-time augmentation.

    Args:
        model (nn.Module): The AdaptiveViT model.
        loader (DataLoader): Validation data loader (yields ``(data, target)``).
        mel_idx (int): Index of the positive (melanoma) class.
        imb_ratio (float | list[float]): Dataset-level ρ.
        n_test (int): Number of TTA passes; 1 = no augmentation.

    Returns:
        tuple[float, float, float]: ``(val_loss, accuracy, balanced_accuracy)``
    """
    model.eval()
    val_losses, LOGITS, PROBS, TARGETS = [], [], [], []

    with torch.no_grad():
        for data, target in tqdm(loader, desc="val"):
            B = data.shape[0]
            imbalance_ratio, _ = _build_imbalance_tensors(imb_ratio, B)
            data, target, imbalance_ratio = (
                data.to(device), target.to(device), imbalance_ratio.to(device)
            )

            logits = torch.zeros((B, args.out_dim), device=device)
            probs  = torch.zeros((B, args.out_dim), device=device)
            for I in range(n_test):
                l = model(get_trans(data, I), imbalance_ratio)
                logits += l
                probs  += l.softmax(1)
            logits /= n_test
            probs  /= n_test

            loss = criterion(logits, target, imbalance_ratio)
            val_losses.append(loss.detach().cpu().item())
            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

    PROBS   = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    acc  = (PROBS.argmax(1) == TARGETS).mean() * 100.0
    if args.out_dim > 2: #multi-class case
        auc = roc_auc_score(TARGETS, PROBS, average='macro', multi_class='ovr')
    else: #binary case
        auc = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
    return float(np.mean(val_losses)), acc, auc


# ─────────────────────────────────────────────
# Dataset factory
# ─────────────────────────────────────────────

def _build_datasets(df_train, df_valid, transforms_train, transforms_val):
    """Instantiate the correct Dataset subclass for the chosen dataset.

    Args:
        df_train (DataFrame): Training split.
        df_valid (DataFrame): Validation split.
        transforms_train: Albumentations pipeline for training.
        transforms_val:   Albumentations pipeline for validation.

    Returns:
        tuple: ``(dataset_train, dataset_valid)``
    """
    d = args.dataset
    if d in ('CBD4905', 'IMBD9810', 'IMBD26k', 'IMBD56k'):
        Cls = ISICCBD4905_Dataset
        return (
            Cls(df_train, 'train', transform=transforms_train),
            Cls(df_valid, 'valid', transform=transforms_val),
        )
    if d in ('Derm7pt', 'Derm7ptClinic'):
        return (
            Derm7pt_Dataset(df_train, 'train', d, transform=transforms_train),
            Derm7pt_Dataset(df_valid, 'valid', d, transform=transforms_val),
        )
    if d == 'ISIC2017':
        return (
            ISIC2017_Dataset(df_train, 'train', transform=transforms_train),
            ISIC2017_Dataset(df_valid, 'valid', transform=transforms_val),
        )
    if d == 'IBDHKUC':
        return (
            IBDHKUC_Dataset(df_train, 'train', transform=transforms_train),
            IBDHKUC_Dataset(df_valid, 'valid', transform=transforms_val),
        )
    if d == 'ISIC2024':
        hdf5 = os.path.join(args.data_dir, 'train-image.hdf5')
        return (
            ISIC2024_Dataset(df_train, 'train', transform=transforms_train, hdf5_path=hdf5),
            ISIC2024_Dataset(df_valid, 'valid', transform=transforms_val,   hdf5_path=hdf5),
        )
    raise ValueError(f"Unknown dataset: {d}")


# ─────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────

def run(df_train, df_valid, transforms_train, transforms_val, mel_idx: int):
    """Build data loaders, model, optimiser, and run the training loop.

    Args:
        df_train (DataFrame): Training split.
        df_valid (DataFrame): Validation split.
        transforms_train: Albumentations pipeline for training.
        transforms_val:   Albumentations pipeline for validation.
        mel_idx (int): Positive class index (melanoma / minority class).
    """
    if args.DEBUG:
        args.n_epochs = 2
        df_train = df_train.sample(150, random_state=args.seed)
        df_valid = pd.concat([
            df_valid[df_valid['target'] == 0].sample(25, random_state=args.seed),
            df_valid[df_valid['target'] == 1].sample(5,  random_state=args.seed),
        ], ignore_index=True)

    assert df_train['target'].nunique() == args.out_dim, (
        f"Classes in dataset ({df_train['target'].nunique()}) "
        f"!= --out-dim ({args.out_dim})"
    )

    dataset_train, dataset_valid = _build_datasets(
        df_train, df_valid, transforms_train, transforms_val
    )

    # ρ and DALoss gamma are computed once from the full training set.
    # This replaces all static per-dataset base_gamma_pos / base_gamma_neg values.
    imb_ratio = calculate_imbalance_ratio_rho(
        df_train, num_classes=args.out_dim, strategy=args.rho_strategy
    )
    base_gamma_pos, base_gamma_neg = calculate_gamma(imb_ratio, num_classes=args.out_dim)
    criterion.__init__(base_gamma_pos=base_gamma_pos, base_gamma_neg=base_gamma_neg)
    print(
        f"{args.dataset} | imb_ratio: {imb_ratio} | "
        f"gamma_pos: {base_gamma_pos:.4f} | gamma_neg: {base_gamma_neg:.4f}"
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset_train),
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset_valid),
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    model = AdaptiveViT(config=config, out_dim=args.out_dim)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer        = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, args.n_epochs - 1
    )
    scheduler_warmup = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine
    )

    model_file = os.path.join(
        args.model_dir, f"{args.save_name}_seed_{args.seed}_best.pth"
    )
    es      = EarlyStopping(patience=10)
    auc_max = 0.
    epoch   = 0
    done    = False
    logs    = []

    print(f"Train: {len(dataset_train)}  |  Valid: {len(dataset_valid)}")

    while epoch < args.n_epochs and not done:
        epoch += 1
        print(time.ctime(), f"Epoch {epoch}")

        train_loss           = train_epoch(model, train_loader, optimizer, imb_ratio)
        val_loss, acc, auc   = val_epoch(model, valid_loader, mel_idx, imb_ratio)

        if es(model, -auc):
            done = True

        scheduler_warmup.step()
        if epoch == 2:
            scheduler_warmup.step()   # bug workaround for GradualWarmupScheduler

        current_lr = scheduler_warmup.get_last_lr()[0]

        log_entry = {
            'Time'       : time.ctime(),
            'Epoch'      : epoch,
            'lr'         : f'{current_lr:.7f}',
            'train loss' : f'{train_loss:.5f}',
            'valid loss' : f'{val_loss:.5f}',
            'acc'        : f'{acc:.4f}',
            'auc'        : f'{auc:.6f}',
        }
        logs.append(log_entry)

        content = (
            f"{time.ctime()} seed {args.seed} | "
            f"Epoch {epoch} | lr: {current_lr:.7f} | "
            f"train loss: {train_loss:.5f} | valid loss: {val_loss:.5f} | "
            f"acc: {acc:.4f} | auc: {auc:.6f} | EStop:[{es.status}]"
        )
        print(content)
        with open(os.path.join(args.log_dir, f"log_{args.save_name}_seed_{args.seed}.txt"), "a") as f:
            f.write(content + "\n")

        if auc > auc_max:
            print(f"auc improved ({auc_max:.6f} → {auc:.6f}). Saving model …")
            torch.save(model.state_dict(), model_file)
            auc_max = auc

    pd.DataFrame(logs).to_csv(
        os.path.join(args.log_dir, f"log_{args.save_name}_seed_{args.seed}.csv"),
        index=False,
    )


# ─────────────────────────────────────────────
# Data loading entry points
# ─────────────────────────────────────────────

def main():
    """Load dataset splits and launch training."""
    d = args.dataset
    if d == 'Derm7ptClinic':
        df_train, df_valid, mel_idx = DatasetDerm7pt.get_clinic_df(args.data_dir, args.out_dim)
    elif d == 'Derm7pt':
        df_train, df_valid, mel_idx = DatasetDerm7pt.get_derm_df(args.data_dir, args.out_dim)
    elif d == 'CBD4905':
        df_train, df_valid, mel_idx = DatasetCBD4905.get_df(args.data_dir)
    elif d == 'IMBD9810':
        df_train, df_valid, mel_idx = DatasetCBD4905.get_df_imbalanced_9810(args.data_dir)
    elif d == 'IMBD26k':
        df_train, df_valid, mel_idx = DatasetCBD4905.get_df_imbalanced_26k(args.data_dir)
    elif d == 'IMBD56k':
        df_train, df_valid, mel_idx = DatasetCBD4905.get_df_imbalanced_56k(args.data_dir)
    elif d == 'ISIC2017':
        df_train, df_valid, mel_idx = DatasetISIC2017.get_df(args.data_dir, args.out_dim)
    elif d == 'IBDHKUC':
        df_train, df_valid, mel_idx = DatasetIBDHKUC.get_df(args.data_dir)
    elif d == 'ISIC2024':
        df_train, df_valid, mel_idx = DatasetISIC2024.get_df(args.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {d}")

    print(f"Dataset: {d} | train: {len(df_train)} | valid: {len(df_valid)}")
    transforms_train, transforms_val = get_transforms(args.image_size)
    run(df_train, df_valid, transforms_train, transforms_val, mel_idx)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()

    # Free any stale GPU allocations before training starts.
    torch.cuda.empty_cache()
    gc.collect()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir,   exist_ok=True)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"seed: {args.seed}")
    set_seed(args.seed)

    writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(args.log_dir), 'tb-log'))

    device_cuda = torch.cuda.is_available()
    if device_cuda:
        print(f"CUDA devices: {torch.cuda.device_count()}")
        if int(args.CUDA_VISIBLE_DEVICES) != 0:
            torch.cuda.set_device(int(args.CUDA_VISIBLE_DEVICES))
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    device = torch.device("cuda" if device_cuda else "cpu")

    # criterion is a module-level singleton; gammas are updated inside run()
    # via calculate_gamma() once df_train is available.
    criterion = DALoss()

    main()

    writer.close()
