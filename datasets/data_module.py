import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


# ─────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────

def make_weighted_sampler(df, num_classes: int) -> WeightedRandomSampler:
    """Build an inverse-frequency ``WeightedRandomSampler`` over a dataset.

    Each sample is assigned a weight equal to the inverse of its class count,
    so every class appears proportionally in every batch regardless of the raw
    class distribution.  Epoch length is preserved (``num_samples = len(df)``).

    Args:
        df (DataFrame): Must contain a ``'target'`` column with integer class labels.
        num_classes (int): Total number of classes (0 … num_classes-1).

    Returns:
        WeightedRandomSampler: Ready to pass as ``sampler=`` to a DataLoader.
    """
    class_counts  = df['target'].value_counts().sort_index()
    class_weights = {c: 1.0 / class_counts.get(c, 1) for c in range(num_classes)}

    sample_weights = torch.tensor(
        [class_weights[t] for t in df['target'].values],
        dtype=torch.float32,
    )

    print("\nWeightedRandomSampler class weights:")
    for c, w in class_weights.items():
        n = class_counts.get(c, 0)
        print(
            f"  Class {c}: n={n:4d}  weight={w:.5f}  "
            f"expected per batch (bs=16): "
            f"{w * n / sum(class_weights.values()) * 16:.1f}"
        )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─────────────────────────────────────────────
# Imbalance ratio ρ
# ─────────────────────────────────────────────

def calculate_imbalance_ratio_rho(
    df,
    num_classes: int,
    strategy: str = 'per_class_avg',
) -> "float | list[float]":
    """Compute the imbalance ratio ρ used to condition DALoss and ADViTFuse.

    **Binary** (num_classes == 2):
        ``ρ = log(pos / neg)``

    **Multi-class** strategies (num_classes > 2):

    +------------------+------------------------------------------------------+
    | ``'per_class'``  | ``ρ_c = log(n_c / n_rest)`` for each class →        |
    |                  | returns a list of C floats.  Most precise; each      |
    |                  | class gets its own distribution-aware gamma.         |
    +------------------+------------------------------------------------------+
    | ``'per_class_avg'`` | Same per-class computation, but returns the      |
    |                  | scalar mean across classes.  Good default.           |
    +------------------+------------------------------------------------------+
    | ``'minmax'``     | ``ρ = log(n_min / n_max)`` — global severity        |
    |                  | signal.  Near 0 = balanced; very negative = severe. |
    +------------------+------------------------------------------------------+
    | ``'tail_head'``  | ``ρ = log(n_tail / n_head)`` — maximum contrast     |
    |                  | between rarest and most common class.               |
    +------------------+------------------------------------------------------+

    Args:
        df (DataFrame): Must contain a ``'target'`` column.
        num_classes (int): Number of classes.
        strategy (str): One of ``'per_class'``, ``'per_class_avg'``,
            ``'minmax'``, ``'tail_head'``.  Ignored for binary classification.

    Returns:
        float | list[float]: Scalar for binary / ``'per_class_avg'`` /
        ``'minmax'`` / ``'tail_head'``; list of C floats for ``'per_class'``.

    Raises:
        ValueError: If an unknown strategy is given.
    """
    class_counts = df['target'].value_counts().sort_index()
    total        = len(df)

    # ── Binary ────────────────────────────────────────────────────────────────
    if num_classes == 2:
        pos = class_counts.get(1, 1)
        neg = class_counts.get(0, 1)
        rho = float(np.log(pos / max(neg, 1)))
        print(f"[rho | binary] pos={pos}, neg={neg}, rho={rho:.4f}")
        return rho

    # ── Multi-class ───────────────────────────────────────────────────────────
    counts     = np.array([class_counts.get(c, 1) for c in range(num_classes)], dtype=float)
    n_min      = counts.min()
    n_max      = counts.max()
    tail_class = int(np.argmin(counts))
    head_class = int(np.argmax(counts))

    print(f"[rho | multiclass | strategy={strategy}]")
    for c in range(num_classes):
        print(f"  Class {c}: {int(counts[c])} samples")
    print(f"  Head class: {head_class} ({int(n_max)} samples)")
    print(f"  Tail class: {tail_class} ({int(n_min)} samples)")

    def _per_class_rhos() -> list:
        rho_list = []
        for c in range(num_classes):
            n_c   = counts[c]
            n_rest = total - n_c
            rho_c = float(np.log(n_c / max(n_rest, 1)))
            rho_list.append(rho_c)
            print(f"  rho[class {c}] = log({int(n_c)}/{int(n_rest)}) = {rho_c:.4f}")
        return rho_list

    if strategy == 'per_class':
        rho_list = _per_class_rhos()
        print(f"  rho (mean scalar) = {np.mean(rho_list):.4f}")
        return rho_list

    if strategy == 'per_class_avg':
        rho_list = _per_class_rhos()
        mean_rho = float(np.mean(rho_list))
        print(f"  rho (mean scalar) = {mean_rho:.4f}")
        return mean_rho

    if strategy == 'minmax':
        rho = float(np.log(n_min / max(n_max, 1)))
        print(f"  rho = log({int(n_min)}/{int(n_max)}) = {rho:.4f}")
        return rho

    if strategy == 'tail_head':
        n_tail = counts[tail_class]
        n_head = counts[head_class]
        rho    = float(np.log(n_tail / max(n_head, 1)))
        print(f"  rho = log(tail={int(n_tail)} / head={int(n_head)}) = {rho:.4f}")
        return rho

    raise ValueError(
        f"Unknown strategy '{strategy}'. "
        f"Choose: 'per_class', 'per_class_avg', 'minmax', 'tail_head'."
    )


# ─────────────────────────────────────────────
# DALoss gamma computation
# ─────────────────────────────────────────────

def calculate_gamma(rho: "float | list[float]", num_classes: int) -> "tuple[float, float]":
    """Derive ``base_gamma_pos`` and ``base_gamma_neg`` for ``DALoss`` from ρ.
    Args:
        rho (float | list[float]): Scalar ρ (binary / aggregated multi-class)
            or list of per-class ρ values as returned by
            ``calculate_imbalance_ratio_rho``.
        num_classes (int): Number of output classes.

    Returns:
        tuple[float, float]: ``(base_gamma_pos, base_gamma_neg)``
    """
    GAMMA_MIN = 0.5
    GAMMA_MAX = 5.0

    # Reduce to a scalar mean ρ for the formula.
    if isinstance(rho, list):
        rho_arr  = np.array(rho, dtype=float)
        rho_mean = float(np.mean(rho_arr))
        rho_head = float(np.max(rho_arr))   # most common class
    else:
        rho_mean = float(rho)
        rho_head = float(rho)

    abs_tanh_mean = float(abs(np.tanh(rho_mean)))
    abs_tanh_head = float(abs(np.tanh(rho_head)))

    # γ− varies continuously with ρ̄
    base_gamma_neg = float(
        np.clip(GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * abs_tanh_mean, GAMMA_MIN, GAMMA_MAX)
    )

    # γ+ is small for binary (no over-focus on positives) and damped by
    # class count for multi-class to avoid head-class suppression.
    if num_classes == 2:
        base_gamma_pos = 0.0
    else:
        pos_raw        = GAMMA_MIN * max(0.0, float(np.tanh(rho_head)))
        pos_dampen     = 1.0 / float(np.sqrt(np.log(num_classes) + 1e-6))
        base_gamma_pos = float(np.clip(pos_raw * pos_dampen, 0.0, GAMMA_MIN))

    print(
        f"[calculate_gamma] rho_mean={rho_mean:.4f} → "
        f"base_gamma_neg={base_gamma_neg:.4f}, "
        f"base_gamma_pos={base_gamma_pos:.4f}"
    )
    return base_gamma_pos, base_gamma_neg


# ─────────────────────────────────────────────
# DALoss bounds (multi-class analysis)
# ─────────────────────────────────────────────

def compute_bounds_multiclass(
    rho_per_class: list,
    num_classes: int,
    pt_easy: float = 0.85,
) -> dict:
    """Compute the full set of DALoss gamma bounds for a multi-class dataset.

    This is an *analysis utility* — it computes collapse ceiling, per-class
    gamma spread, and severity statistics from the per-class ρ values.  It is
    not called during training; use ``calculate_gamma`` for that.

    Args:
        rho_per_class (list[float]): Per-class ρ values of length C, as
            returned by ``calculate_imbalance_ratio_rho(..., strategy='per_class')``.
        num_classes (int): Number of classes C.
        pt_easy (float): Probability threshold below which a sample is
            considered easy (used to compute the collapse ceiling gamma).

    Returns:
        dict: Analysis results including ``gamma_min``, ``gamma_max``,
        ``gamma_ceil``, ``base_gamma_neg``, ``base_gamma_pos``,
        ``per_class_gamma``, and intermediate statistics.
    """
    rho_array  = np.array(rho_per_class, dtype=float)
    rho_mean   = float(np.mean(rho_array))
    rho_head   = float(np.max(rho_array))
    rho_tail   = float(np.min(rho_array))
    idx_head   = int(np.argmax(rho_array))
    idx_tail   = int(np.argmin(rho_array))

    abs_tanh_head = float(abs(np.tanh(rho_head)))
    abs_tanh_tail = float(abs(np.tanh(rho_tail)))
    abs_tanh_mean = float(abs(np.tanh(rho_mean)))

    # Collapse ceiling: gamma that suppresses all easy samples to < 1e-5 weight.
    gamma_collapse = float(np.log(1e-5) / np.log(1 - pt_easy))
    gamma_ceil     = round(gamma_collapse * 0.90, 2)

    gamma_min = float(np.clip(1.0 + abs_tanh_head, 1.0, 2.0))

    p_head = float(np.exp(rho_head) / (1 + np.exp(rho_head)))
    p_tail = float(np.exp(rho_tail) / (1 + np.exp(rho_tail)))

    base_severity  = float(np.log(p_head / p_tail))
    class_factor   = float(np.log(num_classes))
    scarcity_bonus = float(max(0.0, np.log(0.05 / p_tail)))
    severity_scale = float(base_severity * (1 + class_factor) + scarcity_bonus)

    gamma_max = float(np.clip(
        1.0 + abs_tanh_tail * severity_scale,
        gamma_min + 1.5,
        gamma_ceil,
    ))

    base_gamma_neg = float(gamma_min + (gamma_max - gamma_min) * abs_tanh_mean)

    pos_raw        = gamma_min * float(max(0.0, np.tanh(rho_head)))
    pos_dampen     = float(1.0 / np.sqrt(np.log(num_classes) + 1e-6))
    base_gamma_pos = float(np.clip(pos_raw * pos_dampen, 0.0, gamma_min))

    # Per-class gamma spread: tail class gets gamma_max, head gets gamma_min.
    rho_sorted_idx  = np.argsort(rho_array)   # ascending: tail first
    per_class_gamma = np.zeros(num_classes)
    for rank, cidx in enumerate(rho_sorted_idx):
        alpha = 1.0 - (rank / (num_classes - 1))
        per_class_gamma[cidx] = gamma_min + (gamma_max - gamma_min) * alpha

    return {
        'gamma_min'       : round(gamma_min,       4),
        'gamma_max'       : round(gamma_max,        4),
        'gamma_ceil'      : round(gamma_ceil,       4),
        'gamma_collapse'  : round(gamma_collapse,   4),
        'base_gamma_neg'  : round(base_gamma_neg,   4),
        'base_gamma_pos'  : round(base_gamma_pos,   4),
        'rho_mean'        : round(rho_mean,         4),
        'rho_head'        : round(rho_head,         4),
        'rho_tail'        : round(rho_tail,         4),
        'severity_scale'  : round(severity_scale,   4),
        'base_severity'   : round(base_severity,    4),
        'class_factor'    : round(class_factor,     4),
        'scarcity_bonus'  : round(scarcity_bonus,   4),
        'abs_tanh_mean'   : round(abs_tanh_mean,    4),
        'abs_tanh_head'   : round(abs_tanh_head,    4),
        'abs_tanh_tail'   : round(abs_tanh_tail,    4),
        'p_head'          : round(p_head,           4),
        'p_tail'          : round(p_tail,           4),
        'pt_easy'         : pt_easy,
        'num_classes'     : num_classes,
        'head_class_idx'  : idx_head,
        'tail_class_idx'  : idx_tail,
        'per_class_gamma' : [round(float(g), 4) for g in per_class_gamma],
    }
