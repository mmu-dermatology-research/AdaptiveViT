import torch
import torch.nn as nn
from collections import OrderedDict
import timm
from einops import rearrange
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────

class DALoss(nn.Module):
    """Distribution-Aware Adaptive Loss (DALoss).

    A focal-style loss whose per-sample focusing exponent (gamma) is dynamically
    modulated by the class-imbalance ratio of each mini-batch, so the model
    concentrates harder on difficult examples when the dataset is highly skewed.

    Supports binary (C=2) and multi-class (C>2) classification.

    Args:
        base_gamma_pos (float): Base focusing exponent for positive samples.
            Typically kept at 0 so positive-class gradients are not down-weighted.
        base_gamma_neg (float): Base focusing exponent for negative samples.
            Higher values suppress easy negatives more aggressively.
        eps (float): Small constant for numerical stability inside log/clamp.

    Shapes:
        - Binary:      logits [B, 2], targets [B],    imbalance_ratio [B]
        - Multi-class: logits [B, C], targets [B],    imbalance_ratio [B] or [B, C]
        where imbalance_ratio = log(pos_count / neg_count) per class.
    """

    def __init__(self, base_gamma_pos: float = 0.0, base_gamma_neg: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.base_gamma_pos = base_gamma_pos
        self.base_gamma_neg = base_gamma_neg
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, imbalance_ratio: torch.Tensor) -> torch.Tensor:
        """Dispatch to binary or multi-class forward based on the number of classes.

        Args:
            logits (Tensor): Raw model outputs of shape [B, C].
            targets (Tensor): Ground-truth class indices of shape [B].
            imbalance_ratio (Tensor): Per-sample (or per-class) log-imbalance ratio.

        Returns:
            Tensor: Scalar loss value.
        """
        _, C = logits.shape
        if C == 2:
            return self._forward_binary(logits, targets, imbalance_ratio)
        return self._forward_multiclass(logits, targets, imbalance_ratio)

    def _forward_binary(self, logits: torch.Tensor, targets: torch.Tensor, imbalance_ratio: torch.Tensor) -> torch.Tensor:
        """Binary focal loss with distribution-aware gamma.

        Args:
            logits (Tensor): Shape [B, 2].
            targets (Tensor): Binary labels of shape [B] (0 or 1).
            imbalance_ratio (Tensor): Shape [B] — log(pos_count / neg_count).

        Returns:
            Tensor: Scalar loss.
        """
        prob = torch.softmax(logits, dim=-1)[:, 1].clamp(self.eps, 1 - self.eps)  # [B]

        # Dynamic gamma: stronger focus on hard examples when imbalance is high.
        balance_score = 1 - torch.abs(torch.tanh(imbalance_ratio))                # [B]
        gamma_neg = (
            self.base_gamma_neg
            * (1 + torch.tanh(imbalance_ratio))
            * (1 - balance_score + self.eps)
        )                                                                           # [B]
        gamma_pos = torch.full_like(gamma_neg, self.base_gamma_pos)                # [B]

        pt    = torch.where(targets == 1, prob, 1 - prob)                          # [B]
        gamma = torch.where(targets == 1, gamma_pos, gamma_neg)                    # [B]

        focal_weight = (1 - pt) ** gamma
        loss = -focal_weight * torch.where(
            targets == 1,
            torch.log(prob),
            torch.log(1 - prob),
        )
        return loss.mean()

    def _forward_multiclass(self, logits: torch.Tensor, targets: torch.Tensor, imbalance_ratio: torch.Tensor) -> torch.Tensor:
        """Multi-class focal loss with per-class distribution-aware gamma.

        Args:
            logits (Tensor): Shape [B, C].
            targets (Tensor): Class indices of shape [B].
            imbalance_ratio (Tensor): Shape [B] (broadcast to all classes) or
                [B, C] (per-class imbalance ratio).

        Returns:
            Tensor: Scalar loss.
        """
        B, C = logits.shape
        probs     = torch.softmax(logits, dim=-1).clamp(self.eps, 1 - self.eps)    # [B, C]
        targets_oh = F.one_hot(targets, num_classes=C).float()                     # [B, C]

        # Expand imbalance_ratio to [B, C].
        if imbalance_ratio.dim() == 1:
            rho = imbalance_ratio.unsqueeze(1).expand(B, C)                        # [B, C]
        else:
            rho = imbalance_ratio                                                   # [B, C]

        # Dynamic gamma per class.
        balance_score = 1 - torch.abs(torch.tanh(rho))                            # [B, C]
        gamma_neg = (
            self.base_gamma_neg
            * (1 + torch.tanh(rho))
            * (1 - balance_score + self.eps)
        )                                                                           # [B, C]
        gamma_pos = torch.full_like(gamma_neg, self.base_gamma_pos)                # [B, C]

        pt    = torch.where(targets_oh == 1, probs, 1 - probs)                    # [B, C]
        gamma = torch.where(targets_oh == 1, gamma_pos, gamma_neg)                # [B, C]

        focal_weight = (1 - pt) ** gamma                                           # [B, C]
        loss = -focal_weight * torch.where(
            targets_oh == 1,
            torch.log(probs),
            torch.log(1 - probs),
        )                                                                           # [B, C]

        return loss.mean()


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

class QuickGELU(nn.Module):
    """Fast approximation of GELU using a sigmoid gate.

    Computes: x * sigmoid(1.702 * x), which closely tracks the exact GELU
    while being cheaper to evaluate.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Thin wrapper around ``nn.LayerNorm`` with a positional ``d_model`` argument."""

    def __init__(self, d_model: int):
        super().__init__(normalized_shape=d_model)


# ─────────────────────────────────────────────
# Transformer components
# ─────────────────────────────────────────────

class ResidualAttentionBlock(nn.Module):
    """Single transformer block with distribution-aware modulation and a token confidence gate (TCG).

    Extends the standard pre-norm residual attention block with two additions:

    1. **Distribution modulation**: a learnable projection of the imbalance
       embedding ``dist_embed`` is added to every token *before* self-attention,
       injecting dataset-distribution context into the attention computation.

    2. **Token Confidence Gate (TCG)**: a per-token sigmoid gate is applied to
       the MLP branch output, allowing the block to suppress uncertain or
       uninformative token updates.

    Args:
        d_model (int): Token embedding dimensionality.
        n_head (int): Number of self-attention heads.
        attn_mask (Tensor | None): Optional causal or padding mask passed to
            ``nn.MultiheadAttention``.
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn      = nn.MultiheadAttention(d_model, n_head)
        self.ln_1      = LayerNorm(d_model)
        self.mlp       = nn.Sequential(OrderedDict([
            ("c_fc",   nn.Linear(d_model, d_model * 4)),
            ("gelu",   QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2      = LayerNorm(d_model)
        self.attn_mask = attn_mask

        # Token confidence gate (TCG): scalar gate per token.
        self.token_gate       = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.latest_token_gate: torch.Tensor | None = None  # cached for analysis

        # Distribution-aware modulation.
        self.dist_modulator = nn.Linear(d_model, d_model)
        self.ln_dm          = LayerNorm(d_model)
        self.dist_scale     = nn.Parameter(torch.tensor(0.5))  # learnable blending scale

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Run self-attention, casting the mask to the correct dtype/device.

        Args:
            x (Tensor): Token sequence of shape [N, B, D].

        Returns:
            Tensor: Attention output of shape [N, B, D].
        """
        mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x: torch.Tensor, dist_embed: torch.Tensor) -> torch.Tensor:
        """Forward pass with distribution modulation and TCG.

        Args:
            x (Tensor): Token sequence of shape [N, B, D].
            dist_embed (Tensor): Distribution embedding of shape [B, D],
                produced by ``ADViTFuse`` from the per-sample imbalance ratio.

        Returns:
            Tensor: Updated token sequence of shape [N, B, D].
        """
        # 1. Inject distribution context into every token.
        dist_mod = self.ln_dm(self.dist_modulator(dist_embed).unsqueeze(0))  # [1, B, D]
        x = x + self.dist_scale * dist_mod

        # 2. Pre-norm self-attention residual.
        x = x + self._attention(self.ln_1(x))

        # 3. Token confidence gate controls the MLP contribution per token.
        gate = self.token_gate(x)          # [N, B, 1]
        self.latest_token_gate = gate      # cached for downstream analysis
        x = x + gate * self.mlp(self.ln_2(x))

        return x


class Transformer(nn.Module):
    """Stack of ``ResidualAttentionBlock`` layers.

    Args:
        width (int): Token embedding dimensionality (``d_model``).
        layers (int): Number of transformer blocks.
        heads (int): Number of attention heads per block.
        attn_mask (Tensor | None): Optional mask forwarded to every block.
    """

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width   = width
        self.layers  = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, dist_embed: torch.Tensor) -> torch.Tensor:
        """Pass the token sequence through all blocks sequentially.

        Args:
            x (Tensor): Token sequence [N, B, D].
            dist_embed (Tensor): Distribution embedding [B, D].

        Returns:
            Tensor: Output token sequence [N, B, D].
        """
        for block in self.resblocks:
            x = block(x, dist_embed)
        return x


# ─────────────────────────────────────────────
# ADViT-Fuse encoder
# ─────────────────────────────────────────────

class ADViTFuse(nn.Module):
    """Adaptive Distribution-aware ViT Fusion encoder (ADViT-Fuse).

    Fuses three multi-scale patch-embedding sequences (from a CNN backbone) via
    a transformer whose attention is conditioned on the dataset's class-imbalance
    ratio. Learnable *fuse tokens* (analogous to CLS tokens) aggregate global
    information and are projected to class logits.

    Key mechanisms:
    - **Adaptive modulation gate**: when the dataset is balanced (imbalance ≈ 0),
      the gate suppresses the distribution signal and falls back to a learned
      identity residual, so the transformer behaves like a standard ViT.
    - **Imbalance scale**: a sigmoid-gated feature vector amplifies or attenuates
      the distribution embedding dimension-wise.
    - **Fuse tokens**: ``num_classes`` learnable tokens prepended to the sequence;
      their final states are concatenated and classified.

    Args:
        width (int): Transformer embedding dimensionality.
        layers (int): Number of transformer blocks.
        heads (int): Number of attention heads.
        fuse_tokens (int): Number of learnable aggregation tokens (typically = num_classes).
        num_classes (int): Number of output classes.
    """

    def __init__(self, width: int, layers: int, heads: int, fuse_tokens: int = 2, num_classes: int = 2):
        super().__init__()

        self.n_fuse_tokens = fuse_tokens
        self.fuse_tokens   = nn.Parameter(torch.randn(fuse_tokens, 1, width))
        self.ln_pre        = LayerNorm(width)
        self.transformer   = Transformer(width, layers, heads)
        self.classifier    = nn.Sequential(
            nn.LayerNorm(width * fuse_tokens),
            nn.Linear(width * fuse_tokens, num_classes),
        )

        # Distribution-aware modulation components.
        self.dist_embed_layer = nn.Linear(1, width)
        self.imb_scale_layer  = nn.Sequential(nn.Linear(1, width), nn.Sigmoid())
        self.residual_identity = nn.Parameter(torch.rand(1, width))

        # Cached for downstream analysis (not used in the forward computation).
        self.latest_mod_gate: torch.Tensor | None  = None
        self.latest_imb_scale: torch.Tensor | None = None

    def _build_dist_embed(self, imbalance_ratio: torch.Tensor, B: int) -> torch.Tensor:
        """Compute the distribution embedding from the imbalance ratio.

        The embedding is adaptively blended between a distribution-modulated
        signal and a learned identity residual using a modulation gate derived
        from how imbalanced the batch is.

        Args:
            imbalance_ratio (Tensor): Shape [B, 1].
            B (int): Batch size.

        Returns:
            Tensor: Distribution embedding of shape [B, D].
        """
        balance_score = 1.0 - torch.abs(torch.tanh(imbalance_ratio))           # [B, 1]
        mod_gate      = (1.0 - balance_score).clamp(0.0, 1.0)                  # [B, 1]
        self.latest_mod_gate = mod_gate

        imb_scale = self.imb_scale_layer(imbalance_ratio)                       # [B, D]
        self.latest_imb_scale = imb_scale

        dist_embed   = self.dist_embed_layer(imbalance_ratio) * imb_scale       # [B, D]
        res_identity = self.residual_identity.expand(B, -1)                     # [B, D]

        # Blend: high imbalance → use dist_embed; balanced → use identity residual.
        return mod_gate * dist_embed + (1.0 - mod_gate) * res_identity          # [B, D]

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        imbalance_ratio: torch.Tensor,
        return_embeddings: bool = False,
    ):
        """Fuse three patch-embedding sequences and classify.

        Args:
            x1 (Tensor): Patch embeddings from scale 1, shape [N, B, D].
            x2 (Tensor): Patch embeddings from scale 2, shape [N, B, D].
            x3 (Tensor): Patch embeddings from scale 3, shape [N, B, D].
            imbalance_ratio (Tensor): Per-sample log-imbalance ratio, shape [B].
            return_embeddings (bool): If ``True``, also return the fused embedding
                and intermediate modulation tensors for analysis.

        Returns:
            Tensor: Class logits of shape [B, num_classes].
            (optional) Tuple of ``(logits, fuse_embedding, mod_gate, imb_scale)``
                when ``return_embeddings=True``.
        """
        # Concatenate multi-scale tokens and prepend fuse tokens.
        x = torch.cat([x1, x2, x3], dim=0)                                     # [3N, B, D]
        B = x.shape[1]
        fuse_tokens = self.fuse_tokens.expand(-1, B, -1)                        # [F, B, D]
        x = self.ln_pre(torch.cat([fuse_tokens, x], dim=0))                    # [F+3N, B, D]

        # Build distribution embedding and run transformer.
        dist_embed = self._build_dist_embed(imbalance_ratio.view(B, 1), B)     # [B, D]
        x = self.transformer(x, dist_embed)

        # Classify from fuse-token outputs.
        fuse_out    = x[:self.n_fuse_tokens]                                    # [F, B, D]
        fuse_out_cat = fuse_out.permute(1, 0, 2).reshape(B, -1)                # [B, F*D]
        logits      = self.classifier(fuse_out_cat)                             # [B, num_classes]

        if return_embeddings:
            return logits, fuse_out_cat, self.latest_mod_gate, self.latest_imb_scale

        return logits


# ─────────────────────────────────────────────
# Top-level model
# ─────────────────────────────────────────────

class AdaptiveViT(nn.Module):
    """Hybrid CNN-ViT model with distribution-aware adaptive fusion (AdaptiveViT).

    Architecture overview:
    1. **EfficientNet-B0 backbone** extracts multi-scale feature maps at three
       pyramid levels (stages 2, 3, 4), with spatial resolutions 28×28, 14×14,
       and 7×7 respectively for a 224-px input.
    2. Each feature map is patchified with a shared patch size ``p`` and
       projected to the transformer dimension via a scale-specific linear layer.
       All three scales are forced to the same token count ``N0`` (determined by
       the coarsest scale) using a ``view`` reshape — this is the same operation
       as in the original codebase and is required for checkpoint compatibility.
    3. The three patch-embedding sequences are fused by ``ADViTFuse``, which
       conditions self-attention on the per-batch class-imbalance ratio.

    Expected config keys (under ``config['model']``):
        - ``image-size``: Input image resolution (square).
        - ``patch-size``: Patch size used when patchifying feature maps.
        - ``dim``: Transformer embedding dimensionality.
        - ``depth``: Number of transformer blocks.
        - ``heads``: Number of attention heads.
        - ``mlp-dim``, ``emb-dim``, ``dim-head``: Reserved for config compatibility.
        - ``dropout``, ``emb-dropout``: Reserved for config compatibility.

    Args:
        config (dict): Model configuration dictionary (see above).
        out_dim (int): Number of output classes.
        channels (int): Unused; kept for API compatibility.
    """

    def __init__(self, config: dict, out_dim: int, channels: int = 512):
        super().__init__()

        # ── Unpack config ──────────────────────────────────────────────────
        model_cfg  = config['model']
        image_size = model_cfg['image-size']
        patch_size = model_cfg['patch-size']
        dim        = model_cfg['dim']
        depth      = model_cfg['depth']
        heads      = model_cfg['heads']
        num_classes = out_dim

        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})."
        )

        # ── CNN backbone ───────────────────────────────────────────────────
        self.efficient_net = timm.create_model(
            'efficientnet_b0', pretrained=True, features_only=True, out_indices=[2, 3, 4]
        )
        self.feature_channels = self.efficient_net.feature_info.channels()  # e.g. [40, 112, 320]
        print(
            f"AdaptiveViT | backbone: efficientnet_b0 | "
            f"feature channels: {self.feature_channels} | "
            f"patch_size: {patch_size}"
        )

        # ── Patch projection layers ────────────────────────────────────────
        # The patch token count N is fixed by the coarsest feature map (stage 2).
        # For EfficientNet-B0 with patch_size=7 and 224-px input:
        #   Stage 2: 28×28 → 4×4 = 16 patches, each of size 7×7×40  = 1960
        #   Stage 3: 14×14 → 2×2 =  4 patches, but view() forces N back to 16,
        #             so each "patch" absorbs 4/16 of the spatial tokens:
        #             dim = (4 × 7×7×112) / 16 = 1372
        #   Stage 4:  7×7  → 1×1 =  1 patch,  view() forces N to 16:
        #             dim = (1 × 7×7×320) / 16 =  980
        self.patch_size = patch_size

        # Reference patch count from the coarsest scale (stage 2, spatial = image/8).
        spatial_s0   = image_size // 8          # e.g. 224//8 = 28
        n_patches_s0 = (spatial_s0 // patch_size) ** 2   # e.g. (28//7)^2 = 16

        # patch_dim per scale = total_elements_after_rearrange / n_patches_s0
        c0, c1, c2 = self.feature_channels
        spatial_s1 = image_size // 16           # 14 for 224-px input
        spatial_s2 = image_size // 32           # 7

        patch_dim_112 = ((spatial_s0 // patch_size) ** 2 * patch_size ** 2 * c0) // n_patches_s0
        patch_dim_56  = ((spatial_s1 // patch_size) ** 2 * patch_size ** 2 * c1) // n_patches_s0
        patch_dim_28  = ((spatial_s2 // patch_size) ** 2 * patch_size ** 2 * c2) // n_patches_s0

        self.patch_to_embedding_112 = nn.Linear(patch_dim_112, dim)
        self.patch_to_embedding_56  = nn.Linear(patch_dim_56,  dim)
        self.patch_to_embedding_28  = nn.Linear(patch_dim_28,  dim)

        print(
            f"Patch dims — 112: {patch_dim_112}, 56: {patch_dim_56}, 28: {patch_dim_28}"
        )

        # ── ADViT-Fuse encoder ─────────────────────────────────────────────
        self.advit_encoder = ADViTFuse(
            width=dim, layers=depth, heads=heads,
            fuse_tokens=num_classes, num_classes=num_classes,
        )

    def _patchify_and_embed(
        self,
        feature_map: torch.Tensor,
        embed_layer: nn.Linear,
        p: int,
        N0: int,
    ) -> torch.Tensor:
        """Patchify a CNN feature map and project patches to the transformer dimension.

        All three scales share the same token count ``N0`` (set by the coarsest
        scale).  For finer scales whose natural patch count is smaller, the
        spatial and channel dimensions are jointly re-tiled via ``view`` so that
        the total number of elements is preserved while the batch is reshaped to
        exactly ``N0`` tokens.  This is the same operation as in the original
        codebase and must not be changed to keep checkpoint compatibility.

        Args:
            feature_map (Tensor): CNN feature map of shape [B, C, H, W].
            embed_layer (nn.Linear): Scale-specific linear projection.
            p (int): Patch size applied to both spatial dimensions.
            N0 (int): Target number of patch tokens (fixed by the coarsest scale).

        Returns:
            Tensor: Patch embeddings of shape [N0, B, D].
        """
        patches = rearrange(feature_map, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        B, N, C = patches.shape
        patches = patches.view(B, N0, -1)              # unify token count across scales
        return embed_layer(patches).permute(1, 0, 2)   # [N0, B, D]

    def forward(
        self,
        x: torch.Tensor,
        imbalance_ratio: torch.Tensor,
        x_meta=None,
        mask=None,
        return_embeddings: bool = False,
    ):
        """Forward pass.

        Args:
            x (Tensor): Input images of shape [B, 3, H, W].
            imbalance_ratio (Tensor): Per-sample log-imbalance ratio, shape [B].
                Under ``DataParallel`` the full-batch tensor is passed as a keyword
                argument and is therefore *not* automatically scattered; we slice
                it here to match this device's sub-batch.
            x_meta: Unused metadata input; reserved for future extension.
            mask: Unused mask input; reserved for future extension.
            return_embeddings (bool): If ``True``, return intermediate embeddings
                in addition to logits.

        Returns:
            Tensor: Class logits of shape [B, num_classes].
            (optional) Tuple ``(logits, fuse_embedding, mod_gate, imb_scale)``
                when ``return_embeddings=True``.
        """
        B = x.shape[0]

        # DataParallel passes keyword args un-scattered; slice to this device's B.
        if imbalance_ratio is None:
            imbalance_ratio = torch.zeros(B, device=x.device, dtype=x.dtype)
        else:
            imbalance_ratio = imbalance_ratio[:B].to(x.device)

        # ── Multi-scale feature extraction ─────────────────────────────────
        p    = self.patch_size
        feats = self.efficient_net(x)          # [feat_s0, feat_s1, feat_s2]

        # N0: reference token count from the coarsest scale.
        B0, _, H0, W0 = feats[0].shape
        N0 = (H0 // p) * (W0 // p)

        y_112 = self._patchify_and_embed(feats[0], self.patch_to_embedding_112, p, N0)
        y_56  = self._patchify_and_embed(feats[1], self.patch_to_embedding_56,  p, N0)
        y_28  = self._patchify_and_embed(feats[2], self.patch_to_embedding_28,  p, N0)

        # ── Fusion and classification ──────────────────────────────────────
        return self.advit_encoder(
            y_112, y_56, y_28, imbalance_ratio, return_embeddings=return_embeddings
        )