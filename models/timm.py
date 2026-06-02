import torch
import torch.nn as nn
import timm


class ImageNet(nn.Module):
    """This is a timm wrapper for standard image classification backbones.

    Loads any model supported by ``timm`` and, when a ViT backbone is used
    at a non-native resolution, transparently adapts the patch-embedding
    projection and positional embeddings so the model accepts the smaller
    input without modification to the rest of the architecture.

    Currently supported resolution adaptation:
        - ``vit_base_patch16_224`` at ``img_size=128``: the patch size is
          halved to 8×8 so the number of patches (16×16 = 256) matches the
          reduced spatial resolution.

    Args:
        enet_type (str): Any model identifier accepted by ``timm.create_model``
            (e.g. ``'efficientnet_b0'``, ``'vit_base_patch16_224'``).
        out_dim (int): Number of output classes.
        img_size (int): Input image resolution (square). Defaults to 224.
        pretrained (bool): Whether to load ImageNet-pretrained weights.
            Defaults to ``True``.
    """

    def __init__(self, enet_type: str, out_dim: int, img_size: int = 224, pretrained: bool = True):
        super().__init__()
        print(f"ImageNet | backbone: {enet_type} | pretrained: {pretrained} | img_size: {img_size}")

        self.imgnet = timm.create_model(enet_type, pretrained=pretrained, num_classes=out_dim)

        # Adapt ViT patch embedding and positional encoding for non-native resolutions.
        if enet_type == 'vit_base_patch16_224' and img_size == 128:
            adapted_patch_size = img_size // 16  # 8 — keeps the patch grid at 16×16
            self.imgnet.patch_embed = self._adapt_patch_embed(
                self.imgnet.patch_embed, img_size=img_size, patch_size=adapted_patch_size
            )
            self.imgnet.pos_embed = self._adapt_pos_embed(
                self.imgnet.pos_embed, img_size=img_size, patch_size=adapted_patch_size
            )

    def _adapt_patch_embed(self, patch_embed: nn.Module, img_size: int, patch_size: int) -> nn.Module:
        """Replace the patch projection conv to match a new patch size.

        The output channel count is preserved from the original projection so
        that all downstream transformer weights remain compatible.

        Args:
            patch_embed: The existing ``PatchEmbed`` module from the ViT.
            img_size (int): Target input resolution (square).
            patch_size (int): New patch size (square).

        Returns:
            nn.Module: The updated ``patch_embed`` module (modified in-place and returned).
        """
        patch_embed.proj = nn.Conv2d(
            in_channels=3,
            out_channels=patch_embed.proj.out_channels,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )
        patch_embed.num_patches = (img_size // patch_size) ** 2
        patch_embed.img_size   = (img_size, img_size)
        return patch_embed

    def _adapt_pos_embed(self, pos_embed: nn.Parameter, img_size: int, patch_size: int) -> nn.Parameter:
        """Reinitialise positional embeddings when the patch count changes.

        If the existing positional embedding tensor already has the correct
        length it is returned unchanged. Otherwise a new parameter is
        initialised from a standard normal distribution (the model should be
        fine-tuned after this operation).

        Args:
            pos_embed (nn.Parameter): Original positional embedding of shape
                ``[1, num_patches + 1, embed_dim]`` (``+1`` for the CLS token).
            img_size (int): Target input resolution (square).
            patch_size (int): New patch size (square).

        Returns:
            nn.Parameter: Positional embedding of shape
                ``[1, new_num_patches + 1, embed_dim]``.
        """
        num_patches  = (img_size // patch_size) ** 2
        expected_len = num_patches + 1  # +1 for the CLS token

        if pos_embed.shape[1] == expected_len:
            return pos_embed

        print(
            f"Reinitialising positional embeddings: "
            f"{pos_embed.shape[1]} tokens → {expected_len} tokens"
        )
        return nn.Parameter(torch.randn(1, expected_len, pos_embed.shape[2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone forward pass.

        Args:
            x (Tensor): Input images of shape ``[B, 3, H, W]``.

        Returns:
            Tensor: Class logits of shape ``[B, out_dim]``.
        """
        return self.imgnet(x)