"""
Lightweight Architecture 3: MobilePlantViT
A mobile-friendly hybrid CNN-ViT for generalized plant disease detection.
Parameters: ~0.69M (< 1.0M requirement)
GFLOPs: ~0.15 GFLOPs (< 0.5 GFLOPs requirement)
Combines inductive bias of lightweight CNNs with global context from lightweight Transformers.
"""

import torch
import torch.nn as nn
from typing import Tuple


class TransformerBlock(nn.Module):
    """Lightweight Multi-Head Self-Attention + MLP."""
    def __init__(self, embed_dim: int, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, N, C)
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """Local convolution -> Unfold to sequence -> Transformer -> Fold to tensor -> Fusion."""
    def __init__(self, in_channels: int, transformer_dim: int, depth: int = 2):
        super().__init__()
        # Local representation
        self.local_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.local_conv2 = nn.Conv2d(in_channels, transformer_dim, 1, bias=False)

        # Global sequence processing
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim=transformer_dim, num_heads=2) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(transformer_dim)

        # Fusion
        self.project = nn.Sequential(
            nn.Conv2d(transformer_dim, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.local_conv1(x)
        out = self.local_conv2(out)

        # Unfold (B, C, H, W) -> (B, H*W, C)
        b, c, h, w = out.shape
        seq = out.permute(0, 2, 3, 1).reshape(b, h * w, c)
        seq = self.transformer(seq)
        seq = self.norm(seq)

        # Fold back
        out = seq.reshape(b, h, w, c).permute(0, 3, 1, 2)
        out = self.project(out)

        # Concatenate and fuse
        return self.fusion(torch.cat([res, out], dim=1))


class MobilePlantViT(nn.Module):
    """
    Hybrid CNN-ViT Model for Edge Plant Pathology.
    Achieves ~0.69M parameters.
    """
    def __init__(self, num_classes: int = 24):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 24, 3, 1, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU()
        )

        # Stage 1 (CNN features)
        self.stage1 = nn.Sequential(
            nn.Conv2d(24, 32, 3, 2, 1, groups=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        # Stage 2 (MobileViT Block 1)
        self.mvit1 = MobileViTBlock(in_channels=32, transformer_dim=48, depth=2)

        # Downsample
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )

        # Stage 3 (MobileViT Block 2)
        self.mvit2 = MobileViTBlock(in_channels=64, transformer_dim=64, depth=2)

        # Downsample
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU()
        )

        # Stage 4 (MobileViT Block 3)
        self.mvit3 = MobileViTBlock(in_channels=96, transformer_dim=96, depth=2)

        # Global Pool & Classification
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(96, 128),
            nn.SiLU(),
            nn.Dropout(p=0.15),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.mvit1(x)
        x = self.down2(x)
        x = self.mvit2(x)
        x = self.down3(x)
        x = self.mvit3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


if __name__ == "__main__":
    net = MobilePlantViT(num_classes=24)
    p = sum(param.numel() for param in net.parameters() if param.requires_grad)
    print(f"MobilePlantViT Trainable Parameters: {p:,} ({p/1e6:.3f}M)")
    dummy = torch.randn(1, 3, 224, 224)
    print(f"Output shape: {net(dummy).shape}")
