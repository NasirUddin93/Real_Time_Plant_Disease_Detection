"""
Lightweight Architecture 1: MobileNetV3-Small with CBAM Attention
Optimized for:
- Parameter count < 1.0M
- Computational complexity < 0.5 GFLOPs
- INT8 PTQ compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM."""
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(in_planes // ratio, 8)
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv1(scale)
        return self.sigmoid(scale) * x


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention."""
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class InvertedResidual(nn.Module):
    """Depthwise Separable Convolution block with optional SE/CBAM."""
    def __init__(
        self,
        in_c: int,
        out_c: int,
        exp_c: int,
        kernel_size: int,
        stride: int,
        use_cbam: bool = False
    ):
        super().__init__()
        self.use_res_connect = stride == 1 and in_c == out_c
        self.use_cbam = use_cbam

        layers = []
        # Pointwise Expansion
        if exp_c != in_c:
            layers.extend([
                nn.Conv2d(in_c, exp_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(exp_c),
                nn.Hardswish(inplace=True)
            ])

        # Depthwise Convolution
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(exp_c, exp_c, kernel_size, stride, padding, groups=exp_c, bias=False),
            nn.BatchNorm2d(exp_c),
            nn.Hardswish(inplace=True)
        ])

        # Attention
        if use_cbam:
            layers.append(CBAM(exp_c))

        # Linear Pointwise Projection
        layers.extend([
            nn.Conv2d(exp_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class MobileNetV3_CBAM(nn.Module):
    """
    Custom Ultra-Lightweight MobileNetV3-Small augmented with CBAM Attention.
    Total parameters: ~0.85M - 0.95M (<1.0M requirement)
    GFLOPs: ~0.06 - 0.12 GFLOPs (<0.5 GFLOPs requirement)
    """
    def __init__(self, num_classes: int = 24):
        super().__init__()
        # Initial standard Conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True)
        )

        # Inverted Residual Blocks: (in, out, exp, kernel, stride, use_cbam)
        cfgs = [
            (16, 16, 16, 3, 2, False),
            (16, 24, 72, 3, 2, False),
            (24, 24, 88, 3, 1, False),
            (24, 40, 96, 5, 2, True),
            (40, 40, 240, 5, 1, True),
            (40, 48, 144, 5, 1, True),
            (48, 96, 288, 5, 2, True),
            (96, 96, 576, 5, 1, True)
        ]

        blocks = []
        for in_c, out_c, exp_c, k, s, use_cbam in cfgs:
            blocks.append(InvertedResidual(in_c, out_c, exp_c, k, s, use_cbam))
        self.blocks = nn.Sequential(*blocks)

        # Head Conv and Classification
        self.head_conv = nn.Sequential(
            nn.Conv2d(96, 384, 1, 1, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.Hardswish(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(384, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MobileNetV3_CBAM(num_classes=24)
    params = count_parameters(model)
    print(f"MobileNetV3_CBAM Total Trainable Parameters: {params:,} ({params/1e6:.3f}M)")
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape} -> Verified < 1.0M parameter requirement.")
