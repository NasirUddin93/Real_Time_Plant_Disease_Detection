"""
Structured L1-Norm Filter Pruning Engine
Ranks convolutional filters by ||W_j||_1 norm and iteratively zeroes/removes
redundant channels to reduce FLOPs and parameter footprints.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


def compute_layer_l1_norms(conv_layer: nn.Conv2d) -> torch.Tensor:
    """Computes the L1-norm for each output channel/filter in a convolutional layer."""
    # Weight shape: (out_channels, in_channels, kH, kW)
    weights = conv_layer.weight.data
    l1_norms = torch.sum(torch.abs(weights), dim=(1, 2, 3))
    return l1_norms


def prune_conv_layer_l1(
    conv_layer: nn.Conv2d,
    prune_ratio: float = 0.25
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies structured L1-norm pruning mask to a convolutional layer.
    Returns: (pruned_weight_mask, keep_indices)
    """
    l1_norms = compute_layer_l1_norms(conv_layer)
    num_filters = l1_norms.size(0)
    num_to_prune = int(num_filters * prune_ratio)

    if num_to_prune == 0:
        keep_indices = torch.arange(num_filters)
        mask = torch.ones_like(conv_layer.weight.data)
        return mask, keep_indices

    # Sort filter indices by magnitude
    sorted_indices = torch.argsort(l1_norms)
    prune_indices = sorted_indices[:num_to_prune]
    keep_indices = sorted_indices[num_to_prune:]

    # Create binary mask
    mask = torch.ones_like(conv_layer.weight.data)
    mask[prune_indices, :, :, :] = 0.0

    # Apply mask in-place
    conv_layer.weight.data.mul_(mask)

    return mask, keep_indices


def apply_structured_pruning_to_model(
    model: nn.Module,
    prune_ratio: float = 0.25
) -> Dict[str, float]:
    """Applies structured filter pruning across all eligible Conv2d layers in the model."""
    total_filters = 0
    pruned_filters = 0

    for name, module in model.named_modules():
        # Prune standard 2D convolutions (excluding 1x1 depthwise groups)
        if isinstance(module, nn.Conv2d) and module.groups == 1 and module.kernel_size[0] > 1:
            num_filters = module.weight.size(0)
            if num_filters > 8:  # Keep minimal representation
                _, keep_idx = prune_conv_layer_l1(module, prune_ratio)
                pruned = num_filters - len(keep_idx)
                total_filters += num_filters
                pruned_filters += pruned

    effective_pruned_ratio = (pruned_filters / total_filters) if total_filters > 0 else 0.0
    return {
        "total_evaluated_filters": total_filters,
        "pruned_filters": pruned_filters,
        "effective_prune_ratio": effective_pruned_ratio
    }


if __name__ == "__main__":
    test_conv = nn.Conv2d(16, 32, 3, 1, 1)
    mask, keep = prune_conv_layer_l1(test_conv, prune_ratio=0.25)
    print(f"Original filters: 32 | Kept filters: {len(keep)} | Pruned: {32 - len(keep)}")
    print(f"Mask active fraction: {(mask.sum() / mask.numel()).item():.2f}")
