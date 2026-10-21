"""
Explainable AI (XAI) Engine: Grad-CAM & Grad-CAM++
Generates visual class activation heatmaps and computes Perturbation Stability Scores
to assess diagnostic fidelity and prevent reliance on background artifacts.
Compatible with standard CNNs (ResNet-50, MobileNetV3, ShuffleNetV2) and Hybrid ViT (MobilePlantViT).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Captures feature activation maps from target conv layer and weighs them
    by gradient with respect to predicted class logit.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Hook functions
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        use_gradcam_plus_plus: bool = False
    ) -> np.ndarray:
        """Generates normalized (0 to 1) 2D heatmap for input tensor."""
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]  # shape (C, H, W)
        activations = self.activations[0]  # shape (C, H, W)

        if use_gradcam_plus_plus:
            # Grad-CAM++ higher-order weighting
            grad_2 = gradients.pow(2)
            grad_3 = gradients.pow(3)
            sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True)
            alpha_denom = 2 * grad_2 + sum_activations * grad_3
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
            alphas = grad_2 / alpha_denom
            weights = torch.sum(alphas * F.relu(gradients), dim=(1, 2))
        else:
            # Standard Grad-CAM global average pooling of gradients
            weights = torch.mean(gradients, dim=(1, 2))

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam_max = torch.max(cam)
        if cam_max > 0:
            cam = cam / cam_max

        # Interpolate to input resolution
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy()


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_type: str = "jet"
) -> Image.Image:
    """Overlays the Grad-CAM activation heatmap onto the original RGB PIL image."""
    w, h = image.size
    # Generate simple pseudo-color RGB from normalized heatmap
    # Red: high activation, Green: mid activation, Blue: low activation
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.Resampling.BILINEAR)
    h_arr = np.array(heatmap_resized) / 255.0

    # Custom Jet-like color mapping without requiring heavy cv2
    r = np.clip(1.5 - np.abs(4.0 * h_arr - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h_arr - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h_arr - 1.0), 0.0, 1.0)
    color_map = np.stack([r, g, b], axis=-1)
    color_map = np.uint8(255 * color_map)

    heat_img = Image.fromarray(color_map).convert("RGBA")
    base_img = image.convert("RGBA")
    blended = Image.blend(base_img, heat_img, alpha=alpha).convert("RGB")
    return blended


def compute_perturbation_stability_score(
    model: nn.Module,
    input_tensor: torch.Tensor,
    heatmap: np.ndarray,
    target_class: int,
    mask_threshold: float = 0.6
) -> Dict[str, float]:
    """
    Computes Perturbation Stability Score (PSS).
    Systematically masks highest-activation lesion pixels and measures confidence drop.
    A faithful model must exhibit a significant monotonic drop when salient disease features are masked.
    """
    model.eval()
    with torch.no_grad():
        orig_logits = model(input_tensor)
        orig_prob = F.softmax(orig_logits, dim=1)[0, target_class].item()

        # Mask top salient pixels
        perturbed_tensor = input_tensor.clone()
        mask = torch.tensor(heatmap >= mask_threshold, device=input_tensor.device).unsqueeze(0).unsqueeze(0)
        # Replace salient region with image mean
        mean_val = input_tensor.mean()
        perturbed_tensor[mask.repeat(1, input_tensor.shape[1], 1, 1)] = mean_val

        perturbed_logits = model(perturbed_tensor)
        perturbed_prob = F.softmax(perturbed_logits, dim=1)[0, target_class].item()

    drop_absolute = max(0.0, orig_prob - perturbed_prob)
    drop_relative_pct = (drop_absolute / max(orig_prob, 1e-5)) * 100.0

    return {
        "original_confidence": round(orig_prob, 4),
        "perturbed_confidence": round(perturbed_prob, 4),
        "confidence_drop": round(drop_absolute, 4),
        "drop_percentage": round(drop_relative_pct, 2),
        "is_salient_faithful": drop_relative_pct >= 25.0  # Plan threshold: significant drop confirms faithfulness
    }


if __name__ == "__main__":
    from research_work.models.mobilenetv3_cbam import MobileNetV3_CBAM
    from research_work.data.augmentation import generate_synthetic_leaf

    net = MobileNetV3_CBAM(num_classes=24)
    # Choose final conv layer
    target_layer = net.head_conv[0]
    cam_engine = GradCAM(net, target_layer)

    leaf = generate_synthetic_leaf("Tomato", "Tomato___Early_blight")
    dummy_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)

    hm = cam_engine.generate_heatmap(dummy_tensor, target_class=1)
    pss = compute_perturbation_stability_score(net, dummy_tensor, hm, target_class=1)

    print("Grad-CAM Engine Initialized Successfully:")
    print(f"Heatmap shape: {hm.shape} | Range: [{hm.min():.2f}, {hm.max():.2f}]")
    print(f"Perturbation Stability Score metrics: {pss}")
