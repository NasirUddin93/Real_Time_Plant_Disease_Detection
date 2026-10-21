"""
ONNX Runtime Export & Verification Suite
Converts PyTorch models to standard Open Neural Network Exchange (ONNX) format.
Validates graph integrity, operator compatibility, and verifies numerical outputs
against ONNX Runtime inference engine.
"""

import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 14,
    dynamic_batch: bool = True
) -> Dict[str, object]:
    """
    Exports a PyTorch model to ONNX format with optional dynamic batch size.
    Validates output parity against PyTorch forward pass.
    """
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dummy_input = torch.randn(*input_size, requires_grad=False)

    # PyTorch baseline forward pass
    with torch.no_grad():
        torch_out = model(dummy_input)

    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}} if dynamic_batch else None

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    res = {
        "output_path": output_path,
        "file_size_mb": round(file_size_mb, 3),
        "status": "Exported",
        "verified_ort": False,
        "max_discrepancy": 0.0
    }

    # Verify with onnx and onnxruntime if available
    if ONNX_AVAILABLE:
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            res["status"] = "Graph Verified"

            # Check with ONNX Runtime
            ort_session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            ort_outs = ort_session.run(None, ort_inputs)

            # Measure max difference between PyTorch and ONNX Runtime
            diff = abs(torch_out.cpu().numpy() - ort_outs[0]).max()
            res["verified_ort"] = True
            res["max_discrepancy"] = float(diff)
        except Exception as e:
            res["status"] = f"Verification warning: {str(e)}"

    return res


if __name__ == "__main__":
    from research_work.models.mobilenetv3_cbam import MobileNetV3_CBAM
    from research_work.models.shufflenet_se import ShuffleNetV2_SE
    from research_work.models.mobileplantvit import MobilePlantViT

    out_dir = "research_work/deployment/exported_models"
    os.makedirs(out_dir, exist_ok=True)

    models = [
        ("MobileNetV3_CBAM", MobileNetV3_CBAM(num_classes=24)),
        ("ShuffleNetV2_SE", ShuffleNetV2_SE(num_classes=24)),
        ("MobilePlantViT", MobilePlantViT(num_classes=24))
    ]

    for name, m in models:
        path = os.path.join(out_dir, f"{name}.onnx")
        info = export_to_onnx(m, path)
        print(f"Exported {name} -> {path} ({info['file_size_mb']} MB) | Status: {info['status']}")
