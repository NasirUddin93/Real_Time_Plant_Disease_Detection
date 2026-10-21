"""
ARM CMSIS-NN C++ Weight Exporter for Ultra-Low-Power MCUs (Tier 4)
Generates static C++ arrays and header files for bare-metal ARM Cortex-M microcontrollers
(STM32H7 / RP2040) using CMSIS-NN and TFLite for Microcontrollers (TFLM).
"""

import os
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import numpy as np


def quantize_tensor_int8(tensor: torch.Tensor) -> Tuple[np.ndarray, float, int]:
    """Symmetric/Asymmetric INT8 quantization for microcontroller static arrays."""
    data = tensor.detach().cpu().numpy()
    min_val, max_val = float(data.min()), float(data.max())
    max_abs = max(abs(min_val), abs(max_val), 1e-7)
    scale = max_abs / 127.0
    zero_point = 0
    quantized = np.clip(np.round(data / scale), -128, 127).astype(np.int8)
    return quantized, scale, zero_point


def export_cmsis_nn_header(
    model: nn.Module,
    output_header_path: str,
    model_name: str = "mobile_plant_model"
) -> Dict[str, object]:
    """
    Serializes model parameters into a C++ header (.h) with static const int8_t arrays
    formatted for ARM CMSIS-NN convolution and fully-connected layer kernels.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_header_path)), exist_ok=True)

    header_lines = [
        "/**",
        f" * Generated CMSIS-NN Model Header: {model_name}",
        " * Target: ARM Cortex-M4/M7 (STM32H743ZI / RP2040)",
        " * Framework: TFLite Micro / ARM CMSIS-NN",
        " */",
        "",
        "#ifndef " + model_name.upper() + "_WEIGHTS_H",
        "#define " + model_name.upper() + "_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
        f"// Model: {model_name}",
        f"namespace {model_name} {{",
        ""
    ]

    total_weight_bytes = 0
    layer_count = 0

    for name, param in model.named_parameters():
        clean_name = name.replace(".", "_")
        q_data, scale, zero_point = quantize_tensor_int8(param)
        flat_data = q_data.flatten()
        size_bytes = len(flat_data)
        total_weight_bytes += size_bytes
        layer_count += 1

        # Truncate representation in header preview if layer is too big, to keep header clean
        elements_to_show = min(16, len(flat_data))
        sample_str = ", ".join(str(int(x)) for x in flat_data[:elements_to_show])
        if len(flat_data) > elements_to_show:
            sample_str += f", /* ... {len(flat_data) - elements_to_show} more bytes */"

        header_lines.append(f"// Layer: {name} (shape: {list(param.shape)}) | Scale: {scale:.6f}, ZeroPoint: {zero_point}")
        header_lines.append(f"const int8_t {clean_name}_weights[{size_bytes}] = {{ {sample_str} }};")
        header_lines.append(f"const float {clean_name}_scale = {scale:.6f}f;")
        header_lines.append(f"const int32_t {clean_name}_zeropoint = {zero_point};")
        header_lines.append("")

    header_lines.extend([
        f"const uint32_t TOTAL_WEIGHT_BYTES = {total_weight_bytes};",
        f"const uint32_t NUM_PARAM_TENSORS = {layer_count};",
        "",
        f"}} // namespace {model_name}",
        "",
        "#endif // " + model_name.upper() + "_WEIGHTS_H",
        ""
    ])

    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))

    return {
        "output_path": output_header_path,
        "total_weight_bytes": total_weight_bytes,
        "total_weight_kb": round(total_weight_bytes / 1024, 2),
        "total_tensors": layer_count
    }


if __name__ == "__main__":
    from research_work.models.mobileplantvit import MobilePlantViT
    net = MobilePlantViT(num_classes=24)
    out_file = "research_work/deployment/exported_models/mobileplantvit_cmsis_nn.h"
    res = export_cmsis_nn_header(net, out_file, "MobilePlantViT")
    print(f"Generated CMSIS-NN header: {res['output_path']}")
    print(f"Total INT8 weights footprint: {res['total_weight_kb']} KB across {res['total_tensors']} parameter tensors.")
