# Empirical Benchmarking & Hardware Profiling Report

## 1. Architectural Complexity & Host Inference Profile

| Architecture | Params (M) | GFLOPs | FP32 Size | INT8 Size | Host Latency (ms) | Host FPS | Peak RSS (MB) | Budget Satisfied (<1.0M, <0.5GF) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Teacher ResNet-50** | 23.557M | 8.174 | 89.86 MB | 22.46 MB | 208.62 ms | 4.8 | 123.9 MB | NO (Teacher) |
| **MobileNetV3-Small (CBAM)** | 0.551M | 0.08 | 2.1 MB | 0.53 MB | 9.04 ms | 110.7 | 142.0 MB | YES |
| **ShuffleNetV2 (SE)** | 0.242M | 0.063 | 0.92 MB | 0.23 MB | 9.82 ms | 101.8 | 144.1 MB | YES |
| **MobilePlantViT (Hybrid)** | 0.773M | 0.675 | 2.95 MB | 0.74 MB | 435.74 ms | 2.3 | 143.8 MB | NO (Teacher) |

## 2. Cross-Tier Edge Hardware Profiling Projection

| Architecture | Hardware Tier | Latency (ms) | Throughput (FPS) | Power (Watts) | Energy / Inference (mJ) | Real-Time (>= 25 FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Teacher ResNet-50 | Tier 1: Smartphone (Snapdragon/NNAPI) | 93.88 ms | 10.7 FPS | 2.1 W | 197.15 mJ | SUB-REALTIME |
| Teacher ResNet-50 | Tier 2: SBC (Raspberry Pi 4 ARM Cortex-A72) | 260.77 ms | 3.8 FPS | 3.2 W | 834.46 mJ | SUB-REALTIME |
| Teacher ResNet-50 | Tier 3: Edge TPU (RPi 4 + Google Coral USB) | 45.9 ms | 21.8 FPS | 1.9 W | 87.21 mJ | SUB-REALTIME |
| Teacher ResNet-50 | Tier 4: Ultra-Low-Power MCU (ARM Cortex-M7) | 792.76 ms | 1.3 FPS | 0.28 W | 221.97 mJ | SUB-REALTIME |
| MobileNetV3-Small (CBAM) | Tier 1: Smartphone (Snapdragon/NNAPI) | 4.07 ms | 245.7 FPS | 2.1 W | 8.55 mJ | PASS |
| MobileNetV3-Small (CBAM) | Tier 2: SBC (Raspberry Pi 4 ARM Cortex-A72) | 11.3 ms | 88.5 FPS | 3.2 W | 36.16 mJ | PASS |
| MobileNetV3-Small (CBAM) | Tier 3: Edge TPU (RPi 4 + Google Coral USB) | 1.99 ms | 502.5 FPS | 1.9 W | 3.78 mJ | PASS |
| MobileNetV3-Small (CBAM) | Tier 4: Ultra-Low-Power MCU (ARM Cortex-M7) | 34.35 ms | 29.1 FPS | 0.28 W | 9.62 mJ | PASS |
| ShuffleNetV2 (SE) | Tier 1: Smartphone (Snapdragon/NNAPI) | 4.42 ms | 226.2 FPS | 2.1 W | 9.28 mJ | PASS |
| ShuffleNetV2 (SE) | Tier 2: SBC (Raspberry Pi 4 ARM Cortex-A72) | 12.28 ms | 81.4 FPS | 3.2 W | 39.3 mJ | PASS |
| ShuffleNetV2 (SE) | Tier 3: Edge TPU (RPi 4 + Google Coral USB) | 2.16 ms | 463.0 FPS | 1.9 W | 4.1 mJ | PASS |
| ShuffleNetV2 (SE) | Tier 4: Ultra-Low-Power MCU (ARM Cortex-M7) | 37.32 ms | 26.8 FPS | 0.28 W | 10.45 mJ | PASS |
| MobilePlantViT (Hybrid) | Tier 1: Smartphone (Snapdragon/NNAPI) | 196.08 ms | 5.1 FPS | 2.1 W | 411.77 mJ | SUB-REALTIME |
| MobilePlantViT (Hybrid) | Tier 2: SBC (Raspberry Pi 4 ARM Cortex-A72) | 544.67 ms | 1.8 FPS | 3.2 W | 1742.94 mJ | SUB-REALTIME |
| MobilePlantViT (Hybrid) | Tier 3: Edge TPU (RPi 4 + Google Coral USB) | 95.86 ms | 10.4 FPS | 1.9 W | 182.13 mJ | SUB-REALTIME |
| MobilePlantViT (Hybrid) | Tier 4: Ultra-Low-Power MCU (ARM Cortex-M7) | 1655.81 ms | 0.6 FPS | 0.28 W | 463.63 mJ | SUB-REALTIME |