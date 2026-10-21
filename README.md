# EdgePlant AI: Lightweight Edge-Based Deep Learning Framework for Real-Time Plant Disease Detection

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Opset%2014-005CED.svg?logo=onnx&logoColor=white)](https://onnx.ai/)
[![ARM CMSIS-NN](https://img.shields.io/badge/ARM-CMSIS--NN%20Cortex--M-0091BD.svg?logo=arm&logoColor=white)](https://arm-software.github.io/CMSIS_5/NN/html/index.html)
[![Coral Edge TPU](https://img.shields.io/badge/Google-Coral%20Edge%20TPU-FBBC04.svg?logo=google&logoColor=white)](https://coral.ai/)
[![Research Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An ultra-lightweight, multi-crop, hardware-efficient Edge AI framework engineered to execute deep learning plant disease diagnostics directly on battery-constrained field hardware.**

[Key Innovations](#-key-architectural-innovations) •
[Multi-Crop Coverage](#-multi-crop-disease-coverage-24-classes) •
[Empirical Benchmarks](#-empirical-benchmarking-results) •
[Interactive Prototype](#-interactive-web-prototype) •
[Quick Start](#-quick-start--installation) •
[Research Team](#-research-team--academic-affiliation)

</div>

---

## 📌 Executive Summary & Scientific Motivation

Foliar plant pathogens induce severe agricultural crises across developing economies, accounting for up to **40% of global crop losses** (~$220 billion annually). While deep convolutional neural networks (CNNs) and vision transformers (ViTs) frequently achieve $\ge 95\%$ accuracy on clean laboratory datasets, their practical deployment in rural fields is hindered by:

1. **The Laboratory-to-Field Performance Gap ($\Delta_{\text{Field}}$):** Models trained on uniform-backdrop benchmarks (e.g. standard PlantVillage) suffer substantial accuracy collapse when exposed to direct sunlight glare, moving wind blur, shadows, overlapping leaves, and complex soil backgrounds.
2. **Cloud Latency & Rural Connectivity Limits:** Remote farm plots often have absent or unstable cellular connectivity; uploading high-resolution imagery to cloud servers introduces latency ($>3$ seconds) and bandwidth expenses.
3. **Severe Edge Resource Constraints:** Deploying AI on handheld field diagnostic scanners, single-board computers, or microcontrollers requires strict adherence to tight computational budgets:

$$\text{Latency} \le 30\text{ ms} \quad|\quad \text{Throughput} \ge 25\text{ FPS} \quad|\quad \text{Parameters} < 1.0\text{M} \quad|\quad \text{Complexity} < 0.5\text{ GFLOPs} \quad|\quad \text{Power} < 3\text{ W}$$

**EdgePlant AI** bridges this gap through a unified multi-crop pipeline featuring **synthetic environmental field augmentation**, **multi-stage model compression** (Teacher-Student Knowledge Distillation, Structured $\ell_1$-Norm Filter Pruning, PTQ INT8 Quantization), **cross-tier hardware compilation** (ONNX Runtime, TFLite Edge TPU, CMSIS-NN), and **Explainable AI (Grad-CAM/Grad-CAM++)**.

---

## 🌿 Multi-Crop Disease Coverage (24 Classes)

The framework provides unified multi-crop foliar classification across **5 economically vital crop species** covering **24 agricultural classes**:

```
Data Architecture: 24 Diagnostic Classes Across 5 Crops
 ├── 🍅 Tomato (Solanum lycopersicum) [10 Classes]
 │    ├── Bacterial Spot
 │    ├── Early Blight
 │    ├── Late Blight
 │    ├── Leaf Mold
 │    ├── Septoria Leaf Spot
 │    ├── Spider Mites (Two-Spotted)
 │    ├── Target Spot
 │    ├── Yellow Leaf Curl Virus (TYLCV)
 │    ├── Tomato Mosaic Virus (ToMV)
 │    └── Healthy
 ├── 🌾 Rice (Oryza sativa) [5 Classes]
 │    ├── Bacterial Leaf Blight
 │    ├── Brown Spot
 │    ├── Leaf Smut
 │    ├── Rice Blast
 │    └── Healthy
 ├── 🍎 Apple (Malus domestica) [4 Classes]
 │    ├── Apple Scab
 │    ├── Black Rot
 │    ├── Cedar Apple Rust
 │    └── Healthy
 ├── 🥔 Potato (Solanum tuberosum) [3 Classes]
 │    ├── Early Blight
 │    ├── Late Blight
 │    └── Healthy
 └── 🍆 Eggplant / Brinjal (Solanum melongena) [3 Classes]
      ├── Bacterial Wilt
      ├── Little Leaf Disease
      └── Healthy
```

---

## ⚡ Key Architectural Innovations

### 1. Synthetic Environmental Perturbation Pipeline (`FieldAugmentationPipeline`)
Directly combats field performance degradation by mathematically simulating unpredictable outdoor conditions:
- **Solar Glare / Specular Flare:** Generates high-intensity non-linear radial gradient light masks simulating equatorial midday sun.
- **Dynamic Canopy Shadows:** Randomized polygonal darkness overlays ($\alpha \in [0.35, 0.65]$) mimicking leaf shadows and overhead cloud cover.
- **Wind Motion Blur:** Directional blur kernels ($9\times 9$ to $15\times 15$, randomized orientation $\theta \in [0^\circ, 180^\circ]$) simulating leaf flutter.
- **Foliage Occlusion / Cutout:** Rectangular cutouts ($10\%\text{--}30\%$) mimicking overlapping leaves, neighboring weeds, or farmer fingers.
- **Background Clutter Injection:** Synthetic earth tones and dried straw textures to discourage background memorization.

### 2. Multi-Stage Model Compression Pipeline
- **Heavy Oracle Teacher:** `TeacherResNet50` (23.56M parameters, 8.17 GFLOPs) extracts rich diagnostic feature representations.
- **Ultra-Lightweight Backbones:**
  - **`MobileNetV3_CBAM`**: Hard-swish depthwise separable convolutions combined with Channel and Spatial Convolutional Block Attention Modules (**0.551M parameters**, **0.080 GFLOPs**).
  - **`ShuffleNetV2_SE`**: Channel split, channel shuffle, and Squeeze-and-Excitation attention (**0.242M parameters**, **0.063 GFLOPs**).
  - **`MobilePlantViT`**: Hybrid CNN-Transformer architecture combining depthwise convolution with Multi-Head Self-Attention (**0.773M parameters**).
- **Knowledge Distillation (KD):** Soft-logit KL-divergence loss combined with cross-entropy:
  $$\mathcal{L}_{\text{total}} = (1 - \alpha)\mathcal{L}_{\text{CE}}(y, \sigma(z_s)) + \alpha T^2 \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{z_s}{T}\right), \sigma\left(\frac{z_t}{T}\right)\right) \quad (T=4.0, \alpha=0.7)$$
- **Structured $\ell_1$-Norm Channel Pruning:** Measures filter importance $\|\mathcal{W}_j\|_1$ and prunes $20\%\text{--}40\%$ of lowest-ranking Conv2d channels in-place without requiring sparse tensor runtime support.
- **Post-Training Quantization (PTQ INT8):** Calibrated quantization reducing model storage footprint by $4\times$ down to **0.23 MB – 0.74 MB**.

### 3. Cross-Tier Hardware Compilation & Deployment
- **ONNX Runtime Export (`export_onnx.py`):** Standard ONNX opset 14 with dynamic batch size support and numerical parity validation.
- **Bare-Metal MCU C++ Header Generator (`export_cmsis_nn.py`):** Quantizes weights and scales into static `int8_t` C/C++ arrays compatible with ARM CMSIS-NN kernels (`arm_convolve_s8`, `arm_fully_connected_s8`) for STM32H7 and RP2040 microcontrollers.

### 4. Explainable AI & Perturbation Stability (`xai_gradcam.py`)
- Integrated **Grad-CAM** and **Grad-CAM++** visual interpretability modules highlighting symptomatic lesions.
- **Perturbation Stability Score:** Quantifies whether models genuinely focus on pathognomonic symptoms (e.g. concentric circles in Early Blight) rather than background dirt or lighting artifacts.

---

## 📊 Empirical Benchmarking Results

Inference latency profiled over **1,000 continuous iterations** (with warm-up), measuring Mean, Median (P50), P95, P99 latency, real-time FPS throughput, and Resident Set Size (RSS) RAM:

### Model Complexity & Footprint

| Architecture | Role | Parameters | GFLOPs | FP32 Size | INT8 Size | Host Latency | Host FPS | Peak RAM | Budget Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Teacher ResNet-50** | Oracle | 23.56M | 8.17 GF | 89.86 MB | 22.46 MB | 208.62 ms | 4.8 FPS | 123.9 MB | Oracle Baseline |
| **MobileNetV3-Small (CBAM)** | Student | **0.551M** | **0.080 GF** | **2.10 MB** | **0.53 MB** | **9.04 ms** | **110.7 FPS** | 142.0 MB | **PASSED (<1.0M, <0.5GF)** |
| **ShuffleNetV2 (SE)** | Ultra-Light | **0.242M** | **0.063 GF** | **0.92 MB** | **0.23 MB** | **9.82 ms** | **101.8 FPS** | 144.1 MB | **PASSED (<0.5M, <0.1GF)** |
| **MobilePlantViT (Hybrid)** | CNN-ViT | **0.773M** | 0.675 GF | **2.95 MB** | **0.74 MB** | 435.74 ms | 2.3 FPS | 143.8 MB | **PASSED (<1.0M Params)** |

### Cross-Tier Edge Hardware Profiling Projection

| Architecture | Deployment Target | Latency | Throughput | Power | Energy / Inference | Real-Time Goal ($\ge 25\text{ FPS}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3 (CBAM)** | **Tier 1: Smartphone (Snapdragon / NNAPI)** | **4.07 ms** | **245.7 FPS** | 2.10 W | 8.55 mJ | **EXCEEDED ($10\times$)** |
| **MobileNetV3 (CBAM)** | **Tier 2: SBC (Raspberry Pi 4 ARM Cortex-A72)** | **11.30 ms** | **88.5 FPS** | 3.20 W | 36.16 mJ | **EXCEEDED ($3.5\times$)** |
| **MobileNetV3 (CBAM)** | **Tier 3: Edge TPU (Google Coral USB)** | **1.99 ms** | **502.5 FPS** | 1.90 W | **3.78 mJ** | **SUPERIOR ($20\times$)** |
| **MobileNetV3 (CBAM)** | **Tier 4: MCU (ARM Cortex-M7 @ 480 MHz)** | **34.35 ms** | **29.1 FPS** | **0.28 W** | 9.62 mJ | **PASSED ($\ge 25\text{ FPS}$)** |
| **ShuffleNetV2 (SE)** | **Tier 1: Smartphone (Snapdragon / NNAPI)** | **4.42 ms** | **226.2 FPS** | 2.10 W | 9.28 mJ | **EXCEEDED ($9\times$)** |
| **ShuffleNetV2 (SE)** | **Tier 2: SBC (Raspberry Pi 4 ARM Cortex-A72)** | **12.28 ms** | **81.4 FPS** | 3.20 W | 39.30 mJ | **EXCEEDED ($3.2\times$)** |
| **ShuffleNetV2 (SE)** | **Tier 3: Edge TPU (Google Coral USB)** | **2.16 ms** | **463.0 FPS** | 1.90 W | **4.10 mJ** | **SUPERIOR ($18.5\times$)** |
| **ShuffleNetV2 (SE)** | **Tier 4: MCU (ARM Cortex-M7 @ 480 MHz)** | **37.32 ms** | **26.8 FPS** | **0.28 W** | 10.45 mJ | **PASSED ($\ge 25\text{ FPS}$)** |

---

## 🖥️ Interactive Web Prototype

The repository includes a responsive edge diagnostic kiosk and mobile simulator located in [`prototype/`](file:///home/nasir-uddin/Data/Real_Time_Plant_Disease_Detection/prototype/):

- **Procedural Foliage & Lesion Canvas:** Synthesizes realistic foliar lesions across all 24 disease classes on demand.
- **Live Environmental Distortion Sliders:** Dynamically test solar glare, canopy shade, wind motion blur, and leaf occlusion in real time.
- **Live Grad-CAM Heatmap Overlay:** Visual diagnostic attention heatmaps layered directly over leaf canvases with adjustable transparency.
- **Cross-Tier Hardware Telemetry Console:** Switch between Smartphone, Raspberry Pi 4, Google Coral TPU, and Cortex-M MCU to observe real-time latency dials, FPS counters, and energy meters.
- **Interactive Pareto Trade-off Explorer:** Accuracy vs. Latency vs. Power consumption analysis.

To view the prototype, open [`prototype/index.html`](file:///home/nasir-uddin/Data/Real_Time_Plant_Disease_Detection/prototype/index.html) in any modern web browser or start a local server:

```bash
cd prototype
python3 -m http.server 8080
# Open http://localhost:8080 in your browser
```

---

## 📁 Repository Directory Structure

```
Real_Time_Plant_Disease_Detection/
├── .gitignore                          # Git exclusion rules for ML, Python, OS, and large assets
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation & master overview
├── Project_Proposal_With_Chapters.pdf  # Formal thesis proposal & chapter outlines
│
├── research_work/                      # Core Deep Learning Research Pipeline
│   ├── data/
│   │   ├── dataset.py                  # 24-class loader, stratified splits, procedural generator
│   │   └── augmentation.py             # Physical outdoor environmental perturbation pipeline
│   ├── models/
│   │   ├── teacher_resnet.py           # ResNet-50 oracle teacher backbone
│   │   ├── mobilenetv3_cbam.py         # MobileNetV3-Small with CBAM attention (<1.0M params)
│   │   ├── shufflenet_se.py            # ShuffleNetV2 with SE attention (0.24M params)
│   │   └── mobileplantvit.py           # Hybrid CNN-Transformer backbone (0.77M params)
│   ├── compression/
│   │   ├── distillation.py             # Knowledge Distillation engine (KL + CE loss)
│   │   ├── pruning.py                  # Structured L1-norm filter channel pruning
│   │   ├── quantization.py             # Calibrated Post-Training Quantization (PTQ INT8)
│   │   └── train_pipeline.py           # Unified training, evaluation & metrics logging
│   ├── deployment/
│   │   ├── export_onnx.py              # PyTorch to ONNX opset 14 with parity validation
│   │   └── export_cmsis_nn.py          # Bare-metal C++ header export for Cortex-M MCUs
│   └── benchmarking/
│       ├── benchmark_suite.py          # Automated latency, throughput, RAM & power suite
│       ├── xai_gradcam.py              # Grad-CAM / Grad-CAM++ with stability scoring
│       └── results/
│           ├── BENCHMARK_REPORT.md     # Detailed markdown benchmarking report
│           └── benchmark_metrics.json  # Raw metric arrays and tier telemetry logs
│
├── prototype/                          # Interactive Edge AI Kiosk & Simulator
│   ├── index.html                      # Semantic UI with hardware gauges & canvas
│   ├── css/
│   │   └── style.css                   # Responsive dark-theme design system
│   ├── js/
│   │   └── app.js                      # Canvas rendering, perturbation logic & telemetry
│   └── assets/                         # SVG icons and visual assets
│
├── plan/
│   └── RESEARCH_PLAN.md                # 6-Month master research roadmap & RQs
├── overview/
│   └── COMPLETED_TASKS_WALKTHROUGH.md  # Detailed implementation progress report
├── book_Writing/                       # Thesis book chapters drafting workspace
└── related_papers/                     # Reference papers & literature matrix
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites & Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/<your-username>/Real_Time_Plant_Disease_Detection.git
cd Real_Time_Plant_Disease_Detection

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Verify Models & Forward Pass

Test all student architectures and parameter budgets:

```bash
# Test MobileNetV3-Small + CBAM
python3 -m research_work.models.mobilenetv3_cbam

# Test ShuffleNetV2 + SE
python3 -m research_work.models.shufflenet_se

# Test MobilePlantViT
python3 -m research_work.models.mobileplantvit
```

### 3. Run Benchmark Suite

Execute the latency, FPS throughput, and memory profiling suite:

```bash
python3 -m research_work.benchmarking.benchmark_suite
```

### 4. Run Explainable AI (Grad-CAM)

Generate lesion diagnostic heatmaps and verify perturbation stability:

```bash
python3 -m research_work.benchmarking.xai_gradcam
```

### 5. Export to Edge Hardware Runtimes

Export PyTorch weights to ONNX or ARM Cortex-M C++ headers:

```bash
# Export to ONNX format
python3 -m research_work.deployment.export_onnx

# Export to CMSIS-NN C++ header arrays
python3 -m research_work.deployment.export_cmsis_nn
```

---

## 👥 Research Team & Academic Affiliation

### Institution
**Department of Computer Science and Engineering**  
Faculty of Science and Engineering  
**City University**, Dhaka-1340, Bangladesh

### Research Investigators
- **Sheikh Saad** — *Student ID: 0272320005101281*
- **Sabiha Akon Maria** — *Student ID: 0272310005101027*
- **Md Al Mamun** — *Student ID: 0272320005101117*

### Faculty Supervision
- **Mst. Shikha Moni** — *Supervisor*
- **Sharmin Akter** — *Co-Supervisor / Member*

---

## 📄 Citation

If you utilize this framework, model backbones, or benchmark data in your research, please cite:

```bibtex
@misc{edgeplantai2026,
  title={Lightweight Edge-Based Deep Learning Framework for Real-Time Plant Disease Detection},
  author={Saad, Sheikh and Khatun, Sumaiya and Maria, Sabiha Akon and Mamun, Md Al},
  supervisor={Moni, Mst. Shikha and Akter, Sharmin},
  institution={Department of Computer Science and Engineering, City University, Dhaka, Bangladesh},
  year={2026},
  howpublished={\url{https://github.com/<your-username>/Real_Time_Plant_Disease_Detection}}
}
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — free for educational, academic, and non-commercial agricultural research use.
