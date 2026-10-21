# Comprehensive Research Plan: Lightweight Edge-Based Deep Learning Framework for Real-Time Plant Disease Detection

---

**Project Title:** Lightweight Edge-Based Deep Learning Framework for Real-Time Plant Disease Detection  
**Institution:** Department of Computer Science and Engineering, Faculty of Science and Engineering, City University, Dhaka-1340, Bangladesh  
**Research Group:**  
- **Investigators:** Sheikh Saad (0272320005101281), Sumaiya Khatun (0272320005101175), Sabiha Akon Maria (0272310005101027), Md Al Mamun (0272320005101117)  
- **Supervisors:** Mst. Shikha Moni (Supervisor), Sharmin Akter (Co-Supervisor / Member)  
**Implementation Horizon:** September 2026 – February 2027 (6 Months)  
**Document Status:** Master Operational & Experimental Research Plan  

---

## 1. Executive Summary & Project Overview

Agricultural sustainability in developing nations faces severe economic and food security crises caused by foliar plant pathogens, responsible for up to 40% of yield losses worldwide (~$220 billion annually). While deep convolutional neural networks (CNNs) and vision transformers (ViTs) achieve diagnostic accuracies exceeding 95% on benchmark datasets, their deployment in field environments is obstructed by high computational complexity, massive parameter counts, and rigid dependence on cloud infrastructure. Cloud-based diagnosis fails in rural agrarian zones due to absent or unstable 3G/4G connectivity, high latency (>3 seconds per image), and prohibitive cellular transmission costs.

This research formulates an **end-to-end, multi-crop, hardware-efficient edge AI framework** that executes deep learning inference directly on battery-powered edge devices with low latency ($\le 30\text{ ms}$), high throughput ($\ge 25\text{ FPS}$), compact memory footprints ($<1.0\text{M}$ parameters, $<0.5\text{ GFLOPs}$), and energy-conservative power consumption ($<3\text{ W}$). The framework bridges the acute **Laboratory-to-Field Performance Gap** by incorporating specialized environmental augmentation, structured pruning, Knowledge Distillation (KD), Post-Training Quantization (PTQ INT8), and Explainable AI (XAI with Grad-CAM). The target framework will be compiled and evaluated across four distinct hardware tiers: Android smartphones, Raspberry Pi 4 Model B, Google Coral USB Edge TPU, and ARM Cortex-M microcontrollers.

---

## 2. Research Problem Formalization & Hypotheses

### 2.1 Problem Statements
1. **The Laboratory-to-Field Generalization Dilemma:** Current edge-oriented models trained primarily on clean, laboratory-conditioned benchmarks (e.g., PlantVillage with uniform gray/black backdrops and static illumination) experience drastic performance collapse when exposed to uncontrolled field environments characterized by direct sunlight glare, moving wind blur, shadows, leaf overlaps, and weed-laden soil backgrounds.
2. **Incomplete Edge-Hardware Benchmarking:** Existing literature prioritizes top-1/top-5 classification accuracy while routinely omitting critical physical deployment metrics: inference latency jitter, Frames Per Second (FPS), working RAM consumption, thermal throttling dynamics, and battery draw in watts.
3. **Monocultural Over-Specialization:** Existing lightweight architectures are predominantly customized for single crop types (e.g., tomato-only or rice-only), lacking a unified backbone capable of lightweight multi-crop detection without multi-megabyte parameter expansion.

### 2.2 Research Questions (RQs)
- **RQ1 (Compression & Accuracy Trade-off):** How can deep convolutional and hybrid transformer backbones be pruned, distilled, and quantized to achieve $<1.0\text{M}$ parameters and $<0.5\text{ GFLOPs}$ while preserving $\ge 95\%$ multi-class diagnostic accuracy on challenging leaf pathogen datasets?
- **RQ2 (Cross-Tier Hardware-Runtime Co-Optimization):** Which mathematical pairings of lightweight model backbones, compilation runtimes (TFLite, ONNX Runtime, OpenVINO, Edge TPU Compiler, ARM CMSIS-NN), and physical edge compute targets yield the Pareto-optimal trade-off between latency, throughput, memory footprint, and thermal/power consumption?
- **RQ3 (Field Robustness & Transparent Trust):** How robust is the compressed edge pipeline when tested against adverse real-world agricultural conditions (specular reflection, wind motion, complex background clutter), and how effectively can visual Explainable AI (Grad-CAM/LIME) provide faithful, real-time diagnostic explanations to non-expert agriculturalists?

### 2.3 Research Hypotheses
- **$\mathcal{H}_1$:** Combining structured L1-norm filter pruning with teacher-student Knowledge Distillation (using ResNet-50 as teacher) prior to INT8 Post-Training Quantization will constrain accuracy loss to $<1.5\%$ compared to the full-precision FP32 baseline, while reducing model size by $>75\%$.
- **$\mathcal{H}_2$:** Hardware-accelerated inference via dedicated tensor processing (Google Coral USB TPU using Edge TPU compilation) will achieve $\ge 30\text{ FPS}$ at $<2.5\text{ W}$ power consumption, yielding a $4\times$ improvement in energy-delay product over general-purpose edge CPUs (Raspberry Pi 4 ARM Cortex-A72).
- **$\mathcal{H}_3$:** Integrating synthetic environmental perturbations (illumination shift, specular glare, motion blur, and background clutter) into the training pipeline will reduce the laboratory-to-field accuracy degradation gap from $>20\%$ down to $<6\%$.

---

## 3. Literature Synthesis & Benchmark Baseline

A critical analysis of 15 pivotal studies in edge AI and plant pathology establishes the foundational baseline:

| Reference | Target Architecture | Compression / Optimization | Deployment Environment | Key Reported Metrics | Primary Research Gap Addressed in this Study |
|---|---|---|---|---|---|
| **Khan et al. (2023)** | MobileNetV3-Small | PTQ (FP32 $\to$ INT8) | Mobile/Edge simulator | 99.50% Acc, 1.5M $\to$ 0.93M Params | Tested strictly on laboratory PlantVillage; no field evaluation. |
| **Wei et al. (2022)** | DenseNet-121 | Downsampling, OpenVINO | CPU / GPU / VPU | 96.4% Acc across runtimes | Architecture remains too dense ($>7\text{M}$ params) for low-power microcontrollers. |
| **Sun et al. (2021)** | MEAN-SSD (Apple-Inception + SSD) | Lightweight Conv design | Mobile hardware | 83.12% mAP, 12.53 FPS | Insufficient frame rate for real-time video stream; limited to 5 apple diseases. |
| **Chew Jing Xan et al. (2024)** | FD-MobileNet | Quantization + micro-tuning | ARM Cortex-M MCU | 98.44% Acc on rice | Limited to single crop; lacks real-time localization. |
| **Majeed et al. (2024)** | MobileNetV2, EfficientNet-B0 | Hardware benchmarking | Raspberry Pi 4, Coral TPU, Jetson | 98.25% Acc; Coral cost/FPS 0.14 | Comparative benchmarking without unified multi-crop architecture design. |
| **Hoang et al. (2026)** | MobilePlantViT, ShuffleNetV2 | CPU benchmarking + XAI | Desktop CPU edge simulation | ~99.4% Acc; XAI perturbation evaluated | Lacks deployment on physical edge accelerators and MCU targets. |
| **Ashwinkumar et al. (2022)** | OMNCNN (MobileNet) | Feature extraction tuning | CNN workstation | 98.7% Acc | Complex background robustness was not quantified. |
| **Chen et al. (2022)** | MS-DNet | Depthwise separable conv + SE | Mobile-oriented setup | 98.32% Acc | Cross-device thermal and battery power benchmarking was omitted. |
| **Luo et al. (2023)** | LiteCNN | Knowledge distillation | ZYNQ Z7-Lite 7020 FPGA | 95.71% Acc, 0.071 s/frame, 2.41 W | Custom FPGA implementation lacks portability to mass-market mobile devices. |
| **De Vita et al. (2021)** | Custom DNN | Dynamic K-Means compression | Edge environment | 95.0% Acc, 85.2% memory drop | Real-time FPS throughput and power consumption were not measured. |
| **Cui et al. (2026)** | LMRNet | Lightweight CNN (0.90M params) | Android smartphones | 92.41% field Acc, 22.47 FPS | Significant performance drops observed under backlight, occlusion, and glare. |
| **Tonmoy et al. (2025)** | MobilePlantViT | Hybrid CNN-ViT (0.69M params) | Research simulation | High cross-dataset score | Broad physical edge hardware profiling remains unperformed. |
| **Zhao et al. (2026)** | HGS-YOLO | HGNetV2 + HS-FPN + CA + PTQ | D-Robotics RDK X5 | 93.6% mAP@50, 40 FPS, 9.8 W | High power draw (9.8W) limits prolonged battery operation; needs multi-site data. |
| **Remya et al. (2026)** | AMSBFN | Bilinear fusion + attention | GPU workstation | 97.47% Acc | Tested strictly on PlantVillage tomato images; lacks multi-crop edge porting. |
| **Al Kafi et al. (2026)** | LeafAI (LR + MobileNetV3) | Two-stage inference + Grad-CAM | Entry-level laptop | 4,548 s $\to$ 1,010 s inference time | Needs physical multi-tier edge hardware and broader crop validation. |

---

## 4. Methodological Architecture & Phase Breakdown

The research follows a tightly coupled four-phase technical workflow spanning data engineering, neural architecture design, edge compilation, and empirical benchmarking.

```
+---------------------------------------------------------------------------------------------------+
| Phase 1: Data Engineering & Synthetic Field Augmentation Pipeline                                 |
| - PlantVillage (Lab) + Real-world Field Collections (Tomato, Rice, Apple, Potato/Eggplant)        |
| - Albumentations Pipeline: Solar Glare, Shadows, Wind Motion Blur, Occlusion, Background Noise    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| Phase 2: Lightweight Model Architecture Design & Multi-Stage Compression                          |
| - Backbones: MobileNetV3-Small, ShuffleNetV2, MobilePlantViT, HGS-YOLO / YOLO-nano               |
| - Attention Enhancements: Squeeze-and-Excitation (SE), CBAM                                       |
| - Knowledge Distillation (Teacher: ResNet-50 / ConvNeXt -> Student: Lightweight Backbones)        |
| - Structured L1-Norm Filter Pruning -> Iterative Fine-Tuning -> INT8 PTQ Quantization             |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| Phase 3: Hardware Compilation, Runtime Optimization & Cross-Tier Porting                         |
| - Export Formats: TFLite (INT8/FP16), ONNX Runtime, OpenVINO IR, Edge TPU Binary, CMSIS-NN       |
| - Tier 1: Android Smartphone (Snapdragon/MediaTek ARM Cortex-A + GPU/NPU via NNAPI)              |
| - Tier 2: Single-Board Computer (Raspberry Pi 4 Model B, 64-bit Linux)                            |
| - Tier 3: Edge AI Accelerator (Raspberry Pi 4 + Google Coral USB Edge TPU via libedgetpu)        |
| - Tier 4: Ultra-Low-Power MCU (ARM Cortex-M4/M7, STM32H7 / RP2040 via TFLite Micro / CMSIS-NN)   |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| Phase 4: Systematic Empirical Benchmarking, Explainability (XAI) & In-Situ Field Validation       |
| - AI Metrics: Accuracy, Precision, Recall, F1-Score, mAP@50                                      |
| - Hardware Metrics: Latency (P50/P95/P99), FPS, RAM Peak RSS, Power (Watts), Thermal Profile     |
| - Explainable AI: Grad-CAM, Grad-CAM++, Perturbation Stability Score                             |
| - In-Situ Agricultural Field Validation in Savar/Khagan, Bangladesh                               |
+---------------------------------------------------------------------------------------------------+
```

---

### 4.1 Phase 1: Dataset Curation and Field Data Augmentation

#### 4.1.1 Composite Dataset Architecture
To overcome laboratory bias while maintaining sufficient training scale, the project constructs a balanced composite dataset combining controlled and uncontrolled sources:
1. **Benchmark Lab Dataset:** Selected classes from PlantVillage (Mohanty et al., 2016) representing healthy and diseased specimens across primary target crops:
   - **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy.
   - **Rice:** Bacterial Leaf Blight, Brown Spot, Leaf Smut, Blast, Healthy.
   - **Apple:** Apple Scab, Black Rot, Cedar Apple Rust, Healthy.
   - **Local Bangladeshi Crops (Potato & Eggplant):** Early Blight, Late Blight, Bacterial Wilt, Little Leaf, Healthy.
2. **Real-World Field Validation Dataset:** In-situ image acquisition conducted in rural agricultural fields in Savar, Birulia, and Khagan (Dhaka, Bangladesh) using handheld smartphone sensors (12MP–48MP) under natural daylight, cloudy conditions, morning dew, and shadowed foliage.

#### 4.1.2 Synthetic Environmental Perturbation Pipeline
All training batches pass through an environmental corruption pipeline using `Albumentations` to simulate outdoor optical distortions:
- **Solar Glare / Specular Flare:** Randomized high-intensity circular and radial gradient masks mimicking overexposure.
- **Dynamic Shadows:** Irregular polygonal darkness masks ($I_{\text{out}} = I_{\text{in}} \times \alpha$, where $\alpha \in [0.3, 0.6]$) simulating overhead tree canopy or leaf shadows.
- **Wind Motion Blur:** Directional motion blur kernels (kernel size $7 \times 7$ to $15 \times 15$, angle $\theta \in [0^\circ, 180^\circ]$) simulating moving leaves under windy field conditions.
- **Leaf Occlusion & Cutout:** Random erasing of $10\%\text{--}30\%$ bounding box regions with black, mean-pixel, or natural background texture.
- **Background Clutter Injection:** Alpha-blending leaves onto segmented soil, dried straw, weeds, and agricultural plastic mulch backgrounds.

---

### 4.2 Phase 2: Lightweight Model Architecture Design & Compression Pipeline

#### 4.2.1 Candidate Architectures & Architectural Enhancements
The study explores both classification backbones and object detection backbones:
- **Backbone 1: MobileNetV3-Small (Enhanced):** Depthwise separable convolutions with hard-swish activation, augmented with custom Convolutional Block Attention Modules (CBAM) inserted before the final pooling layer.
- **Backbone 2: ShuffleNetV2 (1.0x / 0.5x):** Channel split and channel shuffle operations minimizing memory access cost (MAC), equipped with Squeeze-and-Excitation (SE) squeeze ratio $r=8$.
- **Backbone 3: MobilePlantViT:** Hybrid CNN-Transformer architecture combining depthwise convolution blocks for local inductive bias with lightweight MobileViT transformer blocks for global context modeling, targeted at $\le 0.70\text{M}$ parameters.
- **Backbone 4: Real-Time Detector (HGS-YOLO / YOLOv11-nano):** For continuous multi-leaf localization and bounding-box symptom detection on edge video streams, featuring High-Performance GPU/Edge-friendly Backbone (HGNetV2) and Hybrid Slicing Feature Pyramid Network (HS-FPN).

#### 4.2.2 Multi-Stage Compression Pipeline
To compress models to $<1.0\text{M}$ parameters and $<0.5\text{ GFLOPs}$ without degrading accuracy below $95\%$:

1. **Knowledge Distillation (KD):**
   - **Teacher Model:** High-capacity ResNet-50 / ConvNeXt-Tiny pre-trained on ImageNet and fine-tuned on the composite agricultural dataset (achieving $>98.5\%$ accuracy).
   - **Loss Function:**
     $$\mathcal{L}_{\text{total}} = (1 - \alpha) \mathcal{L}_{\text{CE}}(y, \sigma(z_s)) + \alpha T^2 \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{z_s}{T}\right), \sigma\left(\frac{z_t}{T}\right)\right)$$
     where $T \in [3.0, 5.0]$ is the distillation temperature, $\alpha = 0.7$ balances ground truth and teacher dark knowledge, $z_s$ and $z_t$ are student and teacher logits.

2. **Structured L1-Norm Filter Pruning:**
   - Filters in standard and depthwise convolutional layers are ranked according to their $\ell_1$-norm:
     $$||\mathcal{W}_j||_1 = \sum_{c=1}^{C} \sum_{k_1=1}^{K} \sum_{k_2=1}^{K} |\mathcal{W}_j(c, k_1, k_2)|$$
   - Redundant filters falling below an adaptive threshold ($20\%\text{--}40\%$ pruning ratio) are removed, eliminating computational channels entirely (avoiding sparse matrix overheads on edge CPUs).
   - Iterative fine-tuning for 15 epochs after each pruning pass restores representation capacity.

3. **Post-Training Quantization (PTQ) & Quantization-Aware Training (QAT):**
   - Full 32-bit floating-point weights ($W_{\text{FP32}}$) and activation tensors ($X_{\text{FP32}}$) are mapped to 8-bit signed integers ($\text{INT8}$):
     $$q = \text{clamp}\left(\left\lfloor \frac{r}{S} \right\rceil + Z, -128, 127\right)$$
     where $S$ is the scale factor and $Z$ is the zero-point offset derived from a representative calibration dataset of 500 outdoor field images.
   - If PTQ INT8 degradation exceeds $1.0\%$, Quantization-Aware Training (QAT) with fake-quantization nodes will be executed for 10 epochs.

---

### 4.3 Phase 3: Hardware Compilation, Runtime Optimization & Cross-Tier Porting

The compressed models are compiled into execution graphs tailored for four physical edge tiers:

| Hardware Tier | Physical Platform | Compute Engine / Cores | Memory & Thermal Constraints | Target Runtime & Compilation Pipeline |
|---|---|---|---|---|
| **Tier 1: Smartphone** | Samsung Galaxy / Xiaomi Android device | Octa-core ARM Cortex-A7x/A5x + Adreno/Mali GPU + Hexagon/NPU | 4GB–8GB RAM, passive cooling, battery operated | **TensorFlow Lite (.tflite)** with NNAPI / GPU Delegate; Android APK wrapper with camera preview. |
| **Tier 2: Single-Board Computer** | Raspberry Pi 4 Model B | Quad-core ARM Cortex-A72 @ 1.5 GHz | 4GB/8GB LPDDR4, fan/heatsink cooling, 5V/3A power | **ONNX Runtime (.onnx)** with ARM NEON execution provider & **TFLite runtime** using XNNPACK engine. |
| **Tier 3: Edge AI Accelerator** | Raspberry Pi 4 + Google Coral USB Accelerator | Google Edge TPU coprocessor (4 TOPS @ 2W, USB 3.0) | Dedicated ASIC memory, strict INT8-only requirement | **Edge TPU Compiler (`edgetpu_compiler`)** generating `_edgetpu.tflite` fully offloaded to TPU. |
| **Tier 4: Ultra-Low-Power MCU** | ARM Cortex-M4/M7 (STM32H7 / RP2040) | 133–480 MHz RISC core, no OS (bare-metal / FreeRTOS) | $\le 1\text{MB}$ Flash, 264KB–1MB SRAM, $<500\text{ mW}$ | **TFLite for Microcontrollers (TFLM)** & **ARM CMSIS-NN** library generating static C++ arrays. |

---

### 4.4 Phase 4: Empirical Benchmarking, Explainable AI (XAI) & Field Validation

#### 4.4.1 Comprehensive Evaluation Metrics
The framework will be evaluated across two synchronized vectors:
1. **Predictive Diagnostic AI Metrics:**
   - Multi-class Accuracy, Macro/Micro Precision, Recall, Macro F1-Score:
     $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
   - Detection Metric: Mean Average Precision at IoU 0.50 ($\text{mAP@50}$) and $\text{mAP@[0.5:0.95]}$.
   - Model Compactness: Total Parameter Count ($\le 1.0\text{M}$), Storage Size in Megabytes ($\le 5\text{MB}$ for INT8), Floating-Point Operations ($\le 0.5\text{ GFLOPs}$).
2. **Hardware Profiling Metrics:**
   - **Inference Latency:** Measured over 1,000 continuous iterations after 100 warm-up cycles; reported as Mean, Median (P50), 95th Percentile (P95), and 99th Percentile (P99) in milliseconds ($\text{ms}$).
   - **Throughput:** Frames Per Second ($\text{FPS} = \frac{1000}{\text{Latency}_{\text{ms}}}$). Real-time target: $\ge 25\text{ FPS}$.
   - **Memory Footprint:** Peak Resident Set Size (RSS RAM) logged via OS profiling tools (`valgrind`, `psutil`, or Android Profiler).
   - **Power & Energy Draw:** Real-time electrical power consumption ($P = V \times I$ in watts) logged via an external USB inline power meter / INA219 current sensor during continuous inference; Energy-per-inference ($E = P \times \text{Latency}$ in millijoules).
   - **Thermal Dynamics:** Core SoC temperature ($^\circ\text{C}$) monitored across 30 minutes of continuous inference to measure thermal throttling thresholds.

#### 4.4.2 Explainable AI (XAI) Protocol
To prevent false-positive reliance on background artifacts (e.g., soil, farmer hands, sky), visual interpretability is embedded into the diagnostic feedback:
- **Gradient-weighted Class Activation Mapping (Grad-CAM & Grad-CAM++):**
  $$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right), \quad \alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A^k_{i,j}}$$
  generating coarse localization heatmaps highlighting discriminative lesion regions.
- **Perturbation Stability Score:** Systematic masking of high-activation pixels to verify that prediction confidence drops monotonically, confirming the model relies on true symptomatic patterns rather than spurious noise.

#### 4.4.3 Field Validation Protocol
- An independent field test set of $\ge 1,000$ images acquired across agricultural plots in Savar/Birulia.
- Testing under three lighting conditions: Morning (soft diffuse light), Midday (harsh direct sun with intense shadows), and Overcast/Post-rain (specular water reflections on leaves).
- Calculation of the **Laboratory-to-Field Performance Gap ($\Delta_{\text{Field}}$)**:
  $$\Delta_{\text{Field}} = \text{Accuracy}_{\text{Lab}} - \text{Accuracy}_{\text{Field}}$$
  with a target threshold of $\Delta_{\text{Field}} \le 5.0\%$.

---

## 5. Month-by-Month Work Breakdown & Task Timetable

The six-month research execution plan (September 2026 – February 2027) is structured into distinct, actionable task packages:

```
+===================================================================================================+
| Task / Milestone                    | Sep 2026 | Oct 2026 | Nov 2026 | Dec 2026 | Jan 2027 | Feb 2027 |
+===================================================================================================+
| Task 1: Literature Review & Gaps    | [======] |          |          |          |          |          |
| Task 2: Dataset Curation & Augment  | [======] | [======] |          |          |          |          |
| Task 3: Model Design, Train, Compress|         | [======] | [======] | [======] |          |          |
| Task 4: Edge Porting & Optimization |          |          |          | [======] | [======] |          |
| Task 5: Benchmarking, XAI & Field   |          |          |          |          | [======] | [======] |
| Task 6: Thesis Docs, Paper & Defense|          |          |          |          | [======] | [======] |
+===================================================================================================+
```

### Detailed Monthly Action Items

#### Month 1: September 2026 — Literature Review, Problem Formulation & Dataset Acquisition
- [x] **Task 1.1:** Comprehensive review of 15 key publications; finalize state-of-the-art benchmark matrix.
- [ ] **Task 1.2:** Define mathematical problem formulations, research questions, and experimental hypotheses.
- [ ] **Task 2.1:** Download and sanitize benchmark datasets: PlantVillage (tomato, apple, potato), Rice Disease Dataset (Kaggle/UCI).
- [ ] **Task 2.2:** Conduct field image collection expedition 1 in Savar/Khagan agricultural plots (healthy and infected tomato, rice, and eggplant leaves).
- [ ] **Task 2.3:** Establish unified data directory schema, image resizing standards ($224 \times 224$ and $320 \times 320$), and train/val/test splits (70/15/15 stratified).

#### Month 2: October 2026 — Field Augmentation Pipeline & Baseline Model Implementation
- [ ] **Task 2.4:** Build and test the synthetic environmental perturbation pipeline (glare, shadow, blur, occlusion, clutter) using `Albumentations`.
- [ ] **Task 3.1:** Implement baseline models in PyTorch: ResNet-50 (Teacher), MobileNetV3-Small, ShuffleNetV2, and MobilePlantViT.
- [ ] **Task 3.2:** Train full-precision FP32 teacher and baseline student models; document baseline convergence, loss curves, and top-1 validation accuracies.
- [ ] **Task 3.3:** Set up baseline object detector (HGS-YOLO / YOLOv11-nano) for multi-lesion localization.

#### Month 3: November 2026 — Advanced Model Compression & Knowledge Distillation
- [ ] **Task 3.4:** Implement Knowledge Distillation (KD) training loop transferring representations from ResNet-50 teacher to MobileNetV3 and MobilePlantViT students.
- [ ] **Task 3.5:** Execute structured L1-norm filter pruning targeting $20\%$, $30\%$, and $40\%$ reduction in convolutional channels; run 15-epoch iterative fine-tuning recovery.
- [ ] **Task 3.6:** Analyze parameter scaling and computational complexity: verify that total parameters $< 1.0\text{M}$ and GFLOPs $< 0.5$.
- [ ] **Task 3.7:** Implement Post-Training Quantization (PTQ) to INT8 and evaluate calibration dataset sizes (100 vs. 500 images).

#### Month 4: December 2026 — Cross-Platform Edge Runtime Compilation & Porting
- [ ] **Task 4.1:** Export trained models from PyTorch to ONNX format; validate layer compatibility and operator support.
- [ ] **Task 4.2:** Convert ONNX models to TensorFlow Lite (`.tflite`) with full integer quantization (INT8 weights and activations).
- [ ] **Task 4.3:** Compile INT8 TFLite models using `edgetpu_compiler` for Google Coral USB Accelerator; resolve unsupported operator fallbacks.
- [ ] **Task 4.4:** Set up Raspberry Pi 4 Model B runtime environment (64-bit Raspberry Pi OS, TFLite runtime with XNNPACK, ONNX Runtime).
- [ ] **Task 4.5:** Initialize Android Studio project; integrate TFLite Android Support Library and CameraX API for live preview inference.
- [ ] **Task 4.6:** Initialize ARM Cortex-M bare-metal project using TFLite for Microcontrollers (TFLM) and CMSIS-NN kernel routines.

#### Month 5: January 2027 — Systematic Benchmarking, Explainable AI & Field Evaluation
- [ ] **Task 5.1:** Run automated latency and throughput test suite across Tier 1 (Smartphone), Tier 2 (RPi 4), Tier 3 (Coral TPU), and Tier 4 (MCU).
- [ ] **Task 5.2:** Collect hardware performance data: 1,000-run latency distribution (P50/P95/P99), FPS throughput, peak RSS RAM.
- [ ] **Task 5.3:** Connect physical power meter / INA219 sensor to measure live voltage, current, wattage, and energy-per-inference ($mJ$).
- [ ] **Task 5.4:** Implement Grad-CAM and Grad-CAM++ visualization module; compute perturbation stability scores across diverse disease classes.
- [ ] **Task 5.5:** Conduct field validation expedition 2: execute real-time on-device inference directly in Savar agricultural fields under direct sun, shade, and rain-glare conditions; record the laboratory-to-field performance drop ($\Delta_{\text{Field}}$).
- [ ] **Task 6.1:** Draft Chapter 1 (Introduction), Chapter 2 (Literature Review), and Chapter 3 (Methodology) of the thesis report.

#### Month 6: February 2027 — Final Synthesis, Documentation, Paper Submission & Defense
- [ ] **Task 5.6:** Finalize empirical comparison tables and trade-off Pareto curves (Accuracy vs. Latency vs. Power).
- [ ] **Task 6.2:** Draft Chapter 4 (Experimental Results & Hardware Benchmarking), Chapter 5 (Explainability & Field Validation), and Chapter 6 (Conclusion & Future Work).
- [ ] **Task 6.3:** Complete the end-to-end prototype demo (standalone Android application and standalone Raspberry Pi + Coral TPU kiosk).
- [ ] **Task 6.4:** Prepare research manuscript for submission to an indexed journal (e.g., *IEEE Access*, *Computers and Electronics in Agriculture*, or *Smart Agricultural Technology*).
- [ ] **Task 6.5:** Finalize thesis book formatting, supervisor review revisions, and defense slide deck preparation; conduct formal defense.

---

## 6. Experimental Setup, Hardware & Software Tooling Matrix

```
+---------------------------------------------------------------------------------------------------+
| DEVELOPMENT & TRAINING ENVIRONMENT                                                                |
| - Host Machine: Linux Ubuntu 22.04 / 24.04 LTS x86_64 Workstation                                  |
| - GPUs: NVIDIA RTX 3080/4090 or Google Colab / Kaggle Cloud GPU (CUDA 12.x, cuDNN 8.x)            |
| - Deep Learning Frameworks: PyTorch 2.x, torchvision, TensorFlow 2.x / Keras                      |
| - Optimization & Compression Toolkits: PyTorch Pruning API, Albumentations, ONNX, TFLite Converter|
| - Explainability: PyTorch-Grad-CAM, Captum, OpenCV-Python, Matplotlib                             |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| PHYSICAL TARGET EDGE PLATFORMS                                                                    |
| 1. Tier 1 (Smartphone): Android Phone (ARM64 Cortex-A CPU, Adreno GPU, Android 12+)               |
| 2. Tier 2 (SBC): Raspberry Pi 4 Model B (Quad-core Cortex-A72 @ 1.5GHz, 4GB RAM)                   |
| 3. Tier 3 (ASIC Accelerator): Raspberry Pi 4 + Google Coral USB Accelerator (Edge TPU, 4 TOPS)    |
| 4. Tier 4 (Microcontroller): STM32H743ZI (Cortex-M7 @ 480MHz) or Raspberry Pi Pico (RP2040)        |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| HARDWARE PROFILING & MEASUREMENT INSTRUMENTS                                                      |
| - Power Profiling: UM25C / TC66C USB Digital Multimeter / INA219 I2C Current Sensor + Arduino     |
| - Thermal Logging: `vcgencmd measure_temp` (RPi) / FLIR thermal probe / Android thermal sensor    |
| - Latency Profiling: Python `time.perf_counter_ns()`, Android `SystemClock.elapsedRealtimeNanos()`|
+---------------------------------------------------------------------------------------------------+
```

---

## 7. Risk Analysis, Technical Pitfalls & Mitigation Protocols

| Risk / Technical Bottleneck | Probability | Impact | Root Cause | Concrete Engineering Mitigation Protocol |
|---|---|---|---|---|
| **INT8 Quantization Accuracy Collapse** | Moderate | High | Dynamic range truncation in sensitive attention layers or depthwise convolutions. | 1. Implement Quantization-Aware Training (QAT) to allow weights to adapt during training.<br>2. Use Per-Channel Quantization instead of Per-Tensor Quantization.<br>3. Keep the first convolutional and final classification layers in FP16 / hybrid mode if supported. |
| **Edge TPU Operator Incompatibility** | High | High | Certain activation functions (e.g., hard-swish, GELU) or custom attention ops cannot be mapped to Coral TPU ASIC, causing fallback to slow CPU. | 1. Restrict activation functions to ReLU or ReLU6 in Edge TPU target backbones.<br>2. Run `edgetpu_compiler -s` to inspect per-layer mapping reports early.<br>3. Decouple attention into standard separable convolutions for the TPU branch. |
| **Field Accuracy Degradation ($\Delta_{\text{Field}} > 10\%$)** | High | High | Distribution shift caused by specular reflections, shadows, and camera white-balance differences. | 1. Heavily augment training data with synthetic lighting and shadow masks using `Albumentations`.<br>2. Integrate Color Constancy algorithms (Gray-World algorithm) in the on-device preprocessing pipeline. |
| **Microcontroller SRAM / Flash Overflow** | High | High | Peak activation buffer exceeds 264KB–1MB SRAM on ARM Cortex-M hardware. | 1. Enforce aggressive structural pruning on early feature layers.<br>2. Reduce input image resolution to $160 \times 160$ or $128 \times 128$ specifically for Tier 4 MCU targets.<br>3. Utilize CMSIS-NN memory-reusing arena buffers. |
| **Thermal Throttling on Edge SBC / Phone** | Moderate | Medium | Sustained continuous video stream processing triggers thermal capping, dropping FPS from 30 to $<10$. | 1. Implement frame-skipping / dynamic inference triggering (run inference only every 3rd frame or when motion detector detects a leaf).<br>2. Equip Raspberry Pi 4 with passive aluminum heatsink and active micro-fan. |

---

## 8. Team Roles & Task Allocation Matrix

Under the supervision of **Mst. Shikha Moni** (Supervisor) and **Sharmin Akter** (Co-Supervisor), team members are assigned clear, interlocking responsibilities:

```
                                  +---------------------------------------+
                                  |     Supervisors / Advisors            |
                                  | Mst. Shikha Moni & Sharmin Akter      |
                                  | - Research guidance & methodology     |
                                  | - Quality review & defense oversight  |
                                  +---------------------------------------+
                                                      |
                  +-----------------------------------+-----------------------------------+
                  |                                                                       |
+-----------------------------------+                                   +-----------------------------------+
| Sheikh Saad (Lead & System Arch)  |                                   | Sumaiya Khatun (Data & Field Eng) |
| - Project coordination & git lead |                                   | - Lab dataset integration         |
| - Lightweight architecture design |                                   | - Field data collection (Savar)   |
| - Distillation & Pruning pipeline |                                   | - Albumentations corruption pipe  |
| - Coral TPU & RPi compilation     |                                   | - Benchmark split validation      |
+-----------------------------------+                                   +-----------------------------------+
                  |                                                                       |
+-----------------------------------+                                   +-----------------------------------+
| Sabiha Akon Maria (Edge & Mobile) |                                   | Md Al Mamun (XAI & Benchmarking)  |
| - Android app development         |                                   | - Latency & FPS measurement suite |
| - CameraX & TFLite NNAPI runtime  |                                   | - Power meter & thermal profiling |
| - Microcontroller porting (MCU)   |                                   | - Grad-CAM / Grad-CAM++ pipeline  |
| - Cross-tier runtime verification |                                   | - Statistical results & tables    |
+-----------------------------------+                                   +-----------------------------------+
```

---

## 9. Deliverables, Artifacts & Publication Roadmap

### 9.1 Tangible Deliverables
1. **Lightweight Model Binaries:**
   - Pre-trained PyTorch weights (`.pth`), ONNX graphs (`.onnx`), and quantized TFLite models (`.tflite`, `< 5MB`).
   - Edge TPU compiled binary (`_edgetpu.tflite`) targeting Google Coral.
   - C++ header files with CMSIS-NN array weights for Cortex-M.
2. **Software Prototypes:**
   - **Android Mobile App:** APK supporting real-time camera feed disease classification with visual Grad-CAM overlays and offline operation.
   - **Raspberry Pi + Coral TPU Field Kiosk:** Python-based real-time video stream inspection tool with live FPS and confidence display.
3. **Datasets & Code Repository:**
   - Documented GitHub repository with reproducible training, pruning, and conversion scripts.
   - Savar Field Validation Image Dataset with bounding boxes and environmental metadata.
4. **Academic Documentation:**
   - Complete 6-chapter undergraduate thesis book.
   - Research paper manuscript prepared for submission to an SCIE / Scopus-indexed journal (e.g., *IEEE Access*, *Computers and Electronics in Agriculture*, *Frontiers in Plant Science*).

---

## 10. Summary Checklist for Immediate Next Steps

To initiate research operations immediately:
- [ ] **Step 1:** Set up the project directory structure under `research_work/` (scripts, data pipelines, model definitions, notebooks).
- [ ] **Step 2:** Configure the Python virtual environment with PyTorch 2.x, Albumentations, ONNX, and TFLite tools.
- [ ] **Step 3:** Download benchmark PlantVillage sub-datasets and initialize the dataset preprocessing script.
- [ ] **Step 4:** Plan and schedule the initial field image collection trip to Savar / Birulia.
