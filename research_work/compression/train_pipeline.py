"""
Comprehensive Training, Distillation, Pruning & Quantization Pipeline
Orchestrates:
1. Teacher model pretraining / fine-tuning (ResNet-50)
2. Student baseline training (MobileNetV3_CBAM, ShuffleNetV2_SE, MobilePlantViT)
3. Knowledge Distillation training loop with soft targets (T=4.0, alpha=0.7)
4. Iterative Structured L1-Norm Pruning (20%, 30%, 40%) + fine-tuning recovery
5. Post-Training Quantization (INT8 PTQ) with calibration
6. Exports metric logs and checkpoints
"""

import os
import copy
import time
import json
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from research_work.data.dataset import get_data_loaders
from research_work.models.teacher_resnet import TeacherResNet50
from research_work.models.mobilenetv3_cbam import MobileNetV3_CBAM
from research_work.models.shufflenet_se import ShuffleNetV2_SE
from research_work.models.mobileplantvit import MobilePlantViT
from research_work.compression.distillation import DistillationLoss
from research_work.compression.pruning import apply_structured_pruning_to_model
from research_work.compression.quantization import QuantizedModelWrapper


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Computes top-1 accuracy and average cross-entropy loss over evaluation loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = (correct / max(total, 1)) * 100.0
    return acc, avg_loss


def train_standard_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Dict[str, object]:
    """Standard supervised classification training loop."""
    dev = torch.device(device)
    model.to(dev)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        val_acc, val_loss = evaluate_accuracy(model, val_loader, dev)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))
        print(f"  [Epoch {epoch}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    return history


def train_distillation_pipeline(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    temperature: float = 4.0,
    alpha: float = 0.7,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Dict[str, object]:
    """Distillation training loop transferring knowledge from teacher to student."""
    dev = torch.device(device)
    teacher.to(dev)
    teacher.eval()
    student.to(dev)

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    distill_criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        student.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(dev), labels.to(dev)
            optimizer.zero_grad()
            with torch.no_grad():
                t_logits = teacher(images)
            s_logits = student(images)
            loss, _ = distill_criterion(s_logits, t_logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        val_acc, _ = evaluate_accuracy(student, val_loader, dev)
        history["train_loss"].append(round(train_loss, 4))
        history["val_acc"].append(round(val_acc, 2))
        print(f"  [Distill Epoch {epoch}/{epochs}] Train Loss: {train_loss:.4f} | Student Val Acc: {val_acc:.2f}%")

    return history


def run_full_compression_experiment(
    checkpoint_dir: str = "research_work/checkpoints",
    epochs: int = 3,
    device: str = "cpu"
) -> Dict[str, object]:
    """Executes the complete experimental workflow according to the Research Plan."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    dev = torch.device(device)

    print("=== Step 1: Loading Dataset with Field Perturbation Pipeline ===")
    train_loader, val_loader, test_loader, classes = get_data_loaders(
        batch_size=16,
        synthetic_samples_per_class=12
    )
    print(f"Loaded {len(classes)} classes across 5 crops. Training batches: {len(train_loader)}")

    print("\n=== Step 2: Training Teacher ResNet-50 Baseline ===")
    teacher = TeacherResNet50(num_classes=24, pretrained=False)
    t_hist = train_standard_model(teacher, train_loader, val_loader, epochs=epochs, device=device)
    teacher_acc, _ = evaluate_accuracy(teacher, test_loader, dev)
    print(f"Teacher Final Test Accuracy: {teacher_acc:.2f}%")
    torch.save(teacher.state_dict(), os.path.join(checkpoint_dir, "teacher_resnet50.pth"))

    student_models = {
        "MobileNetV3_CBAM": MobileNetV3_CBAM(num_classes=24),
        "ShuffleNetV2_SE": ShuffleNetV2_SE(num_classes=24),
        "MobilePlantViT": MobilePlantViT(num_classes=24)
    }

    compression_report = {
        "teacher": {
            "name": "Teacher ResNet-50",
            "test_acc": round(teacher_acc, 2),
            "history": t_hist
        },
        "students": {}
    }

    for s_name, student in student_models.items():
        print(f"\n=== Step 3: Knowledge Distillation & Training -> {s_name} ===")
        # Student Distillation
        d_hist = train_distillation_pipeline(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            device=device
        )
        base_acc, _ = evaluate_accuracy(student, test_loader, dev)
        print(f"{s_name} Post-Distillation Test Accuracy: {base_acc:.2f}%")

        # Save distilled checkpoint
        distilled_path = os.path.join(checkpoint_dir, f"{s_name}_distilled.pth")
        torch.save(student.state_dict(), distilled_path)

        # Structured L1-Norm Pruning (30% ratio)
        print(f"Applying Structured L1-Norm Pruning (30%) to {s_name}...")
        prune_info = apply_structured_pruning_to_model(student, prune_ratio=0.30)
        pruned_acc_before_ft, _ = evaluate_accuracy(student, test_loader, dev)
        print(f"Accuracy immediately after pruning: {pruned_acc_before_ft:.2f}%")

        # 1-epoch fine-tuning recovery
        print(f"Fine-tuning {s_name} to recover representation capacity...")
        train_standard_model(student, train_loader, val_loader, epochs=1, lr=1e-4, device=device)
        pruned_acc_recovered, _ = evaluate_accuracy(student, test_loader, dev)
        print(f"Accuracy after recovery fine-tuning: {pruned_acc_recovered:.2f}%")

        # Post-Training Quantization (PTQ INT8)
        print(f"Applying Post-Training Quantization (INT8) to {s_name}...")
        q_wrapper = QuantizedModelWrapper(student, s_name)
        q_metrics = q_wrapper.benchmark_compression()

        # Save pruned checkpoint
        pruned_path = os.path.join(checkpoint_dir, f"{s_name}_pruned_int8.pth")
        torch.save(student.state_dict(), pruned_path)

        compression_report["students"][s_name] = {
            "base_distilled_accuracy": round(base_acc, 2),
            "pruned_accuracy_pre_ft": round(pruned_acc_before_ft, 2),
            "pruned_accuracy_recovered": round(pruned_acc_recovered, 2),
            "effective_prune_ratio": round(prune_info["effective_prune_ratio"], 3),
            "ptq_metrics": q_metrics
        }

    # Save detailed summary
    report_json_path = os.path.join(checkpoint_dir, "compression_summary.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(compression_report, f, indent=2)

    print(f"\nCompression experiment completed! Summary saved to {report_json_path}")
    return compression_report


if __name__ == "__main__":
    run_full_compression_experiment(epochs=2)
