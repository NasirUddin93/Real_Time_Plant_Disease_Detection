"""
Knowledge Distillation (KD) Engine
Transfers dark knowledge from heavy Teacher (ResNet-50) to lightweight Students
(MobileNetV3_CBAM, ShuffleNetV2_SE, MobilePlantViT).

Loss Formulation:
L_total = (1 - alpha) * L_CE(y, sigma(z_s)) + alpha * T^2 * KL(sigma(z_s/T), sigma(z_t/T))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DistillationLoss(nn.Module):
    """Combines hard cross-entropy ground-truth loss with soft teacher distillation loss."""
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Hard task loss
        hard_loss = self.ce_loss(student_logits, labels)

        # Soft distillation loss
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)

        # Combined loss
        total_loss = (1.0 - self.alpha) * hard_loss + self.alpha * soft_loss

        return total_loss, {
            "total_loss": float(total_loss.item()),
            "hard_loss": float(hard_loss.item()),
            "soft_loss": float(soft_loss.item())
        }


def train_distillation_step(
    student: nn.Module,
    teacher: nn.Module,
    criterion: DistillationLoss,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device
) -> Dict[str, float]:
    """Executes a single forward/backward distillation training step."""
    student.train()
    teacher.eval()

    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()

    with torch.no_grad():
        teacher_logits = teacher(images)

    student_logits = student(images)
    loss, loss_metrics = criterion(student_logits, teacher_logits, labels)
    loss.backward()
    optimizer.step()

    # Calculate student training accuracy
    preds = torch.argmax(student_logits, dim=1)
    acc = (preds == labels).float().mean().item() * 100.0
    loss_metrics["accuracy"] = acc

    return loss_metrics


if __name__ == "__main__":
    b_size, n_cls = 4, 24
    s_logits = torch.randn(b_size, n_cls, requires_grad=True)
    t_logits = torch.randn(b_size, n_cls)
    targets = torch.randint(0, n_cls, (b_size,))

    distill_criterion = DistillationLoss(temperature=4.0, alpha=0.7)
    l_tot, metrics = distill_criterion(s_logits, t_logits, targets)
    print("Distillation Loss Verification:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
