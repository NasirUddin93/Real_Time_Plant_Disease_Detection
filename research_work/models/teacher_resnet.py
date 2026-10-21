"""
Teacher Architecture: ResNet-50 for Knowledge Distillation
High-capacity teacher model used to extract soft targets and dark knowledge
for compressing lightweight student models (MobileNetV3, ShuffleNetV2, MobilePlantViT).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class TeacherResNet50(nn.Module):
    """ResNet-50 teacher network for agricultural pathogen classification."""
    def __init__(self, num_classes: int = 24, pretrained: bool = False):
        super().__init__()
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = resnet50(weights=None)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features before final pooling."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x


if __name__ == "__main__":
    teacher = TeacherResNet50(num_classes=24, pretrained=False)
    p = sum(param.numel() for param in teacher.parameters() if param.requires_grad)
    print(f"Teacher ResNet-50 Parameters: {p:,} ({p/1e6:.2f}M)")
