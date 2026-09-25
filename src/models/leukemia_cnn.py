import torch
import torch.nn as nn

class LeukemiaCustomCNN(nn.Module):
    """
    Custom 5-layer Convolutional Neural Network with Batch Normalization
    and Dropout for Blood Smear Single-Cell Classification.
    """
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super(LeukemiaCustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112x112
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56x56
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def get_target_layer_for_cam(self):
        # Target the last conv layer for Grad-CAM
        return self.features[12] # nn.Conv2d(128, 256, ...)


def get_model(backbone: str = "mobilenet_v3_small", num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.3):
    """
    Factory function for instantiating the classification model.
    Supports lightweight MobileNetV3, ResNet, and Custom CNN.
    """
    backbone = backbone.lower()
    
    if backbone == "custom_cnn":
        return LeukemiaCustomCNN(num_classes=num_classes, dropout_rate=dropout_rate)
        
    try:
        import torchvision.models as models
    except ImportError:
        # Fallback to custom CNN if torchvision is not yet ready
        return LeukemiaCustomCNN(num_classes=num_classes, dropout_rate=dropout_rate)
        
    if backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        return model
        
    elif backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        return model
        
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        return model
        
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Choose from ['mobilenet_v3_small', 'resnet18', 'resnet50', 'custom_cnn']")
