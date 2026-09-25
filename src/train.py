import os
import sys
import json
import yaml
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np

from src.dataset import BloodSmearDataset, create_sample_dataset_if_empty
from src.models import get_model

def load_config(config_path: str = "config/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {
        "model": {"backbone": "mobilenet_v3_small", "num_classes": 2, "dropout_rate": 0.3, "weights_path": "weights/leukemia_detector_best.pth"},
        "training": {"batch_size": 8, "epochs": 10, "learning_rate": 0.0003, "weight_decay": 1e-4, "early_stopping_patience": 5},
        "data": {"image_size": 224, "samples_dir": "data/samples"}
    }

def get_transforms(img_size: int = 224):
    try:
        from torchvision import transforms
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return train_transform, val_transform
    except ImportError:
        # Graceful fallback without torchvision
        class BasicTransform:
            def __init__(self, size):
                self.size = size
            def __call__(self, img):
                im = img.resize((self.size, self.size))
                arr = np.array(im, dtype=np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                arr = (arr - mean) / std
                return torch.tensor(arr).permute(2, 0, 1).float()
        return BasicTransform(img_size), BasicTransform(img_size)

def train_pipeline(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    os.makedirs("weights", exist_ok=True)
    
    # 1. Ensure sample data exists
    data_dir = config["data"]["samples_dir"]
    create_sample_dataset_if_empty(data_dir, count_per_class=20)
    
    # 2. Dataset & Loaders
    train_transform, val_transform = get_transforms(config["data"]["image_size"])
    dataset = BloodSmearDataset(root_dir=data_dir, transform=train_transform)
    
    total_len = len(dataset)
    val_len = max(int(0.25 * total_len), 4)
    train_len = total_len - val_len
    
    train_subset, val_subset = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_subset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config["training"]["batch_size"], shuffle=False)
    
    # 3. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[*] Training on compute device: {device}")
    
    backbone = config["model"].get("backbone", "mobilenet_v3_small")
    model = get_model(
        backbone=backbone,
        num_classes=config["model"]["num_classes"],
        pretrained=True,
        dropout_rate=config["model"]["dropout_rate"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    
    # 4. Training Loop
    epochs = config["training"]["epochs"]
    best_val_acc = 0.0
    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    print(f"[*] Commencing training: {epochs} epochs | Backbone: {backbone}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()
            train_total += images.size(0)
            
        scheduler.step()
        epoch_train_loss = train_loss / max(train_total, 1)
        epoch_train_acc = train_correct / max(train_total, 1)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += images.size(0)
                
        epoch_val_loss = val_loss / max(val_total, 1)
        epoch_val_acc = val_correct / max(val_total, 1)
        
        history["epochs"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.1f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.1f}%")
        
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            weights_path = config["model"]["weights_path"]
            torch.save(model.state_dict(), weights_path)
            print(f" --> Best model checkpoint updated at {weights_path}")
            
    # Save training metrics history
    with open("weights/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"[+] Training completed successfully! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    return history

if __name__ == "__main__":
    train_pipeline()
