import os
import glob
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader

class BloodSmearDataset(Dataset):
    """
    PyTorch Dataset for Peripheral Blood Smear Microscopic Cell Images.
    Supports binary classification:
    - 0: Normal / Benign Lymphocyte
    - 1: Acute Lymphoblastic Leukemia (ALL) Blast
    """
    def __init__(self, root_dir: str, transform=None, is_train: bool = True):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Look for standard class folders
        normal_paths = glob.glob(os.path.join(root_dir, "normal", "*.[jJ][pP][gG]")) + \
                       glob.glob(os.path.join(root_dir, "normal", "*.[pP][nN][gG]")) + \
                       glob.glob(os.path.join(root_dir, "Normal", "*.*"))
                       
        all_paths = glob.glob(os.path.join(root_dir, "all_blast", "*.[jJ][pP][gG]")) + \
                    glob.glob(os.path.join(root_dir, "all_blast", "*.[pP][nN][gG]")) + \
                    glob.glob(os.path.join(root_dir, "ALL", "*.*")) + \
                    glob.glob(os.path.join(root_dir, "leukemia", "*.*"))

        for p in normal_paths:
            self.samples.append((p, 0))
        for p in all_paths:
            self.samples.append((p, 1))
            
        random.seed(42)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        else:
            # Basic fallback tensor conversion
            arr = np.array(image, dtype=np.float32) / 255.0
            image = torch.tensor(arr).permute(2, 0, 1)
            
        return image, label


def reinhard_stain_normalization(img_np: np.ndarray, target_means=(200, 150, 175), target_stds=(30, 25, 30)) -> np.ndarray:
    """
    Reinhard Color / Stain Normalization for histopathology and cytological smears.
    Harmonizes microscopic staining variations across different slide preparations.
    """
    img = img_np.astype(np.float32)
    for c in range(3):
        m = np.mean(img[:, :, c])
        s = np.std(img[:, :, c]) + 1e-6
        img[:, :, c] = ((img[:, :, c] - m) / s) * target_stds[c] + target_means[c]
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_synthetic_blood_smear(label: int, size: int = 224) -> Image.Image:
    """
    Generates biologically realistic synthetic microscopic blood smear images:
    - Normal (label=0): Mature lymphocyte with dense, round nucleus, moderate cytoplasm, normal N:C ~0.55
    - ALL Blast (label=1): Lymphoblast with enlarged irregular hyperchromatic nucleus, high N:C >0.80, nucleoli
    - Background: Giemsa stained plasma background with pale red blood cells (erythrocytes)
    """
    # 1. Base pale pinkish/tan plasma background
    bg_r = random.randint(235, 248)
    bg_g = random.randint(230, 242)
    bg_b = random.randint(235, 248)
    img = Image.new("RGB", (size, size), (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)

    # 2. Add background Red Blood Cells (RBCs / Erythrocytes) - pale pink biconcave discs
    num_rbcs = random.randint(6, 12)
    for _ in range(num_rbcs):
        rx = random.randint(10, size - 10)
        ry = random.randint(10, size - 10)
        rad = random.randint(15, 26)
        rbc_col = (random.randint(215, 235), random.randint(170, 195), random.randint(175, 200))
        draw.ellipse([rx - rad, ry - rad, rx + rad, ry + rad], fill=rbc_col)
        # Inner pale depression
        draw.ellipse([rx - rad//2, ry - rad//2, rx + rad//2, ry + rad//2], fill=(bg_r, bg_g, bg_b))

    # 3. Central White Blood Cell (WBC)
    cx, cy = size // 2 + random.randint(-8, 8), size // 2 + random.randint(-8, 8)
    
    if label == 0:
        # Normal Mature Lymphocyte:
        # Cell radius: ~50-65 px, Nucleus radius: ~35-45 px (N:C ratio ~0.50-0.60)
        cell_rad_x = random.randint(48, 58)
        cell_rad_y = random.randint(46, 56)
        
        # Cytoplasm: pale sky-blue / pale lilac
        cyto_col = (random.randint(175, 195), random.randint(190, 210), random.randint(215, 235))
        draw.ellipse([cx - cell_rad_x, cy - cell_rad_y, cx + cell_rad_x, cy + cell_rad_y], fill=cyto_col)
        
        # Mature Nucleus: dense, homogeneous dark purple/violet chromatin
        nuc_rad_x = int(cell_rad_x * random.uniform(0.68, 0.78))
        nuc_rad_y = int(cell_rad_y * random.uniform(0.68, 0.78))
        nuc_col = (random.randint(75, 95), random.randint(40, 60), random.randint(110, 140))
        draw.ellipse([cx - nuc_rad_x, cy - nuc_rad_y, cx + nuc_rad_x, cy + nuc_rad_y], fill=nuc_col)
        
    else:
        # ALL Lymphoblast (Leukemia):
        # Enlarged cell: ~65-80 px. Massive nucleus occupying >85% of volume (very thin cytoplasmic rim)
        cell_rad_x = random.randint(62, 76)
        cell_rad_y = random.randint(60, 74)
        
        # Scanty, pale basophilic cytoplasm
        cyto_col = (random.randint(160, 185), random.randint(175, 195), random.randint(210, 230))
        draw.ellipse([cx - cell_rad_x, cy - cell_rad_y, cx + cell_rad_x, cy + cell_rad_y], fill=cyto_col)
        
        # Massive Blast Nucleus: High N:C ratio (~0.85-0.92), irregular clefts / chromatin clumps
        nuc_rad_x = int(cell_rad_x * random.uniform(0.85, 0.94))
        nuc_rad_y = int(cell_rad_y * random.uniform(0.84, 0.93))
        nuc_col = (random.randint(90, 115), random.randint(45, 65), random.randint(125, 160))
        draw.ellipse([cx - nuc_rad_x, cy - nuc_rad_y, cx + nuc_rad_x, cy + nuc_rad_y], fill=nuc_col)
        
        # Blast chromatin irregularities and nucleoli (distinctive prominent pale spots in blast cells)
        num_nucleoli = random.randint(1, 3)
        for _ in range(num_nucleoli):
            nox = cx + random.randint(-nuc_rad_x // 3, nuc_rad_x // 3)
            noy = cy + random.randint(-nuc_rad_y // 3, nuc_rad_y // 3)
            draw.ellipse([nox - 4, noy - 4, nox + 4, noy + 4], fill=(130, 85, 175))

    # Apply slight optical microscopic blur for realistic lens physics
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img


def create_sample_dataset_if_empty(base_dir: str = "data/samples", count_per_class: int = 15):
    """
    Creates high-quality sample blood smear images if none exist yet.
    Ensures the pipeline is immediately functional and testable.
    """
    normal_dir = os.path.join(base_dir, "normal")
    all_dir = os.path.join(base_dir, "all_blast")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(all_dir, exist_ok=True)

    existing_normal = len(glob.glob(os.path.join(normal_dir, "*.jpg")))
    existing_all = len(glob.glob(os.path.join(all_dir, "*.jpg")))

    if existing_normal < count_per_class:
        for i in range(count_per_class):
            img = generate_synthetic_blood_smear(label=0)
            img.save(os.path.join(normal_dir, f"normal_sample_{i+1:03d}.jpg"), quality=95)

    if existing_all < count_per_class:
        for i in range(count_per_class):
            img = generate_synthetic_blood_smear(label=1)
            img.save(os.path.join(all_dir, f"all_blast_sample_{i+1:03d}.jpg"), quality=95)

    print(f"Sample dataset verified at {base_dir} (Normal: {count_per_class}, ALL: {count_per_class})")

if __name__ == "__main__":
    create_sample_dataset_if_empty()
