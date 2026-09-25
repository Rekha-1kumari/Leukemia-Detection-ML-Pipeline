import os
import sys
import io
import base64
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.models import get_model, GradCAM, overlay_cam_on_image
from src.morphology import analyze_cell_morphology

class LeukemiaPredictor:
    """
    Dual-Evidence Cytological Inference Engine:
    Couples Deep Convolutional Neural Network Feature Embeddings with
    Rigorous Computational Morphometry (N:C Area Ratio & Nuclear Saliency)
    to eliminate random/flickering decisions and enforce clinical concordance.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "model": {"backbone": "mobilenet_v3_small", "num_classes": 2, "weights_path": "weights/leukemia_detector_best.pth", "dropout_rate": 0.3},
                "data": {"image_size": 224, "classes": {0: "Normal", 1: "Acute Lymphoblastic Leukemia (ALL)"}}
            }
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.img_size = self.config["data"]["image_size"]
        self.classes = {0: "Normal", 1: "Acute Lymphoblastic Leukemia (ALL)"}
        
        # Initialize model architecture
        backbone = self.config["model"].get("backbone", "mobilenet_v3_small")
        self.model = get_model(
            backbone=backbone,
            num_classes=self.config["model"]["num_classes"],
            pretrained=False,
            dropout_rate=self.config["model"]["dropout_rate"]
        ).to(self.device)
        
        weights_path = self.config["model"]["weights_path"]
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[+] Loaded model weights from {weights_path}")
            except Exception as e:
                print(f"[!] Warning: Could not load weights ({e}). Running with initialized model.")
        else:
            print(f"[!] Weights not found at {weights_path}.")
            
        self.model.eval()
        
        # Initialize Grad-CAM
        try:
            self.cam_extractor = GradCAM(self.model)
        except Exception as e:
            print(f"[!] Grad-CAM initialization note: {e}")
            self.cam_extractor = None

    def preprocess_image(self, img_pil: Image.Image) -> torch.Tensor:
        im = img_pil.convert("RGB").resize((self.img_size, self.img_size))
        arr = np.array(im, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict(self, img_pil: Image.Image) -> dict:
        """
        Executes multi-modal consensus analysis:
        1. Deep Learning Forward Pass
        2. Biological Morphometry (OpenCV N:C Ratio)
        3. Dual-Evidence Consensus Calibration
        4. Grad-CAM Explainable AI Heatmap
        """
        input_tensor = self.preprocess_image(img_pil)
        
        # 1. Forward Pass (Deep CNN)
        with torch.no_grad():
            logits = self.model(input_tensor)
            raw_probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            p_dl_all = float(raw_probs[1])
            p_dl_normal = float(raw_probs[0])
            
        # 2. Computational Morphology Analysis
        morphology = analyze_cell_morphology(img_pil)
        seg_pil = morphology.pop("segmentation_image")
        nc_ratio = float(morphology.get("nc_ratio", 0.50))

        # 3. Dual-Evidence Consensus Calibration:
        # In hematology: Normal mature lymphocytes have N:C ~ 0.40 - 0.55.
        # Malignant lymphoblasts have N:C >= 0.70 (high nuclear expansion).
        # We calculate morphometric probability p_morph_all via steep logistic sigmoid around N:C = 0.60
        sigmoid_val = 1.0 / (1.0 + np.exp(-16.0 * (nc_ratio - 0.60)))
        p_morph_all = float(np.clip(sigmoid_val, 0.01, 0.99))

        # Check for clinical discordance (e.g. deep CNN disagrees with extreme physical morphometry)
        is_discordant = False
        discordance_note = ""
        if nc_ratio >= 0.70 and p_dl_all < 0.50:
            is_discordant = True
            discordance_note = f"Discrepancy: Deep CNN predicted low probability ({p_dl_all*100:.1f}%) but biological N:C ratio is critically elevated ({nc_ratio:.3f}). Morphological blast criterion overrides negative prediction for patient safety."
            # Hematology clinical safety override: High N:C ratio blast morphology cannot be classified as benign normal
            p_fused_all = max(0.72, 0.40 * p_dl_all + 0.60 * p_morph_all)
        elif nc_ratio <= 0.45 and p_dl_all > 0.65:
            is_discordant = True
            discordance_note = f"Discrepancy: Deep CNN predicted high probability ({p_dl_all*100:.1f}%) but cell exhibits normal mature N:C ratio ({nc_ratio:.3f})."
            p_fused_all = 0.50 * p_dl_all + 0.50 * p_morph_all
        else:
            # Concordant fusion: 70% deep neural network features + 30% biological cytometry
            p_fused_all = 0.70 * p_dl_all + 0.30 * p_morph_all

        p_fused_all = float(np.clip(p_fused_all, 0.005, 0.995))
        p_fused_normal = 1.0 - p_fused_all

        # Determine final consensus prediction
        is_leukemia = (p_fused_all >= 0.50)
        pred_idx = 1 if is_leukemia else 0
        confidence = p_fused_all if is_leukemia else p_fused_normal
        label = self.classes[pred_idx]

        # 4. Grad-CAM Saliency Map
        cam_overlay_pil = None
        if self.cam_extractor is not None:
            try:
                cam = self.cam_extractor.generate(input_tensor, target_class=pred_idx)
                cam_overlay_pil = overlay_cam_on_image(img_pil.resize((self.img_size, self.img_size)), cam, alpha=0.55)
            except Exception as e:
                print(f"[!] Grad-CAM generation note: {e}")
                
        if cam_overlay_pil is None:
            cam_overlay_pil = img_pil.resize((self.img_size, self.img_size))

        # 5. Clinical Risk Tier Stratification
        if is_leukemia and (confidence >= 0.75 or nc_ratio >= 0.75):
            risk_tier = "CRITICAL / HIGH RISK"
            clinical_action = "High lymphoblast proliferation and expanded nuclear footprint identified. Immediate Flow Cytometry (CD10, CD19) and bone marrow examination recommended."
        elif is_leukemia or is_discordant:
            risk_tier = "MODERATE / SUSPICIOUS"
            clinical_action = "Atypical cellular features / elevated N:C ratio detected. Correlate with Complete Blood Count (CBC) and repeat peripheral smear under pathologist supervision."
        else:
            risk_tier = "LOW RISK / NORMAL"
            clinical_action = "Morphologically unremarkable mature lymphocytic profile. Normal physiological baseline."

        # 6. Base64 encoding for API & Web UI
        def pil_to_base64(im):
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "prediction": {
                "class_index": pred_idx,
                "label": label,
                "confidence": round(confidence, 4),
                "is_leukemia": is_leukemia,
                "probabilities": {
                    "Normal": round(p_fused_normal, 4),
                    "Acute Lymphoblastic Leukemia (ALL)": round(p_fused_all, 4)
                }
            },
            "morphology": {
                "nc_ratio": morphology["nc_ratio"],
                "nuclear_circularity": morphology["nuclear_circularity"],
                "nucleus_area_pixels": morphology["nucleus_area_pixels"],
                "cytoplasm_area_pixels": morphology["cytoplasm_area_pixels"],
                "total_cell_area_pixels": morphology["total_cell_area_pixels"],
                "morphology_status": morphology["morphology_status"],
                "risk_indication": morphology["risk_indication"]
            },
            "clinical_decision": {
                "risk_tier": risk_tier,
                "action": clinical_action,
                "is_discordant": is_discordant,
                "discordance_note": discordance_note
            },
            "images": {
                "original_base64": pil_to_base64(img_pil.resize((self.img_size, self.img_size))),
                "gradcam_base64": pil_to_base64(cam_overlay_pil),
                "segmentation_base64": pil_to_base64(seg_pil.resize((self.img_size, self.img_size)))
            },
            "_raw_pil": {
                "original": img_pil.resize((self.img_size, self.img_size)),
                "gradcam": cam_overlay_pil,
                "segmentation": seg_pil.resize((self.img_size, self.img_size))
            }
        }
