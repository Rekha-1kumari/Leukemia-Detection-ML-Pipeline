import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Generates visual explanations for decisions made by CNN-based models.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._find_target_layer()
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _find_target_layer(self):
        """Automatically find the last convolutional layer if not explicitly specified."""
        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError("No Conv2d layer found in the model for Grad-CAM.")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate CAM heatmap for a given input tensor and target class.
        input_tensor: shape (1, C, H, W)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        # Global Average Pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam) # Apply ReLU to focus only on positive contributions
        
        # Interpolate to input image dimensions
        _, _, h, w = input_tensor.shape
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        c_min, c_max = cam.min(), cam.max()
        if c_max - c_min > 1e-8:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()


def overlay_cam_on_image(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Overlay a 2D heatmap [0, 1] onto a PIL Image using a medical colormap (Jet/Plasma).
    Works with pure numpy and PIL without strictly requiring cv2.
    """
    try:
        import matplotlib.cm as cm
        colormap = cm.get_cmap("jet")
        heatmap = colormap(cam)[:, :, :3] # Shape: (H, W, 3), range [0, 1]
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap).resize(img_pil.size, resample=Image.BILINEAR)
    except Exception:
        # Fallback pseudo-color generator if matplotlib cm fails
        h, w = cam.shape
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap[:, :, 0] = (cam * 255).astype(np.uint8) # Red: high activation
        heatmap[:, :, 1] = ((1.0 - np.abs(cam - 0.5) * 2) * 255).astype(np.uint8) # Green
        heatmap[:, :, 2] = ((1.0 - cam) * 255).astype(np.uint8) # Blue
        heatmap_img = Image.fromarray(heatmap).resize(img_pil.size, resample=Image.BILINEAR)

    img_rgb = img_pil.convert("RGB")
    blended = Image.blend(img_rgb, heatmap_img, alpha=alpha)
    return blended
