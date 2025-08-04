import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2

# --- 1. Updated Model with a More Advanced Head and Fine-Tuning Control ---

class FacePADModel(nn.Module):
    """
    An improved ResNet-based binary classifier for Face Presentation Attack Detection.
    
    This model includes:
    - A configurable backbone (e.g., 'resnet18', 'resnet50').
    - A more robust classifier head with BatchNorm and Dropout.
    - Methods to freeze/unfreeze the backbone for effective fine-tuning.
    """
    def __init__(self, backbone_name='resnet50', pretrained=True, dropout_rate=0.5):
        super(FacePADModel, self).__init__()
        
        # Load the specified pretrained backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            num_ftrs = self.backbone.fc.in_features
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            num_ftrs = self.backbone.fc.in_features
        else:
            raise NotImplementedError(f"Backbone '{backbone_name}' not implemented.")

        # Replace the final fully connected layer with a custom head
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 2)  # Binary classification (real vs. attack)
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all layers in the backbone except for the final classifier head."""
        print("Freezing backbone layers...")
        for name, param in self.backbone.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers in the model for full fine-tuning."""
        print("Unfreezing all layers...")
        for param in self.backbone.parameters():
            param.requires_grad = True


# --- 2. Grad-CAM Implementation (Largely unchanged, it's a standard algorithm) ---

class GradCAM:
    """
    Generates Grad-CAM visualizations to explain model predictions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval() # Ensure model is in eval mode
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Target for backprop
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][class_idx] = 1
        
        # Backpropagate
        self.model.zero_grad()
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Pool gradients and compute weights
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels of the activation map
        activations = self.activations[0]
        for i in range(pooled_gradients.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Generate heatmap
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU
        heatmap /= (np.max(heatmap) + 1e-8) # Normalize
        
        return heatmap, class_idx


# --- 3. Improved Overlay Function for Visualization ---

def overlay_heatmap(image_tensor, heatmap, alpha=0.4):
    """
    Overlays a Grad-CAM heatmap on the original image.
    
    Args:
        image_tensor (torch.Tensor): The original input tensor (C, H, W) normalized.
        heatmap (np.ndarray): The generated heatmap (H, W).
        alpha (float): The blending factor for the heatmap.
    
    Returns:
        np.ndarray: The blended image in uint8 format.
    """
    # Denormalize and convert tensor to a displayable CV2 image (BGR format)
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.uint8(img * 255)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize heatmap and apply colormap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Use cv2.addWeighted for safe and correct blending
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    
    return overlayed_img


# --- 4. Example Usage and Demonstration ---

if __name__ == '__main__':
    # --- Model Initialization and Fine-Tuning Demo ---
    print("--- Model Demo ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the improved model
    model = FacePADModel(backbone_name='resnet50', pretrained=True).to(device)
    
    # 1. Start with backbone frozen (for initial training of the head)
    model.freeze_backbone()
    print("Classifier head requires_grad:", model.backbone.fc[0].weight.requires_grad) # Should be True
    print("A backbone layer requires_grad:", model.backbone.layer4[0].conv1.weight.requires_grad) # Should be False
    
    # 2. Unfreeze for full fine-tuning
    model.unfreeze_backbone()
    print("Classifier head requires_grad:", model.backbone.fc[0].weight.requires_grad) # Should be True
    print("A backbone layer requires_grad:", model.backbone.layer4[0].conv1.weight.requires_grad) # Should be True

    # --- Grad-CAM and Overlay Demo ---
    print("\n--- Grad-CAM Demo ---")
    # Create a dummy input tensor (batch, channels, height, width)
    # This should be a preprocessed image tensor
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Find the target layer for Grad-CAM. For ResNet, layer4 is a good choice.
    target_layer = model.backbone.layer4[-1]
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate the heatmap
    heatmap, predicted_class = grad_cam.generate_cam(dummy_input)
    print(f"Generated heatmap with shape: {heatmap.shape}")
    print(f"Model predicted class index: {predicted_class}")
    
    # Create a dummy image tensor (from the same dummy input) to overlay
    # Normalization stats should match what you use in your dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dummy_image_tensor = normalize(torch.rand(3, 224, 224))

    # Overlay the heatmap
    overlayed_image = overlay_heatmap(dummy_image_tensor, heatmap)
    
    print(f"Generated overlayed image with shape: {overlayed_image.shape}")
    
    # You can display or save the image
    # cv2.imshow("Grad-CAM Overlay", overlayed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("grad_cam_example.jpg", overlayed_image)
    print("\nDemonstration complete. You can uncomment the cv2 lines to save/show the image.")