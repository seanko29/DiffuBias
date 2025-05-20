"""
Classifier Inference Tool with GradCAM Visualization

Evaluates pretrained classification models on various data splits (aligned/conflict)
and generates GradCAM visualizations to analyze model attention.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torchvision.transforms as T
import random
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

from models import call_by_name
from utils import GeneralizedCELoss

# For reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


class BaseDataset(torch.utils.data.Dataset):
    """Dataset loader for test images"""
    def __init__(self, image_paths, args, transform=None, mode='test'):
        self.image_paths = image_paths
        self.transform = transform
        self.args = args
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)
            
        # Extract label from filename if available
        try:
            label = int(os.path.basename(image_path).split('_')[-2])
        except (ValueError, IndexError):
            label = -1  # Use -1 for unknown label
            
        # Return original filename for tracking
        return idx, image_tensor, label, image_path


def preprocess_image_for_display(image_tensor):
    """Convert normalized tensor back to displayable image"""
    # First ensure the image tensor is on CPU
    image_tensor = image_tensor.detach().cpu()
    
    # Define mean and std on CPU
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    # Denormalize
    image = image_tensor.clone()
    image = image * std + mean
    image = image.clamp(0, 1)
    
    # Convert to numpy, transpose for display
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    return image


def visualize_gradcam(image_tensor, cam, save_path=None, alpha=0.5):
    """Overlay GradCAM heatmap on original image"""
    # Ensure image_tensor is on CPU before processing
    image_tensor = image_tensor.detach().cpu()
    
    # Prepare original image for display
    orig_image = preprocess_image_for_display(image_tensor.squeeze(0))
    
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (orig_image.shape[1], orig_image.shape[0]))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay heatmap on original image
    superimposed = alpha * heatmap + (1 - alpha) * orig_image
    superimposed = np.clip(superimposed, 0, 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(orig_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # GradCAM heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    # Superimposed image
    axes[2].imshow(superimposed)
    axes[2].set_title('Superimposed')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


class GradCAM:
    """Class for generating GradCAM visualizations"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradient = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_features)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_features(self, module, input, output):
        self.feature_maps = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]
    
    def generate(self, input_image, target_class=None):
        """Generate GradCAM heatmap for input image"""
        # Set model to evaluation mode but enable gradient tracking
        self.model.eval()
        
        # Enable gradient tracking for the input
        input_image.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_image)
        
        # If target class not specified, use predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Create one-hot encoding for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate GradCAM
        with torch.no_grad():
            # Global average pooling of gradients
            weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
            
            # Weight feature maps with gradient importance
            cam = torch.sum(weights * self.feature_maps, dim=1).squeeze()
            
            # Apply ReLU to highlight positive influence
            cam = torch.relu(cam)
            
            # Normalize
            if torch.max(cam) > 0:
                cam = cam / torch.max(cam)
            
            # Convert to numpy
            cam = cam.cpu().numpy()
        
        return cam, target_class


def get_transform(args):
    """Get test transforms based on dataset"""
    if args.exp == 'bffhq':
        # For BFFHQ
        transform = T.Compose([
            T.Resize((224, 224)),  # Resize to model input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    else:
        # Default transforms
        transform = T.Compose([
            T.Resize((args.image_size[0], args.image_size[1])),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    return transform


def setup_gradcam(model):
    """Set up GradCAM with appropriate target layer"""
    # Find appropriate target layer for GradCAM
    if hasattr(model, 'layer4'):
        target_layer = model.layer4[-1]
    elif hasattr(model, 'features'):
        target_layer = model.features[-1]
    else:
        print("Warning: Could not automatically determine target layer for GradCAM.")
        found_target = False
        for name, module in reversed(list(model.named_modules())):
            if not isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                target_layer = module
                found_target = True
                print(f"Using {name} as target layer for GradCAM.")
                break
        if not found_target:
            raise ValueError("Could not find suitable target layer for GradCAM.")
    
    return GradCAM(model, target_layer)


def set_args(args):
    """Set dataset-specific arguments"""
    if args.exp == 'bffhq':
        args.w, args.h = 224, 224
        args.num_classes = 2
        args.num_bias = 2
    else:
        args.w, args.h = 224, 224
        args.num_classes = 2  # Default values


def run_inference_on_specific_images(args):
    """Run inference and generate GradCAM visualizations for specific images"""
    print(f"\n===== Generating GradCAM for specific images =====")
    
    # Load model
    model = call_by_name(args)
    model.cuda()
    
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Setup transforms
    test_transform = get_transform(args)
    
    # Load images from specified directory
    images_dir = args.specific_images_dir
    print(f"Loading images from {images_dir}")
    
    if not os.path.exists(images_dir):
        print(f"Error: Directory not found at {images_dir}")
        return
    
    # Get all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(images_dir, ext)))
    
    if len(image_files) == 0:
        print(f"Error: No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create dataset and dataloader
    image_dataset = BaseDataset(image_files, args, transform=test_transform, mode='test')
    image_loader = torch.utils.data.DataLoader(
        image_dataset, 
        batch_size=1,  # Process one image at a time for GradCAM
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Setup GradCAM
    gradcam = setup_gradcam(model)
    
    # Create directory for GradCAM visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gradcam_base_dir = f"gradcam/specific_images/{timestamp}"
    os.makedirs(gradcam_base_dir, exist_ok=True)
    
    # Run inference and generate GradCAM for each image
    print("\nGenerating GradCAM visualizations...")
    with torch.no_grad():
        for iter_idx, (_, inputs, labels, paths) in enumerate(tqdm(image_loader)):
            inputs = inputs.cuda(non_blocking=True)
            
            # Get model prediction
            outputs = model(inputs)
            predictions = outputs.argmax(1)
            pred_class = predictions[0].item()
            
            # Get true label if available
            true_class = labels[0].item() if labels[0].item() != -1 else "unknown"
            
            # Generate GradCAM
            with torch.enable_grad():
                # Clone and require grad for the image
                image_tensor = inputs.clone().detach().requires_grad_(True)
                
                # Generate GradCAM for predicted class
                cam, _ = gradcam.generate(image_tensor, pred_class)
                
                # Get filename from path
                image_name = os.path.basename(paths[0])
                
                # Save GradCAM visualization
                save_path = os.path.join(
                    gradcam_base_dir,
                    f"img{iter_idx:03d}_{image_name}_true{true_class}_pred{pred_class}.png"
                )
                visualize_gradcam(image_tensor, cam, save_path=save_path)
    
    print(f"\nGradCAM visualizations saved to {gradcam_base_dir}")
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Classifier Inference Tool with GradCAM Visualization")
    
    # Dataset and model parameters
    parser.add_argument("--exp", type=str, default='bffhq', 
                        help="Dataset name (bffhq, bar, cifar10c, etc.)")
    parser.add_argument("--pct", type=str, default="5pct", 
                        help="Percent of data used in training (1pct, 5pct, etc.)")
    parser.add_argument("--etc", type=str, default='vanilla', 
                        help="Experiment configuration")
    
    # Paths
    parser.add_argument("--data_root", type=str, 
                        default="/home/sean/debiasing/aaai/data/debias/bffhq", 
                        help="Dataset root directory")
    parser.add_argument("--model_path", type=str,
                        default="/home/sean/debiasing/aaai/save/final_results/bffhq/bffhq-1pct-vanilla/best.path.tar",
                        help="Path to the pretrained model checkpoint")
    parser.add_argument("--output_path", type=str,
                        default="./test_results.csv",
                        help="Path to save inference results")
    parser.add_argument("--specific_images_dir", type=str,
                        default="/home/sean/debiasing/aaai/save/saved_gradcam",
                        help="Directory containing specific images to analyze")
    
    # Test mode selection
    parser.add_argument("--conflict_only", action="store_true",
                        help="Run inference only on the conflict dataset")
    parser.add_argument("--aligned_only", action="store_true",
                        help="Run inference only on the aligned dataset")
    parser.add_argument("--specific_images", action="store_true",
                        help="Run GradCAM on specific images in the provided directory")
    
    # Other parameters
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--pretrained", type=bool, default=True, help="Use pretrained model")
    parser.add_argument("--num_gradcam_samples", type=int, default=30, 
                        help="Number of GradCAM samples to generate per class")
    
    args = parser.parse_args()
    
    # Set dataset-specific arguments
    set_args(args)
    
    # Update paths
    args.image_size = (args.w, args.h)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        exit(1)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Dataset: {args.exp}")
    print(f"  Percentage: {args.pct}")
    print(f"  Model path: {args.model_path}")
    print(f"  Data root: {args.data_root}")
    print(f"  Image size: {args.image_size}")
    print(f"  GradCAM samples per class: {args.num_gradcam_samples}")
    
    # Determine which data to analyze
    if args.specific_images:
        print(f"Analyzing specific images from: {args.specific_images_dir}")
        run_inference_on_specific_images(args)
    elif args.conflict_only and args.aligned_only:
        print("Running inference on both conflict and aligned datasets")
        run_inference_on_conflict(args)
        run_inference_on_aligned(args)
    elif args.conflict_only:
        print("Running inference only on conflict dataset")
        run_inference_on_conflict(args)
    elif args.aligned_only:
        print("Running inference only on aligned dataset")
        run_inference_on_aligned(args)
    else:
        print("Running inference on full test dataset")
        run_inference(args) 