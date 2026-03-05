import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
from safetensors.torch import load_file

from vit import create_hybrid_vit


# Path to the image
IMAGE_PATH = "/home/bilal/SAM3-Demo/prompt.png" 

# Path to the saved model weights
CHECKPOINT_PATH = "/home/bilal/Research/model.safetensors"

# Model Architecture Params
MODEL_SIZE = "hybrid_base"         # 'hybrid_base', 'co4_base'
IMAGE_SIZE = 224
NUM_CLASSES = 1000   
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # ImageNet transformation constants

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

def get_transform(crop_size: int = 224, resize_size: int = 256) -> transforms.Compose:
    """Standard ImageNet Validation transforms with ViT-required Augmentations."""
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def load_imagenet_labels():
    """Fetches ImageNet-1K labels for readable output."""
    try:
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        return requests.get(url).json()
    except Exception as e:
        print(f"Warning: Could not load class names ({e}). Using indices.")
        return [str(i) for i in range(1000)]

def run_inference():
    print(f"🖥️  Running inference on {DEVICE}...")
    
    # 1. Load Image
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
    
    raw_image = Image.open(IMAGE_PATH).convert("RGB")
    
    # 2. Preprocess (Validation Transform)
    transform = get_transform(crop_size=IMAGE_SIZE, resize_size=256)
    input_tensor = transform(raw_image).unsqueeze(0).to(DEVICE) # Add batch dim
    
    # 3. Build Model
    print(f"🏗️  Building model ({MODEL_SIZE})...")
    model = create_hybrid_vit(
        model_size=MODEL_SIZE,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        dropout=0.0,
        drop_path_rate=0.0
    )
    model.to(DEVICE)
    model.eval()
    
    # 4. Load Weights
    print(f"📂 Loading weights from {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    if CHECKPOINT_PATH.endswith(".safetensors"):
        # Load safetensors directly to the device (faster & memory efficient)
        state_dict = load_file(CHECKPOINT_PATH, device=DEVICE)
    else:
        # Fallback for standard .pth files
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Handle case where weights might be wrapped in 'model' key or have '_orig_mod' prefix (torch.compile)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    # Clean up state dict keys if they have 'module.' or '_orig_mod.' prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "").replace("_orig_mod.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    
    # 5. Inference
    labels = load_imagenet_labels()
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probs, 5)
        
    # 6. Output Results
    print("\n📊 Results:")
    print("-" * 40)
    for i in range(5):
        idx = top5_idx[0][i].item()
        prob = top5_prob[0][i].item()
        class_name = labels[idx]
        print(f"{i+1}. {class_name:<25} ({prob*100:.2f}%)")
    print("-" * 40)

if __name__ == "__main__":
    run_inference()