import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from accelerate import Accelerator
from tqdm import tqdm
from safetensors.torch import load_file
from typing import Optional, Tuple

from vit import create_hybrid_vit


CHECKPOINT_PATH = "/home/bilal/Research/model.safetensors"
CACHE_DIR = "/home/bilal/imagenet_cache"
HF_TOKEN = None # Optional if logged in

BATCH_SIZE = 256
NUM_WORKERS = 8
PIN_MEMORY = True

MODEL_SIZE = "hybrid_base"         # 'hybrid_base', 'co4_base'
IMAGE_SIZE = 224
NUM_CLASSES = 1000
# ==========================================

class ImageNetValDataset(Dataset):
    """
    Minimal ImageNet Validation Dataset.
    Loads from Hugging Face and applies standard validation transforms.
    """
    def __init__(self, cache_dir: str, token: str = None, image_size: int = 224):
        # 1. Load the validation split from Hugging Face
        print("  ⏳ Loading Hugging Face dataset index...")
        self.dataset = load_dataset(
            "ILSVRC/imagenet-1k", 
            split="validation",
            cache_dir=cache_dir, 
            token=token, 
            keep_in_memory=False,
            trust_remote_code=True
        )
        
        # 2. Define standard ImageNet validation transforms
        # (256 resize -> 224 center crop is the standard protocol)
        resize_size = int(image_size * 256 / 224)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        
        # Ensure RGB (grayscale images cause crash in transforms)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return self.transform(image), sample["label"]

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)) -> list:
    """Compute top-k accuracy for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def evaluate():
    # 1. Initialize Accelerator (handles device placement & AMP automatically)
    accelerator = Accelerator(mixed_precision="fp16") # Change to 'bf16' if on A100
    accelerator.print(f"🚀 Starting evaluation on {accelerator.device}")

    # 2. Data Module
    accelerator.print("📦 Setting up data loader...")
    val_dataset = ImageNetValDataset(
        cache_dir=CACHE_DIR, 
        token=HF_TOKEN, 
        image_size=IMAGE_SIZE
    )
    
    # Get only val loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    # 3. Model Setup
    accelerator.print(f"🏗️  Creating model ({MODEL_SIZE})...")
    model = create_hybrid_vit(
        model_size=MODEL_SIZE,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        dropout=0.0,
        drop_path_rate=0.0
    )

    # 4. Load Checkpoint
    accelerator.print(f"📂 Loading weights from {CHECKPOINT_PATH}...")
    
    if CHECKPOINT_PATH.endswith(".safetensors"):
        weights = load_file(CHECKPOINT_PATH, device="cpu")
    else:
        weights = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    # Handle state dict cleaning
    if 'model' in weights: 
        weights = weights['model']
        
    clean_weights = {
        k.replace("module.", "").replace("_orig_mod.", ""): v 
        for k, v in weights.items()
    }
    
    model.load_state_dict(clean_weights)
    
    # 5. Prepare with Accelerator
    model, val_loader = accelerator.prepare(model, val_loader)
    model.eval()
    
    # 6. Validation Loop
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total_samples = 0
    
    accelerator.print("\n📊 Running validation loop...")
    progress_bar = tqdm(val_loader, disable=not accelerator.is_local_main_process)
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Gather metrics across GPUs (if multi-gpu)
            gathered_outputs = accelerator.gather(outputs)
            gathered_labels = accelerator.gather(labels)
            gathered_loss = accelerator.gather(loss).mean()
            
            batch_size = gathered_labels.size(0)
            
            # Calculate Acc
            acc1, acc5 = accuracy(gathered_outputs, gathered_labels, topk=(1, 5))
            
            # Update stats
            total_loss += gathered_loss.item() * batch_size
            top1_correct += acc1 * batch_size / 100.0
            top5_correct += acc5 * batch_size / 100.0
            total_samples += batch_size
            
            # Update progress bar description
            progress_bar.set_description(f"Acc@1: {top1_correct/total_samples*100:.2f}%")

    # 7. Final Metrics
    final_loss = total_loss / total_samples
    final_acc1 = (top1_correct / total_samples) * 100
    final_acc5 = (top5_correct / total_samples) * 100

    accelerator.print("\n" + "="*40)
    accelerator.print(f"✅ Evaluation Complete")
    accelerator.print("="*40)
    accelerator.print(f"Images Evaluated: {total_samples}")
    accelerator.print(f"Loss:             {final_loss:.4f}")
    accelerator.print(f"Top-1 Accuracy:   {final_acc1:.2f}%")
    accelerator.print(f"Top-5 Accuracy:   {final_acc5:.2f}%")
    accelerator.print("="*40)

if __name__ == "__main__":
    evaluate()