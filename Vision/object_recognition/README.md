## 🏗️ Model Architectures
We provide two primary configurations pre-trained on imagenet-1k. Both utilize a patch size of 16 and an embedding dimension of 768.

| Configuration | Depth | Mod Layer Heads |  MHSA Layer Heads | Attention Pattern | Description |
| --- | --- | --- | --- | --- | --- |
| **`hybrid_base`** | 6 | 1 | 12 | `['modulated', 'modulated', 'standard', 'modulated', 'modulated', 'standard']` | A hybrid architecture mixing standard Multi-Head Self-Attention with Modulated Attention layers. |
| **`co4_base`** | 6 | 1 | 0 | `['modulated'] * 6` (All layers) | Uses strictly Modulated Attention throughout the network with a single-head design. |

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ARIA-Funded-TREND/HMS.git
cd Vision/object_recognition

```


2. **Install dependencies:**
It is recommended to use a virtual environment (Conda or venv).
```bash
pip install -r requirements.txt

```



## 📥 Model Weights

Pre-trained weights are hosted on Hugging Face. Download the weights from the link:

* **Pre-trained Weights:** [`hybrid_base.safetensors`](https://huggingface.co/engrbilal/HMS_Vision/tree/main)

## 🖼️ Inference

To run inference on a single image, use the `inference.py` script.

**Usage:**

1. Open `inference.py`.
2. Update the `IMAGE_PATH` to point to your input image.
3. Update `CHECKPOINT_PATH` to the location of your downloaded `.safetensors` file.
4. Set the `MODEL_SIZE` variable to either `'hybrid_base'` or `'co4_base'`.

```bash
python inference.py

```

**Output Example:**

```text
📊 Results:
----------------------------------------
1. goldfish                  (82.45%)
2. cock                      (12.10%)
3. hen                       (3.15%)
4. ostrich                   (0.85%)
5. tench                     (0.22%)
----------------------------------------

```

## 📊 Evaluation

To evaluate the model on the full **ImageNet-1K Validation Set**, use the `evaluate.py` script. This script handles data loading directly from Hugging Face and calculates Top-1 and Top-5 accuracy.

**Usage:**

1. Open `evaluate.py`.
2. Update `CHECKPOINT_PATH` and `MODEL_SIZE`.
3. (Optional) Set `CACHE_DIR` if you have a specific location for the ImageNet dataset.

**Run the evaluation:**

```bash
# Run on single GPU
python evaluate.py

# OR with Accelerate (recommended for multi-GPU)
accelerate launch evaluate.py

```
## 🧪 Training
We provide a reference training notebook, **sample_training_code**, to demonstrate how to train the provided architectures within this repository. The notebook is intended as a starter template and includes the core components required to run training.

> **_NOTE:_** The provided training notebook serves as a reference implementation for training the architectures in this repository. While it is not configured with fully optimized hyperparameters (e.g., learning rate schedules, augmentation strategies, regularization, or batch sizing), it contains all necessary components to reproduce the vision results reported in the paper. With appropriate configuration and compute resources, the results shown in the paper can be replicated using this notebook.

