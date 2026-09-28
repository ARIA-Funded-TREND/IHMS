# Gated MLP Implementation

This directory contains the **Gated MLP (gMLP) variant** of the vision-guided quadrupedal locomotion architecture.

**Gated MLP Mechanism Overview:**
In this implementation, the standard self-attention mechanisms within the Cross-Modal Transformers are entirely replaced with **Gated Multi-Layer Perceptrons** (specifically using a Spatial Gating Unit). By removing quadratic self-attention, the overall computational complexity is reduced to **$O(N)$** (linear complexity with respect to token sequence length). This allows the policy to scale efficiently to longer sequences or higher-resolution spatial inputs without the heavy processing cost usually associated with Transformers.

---

### 🧠 Network Architecture Location

All neural network architectures—including the Spatial Gating Units, the gMLP block structures, and the core policy models—are located here:

**`torchrl/networks/nets.py`**

---

### 📁 Directory Overview

* **`a1_hardware/`**: Scripts and interfaces for deploying trained policies to the physical Unitree A1 robot (includes RealSense and TensorRT conversion).
* **`build/`**: Compiled C++ binaries and shared libraries (e.g., for the robot interface).
* **`config/`**: JSON configuration files for defining RL training parameters, environments, and terrain setups.
* **`figures/`**: Images, diagrams, and visualization assets used in the documentation.
* **`log_mpc/`**: Default directory for storing training checkpoints, logs, and TensorBoard data.
* **`mpc_controller/`**: Implementation of the Model Predictive Control (MPC) logic used for visual-MPC training.
* **`starter/`**: Python entry-point scripts for training, evaluating, and rendering the simulation GUI.
* **`third_party/`**: External submodules and dependencies, primarily the `unitree_legged_sdk`.
* **`torchrl/`**: The core reinforcement learning library handling PPO algorithms, replay buffers, and the gMLP neural networks.
* **`vision4leg/`**: The main Python package containing PyBullet simulation environments, robot configurations, and task definitions.
* **`vision4leg.egg-info/`**: Auto-generated metadata directory for Python package installation.

To change the speed of reality for any configuration, you can locate `get_image_interval` in `config/mpc/locotransformer/thin-wide.json` for complicated environment and `config/mpc/locotransformer/thin-random-shape.json` for easy environment. Changing it from 1, to 5 means, skipping 5 frames. 
---

### 🏆 Best Pre-trained Models & Inference

The best-performing model checkpoints are located inside the **`log_mpc/`** folder. 

**Inference Command**  
To evaluate the models and generate a rollout video, use the following command. It is critical that the `--id` argument **exactly matches the name of the folder** inside `log_mpc`. This ensures the specific network configuration and checkpoint you want to evaluate is correctly loaded:

```bash
python starter/locotransformer_diff_env.py \
  --config config/mpc/locotransformer/thin-wide.json \
  --save_dir DiffVideo/ \
  --video_dir DiffVideo \
  --log_dir log_mpc \
  --id [YOUR_MODEL_ID] \
  --add_tag _ThinWide \
  --env_name A1MoveGroundMPC \
  --seed 0 \
  --env_seed 1