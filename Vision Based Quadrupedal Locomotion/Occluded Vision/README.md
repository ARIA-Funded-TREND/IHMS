# Occluded Vision Implementation

This directory contains the **Occluded Vision variant** of the vision-guided quadrupedal locomotion architecture, covering both the **standard baseline** and the **CO^4** (Apical Amplification modulated QKV) implementations.

**Occluded Vision Overview:**

Similar to the Original Implementation, this variant evaluates model robustness under visual impairment. During training and inference, the models receive input depth images where **20% of the image patches are randomly occluded (masked)**. This forces both the standard Cross-Modal Transformer and the CO^4 architecture to leverage proprioceptive feedback and learn robust, partial-observability representations.

---

### 🧠 Network Architecture Location

All neural network architectures—including the standard baseline models, the CO^4 Apical Amplification structures, and the patch occlusion logic—are located here:

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
* **`torchrl/`**: The core reinforcement learning library handling PPO algorithms, replay buffers, and the occluded vision networks.
* **`vision4leg/`**: The main Python package containing PyBullet simulation environments, robot configurations, and task definitions.
* **`vision4leg.egg-info/`**: Auto-generated metadata directory for Python package installation.

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