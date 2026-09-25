# Active Precision Implementation

This directory contains the **Active Precision variant** of the vision-guided quadrupedal locomotion architecture.

**Active Precision Mechanism Overview:**
In this implementation, a local error—referred to as the *prediction error*—is computed individually for the Query, Key, and Value (QKV) matrices. This prediction error is then scaled with their modulated counterparts and utilized as a precision term to dynamically adjust the network's focus and computational allocation.

---

### 🧠 Network Architecture Location

All neural network architectures—including the prediction error computation, precision scaling logic, and base models—are located here:

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
* **`torchrl/`**: The core reinforcement learning library handling PPO algorithms, replay buffers, and the Active Precision neural networks.
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