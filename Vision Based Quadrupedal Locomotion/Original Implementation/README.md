# Original Implementation (Baseline & CO^4)

This directory contains the **baseline architecture** for the vision-guided quadrupedal locomotion models.

Additionally, it includes the implementation of **CO^4**, where the Query, Key, and Value (QKV) attention matrices are all modulated using the **Apical Amplification equation**.

---

### 🧠 Network Architecture Location

All neural network architectures—including the core MLPs, CNNs, Cross-Modal Transformers, and the CO^4 Apical Amplification implementations—are located here:

**`torchrl/networks/nets.py`**

---

### 📁 Directory Overview

* **`a1_hardware/`**: Scripts and interfaces for deploying trained policies to the physical Unitree A1 robot (includes RealSense and TensorRT conversion).
* **`build/`**: Compiled C++ binaries and shared libraries (e.g., for the robot interface).
* **`config/`**: JSON configuration files for defining RL training parameters, environments, and terrain setups.
* **`figures/`**: Images, diagrams, and visualization assets used in the documentation.
* **`log_mpc/`**: Default directory for storing training checkpoints, logs, and TensorBoard data.
* **`mpc_controller/`**: Implementation of the Model Predictive Control (MPC) logic used for visual-MPC training.
* **`starter/`**: Python entry-point scripts for training (`ppo_locotransformer.py`), evaluating, and rendering the simulation GUI.
* **`third_party/`**: External submodules and dependencies, primarily the `unitree_legged_sdk`.
* **`torchrl/`**: The core reinforcement learning library handling PPO algorithms, replay buffers, and neural networks.
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