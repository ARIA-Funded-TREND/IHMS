# TopK Implementation

This directory contains the **TopK attention variant** of the vision-guided quadrupedal locomotion architecture.

**TopK Mechanism Overview:**
In this implementation, the Top-K selection is applied *after* the QKV matrices are modulated. Specifically, we select only **one token** (Top-1) to perform the attention computation. The result is then scattered back to its original size to make it ready for processing by the next layer. This approach drastically reduces computational overhead during inference.

---

### 🧠 Network Architecture Location

All neural network architectures—including the TopK routing, attention scatter logic, and base models—are located here:

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
* **`torchrl/`**: The core reinforcement learning library handling PPO algorithms, replay buffers, and the TopK neural networks.
* **`vision4leg/`**: The main Python package containing PyBullet simulation environments, robot configurations, and task definitions.
* **`vision4leg.egg-info/`**: Auto-generated metadata directory for Python package installation.