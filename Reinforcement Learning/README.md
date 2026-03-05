
# Reinforcement Learning

This repository contains the implementation of Reinforcement Learning (RL) algorithm for **Scalable Machines with Intrinsic Higher Mental States**, using Google’s PI framework ([Brain Tokyo Workshop](https://github.com/google/brain-tokyo-workshop)) as baseline.

The implementation utilizes **CO4** and is designed to be highly modular, with individual classes handling specific logic to maintain scalability despite the increased code length.

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.x**
* **CPU-Heavy Environment**: Rollouts run on the CPU; GPU acceleration is currently not utilized for this specific implementation.

### Installation

1. **Clone and Navigate**
```bash
cd AttentionNeuron

```

2. **Setup Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

```


3. **Install Dependencies**
```bash
pip install -r requirements_comprehensive.txt

```



---

## 🛠 Usage

### Training an Agent

To begin training, use the `train_agent.py` script with the desired configuration file.

```bash
python train_agent.py --config configs/x.gin --log-dir log/x --num-workers 6

```

* `--config`: Path to your `.gin` configuration.
* `--log-dir`: Directory where logs and model checkpoints will be saved.
* `--num-workers`: Number of parallel workers for rollouts.
* `x_cmi.gin`: File for experiment hyperparameter for Co4.

### Evaluation

To evaluate a trained model and check its performance:

```bash
python eval_agent.py --log-dir log/x --model-filename best.npz

```

---


## ⚙️ Configuration

Each task has specific parameters in its corresponding `config.gin` file. Below are the key tunable settings:

### Common to All Tasks

* `shuffle_on_reset` or `permute_obs`: Defines whether to shuffle observations during rollouts.
* `render`: Toggle environment visualization.
* `v`: Verbosity toggle for logging.

### Task-Specific Tuning

| Task | Parameter | Description |
| --- | --- | --- |
| **CartPoleSwingUp** | `num_noise_channels` | Defines the number of noisy input channels to the agent. |
| **CarRacing** | `bkg` | Set background image (e.g., `bkg="mt_fuji"` loads `tasks/bkg/mt_fuji.jpg`). Use `None` for default. |
| **CarRacing** | `patch_size` | Defines the size of the patch to be shuffled. |

---

## 📊 Training Results: Google vs. Co4

Below is a comparison of training progress across various RL environments, showcasing the performance of the **Co4** implementation against the baseline.


| **Environment** | **Acrobot** | **CarRacing** |
| :--- | :---: | :---: |
| **Results** | ![Acrobot](/assets/AcrobotGooglevsCMI.png) | ![CarRacing](/assets/CarRacingGooglevsCMI.png) |

| **Environment** | **CartPole** | **PyAnt** |
| :--- | :---: | :---: |
| **Results** | ![CartPole](/assets/CartPoleGooglevsCMI.png) | ![PyAnt](/assets/PyAntGooglevsCMI.png) |

---

## 📂 Project Structure

* `AttentionNeuron/`: Core logic and agent implementation.
* `configs/`: `.gin` files for experiment hyperparameters.
* `log/`: Training logs and saved `.npz` model weights.
* `images/`: Visualization plots for documentation.



## Demo


### Cartpole
https://github.com/user-attachments/assets/555d802d-bca4-49ee-8d0e-c7ce666d3e7c

A Comparison of cartpole trained on 1K, 5K and 10K episodes with Transformer and Co4 with intrinsic higher mental states shows a much better understanding of environment, when trained with Co4.


### PyAnt
https://github.com/user-attachments/assets/6af058fa-77ff-4da8-8517-106920ad2f88

A Comparison of PyAnt trained on 1K episodes with Transformer and Co4 with intrinsic higher mental states shows faster learning, when trained with Co4.
