from pathlib import Path
import os
import sys
import time
import csv
import pickle
import random
import argparse

import numpy as np
import torch

sys.path.append(".")

import torchrl.networks as networks
import torchrl.policies as policies
from torchrl.utils import get_params
from vision4leg.get_env import get_env


# -----------------------------
# ARGUMENTS
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="RL Eval")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_seed", type=int, default=0)
    parser.add_argument("--env_name", type=str, default="A1MoveGround")
    parser.add_argument("--num_episodes", type=int, default=5)

    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--video_dir", type=str, default="./video")  # reused for logs
    parser.add_argument("--id", type=str, default=None)

    parser.add_argument("--snap_check", type=str, default="best")
    parser.add_argument("--no_cuda", action="store_true", default=False)

    return parser.parse_args()


args = get_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# -----------------------------
# SEEDS
# -----------------------------
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

# -----------------------------
# LOAD PARAMS
# -----------------------------
PARAM_PATH = os.path.join(
    args.log_dir,
    args.id,
    args.env_name,
    str(args.seed),
    "params.json"
)

params = get_params(PARAM_PATH)

params["env"]["env_build"]["enable_rendering"] = False

# -----------------------------
# ENV
# -----------------------------
env = get_env(params["env_name"], params["env"])

if hasattr(env, "_obs_normalizer"):
    norm_path = os.path.join(
        args.log_dir,
        args.id,
        params["env_name"],
        str(args.seed),
        "model",
        f"_obs_normalizer_{args.snap_check}.pkl"
    )
    with open(norm_path, "rb") as f:
        env._obs_normalizer = pickle.load(f)

env.eval()

# -----------------------------
# POLICY NETWORK
# -----------------------------
params["net"]["activation_func"] = torch.nn.ReLU

encoder = networks.LocoTransformerEncoder(
    in_channels=env.image_channels,
    state_input_dim=env.observation_space.shape[0],
    **params["encoder"]
)

pf = policies.GaussianContPolicyLocoTransformer(
    encoder=encoder,
    state_input_shape=env.observation_space.shape[0],
    visual_input_shape=(env.image_channels, 64, 64),
    output_shape=env.action_space.shape[0],
    **params["net"],
    **params["policy"]
)

MODEL_PATH = os.path.join(
    args.log_dir,
    args.id,
    params["env_name"],
    str(args.seed),
    "model",
    f"model_pf_{args.snap_check}.pth"
)

pf.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
pf.eval()

# -----------------------------
# OUTPUT PATH
# -----------------------------
output_dir = os.path.join(
    args.video_dir,
    args.id,
    params["env_name"],
    str(args.seed)
)
Path(output_dir).mkdir(parents=True, exist_ok=True)

csv_path = os.path.join(output_dir, "eval_log.csv")

# -----------------------------
# EVAL LOOP
# -----------------------------
results = []

for ep in range(args.num_episodes):

    env.seed(args.env_seed)
    random.seed(args.env_seed)
    np.random.seed(args.env_seed)
    torch.manual_seed(args.env_seed)

    obs = env.reset()

    ep_reward = 0.0
    ep_steps = 0

    start_time = time.time()

    while True:

        obs_t = torch.Tensor(obs).unsqueeze(0)

        with torch.no_grad():
            action = pf.eval_act(obs_t)

        obs, rew, done, info = env.step(action)

        ep_reward += rew
        ep_steps += 1

        if done:
            break

    fps = ep_steps / (time.time() - start_time)

    # safe position extraction
    try:
        pos = env.env.env.env.env.env.env._gym_env._robot.GetBasePosition()
        x_pos = pos[0]
    except:
        x_pos = None

    results.append([
        ep,
        ep_reward,
        ep_steps,
        fps,
        x_pos
    ])

    print(
        f"[EP {ep}] reward={ep_reward:.3f} "
        f"steps={ep_steps} fps={fps:.2f} x={x_pos}"
    )

# -----------------------------
# SAVE CSV
# -----------------------------
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "steps", "fps", "final_x_position"])
    writer.writerows(results)

print(f"\nSaved results to: {csv_path}")