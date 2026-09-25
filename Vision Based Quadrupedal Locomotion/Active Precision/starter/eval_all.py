# from pathlib import Path
# import os
# import sys
# import time
# import csv
# import pickle
# import random
# import argparse

# import numpy as np
# import torch

# sys.path.append(".")

# import torchrl.networks as networks
# import torchrl.policies as policies
# from torchrl.utils import get_params
# from vision4leg.get_env import get_env


# # -----------------------------
# # ARGUMENTS
# # -----------------------------
# def get_args():
#     parser = argparse.ArgumentParser(description="RL Eval")

#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--env_seed", type=int, default=0)
#     parser.add_argument("--env_name", type=str, default="A1MoveGround")
#     parser.add_argument("--num_episodes", type=int, default=5)

#     parser.add_argument("--log_dir", type=str, default="./log")
#     parser.add_argument("--video_dir", type=str, default="./video")  # reused for logs
#     parser.add_argument("--id", type=str, default=None)

#     parser.add_argument("--snap_check", type=str, default="best")
#     parser.add_argument("--no_cuda", action="store_true", default=False)

#     return parser.parse_args()


# args = get_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()

# # -----------------------------
# # SEEDS
# # -----------------------------
# np.random.seed(args.seed)
# random.seed(args.seed)
# torch.manual_seed(args.seed)

# # -----------------------------
# # LOAD PARAMS
# # -----------------------------
# PARAM_PATH = os.path.join(
#     args.log_dir,
#     args.id,
#     args.env_name,
#     str(args.seed),
#     "params.json"
# )

# params = get_params(PARAM_PATH)

# params["env"]["env_build"]["enable_rendering"] = False

# # -----------------------------
# # ENV
# # -----------------------------
# env = get_env(params["env_name"], params["env"])

# if hasattr(env, "_obs_normalizer"):
#     norm_path = os.path.join(
#         args.log_dir,
#         args.id,
#         params["env_name"],
#         str(args.seed),
#         "model",
#         f"_obs_normalizer_{args.snap_check}.pkl"
#     )
#     with open(norm_path, "rb") as f:
#         env._obs_normalizer = pickle.load(f)

# env.eval()

# # -----------------------------
# # POLICY NETWORK
# # -----------------------------
# params["net"]["activation_func"] = torch.nn.ReLU

# encoder = networks.LocoTransformerEncoder(
#     in_channels=env.image_channels,
#     state_input_dim=env.observation_space.shape[0],
#     **params["encoder"]
# )

# pf = policies.GaussianContPolicyLocoTransformer(
#     encoder=encoder,
#     state_input_shape=env.observation_space.shape[0],
#     visual_input_shape=(env.image_channels, 64, 64),
#     output_shape=env.action_space.shape[0],
#     **params["net"],
#     **params["policy"]
# )

# MODEL_PATH = os.path.join(
#     args.log_dir,
#     args.id,
#     params["env_name"],
#     str(args.seed),
#     "model",
#     f"model_pf_{args.snap_check}.pth"
# )

# pf.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
# pf.eval()

# # -----------------------------
# # OUTPUT PATH
# # -----------------------------
# output_dir = os.path.join(
#     args.video_dir,
#     args.id,
#     params["env_name"],
#     str(args.seed)
# )
# Path(output_dir).mkdir(parents=True, exist_ok=True)

# csv_path = os.path.join(output_dir, "eval_log.csv")

# # -----------------------------
# # EVAL LOOP
# # -----------------------------
# results = []

# for ep in range(args.num_episodes):

#     env.seed(args.env_seed)
#     random.seed(args.env_seed)
#     np.random.seed(args.env_seed)
#     torch.manual_seed(args.env_seed)

#     obs = env.reset()

#     ep_reward = 0.0
#     ep_steps = 0

#     start_time = time.time()

#     while True:

#         obs_t = torch.Tensor(obs).unsqueeze(0)

#         with torch.no_grad():
#             action = pf.eval_act(obs_t)

#         obs, rew, done, info = env.step(action)

#         ep_reward += rew
#         ep_steps += 1

#         if done:
#             break

#     fps = ep_steps / (time.time() - start_time)

#     # safe position extraction
#     try:
#         pos = env.env.env.env.env.env.env._gym_env._robot.GetBasePosition()
#         x_pos = pos[0]
#     except:
#         x_pos = None

#     results.append([
#         ep,
#         ep_reward,
#         ep_steps,
#         fps,
#         x_pos
#     ])

#     print(
#         f"[EP {ep}] reward={ep_reward:.3f} "
#         f"steps={ep_steps} fps={fps:.2f} x={x_pos}"
#     )

# # -----------------------------
# # SAVE CSV
# # -----------------------------
# with open(csv_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["episode", "reward", "steps", "fps", "final_x_position"])
#     writer.writerows(results)

# print(f"\nSaved results to: {csv_path}")






from pathlib import Path
import cv2
import pickle
import pybullet
import numpy as np
import time
import os
import sys
import glob
import re
sys.path.append(".")
import matplotlib.pyplot as plt
import torchrl.networks as networks
import torchrl.policies as policies
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import gym.wrappers as wrappers
import gym
import torch.nn.functional as F
from torchrl.utils import get_params
from vision4leg.get_env import get_env
import seaborn as sns
import random
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter # Added for TensorBoard logging
from vidgear.gears import WriteGear

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--env_seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--env_name', type=str, default='A1MoveGround')
    parser.add_argument('--num_episodes', type=int, default=5, # Changed default to 5 for reliable eval
                        help='number of episodes')
    parser.add_argument("--config", type=str,   default=None,
                        help="config file", )
    parser.add_argument('--save_dir', type=str, default='./snapshots',
                        help='directory for snapshots (default: ./snapshots)')
    parser.add_argument('--video_dir', type=str, default='./video')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='directory for tensorboard logs (default: ./log)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--device", type=int, default=0,
                        help="gpu specification", )
    parser.add_argument('--add_tag', type=str, default='',
                        help='directory for snapshots (default: ./snapshots)')
    parser.add_argument('--snap_check', type=str, default='best')
    # tensorboard
    parser.add_argument("--id", type=str,   default=None,
                        help="id for tensorboard", )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

args = get_args()
device = f"cuda:{args.device}" if args.cuda else "cpu"

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

PARAM_PATH = os.path.join(
    args.log_dir,
    args.id,
    args.env_name,
    str(args.seed),
    "params.json"
)

params = get_params(PARAM_PATH)
params["env"]["env_build"]["enable_rendering"] = False
env = get_env(
    params['env_name'],
    params['env'])

params['net']['activation_func'] = torch.nn.ReLU

# Initialize networks
encoder = networks.LocoTransformerEncoder(
    in_channels=env.image_channels,
    state_input_dim=env.observation_space.shape[0],
    **params["encoder"]
)
encoder.to(device)

pf = policies.GaussianContPolicyLocoTransformer(
    encoder=encoder,
    state_input_shape=env.observation_space.shape[0],
    visual_input_shape=(env.image_channels, 64, 64),
    output_shape=env.action_space.shape[0],
    **params["net"],
    **params["policy"]
)
pf.to(device)

model_dir = os.path.join(
    args.log_dir,
    args.id,
    params['env_name'],
    str(args.seed),
    "model"
)

# --- NEW: Setup TensorBoard ---
tb_log_dir = os.path.join(args.log_dir, args.id, params['env_name'], str(args.seed), "eval_tb")
tb_writer = SummaryWriter(log_dir=tb_log_dir)

# --- NEW: Find all models in the directory ---
model_files = glob.glob(os.path.join(model_dir, "model_pf_*.pth"))
checkpoints = []
for mf in model_files:
    match = re.search(r'model_pf_(.+)\.pth', os.path.basename(mf))
    if match:
        checkpoints.append(match.group(1))

# Helper to sort numeric checkpoints
def get_step(ckpt):
    try:
        return int(ckpt)
    except ValueError:
        return -1

checkpoints.sort(key=get_step)

print(f"Found {len(checkpoints)} checkpoints to evaluate in {model_dir}")

best_reward = -float('inf')
best_checkpoint = None

# ==========================================
# PHASE 1: EVALUATE ALL MODELS
# ==========================================
for ckpt in checkpoints:
    print(f"\n--- Evaluating Checkpoint: {ckpt} ---")
    
    # Load corresponding normalizer
    norm_path = os.path.join(model_dir, f"_obs_normalizer_{ckpt}.pkl")
    if hasattr(env, "_obs_normalizer") and os.path.exists(norm_path):
        with open(norm_path, 'rb') as f:
            env._obs_normalizer = pickle.load(f)

    # Load corresponding model
    model_path = os.path.join(model_dir, f"model_pf_{ckpt}.pth")
    pf.load_state_dict(torch.load(model_path, map_location=device))
    pf.eval()

    ckpt_rewards = []
    ckpt_distances = []

    for ep in range(args.num_episodes):
        env.seed(args.env_seed + ep)
        obs = env.reset()
        epi_reward = 0
        
        while True:
            ob_t = torch.Tensor(obs).unsqueeze(0).to(device)
            action = pf.eval_act(ob_t)
            obs, rew, done, info = env.step(action)
            epi_reward += rew
            
            if done:
                ckpt_rewards.append(epi_reward)
                # Note: Kept your original logic for distance extraction here
                pos = [0,0,0] # env.env.env.env.env.env.env._gym_env._robot.GetBasePosition()
                epi_moving = pos[0] 
                ckpt_distances.append(epi_moving)
                break

    avg_reward = np.mean(ckpt_rewards)
    avg_dist = np.mean(ckpt_distances)
    
    print(f"Ckpt {ckpt} | Avg Reward: {avg_reward:.2f} | Avg Dist: {avg_dist:.2f}")

    # Log to TensorBoard (only if it's a numeric step, ignoring 'best' or 'latest')
    step_val = get_step(ckpt)
    if step_val != -1:
        tb_writer.add_scalar('Eval/Reward', avg_reward, step_val)
        tb_writer.add_scalar('Eval/Move_Distance', avg_dist, step_val)

    # Track best model
    if avg_reward > best_reward:
        best_reward = avg_reward
        best_checkpoint = ckpt

tb_writer.close()

if best_checkpoint is None:
    print("No checkpoints were successfully evaluated. Exiting.")
    sys.exit(0)

print(f"\n==========================================")
print(f"Evaluation Complete!")
print(f"Best Checkpoint Identified: {best_checkpoint} (Reward: {best_reward:.2f})")
print(f"==========================================\n")

# ==========================================
# PHASE 2: LOAD BEST MODEL AND RENDER VIDEO
# ==========================================
print(f"Loading best checkpoint ({best_checkpoint}) for video rendering...")

# Load best normalizer
norm_path = os.path.join(model_dir, f"_obs_normalizer_{best_checkpoint}.pkl")
if hasattr(env, "_obs_normalizer") and os.path.exists(norm_path):
    with open(norm_path, 'rb') as f:
        env._obs_normalizer = pickle.load(f)

# Load best model
best_model_path = os.path.join(model_dir, f"model_pf_{best_checkpoint}.pth")
pf.load_state_dict(torch.load(best_model_path, map_location=device))
pf.eval()

video_output_path = os.path.join(args.video_dir, args.id, params['env_name'], str(args.seed))
Path(video_output_path).mkdir(parents=True, exist_ok=True)

output_params = {"-vcodec": "libx264", "-crf": 0, "-preset": "fast"}
writer = WriteGear(
    output_filename=os.path.join(
        video_output_path, f'Output_{best_checkpoint}{args.add_tag}.mp4'
    ),
    logging=True, **output_params
)

env.seed(0)
obs = env.reset()
reward_vid = 0
step_vid = 0

while True:
    frame_1 = obs[-64 * 64:]
    frame_1 = frame_1.reshape((64, 64))

    ob_t = torch.Tensor(obs).unsqueeze(0).to(device)
    action = pf.eval_act(ob_t)
    
    obs, rew, done, info = env.step(action)
    reward_vid += rew
    step_vid += 1

    # --- VIDEO RENDERING AND WRITING LOGIC ---
    img = env.render(mode='rgb_array')
    
    sim_model = env.robot.quadruped
    pyb = env.pybullet_client
    root_vel_sim, root_ang_vel_sim = pyb.getBaseVelocity(sim_model)

    w, h, _ = img.shape
    # Normalize the camera frame
    frame_1 = (frame_1 - np.min(frame_1)) / (np.max(frame_1) - np.min(frame_1))
    
    # Resize and convert to match RGB image dimensions
    frame_resize = cv2.resize(frame_1, (w, w))
    frame_resize = (frame_resize * 255).astype(img.dtype)
    frame_resize = frame_resize.reshape((w, w, 1))
    frame_resize = np.repeat(frame_resize, 3, axis=2)

    # Write the concatenated frames to the video
    writer.write(np.concatenate([img, frame_resize], axis=1), rgb_mode=True)
    # -----------------------------------------

    if done:
        # Saving just the best result dictionary
        pos = [0,0,0] # Extract real pos here if uncommented
        with open(os.path.join(video_output_path, "eval_result_best.pkl"), "wb+") as file:
            pickle.dump(
                dict(reward=[reward_vid],
                     move_distance=[pos[0]]
                     ), file)
        break

writer.close()
print(f"Video rendering for best model ({best_checkpoint}) complete. Reward: {reward_vid:.2f}, Steps: {step_vid}")