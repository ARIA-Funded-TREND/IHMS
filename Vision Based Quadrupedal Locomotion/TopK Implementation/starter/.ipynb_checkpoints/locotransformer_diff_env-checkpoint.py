from pathlib import Path
import cv2
import pickle
import pybullet
import numpy as np
import time
import os
import sys
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


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--env_seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--env_name', type=str, default='A1MoveGround')
    parser.add_argument('--num_episodes', type=int, default=1,
                        help='number of episodes')
    # CHANGED: Config is now expected as an argument for loading the environment
    parser.add_argument("--config", type=str, default=None, required=True,
                        help="config file for the environment", )
    parser.add_argument('--save_dir', type=str, default='./snapshots',
                        help='directory for snapshots (default: ./snapshots)')
    parser.add_argument('--video_dir', type=str, default='./video')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='directory for tensorboard logs (default: ./log)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--device", type=int, default=0,
                        help="gpu secification", )
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

np.random.seed(0)
random.seed(0)

# Path to the specific log parameters for network sizing/architecture
MODEL_PARAM_PATH = os.path.join(
    args.log_dir,
    args.id,
    args.env_name,
    str(args.seed),
    "params.json"
)
model_params = get_params(MODEL_PARAM_PATH)

# CHANGED: Load environment parameters from the custom argument configuration file instead
env_params = get_params(args.config)
env_params["env"]["env_build"]["enable_rendering"] = False

env = get_env(
    env_params['env_name'],
    env_params['env'])
########################################################### ADDED FOR MPC ##########################################
wrapper = env
while hasattr(wrapper, "env"):
    wrapper = wrapper.env

gym_env = wrapper._gym_env
######################################################################################################################


if hasattr(env, "_obs_normalizer"):
    NORM_PATH = "{}/{}/{}/{}/model/_obs_normalizer_{}.pkl".format(
        args.log_dir,
        args.id,
        env_params['env_name'],
        args.seed,
        args.snap_check
    )
    with open(NORM_PATH, 'rb') as f:
        env._obs_normalizer = pickle.load(f)
        print(env._obs_normalizer._mean)
        print(env._obs_normalizer._var)

env.eval()

model_params['net']['activation_func'] = torch.nn.ReLU

obs_normalizer = env._obs_normalizer if hasattr(env, "_obs_normalizer") \
    else None

encoder = networks.LocoTransformerEncoder(
    in_channels=env.image_channels,
    state_input_dim=env.observation_space.shape[0],
    **model_params["encoder"]
)

pf = policies.GaussianContPolicyLocoTransformer(
    encoder=encoder,
    state_input_shape=env.observation_space.shape[0],
    visual_input_shape=(env.image_channels, 64, 64),
    output_shape=env.action_space.shape[0],
    **model_params["net"],
    **model_params["policy"]
)

PATH = "{}/{}/{}/{}/model/model_pf_{}.pth".format(
    args.log_dir,
    args.id,
    env_params['env_name'],
    args.seed,
    args.snap_check
)

current_pf_dict = pf.state_dict()
current_pf_dict.update(torch.load(
    PATH,
    map_location="cuda:0")
)
pf.load_state_dict(
    torch.load(
        PATH,
        map_location="cuda:0"
    )
)

pf.eval()

action_weights = []
num_episodes = args.num_episodes
success = 0
count = 0
total_success = 0

rewards = []
task_names = []

video_output_path = "{}/{}/{}/{}".format(
    args.video_dir,
    args.id,
    env_params['env_name'],
    args.seed
)

Path(video_output_path).mkdir(parents=True, exist_ok=True)

total_reward = []
total_moving_distance = []


from vidgear.gears import WriteGear
output_params = {"-vcodec": "libx264", "-crf": 0, "-preset": "fast"}
video_path = os.path.join(
    video_output_path,
    f"Output_{args.snap_check}{args.add_tag}.mp4"
)

dump_path = os.path.splitext(video_path)[0] + ".pkl"

writer = WriteGear(
        output_filename=video_path,
        logging=True, **output_params
)
# Find the specific MultiheadAttention module inside the policy
attn_module = None
for name, module in pf.named_modules():
    if module.__class__.__name__ == 'MultiheadAttention':
        attn_module = module
        break
if attn_module is None:
    print("Warning: Could not find MultiheadAttention module to extract visualization.")

for _ in range(5):
    t = time.time()
    count = 0
    morpho_action_weights = []

    env.seed(args.env_seed)
    random.seed(args.env_seed)
    torch.manual_seed(args.env_seed)
    np.random.seed(args.env_seed)
    
    reward = 0
    step = 0

    epi_reward = 0
    epi_moving = 0

    env.seed(0)
    random.seed(0)
    np.random.seed(0)
    obs = env.reset()

    while True:
        frame_1 = obs[-64 * 64:]
        frame_1 = frame_1.reshape((64, 64))

        ob_t = torch.Tensor(obs).unsqueeze(0)
        action = pf.eval_act(ob_t)
        count += 1

        morpho_action_weights.append(action)
        obs, rew, done, info = env.step(action)

        epi_reward += rew
        # pos = env.env.env.env.env.env.env._gym_env._robot.GetBasePosition()    # Comment Out in Case of MPC
        pos, base_orn = gym_env.pybullet_client.getBasePositionAndOrientation(
            gym_env.robot.quadruped
        )            # Comment out in case of RL
        epi_moving = pos[0]

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
        frame_repeat = np.repeat(frame_resize, 3, axis=2)

        
        # --- NEW: ATTENTION EXTRACTION AND OVERLAY ---
        attn_overlay = np.zeros_like(frame_repeat) # Fallback if history is empty
        
        if attn_module is not None and len(attn_module.attn_history) > 0:
            # We want the most recent attention map, so we take the LAST item 
            # and clear the rest of the history to prevent memory leaks/desyncs
            attn_matrix = attn_module.attn_history[-1] 
            attn_module.attn_history.clear()
            
            # Safety check: ensure the matrix has more than just the 1 state token
            # A typical LocoTransformer visual sequence will have 65+ tokens (1 state + 64 visual)
            if attn_matrix.shape[0] > 1 and attn_matrix.shape[1] > 1:
                
                # Target: state token (index 0). Source: visual tokens (index 1 to end)
                vis_attn = attn_matrix[0, 1:] 
                
                if len(vis_attn) > 0:
                    grid_size = int(np.sqrt(len(vis_attn)))
                    
                    # Only attempt to render if we have a valid 2D grid
                    if grid_size > 0:
                        vis_attn = vis_attn[:grid_size**2] # Slice to ensure perfect square
                        attn_grid = vis_attn.reshape((grid_size, grid_size))
                        
                        # Normalize and Resize
                        attn_grid_norm = cv2.normalize(attn_grid, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        attn_heatmap = cv2.resize(attn_grid_norm, (w, w), interpolation=cv2.INTER_CUBIC)
                        
                        # Colorize and blend
                        attn_color = cv2.applyColorMap(attn_heatmap, cv2.COLORMAP_JET)
                        attn_overlay = cv2.addWeighted(frame_repeat, 0.6, attn_color, 0.4, 0)
        # ---------------------------------------------
        # ---------------------------------------------

        # Write the concatenated frames: [Simulator RGB | Raw Depth | Attention Overlay]
        writer.write(np.concatenate([img, frame_repeat, attn_overlay], axis=1), rgb_mode=True)
        # -----------------------------------------
        reward += rew
        step += 1
        
        if done:
            total_reward.append(epi_reward)
            total_moving_distance.append(epi_moving)

            with open(dump_path, "wb+") as file:
                pickle.dump(
                    dict(reward=total_reward,
                         move_distance=total_moving_distance, step=step
                         ), file)
            break

    print("finish episode")
    fps = count / (time.time() - t)
    print("Reward:", reward / num_episodes, "FPS:", fps)
    print("Step Counts:", step)

    rewards.append(reward)
writer.close()