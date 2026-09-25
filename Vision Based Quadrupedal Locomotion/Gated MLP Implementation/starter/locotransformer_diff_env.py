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
writer = WriteGear(
        output_filename=os.path.join(
            video_output_path, 'Output_{}{}.mp4'.format(args.snap_check, args.add_tag)),
        logging=True, **output_params
)

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
        pos = env.env.env.env.env.env.env._gym_env._robot.GetBasePosition()
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

        # Write the concatenated frames to the video
        writer.write(np.concatenate([img, frame_repeat], axis=1), rgb_mode=True)
        # -----------------------------------------

        reward += rew
        step += 1
        
        if done:
            total_reward.append(epi_reward)
            total_moving_distance.append(epi_moving)

            with open("{}.pkl".format("eval_result"), "wb+") as file:
                pickle.dump(
                    dict(reward=total_reward,
                         move_distance=total_moving_distance
                         ), file)
            break

    print("finish episode")
    fps = count / (time.time() - t)
    print("Reward:", reward / num_episodes, "FPS:", fps)
    print("Step Counts:", step)

    rewards.append(reward)
writer.close()