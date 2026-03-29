import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import gymnasium

from pacsimEnv import pacsimEnv
from gymnasium.wrappers import FrameStackObservation, NormalizeReward, NormalizeObservation

import cv2
from tqdm.rich import tqdm, trange

pacsimArgs = {
    "cam_sim": True
}
pacsim = pacsimEnv(pacsimArgs)
env = FrameStackObservation(pacsim, stack_size=3)

import networks

import subprocess
import argparse
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLI args: choose model type and path
parser = argparse.ArgumentParser(description="Actor inference runner")
parser.add_argument('--model-type', choices=['state', 'vision'], default='state', help='Type of model to run')
parser.add_argument('--model-path', type=str, default=None, help='Path to model to load (overrides defaults)')
parser.add_argument('--noisy', action='store_true', help='Use noisy variant when applicable')
parser.add_argument('--dagger', action='store_true', help='If running a vision model, load the DAgger-trained vision model (resnet_dagger_... prefix)')
parser.add_argument('--latent', action='store_true', help='Prefer latent (VAE) vision models when loading')
args = parser.parse_args()

# Determine default model path if not provided
if args.model_path:
    model_path = args.model_path
else:
    if args.model_type == 'state':
        model_path = 'networks/state_actor_noisy' if args.noisy else 'networks/state_actor'
    else:
        basename = 'data_state_actor_noisy' if args.noisy else 'data_state_actor'

        if args.dagger:
            model_path = f'networks/resnet_dagger_{basename}'
        else:
            model_path = f'networks/resnet_bc_pretrain_{basename}'

# Try common extensions and latent/structured suffixes
found = False
if getattr(args, 'latent', False):
    suffixes = ['_latent', '', '_structured']
else:
    suffixes = ['_structured', '', '_latent']
exts = ['', '.pt', '.pth', '.ckpt', '.pkl']

for s in suffixes:
    for e in exts:
        candidate = model_path + s + e
        if os.path.isfile(candidate):
            model_path = candidate
            found = True
            break
    if found:
        break

if not found:
    print(f"Warning: Could not find specific model file for {model_path}. Trying base path.")

print(f"Loading model: {model_path} (type={args.model_type})")
actor = torch.load(model_path, weights_only=False)

actor = actor.to(device)

model_base = os.path.splitext(os.path.basename(model_path))[0]
video_name = f"{model_base}_{args.model_type}_{int(time.time())}.mp4"
video_path = os.path.join("..", "viz", video_name)
os.makedirs(os.path.dirname(video_path), exist_ok=True)
print(f"Writing video to: {video_path}")

resetArgs = {
    "map_files" : ["/root/workspace/tracks/FSE23.yaml"],
    "noAugment": True,
}
obs = env.reset(options=resetArgs)[0]

count = 0

def obs_to_image(obs):
    leftImage = np.transpose(obs["cameraLeft"][2], (2, 1, 0))
    frontImage = np.transpose(obs["cameraFront"][2], (2, 1, 0))
    rightImage = np.transpose(obs["cameraRight"][2], (2, 1, 0))
    camImageStitched = np.zeros((frontImage.shape[0], 3*frontImage.shape[1], frontImage.shape[2]), dtype=np.uint8)
    camImageStitched[:,0:frontImage.shape[1]] = leftImage
    camImageStitched[:,frontImage.shape[1]:2*frontImage.shape[1]] = frontImage
    camImageStitched[:,2*frontImage.shape[1]:3*frontImage.shape[1]] = rightImage
    return camImageStitched

im = obs_to_image(obs)

command = [
    'ffmpeg',
    '-y',                     # Overwrite output file
    '-f', 'rawvideo',         # Input format
    '-vcodec', 'rawvideo',
    '-s', f'{im.shape[1]}x{im.shape[0]}', # Size
    '-pix_fmt', 'bgr24',       # OpenCV uses BGR, so we tell FFmpeg to expect BGR
    '-r', str(10),           # Input framerate
    '-i', '-',                # Input comes from a pipe
    '-c:v', 'libx264',        # Video codec to use
    '-pix_fmt', 'yuv420p',    # Output pixel format (standard for compatibility)
    '-preset', 'medium',      # Encoding speed vs compression
    '-crf', '17',             # QUALITY CONTROL: 0 (lossless) to 51 (worst)
    video_path
]
process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

cv2.imwrite("/root/workspace/viz/render/outRender"+str(count)+".png",im)
count += 1

done = False
actor.eval()

actions = []
info = None

while not done:
    # Choose processing depending on model type
    with torch.no_grad():
        if args.model_type == 'vision':
            vision, sensors = networks.flattenFuncVisionSingle(obs)
            vision = vision.to(device)
            sensors = sensors.to(device)
            action = actor.get_action(vision, sensors)[-1].cpu().detach().numpy()[0]
        else:
            obs_flattened = networks.flattenFuncStateSingle(obs).to(device)
            action = actor.get_action(obs_flattened)[-1].cpu().detach().numpy()[0]

    actions.append(action)
    obsNew, reward, terminated, truncated, info = env.step(action)

    camView = obs_to_image(obsNew)
    cv2.imwrite("/root/workspace/viz/render/outRender"+str(count)+".png",camView)
    process.stdin.write(camView)

    obs = obsNew.copy()
    done = terminated or truncated or (count>1000)
    count += 1

print(info)
process.stdin.close()
process.wait()