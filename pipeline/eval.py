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

pacsim = pacsimEnv(None)
env = FrameStackObservation(pacsim, stack_size=3)

import networks

import subprocess
import time

import os
import csv
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLI args: choose model type and path
parser = argparse.ArgumentParser(description="Evaluator for models")
parser.add_argument('--model-type', choices=['state', 'vision'], default='state', help='Type of model to evaluate')
parser.add_argument('--model-path', type=str, default=None, help='Path to model to load (overrides defaults)')
parser.add_argument('--noisy', action='store_true', help='Use noisy variant when applicable')
parser.add_argument('--dagger', action='store_true', help='If evaluating a vision model, load the DAgger-trained vision model (resnet_dagger_... prefix)')
parser.add_argument('--latent', action='store_true', help='Prefer latent (VAE) vision models when loading')
args = parser.parse_args()

# Determine default model path if not provided
if args.model_path:
    model_path = args.model_path
else:
    if args.model_type == 'state':
        model_path = 'networks/state_actor_noisy' if args.noisy else 'networks/state_actor'
    else:
        # For vision models, choose between BC pretrain or DAgger model variants
        basename = 'data_state_actor_noisy' if args.noisy else 'data_state_actor'

        if args.dagger:
            model_path = f'networks/resnet_dagger_{basename}'
        else:
            model_path = f'networks/resnet_bc_pretrain_{basename}'

# try common extensions and latent/structured suffixes
# We search for these unless the path is already a valid file and no latent/structured preference was forced
found = False
if getattr(args, 'latent', False):
    suffixes = ['_latent', '', '_structured']
else:
    # If not explicitly latent, prefer structured over the bare name if available
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
actor.eval()

obs = env.reset()[0]

im = env.render()

count = 0

count += 1

done = False
# actor.eval()
normalizeVision = True

actions = []
info = None

mapFiles = []


mapFiles.append("/root/workspace/tracks/FSE22.yaml")
mapFiles.append("/root/workspace/tracks/FSE22_test.yaml")
mapFiles.append("/root/workspace/tracks/FSE23.yaml")
mapFiles.append("/root/workspace/tracks/FSG19.yaml")
mapFiles.append("/root/workspace/tracks/FSG21.yaml")
mapFiles.append("/root/workspace/tracks/FSG23.yaml")
mapFiles.append("/root/workspace/tracks/FSS19.yaml")
mapFiles.append("/root/workspace/tracks/FSS22_V1.yaml")
mapFiles.append("/root/workspace/tracks/FSS22_V2.yaml")
mapFiles.append("/root/workspace/tracks/FSO20.yaml")

mapFiles.append("/root/workspace/tracks/FSI24.yaml")
mapFiles.append("/root/workspace/tracks/FSCZ24.yaml")


mapFiles.append("/root/workspace/tracks/FSG24.yaml")
mapFiles.append("/root/workspace/tracks/FSE24.yaml")
mapFiles.append("/root/workspace/tracks/FSG25.yaml")
mapFiles.append("/root/workspace/tracks/FSCZ25.yaml")


# Prepare output CSV named by configuration and timestamp
os.makedirs('eval', exist_ok=True)
model_base = os.path.splitext(os.path.basename(model_path))[0]

# Detect type if not already in filename (consistent with bc/dagger outputs)
if args.model_type == 'vision':
    m_type = 'latent' if hasattr(actor, 'vae') else 'structured'
    if m_type not in model_base:
        model_base = f"{model_base}_{m_type}"

timestamp = int(time.time())
csv_fname = f"eval/test_{model_base}_{timestamp}.csv"
f = open(csv_fname, 'w', newline='', encoding='utf-8')
writer = csv.writer(f)
# write the header
writer.writerow(["Track", "success", "time"])
print(f"Writing evaluation results to: {csv_fname}")

for i in mapFiles:
    options = {"map_files" : [i]}
    options["noAugment"] = True
    obs = env.reset(options=options)[0]
    done = False
    while not done:
        # Choose processing depending on model type
        if args.model_type == 'vision':
            vision, sensors = networks.flattenFuncVisionSingle(obs)
            vision = vision.to(device)
            sensors = sensors.to(device)
            with torch.no_grad():
                # both state and vision actors return the 'mean' as the last element
                action_tensor = actor.get_action(vision, sensors)[-1]
        else:
            obs_flattened = networks.flattenFuncStateSingle(obs).to(device)
            with torch.no_grad():
                action_tensor = actor.get_action(obs_flattened)[-1]

        action = action_tensor.cpu().detach().numpy()[0]

        actions.append(action)
        obsNew, reward, terminated, truncated, info = env.step(action)

        im = env.render()

        obs = obsNew.copy()
        done = terminated or truncated
        count += 1

    track_name = os.path.basename(i).replace('.yaml', '')
    writer.writerow([track_name, str(info["laptime"] > 0.0), str(info["laptime"])])