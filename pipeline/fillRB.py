import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium
from gymnasium.wrappers import FrameStackObservation

import argparse
import os

from tqdm.rich import tqdm, trange

parser = argparse.ArgumentParser(description="fillRB model loader")
parser.add_argument("--model", type=str, default=None, help="Path to model to load (overrides flags)")
parser.add_argument("--noisy", action="store_true", help="Load 'networks/state_actor_noisy' instead of default 'networks/state_actor'")
args = parser.parse_args()

import networks
from pacsimEnv import pacsimEnv
pacsim = pacsimEnv(None)
env = FrameStackObservation(pacsim, stack_size=3)

from buffers import DictReplayBuffer, save_replay_buffer

buffer_size = 8000
rb = DictReplayBuffer(
    buffer_size,
    env.observation_space,
    env.action_space,
    "cpu",
    n_envs=1,
    handle_timeout_termination=False,
)

if args.model:
    model_path = args.model
else:
    model_path = "networks/state_actor_noisy" if args.noisy else "networks/state_actor"

print(f"Loading model: {model_path}")
model_basename = os.path.splitext(os.path.basename(model_path))[0]
save_filename = f"data_{model_basename}.pkl"
model = torch.load(model_path, weights_only=False)
model = model.to("cpu")

pbar = tqdm(total=buffer_size)
while rb.size() < buffer_size:
    obs = env.reset()[0]
    done = False
    while not done and rb.size() < buffer_size:
        obsFiltered = networks.filterObservationForState(obs)
        obs_flattened = networks.flattenFuncStateSingle(obsFiltered)
        pi, log_pi, mean = model.get_action(obs_flattened)
        action = mean.detach().numpy()[0]
        obsNew, reward, terminated, truncated, info = env.step(action)

        rb.add(obs, obsNew, action, reward, terminated, info)
        pbar.update(1)
        obs = obsNew.copy()
        done = terminated or truncated
pbar.close()


print("Writing replay buffer to:", save_filename)
save_replay_buffer(save_filename, rb)
