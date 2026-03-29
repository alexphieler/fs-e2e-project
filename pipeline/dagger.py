import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym

from pacsimEnv import pacsimEnv
from gymnasium.wrappers import FrameStackObservation

import networks
from torch.utils.tensorboard import SummaryWriter
import argparse, os
from buffers import DictReplayBuffer, load_replay_buffer

from tqdm.rich import tqdm, trange

pacsim = pacsimEnv(None)
env = FrameStackObservation(pacsim, stack_size=3)


device = "cuda"


# CLI args: choose replay buffer path or use noisy buffer; optionally specify pretrained BC model
parser = argparse.ArgumentParser(description="DAgger trainer using BC pretraining")
parser.add_argument("--buffer", type=str, default=None, help="Path to replay buffer pickle file")
parser.add_argument("--noisy", action="store_true", help="Use the noisy state buffer (data_state_actor_noisy.pkl)")
parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained BC model to initialize vision actor")
parser.add_argument("--latent", action="store_true", help="Prefer latent BC models (with _latent suffix)")
args = parser.parse_args()

if args.buffer:
    rb_path = args.buffer
else:
    rb_path = "data_state_actor_noisy.pkl" if args.noisy else "data_state_actor.pkl"
# Fallback order: chosen path -> regular noisy -> default -> data.pkl
if not os.path.exists(rb_path):
    for fallback in ["data_state_actor_noisy.pkl", "data_state_actor.pkl", "data.pkl"]:
        if os.path.exists(fallback):
            rb_path = fallback
            break

rb = load_replay_buffer(rb_path)
print(f"Loaded replay buffer: {rb_path}")

# TensorBoard writer
rb_basename = os.path.splitext(os.path.basename(rb_path))[0]
suffix = "_latent" if args.latent else ""
writer = SummaryWriter(log_dir=f"runs/dagger_{rb_basename}{suffix}")
writer.add_text('info', f"replay_buffer: {rb_path}")

count = 0

rbExpert = DictReplayBuffer(
    4000,
    env.observation_space,
    env.action_space,
    "cpu",
    n_envs=1,
    handle_timeout_termination=False,
)

rbExpertShort = DictReplayBuffer(
    30,
    env.observation_space,
    env.action_space,
    "cpu",
    n_envs=1,
    handle_timeout_termination=False,
)

alpha = 0.8
batch_size = 32
batch_size_expert = int(batch_size*alpha)
batch_size_expert_short = int((batch_size-batch_size_expert)*0.3)
batch_size_bc = batch_size-batch_size_expert-batch_size_expert_short

# Load pretrained vision actor (BC) — prefer CLI provided path or one derived from the replay buffer basename
if args.pretrained:
    pretrained_path = args.pretrained
else:
    candidate = f"networks/resnet_bc_pretrain_{rb_basename}"
    pretrained_path = None
    # try candidate and common fallbacks, also try latent/structured suffixes
    if getattr(args, 'latent', False):
        suffixes = ['_latent', '', '_structured']
    else:
        # Default to preferring structured over bare name if --latent is not set
        suffixes = ['_structured', '', '_latent']
    exts = ['', '.pt', '.pth', '.pkl']
    for base in [candidate, "networks/resnet_bc_pretrain"]:
        for s in suffixes:
            for e in exts:
                p = base + s + e
                if os.path.isfile(p):
                    pretrained_path = p
                    break
            if pretrained_path:
                break
        if pretrained_path:
            break
    if pretrained_path is None:
        pretrained_path = "networks/resnet_bc_pretrain"

print(f"Loading pretrained vision actor: {pretrained_path}")
actor = torch.load(pretrained_path, weights_only=False)
actor = actor.to(device)

# Load state actor (supports clean, noisy variants)
if args.noisy:
    state_candidate = "networks/state_actor_noisy"
else:
    state_candidate = "networks/state_actor"

state_actor_path = None
candidates = [state_candidate, state_candidate.replace('_noisy',''), 'networks/state_actor']
for p in candidates:
    if os.path.isfile(p):
        state_actor_path = p
        break
    for ext in ['.pt', '.pth', '.pkl']:
        if os.path.isfile(p + ext):
            state_actor_path = p + ext
            break
    if state_actor_path:
        break
if state_actor_path is None:
    # fallback to unqualified name
    state_actor_path = "networks/state_actor_noisy" if args.noisy else "networks/state_actor"

print(f"Loading state actor: {state_actor_path}")
state_actor = torch.load(state_actor_path, weights_only=False).to(device)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)

normalizeVision = True

train_steps = 300
batches_per_iter = 120
beta = 0.6
beta_decay = 0.965
log_step = 0
for outer_iter in tqdm(range(0, train_steps)):


    obs = env.reset()[0]
    done = False
    actor.eval()
    print("beta: " + str(beta))
    while not done:
        obsFiltered = networks.filterObservationForState(obs)
        obs_flattened = networks.flattenFuncStateSingle(obsFiltered).to(device)
        _, _, state_mean = state_actor.get_action(obs_flattened)
        state_action = state_mean.detach().cpu().numpy()[0]

        vision, sensors = networks.flattenFuncVisionSingle(obs)
        vision = vision.to(device)
        sensors = sensors.to(device)

        vision_action, _ , _, vision_mean = actor.get_action(vision, sensors)
        vision_action = vision_mean[0].detach().cpu().numpy()

        pi = beta*state_action + (1.0-beta)*vision_action

        obsNew, reward, terminated, truncated, info = env.step(pi)

        rbExpert.add(obs, obsNew, state_action, reward, terminated, info)
        rbExpertShort.add(obs, obsNew, state_action, reward, terminated, info)
        obs = obsNew.copy()
        done = terminated or truncated
        # count += 1
    actor.train()
    beta = beta*beta_decay

    print("new iter")
    # Log beta value for this iteration
    try:
        writer.add_scalar('misc/beta', beta, outer_iter)
    except Exception as e:
        print(f"TensorBoard logging failed: {e}")

    for i in range(batches_per_iter):
        bufferSamplesBC = rb.sample(batch_size_bc)
        observationsBC = bufferSamplesBC[0]
        bufferActionsBC = bufferSamplesBC[1].to(device)
        visionBC, sensorsBC = networks.flattenFuncVision(observationsBC)
        visionBC = visionBC.to(device)
        sensorsBC = sensorsBC.to(device)

        bufferSamplesExpert = rbExpert.sample(batch_size_expert)
        observationsExpert = bufferSamplesExpert[0]
        bufferActionsExpert = bufferSamplesExpert[1].to(device)
        visionExpert, sensorsExpert = networks.flattenFuncVision(observationsExpert)
        visionExpert = visionExpert.to(device)
        sensorsExpert = sensorsExpert.to(device)

        bufferSamplesExpertShort = rbExpertShort.sample(batch_size_expert_short)
        observationsExpertShort = bufferSamplesExpertShort[0]
        bufferActionsExpertShort = bufferSamplesExpertShort[1].to(device)
        visionExpertShort, sensorsExpertShort = networks.flattenFuncVision(observationsExpertShort)
        visionExpertShort = visionExpertShort.to(device)
        sensorsExpertShort = sensorsExpertShort.to(device)

        vision = torch.cat((visionBC, visionExpert, visionExpertShort))
        sensors = torch.cat((sensorsBC, sensorsExpert, sensorsExpertShort))
        actionLabels = torch.cat((bufferActionsBC, bufferActionsExpert, bufferActionsExpertShort))

        action, _, _, mean = actor.get_action(vision, sensors)

        actor_loss = torch.nn.functional.mse_loss(mean, actionLabels)

        # Optimize the actor
        print(actor_loss.item())
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Logging
        try:
            log_step += 1
            writer.add_scalar('loss/actor', actor_loss.item(), log_step)
        except Exception as e:
            print(f"TensorBoard logging failed: {e}")

# Save model with name reflecting the buffer used and actor type
model_type = 'latent' if hasattr(actor, 'vae') else 'structured'
save_name = f"networks/resnet_dagger_{rb_basename}_{model_type}.pth"

torch.save(actor, save_name)
print(f"Saved actor to: {save_name}")
# TensorBoard finalize
try:
    writer.add_text('model/saved', save_name)
    writer.flush()
    writer.close()
except Exception as e:
    print(f"TensorBoard finalize failed: {e}")
