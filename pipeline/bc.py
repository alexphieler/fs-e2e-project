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

from tqdm.rich import tqdm, trange
from buffers import load_replay_buffer

pacsim = pacsimEnv(None)
env = FrameStackObservation(pacsim, stack_size=3)

import networks
from torch.utils.tensorboard import SummaryWriter

imgDim = (3,3,3, int(306/1), int(256/1))
sensor_dim = 3*(4+3+1)
action_dim = 5
device = "cuda"

# Actor will be initialized after parsing CLI args so we can choose latent or structured

# CLI args: choose replay buffer path or use noisy buffer
import argparse, os
parser = argparse.ArgumentParser(description="Behavior cloning trainer")
parser.add_argument("--buffer", type=str, default=None, help="Path to replay buffer pickle file")
parser.add_argument("--noisy", action="store_true", help="Use the noisy state buffer (data_state_actor_noisy.pkl)")
parser.add_argument("--latent", action="store_true", help="Use LatentHierarchicalActor (uses a ConvVAE encoder)")
parser.add_argument("--vae-path", type=str, default=None, help="Path to pretrained VAE weights (.pth/.pt). Optional when using --latent.")
parser.add_argument("--unfreeze-vae", action="store_true", help="Allow fine-tuning of the VAE weights during BC training")
args = parser.parse_args()

if args.buffer:
    rb_path = args.buffer
else:
    rb_path = "data_state_actor_noisy.pkl" if args.noisy else "data_state_actor.pkl"
# Fallback: prefer regular noisy then original data.pkl
if not os.path.exists(rb_path):
    for fallback in ["data_state_actor_noisy.pkl", "data_state_actor.pkl", "data.pkl"]:
        if os.path.exists(fallback):
            rb_path = fallback
            break

rb = load_replay_buffer(rb_path)
print(f"Loaded replay buffer: {rb_path}")

# TensorBoard writer
rb_basename = os.path.splitext(os.path.basename(rb_path))[0]
writer = SummaryWriter(log_dir=f"runs/bc_{rb_basename}")
writer.add_text('info', f"replay_buffer: {rb_path}")

count = 0

# Initialize actor depending on CLI flags (structured vs latent)
if getattr(args, 'latent', False):
    # Create ConvVAE and optionally load weights
    vae = networks.ConvVAE()
    vae_loaded = False
    # Determine VAE path: use --vae-path if provided, otherwise look for common defaults
    vae_path = getattr(args, 'vae_path', None)
    if not vae_path:
        # Common filenames (no-extension, .pth, .pt)
        candidates = ["networks/vae", "networks/vae.pth", "networks/vae.pt"]
        for cand in candidates:
            if os.path.exists(cand):
                vae_path = cand
                break
    if vae_path:
        if os.path.exists(vae_path):
            try:
                state = torch.load(vae_path, map_location='cpu', weights_only=False)
                
                # Support a few common checkpoint layouts
                load_target = None
                if isinstance(state, dict):
                    for key in ('state_dict', 'model', 'vae'):
                        if key in state:
                            load_target = state[key]
                            break
                    if load_target is None:
                        load_target = state
                elif isinstance(state, nn.Module):
                    vae = state
                    load_target = None
                if load_target is not None:
                    vae.load_state_dict(load_target)
                vae_loaded = True
                print(f"Loaded VAE from: {vae_path}")
            except Exception as e:
                print(f"Failed to load VAE from {vae_path}: {e}")
        else:
            print(f"VAE path not found: {vae_path}. Using random initialized VAE.")
    else:
        # No path supplied and no default found
        print("No VAE path provided and no default 'models/vae' found; using randomly initialized VAE.")
    actor = networks.LatentHierarchicalActor(
        vae, 
        imgDim, 
        sensor_dim, 
        action_dim,
        freeze_vae=not getattr(args, 'unfreeze_vae', False)
    ).to(device)
    writer.add_text('model/type', f'latent (vae_loaded={vae_loaded}, frozen={not getattr(args, "unfreeze_vae", False)})')
else:
    actor = networks.StructuredHierarchicalActor(imgDim, sensor_dim, action_dim).to(device)
    writer.add_text('model/type', 'structured')

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=0.9998)

train_steps = 10000
batch_size = 16
for _ in tqdm(range(0, train_steps)):
    bufferSamples = rb.sample(batch_size)
    observations = bufferSamples[0]
    bufferActions = bufferSamples[1].to(device)
    vision, sensors = networks.flattenFuncVision(observations)
    vision = vision.to(device)
    sensors = sensors.to(device)
    # print(f"Vision tensor shape: {vision.shape}")
    # print(f"Sensors tensor shape: {sensors.shape}")

    action, _, _, mean = actor.get_action(vision, sensors)

    actor_loss = torch.nn.functional.mse_loss(mean, bufferActions)
    # Optimize the actor
    print(actor_loss.item())
    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optimizer.step()
    scheduler.step()

    # Logging to TensorBoard
    count += 1
    try:
        writer.add_scalar('loss/actor', actor_loss.item(), count)
        writer.add_scalar('lr', scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else 0.0, count)
    except Exception as e:
        # Don't crash training if writer fails
        print(f"TensorBoard logging failed: {e}")

    # print(bufferActions)
    # print(mean)

# Save model with name reflecting the buffer used
rb_basename = os.path.splitext(os.path.basename(rb_path))[0]
# Include model type in filename (latent vs structured)
model_type = 'latent' if getattr(args, 'latent', False) else 'structured'
save_name = f"networks/resnet_bc_pretrain_{rb_basename}_{model_type}.pth"

torch.save(actor, save_name)
print(f"Saved actor to: {save_name}")
# TensorBoard finalize
try:
    writer.add_text('model/saved', save_name)
    writer.flush()
    writer.close()
except Exception as e:
    print(f"TensorBoard finalize failed: {e}")