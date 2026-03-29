# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# from cleanrl_utils.buffers import ReplayBuffer
from buffers import DictReplayBuffer

from networks import Actor, SoftQNetwork
import networks

from pacsimEnv import pacsimEnv
from gymnasium.wrappers import FrameStackObservation
import gymnasium

from tqdm.rich import tqdm, trange


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "pacsimEnv"
    """the environment id of the task"""
    total_timesteps: int = 4000_000
    # total_timesteps: int = 30000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(3e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    # tau: float = 0.001
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1e4
    """timestep to start learning"""
    policy_lr: float = 3e-4
    # policy_lr: float = 2e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    # q_lr: float = 5e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    noise_augment: bool = False
    """apply noise augementation to actions during training"""
    reward_mode: str = "conservative"
    """the reward mode for the environment (conservative or aggressive)"""


# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         if capture_video and idx == 0:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env.action_space.seed(seed)
#         return env

#     return thunk

def make_env(seed, reward_mode="conservative"):
    def thunk():
        pacsimArgs = {"cam_sim" : False}
        if reward_mode == "aggressive":
            pacsimArgs.update({
                "lambda_progress": 0.02,
                "lambda_tracking": 0.003,
                "lambda_finish": 10.0,
                "lambda_collition": 10.0,
                "lambda_stand": 0.5,
                "lambda_constant": 0.1,
                "lambda_slipAngle": 0.005,
                "lambda_slipRatio": 0.05,
                "lambda_actionRate": 0.002,
                "lambda_lateral_consistency": 0.001,
                "lambda_longitudinal_consistency": 0.0002,
            })
        elif reward_mode == "conservative":
            pacsimArgs.update({
                "lambda_progress": 0.007,
                "lambda_tracking": 0.01,
                "lambda_finish": 10.0,
                "lambda_collition": 10.0,
                "lambda_stand": 0.5,
                "lambda_constant": 0.02,
                "lambda_slipAngle": 0.005,
                "lambda_slipRatio": 0.05,
                "lambda_actionRate": 0.002,
                "lambda_lateral_consistency": 0.001,
                "lambda_longitudinal_consistency": 0.0002,
            })
            
        pacsim = pacsimEnv(pacsimArgs)
        env = FrameStackObservation(pacsim, stack_size=3)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # outermost so info dict is preserved
        env.action_space.seed(seed)
        return env
    return thunk


if __name__ == "__main__":

    args = tyro.cli(Args)
    envId = args.env_id
    if getattr(args, "noise_augment", False):
        envId = envId + "_noiseAugmented"
    run_name = f"{envId}_{args.reward_mode}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # state based policies are faster on cpu
    device = "cpu"
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, args.reward_mode) for i in range(args.num_envs)]
    )
    print(args)
    # noise augmentation flags
    print("noise_augment:", getattr(args, "noise_augment", False))

    obs_space_filtered = networks.filterObservationForState(envs.single_observation_space)
    actor = Actor(obs_space_filtered).to(device)
    qf1 = SoftQNetwork(obs_space_filtered).to(device)
    qf2 = SoftQNetwork(obs_space_filtered).to(device)
    qf1_target = SoftQNetwork(obs_space_filtered).to(device)
    qf2_target = SoftQNetwork(obs_space_filtered).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        # target_entropy = -torch.prod(action_space_shape).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    rb = DictReplayBuffer(
        args.buffer_size,
        gymnasium.spaces.Dict(networks.filterObservationForState(envs.single_observation_space)),
        # envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    # obs = networks.filterObservationForState(obs)
    for global_step in tqdm(range(args.total_timesteps)):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs_flattened = networks.flattenFuncState(obs)
            actions, _, _ = actor.get_action(torch.Tensor(obs_flattened).to(device))
            actions = actions.detach().cpu().numpy()

        # 1. Define base noise constants
        STEER_NOISE_BASE = 0.095
        TORQUE_NOISE_BASE = 0.15
        
        # 2. Determine your MAX target multiplier based on flags
        # If normal is on, we aim for 1x.
        if getattr(args, "noise_augment", False):
            target_max_multiplier = 1.0
        else:
            target_max_multiplier = 0.0 # Noise disabled

        # 3. Define the Schedule
        #   Step 0 to Start:       1.0 Noise (Standard noise training)
        #   Start to End:          Ramp linearly 1.0 -> Max
        #   End to Finish:         Hold Max (Robustness training)
        anneal_start_step = int(0.3 * args.total_timesteps) # Start noise at 50%
        anneal_end_step   = int(0.5 * args.total_timesteps) # Reach max noise at 70%
        
        # current_multiplier = 1.0 if target_max_multiplier > 0 else 0.0
        current_multiplier = 0.0

        if target_max_multiplier > 0.0 and global_step > anneal_start_step:
            if global_step >= anneal_end_step:
                # We have finished the ramp, hold max noise
                current_multiplier = target_max_multiplier
            else:
                # Linear Ramp Calculation
                steps_elapsed = global_step - anneal_start_step
                ramp_duration = anneal_end_step - anneal_start_step
                progress = steps_elapsed / ramp_duration
                current_multiplier = progress * (target_max_multiplier)

        # 4. Apply the calculated noise
        if current_multiplier > 0:
            # Create weight vector
            # Assuming action space is [steer, fl_wheel, fr_wheel, rl_wheel, rr_wheel]
            base_weights = np.array([STEER_NOISE_BASE, TORQUE_NOISE_BASE, TORQUE_NOISE_BASE, TORQUE_NOISE_BASE, TORQUE_NOISE_BASE])
            
            # Scale weights by the current annealed multiplier
            current_sigma = base_weights * current_multiplier
            
            noise = np.random.normal(loc=0, scale=current_sigma)
            
            # Save intended actions for the buffer (Critical for DAgger/BC logic usually, 
            # though SAC technically learns off-policy so storing the noisy action is standard. 
            # Your code stores 'actionsIntended', so we keep that flow.)
            actionsIntended = actions.copy()
            
            actions = actions + noise
            actions = np.clip(actions, -1.0, 1.0)
            
        else:
            actionsIntended = actions.copy()

        # Log the noise level occasionally to Tensorboard so you can debug
        if global_step % 1000 == 0:
            writer.add_scalar("debug/noise_multiplier", current_multiplier, global_step)
        # print(actions)
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # print(infos)
        if "episode" in infos:
            # "_episode" mask indicates which envs finished an episode
            for idx, finished in enumerate(infos["_episode"]):
                if finished:
                    ep_return = infos["episode"]["r"][idx]
                    ep_length = infos["episode"]["l"][idx]
                    print(f"global_step={global_step}, episodic_return={ep_return}, episodic_length={ep_length}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actionsIntended, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            obs_flattened = networks.flattenFuncState(data.observations).to(device)
            next_obs_flattened = networks.flattenFuncState(data.next_observations).to(device)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(next_obs_flattened)
                qf1_next_target = qf1_target(next_obs_flattened, next_state_actions)
                qf2_next_target = qf2_target(next_obs_flattened, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.to(device).flatten() + (1 - data.dones.to(device).flatten()) * args.gamma * (min_qf_next_target).view(-1)

            data_actions = data.actions.to(device)
            qf1_a_values = qf1(obs_flattened, data_actions).view(-1)
            qf2_a_values = qf2(obs_flattened, data_actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()

            torch.nn.utils.clip_grad_norm_(qf1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(qf2.parameters(), max_norm=1.0)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(obs_flattened)
                    qf1_pi = qf1(obs_flattened, pi)
                    qf2_pi = qf2(obs_flattened, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(obs_flattened)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()

    # Save model according to augmentation mode used
    reward_suffix = f"_{args.reward_mode}"
    if getattr(args, "noise_augment", False):
        torch.save(actor, f"networks/state_actor_noisy{reward_suffix}")
        torch.save(qf1, f"networks/state_qf1_noisy{reward_suffix}")
        torch.save(qf2, f"networks/state_qf2_noisy{reward_suffix}")
    else:
        torch.save(actor, f"networks/state_actor{reward_suffix}")
        torch.save(qf1, f"networks/state_qf1{reward_suffix}")
        torch.save(qf2, f"networks/state_qf2{reward_suffix}")