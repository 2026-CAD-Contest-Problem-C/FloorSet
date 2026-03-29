"""
Phase 1b: PPO training loop for placement policy.
"""
import os
import random
import collections
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ..models.full_agent import FloorplanAgent
from ..env.floorplan_env import FloorplanEnv
from ..training.losses import ppo_policy_loss, ppo_value_loss
from ..data.dataset import FloorSetDataset
from ..utils.logger import Logger
from ..utils.seed import set_seed


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores transitions for one PPO update cycle."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def clear(self):
        self.__init__()

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def __len__(self):
        return len(self.actions)

    def compute_returns(self, gamma: float = 1.0, gae_lambda: float = 0.95):
        """Compute GAE advantages and discounted returns."""
        n = len(self.rewards)
        advantages = torch.zeros(n)
        returns = torch.zeros(n)

        last_gae = 0.0
        last_value = 0.0

        values = [float(v) if isinstance(v, torch.Tensor) else v for v in self.values]
        rewards = [float(r) for r in self.rewards]
        dones = [float(d) for d in self.dones]

        for t in reversed(range(n)):
            if dones[t]:
                next_value = 0.0
                last_gae = 0.0
            else:
                next_value = values[t + 1] if t + 1 < n else 0.0

            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        # Normalise advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp(min=1e-8)
        advantages = (advantages - adv_mean) / adv_std

        return advantages, returns


# ---------------------------------------------------------------------------
# Moving average baseline
# ---------------------------------------------------------------------------

class MovingAvgBaseline:
    def __init__(self, alpha: float = 0.99):
        self.alpha = alpha
        self.value = 0.0
        self._initialised = False

    def update(self, reward: float) -> float:
        if not self._initialised:
            self.value = reward
            self._initialised = True
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * reward
        return self.value

    def get(self) -> float:
        return self.value


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_rl(cfg: dict, gnn_ckpt: Optional[str] = None, output_dir: str = '.'):
    """Run PPO training."""
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"RL training on {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Build agent
    agent = FloorplanAgent.from_config(cfg).to(device)

    # Load pretrained GNN if available
    if gnn_ckpt and os.path.exists(gnn_ckpt):
        ckpt = torch.load(gnn_ckpt, map_location=device, weights_only=True)
        if 'gnn_state' in ckpt:
            agent.gnn_encoder.load_state_dict(ckpt['gnn_state'])
            print(f"Loaded pretrained GNN from {gnn_ckpt}")
        else:
            agent.gnn_encoder.load_state_dict(ckpt)

    # Build environment
    env = FloorplanEnv(
        agent=agent,
        grid_size=cfg['placement']['grid_size'],
        alpha=cfg['reward']['alpha'],
        overlap_pen=cfg['reward']['overlap_pen'],
        device=str(device),
    )

    # Optimizers (separate for actor and critic / GNN)
    placement_params = list(agent.placement_policy.parameters()) + \
                       list(agent.shape_policy.parameters())
    gnn_params = list(agent.gnn_encoder.parameters())

    optimizer_actor = torch.optim.Adam(placement_params, lr=cfg['placement']['ppo_lr_actor'])
    optimizer_critic = torch.optim.Adam(
        placement_params, lr=cfg['placement']['ppo_lr_critic']
    )
    optimizer_gnn = torch.optim.Adam(gnn_params, lr=cfg['placement']['ppo_lr_actor'] * 0.1)

    logger = Logger(project='floorset-rl', enabled=False)
    baseline = MovingAvgBaseline()
    buffer = RolloutBuffer()

    curriculum_stages = cfg['curriculum']['stages']
    data_root = cfg['data']['dataset_path']
    # RL needs full data (block_meta, b2b_connectivity, etc.) — use separate cache dir
    cache_dir = cfg['data'].get('rl_cache_dir', cfg['data']['cache_dir'])
    train_data_path = cfg['data'].get('train_data_path', None)
    freeze_gnn_steps = cfg['placement']['freeze_gnn_steps']

    global_step = 0
    best_val_reward = float('-inf')

    # Build datasets for all curriculum stages
    datasets = []
    for stage in curriculum_stages:
        ds = FloorSetDataset(
            root=data_root, split='train',
            block_count_range=tuple(stage),
            cache_dir=cache_dir, training=True,
            train_data_path=train_data_path,
            lite=False,
        )
        datasets.append(ds)
        print(f"Stage {stage}: {len(ds)} samples")

    # Curriculum loop
    for stage_idx, stage in enumerate(curriculum_stages):
        print(f"\n=== Curriculum Stage {stage_idx+1}: {stage} blocks ===")
        ds_current = datasets[stage_idx]
        ds_replay = datasets[:stage_idx]  # previous stages for replay

        if len(ds_current) == 0:
            print("No data for this stage, skipping")
            continue

        n_updates = 2000  # Updates per stage
        rollout_buf_size = cfg['placement']['rollout_buf']

        for update in tqdm(range(n_updates), desc=f"Stage {stage_idx+1}"):
            global_step += 1

            # GNN freeze schedule
            gnn_frozen = (global_step < freeze_gnn_steps)
            for p in agent.gnn_encoder.parameters():
                p.requires_grad = not gnn_frozen

            # Sample netlist
            replay_frac = cfg['curriculum']['replay_frac']
            if ds_replay and random.random() < replay_frac:
                ds = random.choice(ds_replay)
                idx = random.randint(0, max(len(ds) - 1, 0))
            else:
                idx = random.randint(0, max(len(ds_current) - 1, 0))
                ds = ds_current

            data = ds[idx]
            if data is None:
                continue

            # Run episode
            episode_buffer = _run_episode(agent, env, data, device)
            if not episode_buffer:
                continue

            # Accumulate into main buffer
            for transition in episode_buffer:
                buffer.add(*transition)

            if len(buffer) >= rollout_buf_size * data.num_nodes:
                # PPO update
                advantages, returns = buffer.compute_returns(
                    gamma=cfg['placement']['gamma'],
                    gae_lambda=cfg['placement']['gae_lambda'],
                )

                _ppo_update(
                    agent, buffer, advantages, returns,
                    optimizer_actor, optimizer_critic,
                    optimizer_gnn if not gnn_frozen else None,
                    cfg, device,
                )

                # Log reward
                ep_rewards = [r for r in buffer.rewards if r != 0.0]
                if ep_rewards:
                    mean_reward = sum(ep_rewards) / len(ep_rewards)
                    baseline.update(mean_reward)
                    if update % 100 == 0:
                        print(f"  Step {global_step}: mean_reward={mean_reward:.4f} "
                              f"baseline={baseline.get():.4f}")

                buffer.clear()

            # Periodic evaluation
            if global_step % 500 == 0:
                val_reward = _evaluate_stage(agent, env, ds_current, device, n_eval=5)
                print(f"  Val reward (stage {stage_idx+1}): {val_reward:.4f}")
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    ckpt_path = os.path.join(output_dir, 'best_agent.pt')
                    agent.save(ckpt_path)
                    print(f"  Saved best agent (reward={val_reward:.4f})")

        # Save stage checkpoint
        ckpt_path = os.path.join(output_dir, f'agent_stage{stage_idx+1}.pt')
        agent.save(ckpt_path)
        print(f"Saved stage {stage_idx+1} checkpoint")

    # Final save
    agent.save(os.path.join(output_dir, 'agent_final.pt'))
    print("RL training complete.")


def _run_episode(agent, env, data, device):
    """Run one full episode. Returns list of (state, action, reward, value, log_prob, done)."""
    obs = env.reset(data)
    if obs is None:
        return []

    transitions = []
    done = False

    while not done:
        # Move obs to device
        obs_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in obs.items()
        }

        with torch.no_grad():
            action, log_prob, entropy = agent.placement_policy.get_action(obs_dev, deterministic=False)
            value = agent.placement_policy.get_value(obs_dev)

        next_obs, reward, done, info = env.step(action)

        transitions.append((obs, action, reward, float(value.item()), float(log_prob.item()), done))
        obs = next_obs if not done else obs

    return transitions


def _ppo_update(agent, buffer, advantages, returns, optimizer_actor, optimizer_critic, optimizer_gnn, cfg, device):
    """Run PPO mini-batch updates."""
    n = len(buffer)
    ppo_epochs = cfg['placement']['ppo_epochs']
    clip_eps = cfg['placement']['ppo_clip']
    entropy_coef = cfg['placement']['entropy_coef']
    value_coef = cfg['placement']['value_coef']
    grad_clip = cfg['placement']['grad_clip']

    # Old log probs (already computed during rollout)
    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
    adv = advantages.to(device)
    ret = returns.to(device)

    for _ in range(ppo_epochs):
        # Process each transition
        for i in range(n):
            obs = buffer.states[i]
            action = buffer.actions[i]

            obs_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in obs.items()
            }

            new_log_prob, entropy, value = agent.placement_policy.evaluate_action(obs_dev, action)

            old_lp = old_log_probs[i].to(device)
            ratio = torch.exp(new_log_prob - old_lp)

            surr1 = ratio * adv[i]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[i]
            policy_loss = -torch.min(surr1, surr2)
            value_loss = (value - ret[i]) ** 2
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            if optimizer_gnn is not None:
                optimizer_gnn.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(agent.placement_policy.parameters(), grad_clip)
            if optimizer_gnn is not None:
                nn.utils.clip_grad_norm_(agent.gnn_encoder.parameters(), grad_clip)

            optimizer_actor.step()
            optimizer_critic.step()
            if optimizer_gnn is not None:
                optimizer_gnn.step()


def _evaluate_stage(agent, env, ds, device, n_eval: int = 5) -> float:
    """Run n_eval episodes and return mean reward."""
    agent.eval()
    rewards = []
    indices = random.sample(range(min(len(ds), 100)), min(n_eval, len(ds)))
    for idx in indices:
        data = ds[idx]
        if data is None:
            continue
        obs = env.reset(data)
        if obs is None:
            continue
        done = False
        ep_reward = 0.0
        while not done:
            obs_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
            with torch.no_grad():
                action, _, _ = agent.placement_policy.get_action(obs_dev, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward = reward if reward != 0.0 else ep_reward
        rewards.append(ep_reward)
    agent.train()
    return sum(rewards) / max(len(rewards), 1)
