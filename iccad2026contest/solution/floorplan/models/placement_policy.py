"""Actor-Critic placement policy over discretised canvas grid."""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PlacementPolicy(nn.Module):
    """
    Actor-Critic architecture.

    Observation:
      - current_emb: (128,) embedding of block being placed
      - graph_emb:   (128,) whole-netlist embedding
      - canvas:      (grid_size, grid_size) occupancy map
      - legal_mask:  (grid_size, grid_size) bool mask
      - step_frac:   scalar

    Action: integer in [0, grid_size * grid_size)
    """

    def __init__(
        self,
        node_emb_dim: int = 128,
        grid_size: int = 256,
        canvas_channels: int = 1,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.node_emb_dim = node_emb_dim
        self.canvas_feat_dim = 64

        # Canvas encoder: conv2d -> flatten -> linear
        # Input canvas is grid_size x grid_size; we downsample first
        self.canvas_encoder = nn.Sequential(
            nn.Conv2d(canvas_channels, 16, kernel_size=8, stride=4, padding=2),  # /4
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, self.canvas_feat_dim),
            nn.ReLU(),
        )

        # Actor MLP: takes [current_emb, graph_emb, canvas_feat, step_frac] -> logits over grid
        actor_in = node_emb_dim + node_emb_dim + self.canvas_feat_dim + 1
        self.actor_net = nn.Sequential(
            nn.Linear(actor_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Spatial upsampling head: 256-dim -> grid_size x grid_size logits
        # We decode to a coarser grid and upsample
        self.spatial_head = nn.Sequential(
            nn.Linear(256, 64 * 8 * 8),  # 64 channels, 8x8
            nn.ReLU(),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),    # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),    # 256x256
        )

        # Critic MLP
        critic_in = actor_in
        self.critic_net = nn.Sequential(
            nn.Linear(critic_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _encode_obs(self, obs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation dict into (actor_features, critic_features).

        Both are the same here (shared features).
        """
        device = obs['current_emb'].device

        # Canvas: add batch and channel dims: (1, 1, H, W)
        canvas = obs['canvas'].unsqueeze(0).unsqueeze(0).float()
        canvas_feat = self.canvas_encoder(canvas).squeeze(0)  # (canvas_feat_dim,)

        step_frac = torch.tensor([obs.get('step_frac', 0.0)], device=device, dtype=torch.float32)

        feat = torch.cat([obs['current_emb'], obs['graph_emb'], canvas_feat, step_frac], dim=0)
        return feat, feat

    def get_action(
        self,
        obs: dict,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample (or take argmax) action from actor.

        Returns:
            action: int (flattened cell index)
            log_prob: scalar tensor
            entropy: scalar tensor
        """
        actor_feat, _ = self._encode_obs(obs)
        logits = self._compute_logits(actor_feat)  # (grid_size * grid_size,)

        # Apply legal mask (log softmax trick)
        legal_mask = obs['legal_mask'].flatten().bool()
        logits_masked = logits.clone()
        logits_masked[~legal_mask] = -1e9

        dist = Categorical(logits=logits_masked)

        if deterministic:
            action = int(logits_masked.argmax().item())
            log_prob = dist.log_prob(torch.tensor(action, device=logits.device))
        else:
            action_tensor = dist.sample()
            action = int(action_tensor.item())
            log_prob = dist.log_prob(action_tensor)

        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_action(
        self,
        obs: dict,
        action: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate stored action under current policy (for PPO update).

        Returns:
            log_prob: scalar tensor
            entropy: scalar tensor
            value: scalar tensor
        """
        actor_feat, critic_feat = self._encode_obs(obs)
        logits = self._compute_logits(actor_feat)

        legal_mask = obs['legal_mask'].flatten().bool()
        logits_masked = logits.clone()
        logits_masked[~legal_mask] = -1e9

        dist = Categorical(logits=logits_masked)
        action_tensor = torch.tensor(action, device=logits.device)
        log_prob = dist.log_prob(action_tensor)
        entropy = dist.entropy()

        value = self.critic_net(critic_feat).squeeze(-1)
        return log_prob, entropy, value

    def get_value(self, obs: dict) -> torch.Tensor:
        """Compute state value estimate."""
        _, critic_feat = self._encode_obs(obs)
        return self.critic_net(critic_feat).squeeze(-1)

    def _compute_logits(self, actor_feat: torch.Tensor) -> torch.Tensor:
        """Compute per-cell logits using spatial decoder."""
        h = self.actor_net(actor_feat)  # (256,)
        spatial = self.spatial_head(h)  # (64*8*8,)
        spatial = spatial.view(1, 64, 8, 8)
        logit_map = self.upsample(spatial)  # (1, 1, grid_size, grid_size)
        logit_map = logit_map.squeeze(0).squeeze(0)  # (grid_size, grid_size)

        # Interpolate to exact grid_size if needed
        if logit_map.shape[0] != self.grid_size or logit_map.shape[1] != self.grid_size:
            logit_map = F.interpolate(
                logit_map.unsqueeze(0).unsqueeze(0),
                size=(self.grid_size, self.grid_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0).squeeze(0)

        return logit_map.flatten()  # (grid_size * grid_size,)
