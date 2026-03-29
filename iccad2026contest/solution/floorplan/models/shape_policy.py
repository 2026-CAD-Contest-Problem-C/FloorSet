"""Shape policy: predict log aspect ratio for each soft block."""
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ShapePolicy(nn.Module):
    """
    MLP that maps node embeddings to block dimensions (w, h).

    For soft/fixed-shape blocks:
      - Outputs log_r ~ N(mu, sigma) where r = w/h
      - w = sqrt(area * r), h = sqrt(area / r)

    For preplaced blocks: uses ground-truth dimensions from block_meta.
    For fixed-shape blocks: uses ground-truth dimensions from block_meta.
    """

    def __init__(
        self,
        node_emb_dim: int = 128,
        hidden_dim: int = 256,
        log_r_min: float = -2.0,
        log_r_max: float = 2.0,
    ):
        super().__init__()
        self.log_r_min = log_r_min
        self.log_r_max = log_r_max

        # MLP: node_emb_dim -> hidden_dim -> hidden_dim//2 -> 1 (mu)
        self.mu_net = nn.Sequential(
            nn.Linear(node_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Learned log-std (shared across all blocks, per-instance)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        node_emb: torch.Tensor,
        block_meta: List[dict],
        greedy: bool = False,
    ) -> Tuple[List[float], List[float], torch.Tensor]:
        """
        Args:
            node_emb: (k, node_emb_dim)
            block_meta: list of k dicts with keys:
                'area', 'is_fixed_shape', 'is_preplaced', 'gt_w', 'gt_h'
            greedy: if True, use mu directly (evaluation mode)

        Returns:
            w_list: list of k widths
            h_list: list of k heights
            log_prob: scalar tensor (sum of log probs for sampled blocks)
        """
        k = len(block_meta)
        w_list = []
        h_list = []
        log_prob = torch.tensor(0.0, device=node_emb.device)

        # Indices of soft (non-fixed, non-preplaced) blocks
        soft_idx = []
        for i, meta in enumerate(block_meta):
            if meta['is_preplaced'] and meta['gt_w'] is not None:
                w_list.append(meta['gt_w'])
                h_list.append(meta['gt_h'])
            elif meta['is_fixed_shape'] and meta['gt_w'] is not None:
                w_list.append(meta['gt_w'])
                h_list.append(meta['gt_h'])
            else:
                soft_idx.append(i)
                w_list.append(None)
                h_list.append(None)

        if soft_idx:
            soft_emb = node_emb[soft_idx]  # (n_soft, dim)
            mu = self.mu_net(soft_emb).squeeze(-1)  # (n_soft,)
            mu = torch.clamp(mu, self.log_r_min, self.log_r_max)

            std = torch.exp(self.log_std).expand_as(mu).clamp(min=1e-3)
            dist = Normal(mu, std)

            if greedy:
                log_r = mu
            else:
                log_r = dist.rsample()
                log_r = torch.clamp(log_r, self.log_r_min, self.log_r_max)
                log_prob = dist.log_prob(log_r).sum()

            for j, i in enumerate(soft_idx):
                area = max(block_meta[i]['area'], 1e-8)
                r = math.exp(float(log_r[j].item()))
                w = math.sqrt(area * r)
                h = math.sqrt(area / r)
                w_list[i] = w
                h_list[i] = h

        # Sanity: fill any remaining Nones (shouldn't happen)
        for i, meta in enumerate(block_meta):
            if w_list[i] is None:
                area = max(meta['area'], 1e-8)
                s = math.sqrt(area)
                w_list[i] = s
                h_list[i] = s

        return w_list, h_list, log_prob

    def log_prob_given(
        self,
        node_emb: torch.Tensor,
        block_meta: List[dict],
        w_list: List[float],
        h_list: List[float],
    ) -> torch.Tensor:
        """Compute log prob of given (w, h) assignments under current policy."""
        soft_idx = [
            i for i, m in enumerate(block_meta)
            if not (m['is_preplaced'] or m['is_fixed_shape']) or m['gt_w'] is None
        ]
        if not soft_idx:
            return torch.tensor(0.0, device=node_emb.device)

        soft_emb = node_emb[soft_idx]
        mu = self.mu_net(soft_emb).squeeze(-1)
        mu = torch.clamp(mu, self.log_r_min, self.log_r_max)
        std = torch.exp(self.log_std).expand_as(mu).clamp(min=1e-3)
        dist = Normal(mu, std)

        # Recover log_r from w, h
        log_r_values = []
        for i in soft_idx:
            w = w_list[i]
            h = h_list[i]
            if h > 1e-12:
                log_r = math.log(w / h)
            else:
                log_r = 0.0
            log_r_values.append(log_r)

        log_r_tensor = torch.tensor(log_r_values, device=node_emb.device, dtype=torch.float32)
        return dist.log_prob(log_r_tensor).sum()
