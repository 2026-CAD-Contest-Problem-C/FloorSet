"""Loss functions and reward computation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GNN Pretraining Losses
# ---------------------------------------------------------------------------

class MaskedFeatureHead(nn.Module):
    """Task A: predict masked node features."""

    def __init__(self, hidden_dim: int = 128, feat_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        return self.net(node_emb)


class LinkPredHead(nn.Module):
    """Task B: predict link existence between node pairs."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_emb_i: torch.Tensor, node_emb_j: torch.Tensor) -> torch.Tensor:
        """Returns logit for edge (i, j) existing."""
        pair = torch.cat([node_emb_i, node_emb_j], dim=-1)
        return self.net(pair).squeeze(-1)


class HPWLHead(nn.Module):
    """Task C: predict HPWL from graph embedding."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        return self.net(graph_emb).squeeze(-1)


def masked_feature_loss(
    node_emb: torch.Tensor,
    original_x: torch.Tensor,
    mask: torch.Tensor,
    head: MaskedFeatureHead,
) -> torch.Tensor:
    """MSE loss on masked node features."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=node_emb.device)
    pred = head(node_emb[mask])
    target = original_x[mask]
    return F.mse_loss(pred, target)


def link_prediction_loss(
    node_emb: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    head: LinkPredHead,
    neg_ratio: int = 1,
) -> torch.Tensor:
    """Binary cross-entropy link prediction loss."""
    if edge_index.shape[1] == 0 or num_nodes < 2:
        return torch.tensor(0.0, device=node_emb.device)

    # Positive edges (remove self-loops)
    src, dst = edge_index[0], edge_index[1]
    non_self = src != dst
    src_pos = src[non_self]
    dst_pos = dst[non_self]

    if src_pos.shape[0] == 0:
        return torch.tensor(0.0, device=node_emb.device)

    n_pos = src_pos.shape[0]
    n_neg = min(n_pos * neg_ratio, num_nodes * (num_nodes - 1) - n_pos)

    pos_logits = head(node_emb[src_pos], node_emb[dst_pos])
    pos_labels = torch.ones(n_pos, device=node_emb.device)

    # Random negative edges
    neg_src = torch.randint(0, num_nodes, (n_neg,), device=node_emb.device)
    neg_dst = torch.randint(0, num_nodes, (n_neg,), device=node_emb.device)
    neg_logits = head(node_emb[neg_src], node_emb[neg_dst])
    neg_labels = torch.zeros(n_neg, device=node_emb.device)

    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([pos_labels, neg_labels])
    return F.binary_cross_entropy_with_logits(logits, labels)


def hpwl_prediction_loss(
    graph_emb: torch.Tensor,
    hpwl_base: torch.Tensor,
    head: HPWLHead,
) -> torch.Tensor:
    """MSE loss on log(HPWL+1) prediction."""
    pred = head(graph_emb)
    target = torch.log(hpwl_base.float() + 1.0)
    return F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# RL Reward
# ---------------------------------------------------------------------------

def compute_reward(
    hpwl: float,
    area: float,
    overlap: float,
    hpwl_base: float,
    area_base: float,
    alpha: float = 0.5,
    overlap_pen: float = -10.0,
) -> float:
    """
    Compute episode reward.

    Returns a negative cost in approximately [-7.4, -0.7] for feasible solutions.
    """
    if overlap > 1e-6:
        return overlap_pen

    hpwl_gap = (hpwl - hpwl_base) / max(hpwl_base, 1e-8)
    area_gap = (area - area_base) / max(area_base, 1e-8)
    cost = 1.0 + alpha * (hpwl_gap + area_gap)
    return -cost


# ---------------------------------------------------------------------------
# PPO Losses
# ---------------------------------------------------------------------------

def ppo_policy_loss(
    log_prob_new: torch.Tensor,
    log_prob_old: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO clipped policy loss."""
    ratio = torch.exp(log_prob_new - log_prob_old)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
    return -torch.min(surr1, surr2).mean()


def ppo_value_loss(
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
) -> torch.Tensor:
    """Value function MSE loss."""
    return F.mse_loss(value_pred, value_target)
