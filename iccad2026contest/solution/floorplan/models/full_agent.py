"""FloorplanAgent: ties GNNEncoder, ShapePolicy, and PlacementPolicy together."""
import torch
import torch.nn as nn
from typing import Optional

from .gnn_encoder import GNNEncoder
from .shape_policy import ShapePolicy
from .placement_policy import PlacementPolicy


class FloorplanAgent(nn.Module):
    """
    Full agent combining all three components:
      1. GNNEncoder: encode netlist graph
      2. ShapePolicy: choose (w, h) for each block
      3. PlacementPolicy: place blocks one by one on canvas
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        gnn_dropout: float = 0.1,
        shape_hidden_dim: int = 256,
        log_r_min: float = -2.0,
        log_r_max: float = 2.0,
        grid_size: int = 256,
    ):
        super().__init__()

        self.gnn_encoder = GNNEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=gnn_dropout,
        )
        self.shape_policy = ShapePolicy(
            node_emb_dim=hidden_dim,
            hidden_dim=shape_hidden_dim,
            log_r_min=log_r_min,
            log_r_max=log_r_max,
        )
        self.placement_policy = PlacementPolicy(
            node_emb_dim=hidden_dim,
            grid_size=grid_size,
        )

        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

    def encode(self, data):
        """Run GNN encoder. Returns (node_emb, graph_emb)."""
        return self.gnn_encoder.encode_single(data)

    def decide_shapes(self, node_emb, block_meta, greedy=False):
        """Run shape policy. Returns (w_list, h_list, log_prob)."""
        return self.shape_policy(node_emb, block_meta, greedy=greedy)

    @classmethod
    def from_config(cls, cfg: dict) -> 'FloorplanAgent':
        return cls(
            hidden_dim=cfg['gnn']['hidden_dim'],
            num_gnn_layers=cfg['gnn']['num_layers'],
            gnn_dropout=cfg['gnn']['dropout'],
            shape_hidden_dim=cfg['shape_policy']['hidden_dim'],
            log_r_min=cfg['shape_policy']['log_r_min'],
            log_r_max=cfg['shape_policy']['log_r_max'],
            grid_size=cfg['placement']['grid_size'],
        )

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, device=None):
        state = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state)
