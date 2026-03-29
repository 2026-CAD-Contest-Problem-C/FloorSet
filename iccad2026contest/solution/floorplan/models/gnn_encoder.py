"""3-layer GIN encoder with edge feature injection."""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GINEConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GINLayer(nn.Module):
    """GIN layer with edge feature injection via linear projection."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.BatchNorm1d(out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        # Edge feature -> same dim as node
        edge_emb = self.edge_proj(edge_attr)  # (E, in_dim)

        # Aggregate: sum neighbour (node + edge) features
        src, dst = edge_index[0], edge_index[1]
        n = x.shape[0]
        agg = torch.zeros_like(x)
        # Contribution: neighbour_x + edge_emb
        contrib = x[src] + edge_emb
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(contrib), contrib)

        out = self.mlp((1 + self.eps) * x + agg)
        return F.dropout(out, p=self.dropout, training=self.training)


class GNNEncoder(nn.Module):
    """
    3-layer GIN encoder.

    Input:  PyG Data/Batch with x (node feats, 12-dim) and edge_index, edge_attr (3-dim)
    Output: (node_emb: (N, hidden_dim), graph_emb: (B, hidden_dim))
    """

    NODE_FEAT_DIM = 12
    EDGE_FEAT_DIM = 3

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(self.NODE_FEAT_DIM, hidden_dim)

        self.layers = nn.ModuleList([
            GINLayer(hidden_dim, hidden_dim, self.EDGE_FEAT_DIM, dropout)
            for _ in range(num_layers)
        ])

        # Skip connections: project residual if needed
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * (num_layers + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, data):
        """
        Args:
            data: PyG Batch object

        Returns:
            node_emb: (total_nodes, hidden_dim)
            graph_emb: (batch_size, hidden_dim)
        """
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Collect all layer outputs for JK (jumping knowledge) aggregation
        all_x = [x]
        for i, layer in enumerate(self.layers):
            h = layer(x, edge_index, edge_attr)
            x = self.layer_norms[i](h + x)  # Residual
            all_x.append(x)

        # JK concat
        node_emb = self.output_proj(torch.cat(all_x, dim=-1))  # (N, hidden_dim)

        # Graph-level mean pooling
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=node_emb.device)

        graph_emb = global_mean_pool(node_emb, batch)  # (B, hidden_dim)

        return node_emb, graph_emb

    def encode_single(self, data):
        """
        Convenience method for single graph inference.

        Args:
            data: PyG Data (not batched)

        Returns:
            node_emb: (k, hidden_dim)
            graph_emb: (1, hidden_dim) — squeeze to (hidden_dim,) if desired
        """
        # Add batch dimension
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)

        node_emb, graph_emb = self.forward(data)
        return node_emb, graph_emb.squeeze(0)


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Mean pooling over nodes in each graph."""
    try:
        from torch_geometric.nn import global_mean_pool as pyg_gmp
        return pyg_gmp(x, batch)
    except ImportError:
        pass

    # Fallback implementation
    num_graphs = int(batch.max().item()) + 1
    out = torch.zeros(num_graphs, x.shape[1], device=x.device, dtype=x.dtype)
    count = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
    out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
    count.scatter_add_(0, batch.unsqueeze(1), torch.ones(x.shape[0], 1, device=x.device))
    return out / count.clamp(min=1)
