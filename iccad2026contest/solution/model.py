"""
FloorplanTransformer: Graph Transformer for floorplan placement.

Input:
  - node_feat  : [N, 6]  (area_target, fixed, preplaced, MIB, cluster, boundary)
  - b2b_edges  : [E, 3]  (block_i, block_j, weight)  — padded with -1
  - p2b_edges  : [P, 3]  (pin_idx, block_idx, weight) — padded with -1
  - pins_pos   : [K, 2]  (x, y) of fixed pins        — padded with -1
  - N          : int, actual block count (un-padded)

Output:
  - positions  : [N, 4]  (x, y, w, h)
    w = sqrt(area) * exp( log_r / 2)
    h = sqrt(area) * exp(-log_r / 2)
    guarantees w*h == area_target exactly
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SinusoidalPosEnc(nn.Module):
    """1-D sinusoidal encoding for block index order."""
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [N, d]
        return x + self.pe[:x.size(0)]


# ---------------------------------------------------------------------------
# Attention with edge bias
# ---------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """
    Full N×N self-attention (safe for N≤120) with an additive edge-weight bias.

    edge_bias[i, j] = learned_proj(b2b_weight(i,j))   if edge exists
                    = 0                                 otherwise
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # edge weight (scalar) → bias per head
        self.edge_bias_proj = nn.Linear(1, n_heads)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_bias: torch.Tensor) -> torch.Tensor:
        """
        x         : [N, d_model]
        edge_bias : [N, N, n_heads]  (pre-built from b2b connectivity)
        returns   : [N, d_model]
        """
        N, d = x.shape
        scale = self.d_head ** -0.5

        q = self.q(x).view(N, self.n_heads, self.d_head).transpose(0, 1)  # [H, N, dh]
        k = self.k(x).view(N, self.n_heads, self.d_head).transpose(0, 1)
        v = self.v(x).view(N, self.n_heads, self.d_head).transpose(0, 1)

        attn = torch.bmm(q, k.transpose(1, 2)) * scale                    # [H, N, N]
        attn = attn + edge_bias.permute(2, 0, 1)                           # [H, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.bmm(attn, v)                                           # [H, N, dh]
        out = out.transpose(0, 1).contiguous().view(N, d)                  # [N, d]
        return self.out_proj(out)

    def build_edge_bias(
        self,
        N: int,
        b2b_edges: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Build [N, N, n_heads] edge-bias matrix from padded b2b_edges [E, 3].
        Edges padded with -1 are ignored.
        """
        bias = torch.zeros(N, N, self.n_heads, device=device)
        if b2b_edges.numel() == 0:
            return bias
        valid = b2b_edges[:, 0] >= 0
        edges = b2b_edges[valid]
        if edges.numel() == 0:
            return bias
        i = edges[:, 0].long()
        j = edges[:, 1].long()
        w = edges[:, 2:3].float()                      # [E, 1]
        b = self.edge_bias_proj(w)                     # [E, n_heads]
        # mask out-of-range indices (padding artefacts)
        valid2 = (i < N) & (j < N)
        i, j, b = i[valid2], j[valid2], b[valid2]
        bias[i, j] += b
        bias[j, i] += b                                # symmetric
        return bias


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = GraphAttentionLayer(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_bias: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x, edge_bias)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# Pin context encoder
# ---------------------------------------------------------------------------

class PinEncoder(nn.Module):
    """Aggregate variable-length pin positions into a fixed context vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pins_pos: torch.Tensor) -> torch.Tensor:
        """
        pins_pos : [K, 2], padded with -1
        returns  : [d_model]
        """
        valid = (pins_pos[:, 0] >= 0)
        pins = pins_pos[valid]                         # [k, 2]
        if pins.numel() == 0:
            return torch.zeros(self.proj[0].out_features,
                               device=pins_pos.device)
        emb = self.proj(pins)                          # [k, d]
        # normalise coordinates before encoding
        return emb.mean(0)                             # [d]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FloorplanTransformer(nn.Module):
    """
    Graph Transformer that predicts (x, y, log_r) per block.

    (x, y)  — placement coordinates (un-normalised)
    log_r   — log aspect ratio; w = sqrt(area)*exp(log_r/2),
                                h = sqrt(area)*exp(-log_r/2)
              guarantees w*h == area_target exactly
    """

    NODE_IN = 6   # area(1) + fixed(1) + preplaced(1) + MIB(1) + cluster(1) + boundary(1)

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Node embedding: raw features → d_model
        self.node_embed = nn.Sequential(
            nn.Linear(self.NODE_IN, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = SinusoidalPosEnc(d_model)

        # Pin context
        self.pin_encoder = PinEncoder(d_model)
        self.pin_cross = nn.Linear(d_model * 2, d_model)

        # Transformer stack
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output head: (x, y, log_r)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        area_target: torch.Tensor,    # [N] (valid blocks only, un-padded)
        b2b_edges: torch.Tensor,      # [E, 3] padded with -1
        p2b_edges: torch.Tensor,      # [P, 3] padded with -1  (unused in attn, reserved)
        pins_pos: torch.Tensor,       # [K, 2] padded with -1
        constraints: torch.Tensor,   # [N, 5]
    ) -> torch.Tensor:
        """Returns positions [N, 4] = (x, y, w, h)."""
        N = area_target.shape[0]
        device = area_target.device

        # --- Node features ---
        # Normalise area by sqrt for better scaling
        area_norm = (area_target / (area_target.mean() + 1e-8)).unsqueeze(-1)  # [N,1]
        node_feat = torch.cat([area_norm, constraints.float()], dim=-1)        # [N,6]
        x = self.node_embed(node_feat)                                         # [N, d]
        x = self.pos_enc(x)

        # --- Pin global context ---
        pin_ctx = self.pin_encoder(pins_pos)                                   # [d]
        pin_ctx = pin_ctx.unsqueeze(0).expand(N, -1)                          # [N, d]
        x = self.pin_cross(torch.cat([x, pin_ctx], dim=-1))                   # [N, d]

        # --- Build edge bias (shared across layers) ---
        edge_bias = self.layers[0].attn.build_edge_bias(N, b2b_edges, device)

        # --- Transformer ---
        for layer in self.layers:
            x = layer(x, edge_bias)

        # --- Output ---
        out = self.head(x)                         # [N, 3]: (x, y, log_r)
        px = out[:, 0]
        py = out[:, 1]
        log_r = out[:, 2].clamp(-3.0, 3.0)       # limit aspect ratio to e^3 ≈ 20

        sqrt_area = area_target.float().sqrt()
        w = sqrt_area * (log_r / 2).exp()
        h = sqrt_area * (-log_r / 2).exp()

        return torch.stack([px, py, w, h], dim=-1)  # [N, 4]
