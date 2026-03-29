"""Convert raw FloorSet sample to PyG Data object."""
import torch
import numpy as np

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None

from .feature_extractor import extract_node_features


def build_graph(
    area_target: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    placement_constraints: torch.Tensor,
    fp_sol: torch.Tensor = None,
    metrics_sol: torch.Tensor = None,
    training: bool = True,
) -> "Data":
    """
    Convert a FloorSet-Lite sample into a PyG Data object.

    Args:
        area_target: (n_blocks,) area targets
        b2b_connectivity: (E, 3) block-to-block edges [i, j, weight]
        p2b_connectivity: (E, 3) pin-to-block edges [pin_idx, block_idx, weight]
        pins_pos: (n_pins, 2) pin positions
        placement_constraints: (n_blocks, 5) constraint flags
        fp_sol: (n_blocks, 4) ground truth [w, h, x, y] (optional)
        metrics_sol: (8,) metrics tensor (optional)
        training: whether in training mode (affects Laplacian PE sign augmentation)

    Returns:
        PyG Data object with fields:
            x, edge_index, edge_attr, num_nodes,
            hpwl_base, area_base, block_meta
    """
    assert Data is not None, "torch_geometric is required"

    # Determine valid (non-padded) block count
    num_nodes = _count_valid_nodes(area_target, placement_constraints)

    # Build node features (num_nodes, 12)
    x = extract_node_features(
        area_target, placement_constraints,
        b2b_connectivity, p2b_connectivity,
        num_nodes, training=training,
    )

    # Build edges (undirected b2b + self-loops)
    edge_index, edge_attr = _build_edges(b2b_connectivity, num_nodes)

    # HPWL and area baselines from metrics_sol
    hpwl_base = 0.0
    area_base = 0.0
    if metrics_sol is not None:
        m = metrics_sol.float()
        if m.shape[0] >= 8:
            area_base = float(m[0].item())
            hpwl_b2b = float(m[6].item())
            hpwl_p2b = float(m[7].item())
            hpwl_base = hpwl_b2b + hpwl_p2b
        elif m.shape[0] >= 7:
            area_base = float(m[0].item())
            hpwl_base = float(m[6].item())

    # Block metadata
    block_meta = _build_block_meta(
        area_target, placement_constraints, fp_sol, num_nodes
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
    )
    data.hpwl_base = hpwl_base
    data.area_base = area_base
    data.block_meta = block_meta
    data.area_target = area_target[:num_nodes].float()
    data.pins_pos = pins_pos.float() if pins_pos is not None else torch.zeros(0, 2)
    data.b2b_connectivity = b2b_connectivity.float() if b2b_connectivity is not None else torch.zeros(0, 3)
    data.p2b_connectivity = p2b_connectivity.float() if p2b_connectivity is not None else torch.zeros(0, 3)

    return data


def _count_valid_nodes(area_target: torch.Tensor, constraints: torch.Tensor) -> int:
    """Count non-padded blocks. Padding uses -1 or 0 area targets."""
    n = area_target.shape[0]
    for i in range(n - 1, -1, -1):
        if area_target[i].item() > 0:
            return i + 1
    return n


def _build_edges(b2b_connectivity: torch.Tensor, num_nodes: int):
    """Build undirected COO edge_index with self-loops and edge_attr."""
    src_list, dst_list, attr_list = [], [], []

    # Self-loops (vectorized)
    self_idx = torch.arange(num_nodes, dtype=torch.long)
    self_attr = torch.zeros(num_nodes, 3)
    self_attr[:, 0] = 1.0  # is_self_loop

    if b2b_connectivity is None or b2b_connectivity.shape[0] == 0:
        edge_index = torch.stack([self_idx, self_idx], dim=0)
        return edge_index, self_attr

    # B2B edges (vectorized, both directions)
    valid_mask = (
        (b2b_connectivity[:, 0] >= 0) & (b2b_connectivity[:, 0] < num_nodes) &
        (b2b_connectivity[:, 1] >= 0) & (b2b_connectivity[:, 1] < num_nodes)
    )
    b2b = b2b_connectivity[valid_mask]

    if b2b.shape[0] == 0:
        edge_index = torch.stack([self_idx, self_idx], dim=0)
        return edge_index, self_attr

    i_idx = b2b[:, 0].long()
    j_idx = b2b[:, 1].long()
    w = b2b[:, 2].float() if b2b.shape[1] > 2 else torch.ones(b2b.shape[0])
    lw = torch.log(w + 1.0)

    b2b_attr = torch.stack([torch.zeros_like(w), w, lw], dim=1)

    src = torch.cat([self_idx, i_idx, j_idx], dim=0)
    dst = torch.cat([self_idx, j_idx, i_idx], dim=0)
    edge_attr = torch.cat([self_attr, b2b_attr, b2b_attr], dim=0)

    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, edge_attr


def _build_block_meta(
    area_target: torch.Tensor,
    placement_constraints: torch.Tensor,
    fp_sol: torch.Tensor,
    num_nodes: int,
) -> list:
    """Build list of per-block metadata dicts."""
    meta = []
    for i in range(num_nodes):
        area = float(area_target[i].item()) if i < area_target.shape[0] else 1.0
        c = placement_constraints[i].tolist() if (
            placement_constraints is not None and i < placement_constraints.shape[0]
        ) else [0, 0, 0, 0, 0]

        is_fixed = bool(c[0] > 0.5)
        is_preplaced = bool(c[1] > 0.5)

        gt_w, gt_h, gt_x, gt_y = None, None, None, None
        if fp_sol is not None and i < fp_sol.shape[0]:
            row = fp_sol[i]
            if not torch.all(row == -1):
                gt_w = float(row[0].item())
                gt_h = float(row[1].item())
                gt_x = float(row[2].item())
                gt_y = float(row[3].item())

        meta.append({
            'area': area,
            'is_fixed_shape': is_fixed,
            'is_preplaced': is_preplaced,
            'is_mib': bool(c[2] > 0.5),
            'is_cluster': bool(c[3] > 0.5),
            'is_boundary': bool(c[4] > 0.5),
            'gt_w': gt_w,
            'gt_h': gt_h,
            'gt_x': gt_x,
            'gt_y': gt_y,
        })
    return meta
