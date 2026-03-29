"""Laplacian Positional Encoding for graph nodes."""
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def compute_laplacian_pe(
    adj: np.ndarray,
    num_nodes: int,
    k: int = 2,
    training: bool = True,
) -> np.ndarray:
    """
    Compute Laplacian Positional Encoding.

    Args:
        adj: (num_nodes, num_nodes) adjacency matrix (weighted, symmetric).
        num_nodes: number of nodes.
        k: number of eigenvectors to return (excluding trivial).
        training: if True, randomly flip eigenvector signs for augmentation.

    Returns:
        pe: (num_nodes, min(k, num_nodes-2)) positional encoding.
    """
    k_max = min(num_nodes - 1, 8)
    k_use = min(k, k_max)

    if num_nodes <= 2 or k_use < 1:
        return np.zeros((num_nodes, k), dtype=np.float32)

    # Build normalised Laplacian
    degree = adj.sum(axis=1)
    degree = np.maximum(degree, 1e-8)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    A = adj
    L = np.diag(degree) - A
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    L_sparse = csr_matrix(L_norm, dtype=np.float64)

    try:
        # k+1 to skip the trivial 0 eigenvector
        num_eigs = min(k_use + 1, num_nodes - 1)
        eigenvalues, eigenvectors = eigsh(L_sparse, k=num_eigs, which='SM', tol=1e-4, maxiter=500)
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        # Skip the first (trivial, all-equal) eigenvector
        pe = eigenvectors[:, 1:k_use + 1]
    except Exception:
        pe = np.zeros((num_nodes, k_use), dtype=np.float32)

    # Pad to k columns if needed
    if pe.shape[1] < k:
        pad = np.zeros((num_nodes, k - pe.shape[1]), dtype=np.float32)
        pe = np.concatenate([pe, pad], axis=1)

    pe = pe.astype(np.float32)

    # Sign augmentation / normalisation
    for col in range(pe.shape[1]):
        if training:
            if np.random.random() < 0.5:
                pe[:, col] *= -1
        else:
            # Positive-mean convention
            if pe[:, col].mean() < 0:
                pe[:, col] *= -1

    return pe


def extract_node_features(
    area_target: torch.Tensor,
    placement_constraints: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    num_nodes: int,
    training: bool = True,
) -> torch.Tensor:
    """
    Build a (num_nodes, 12) node feature matrix.

    Features per node:
      [0]  log(area_target + 1)
      [1]  sqrt(area_target)  (side length if square)
      [2]  is_fixed_shape      (constraints[:, 0])
      [3]  is_preplaced        (constraints[:, 1])
      [4]  is_mib              (constraints[:, 2])
      [5]  is_cluster          (constraints[:, 3])
      [6]  is_boundary         (constraints[:, 4])
      [7]  weighted_degree_b2b (normalised)
      [8]  weighted_degree_p2b (normalised)
      [9]  node_count_norm     (num_nodes / 120)
      [10] laplacian_pe_0
      [11] laplacian_pe_1
    """
    features = torch.zeros(num_nodes, 12)

    area = area_target[:num_nodes].float()
    features[:, 0] = torch.log(area + 1.0)
    features[:, 1] = torch.sqrt(area.clamp(min=1e-8))

    if placement_constraints is not None and placement_constraints.shape[0] >= num_nodes:
        c = placement_constraints[:num_nodes].float()
        features[:, 2] = c[:, 0]  # fixed
        features[:, 3] = c[:, 1]  # preplaced
        features[:, 4] = c[:, 2]  # mib
        features[:, 5] = c[:, 3]  # cluster
        features[:, 6] = c[:, 4]  # boundary

    # Weighted degrees (vectorized)
    deg_b2b = torch.zeros(num_nodes)
    if b2b_connectivity is not None and b2b_connectivity.shape[0] > 0:
        valid = (b2b_connectivity[:, 0] >= 0) & (b2b_connectivity[:, 0] < num_nodes) & \
                (b2b_connectivity[:, 1] >= 0) & (b2b_connectivity[:, 1] < num_nodes)
        b2b = b2b_connectivity[valid]
        if b2b.shape[0] > 0:
            w = b2b[:, 2].float() if b2b.shape[1] > 2 else torch.ones(b2b.shape[0])
            i_idx = b2b[:, 0].long()
            j_idx = b2b[:, 1].long()
            deg_b2b.scatter_add_(0, i_idx, w)
            deg_b2b.scatter_add_(0, j_idx, w)

    deg_p2b = torch.zeros(num_nodes)
    if p2b_connectivity is not None and p2b_connectivity.shape[0] > 0:
        valid = (p2b_connectivity[:, 1] >= 0) & (p2b_connectivity[:, 1] < num_nodes)
        p2b = p2b_connectivity[valid]
        if p2b.shape[0] > 0:
            w = p2b[:, 2].float() if p2b.shape[1] > 2 else torch.ones(p2b.shape[0])
            j_idx = p2b[:, 1].long()
            deg_p2b.scatter_add_(0, j_idx, w)

    max_b2b = deg_b2b.max().clamp(min=1e-8)
    max_p2b = deg_p2b.max().clamp(min=1e-8)
    features[:, 7] = deg_b2b / max_b2b
    features[:, 8] = deg_p2b / max_p2b
    features[:, 9] = num_nodes / 120.0

    # Laplacian PE (vectorized adjacency matrix construction)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    if b2b_connectivity is not None and b2b_connectivity.shape[0] > 0:
        valid = (b2b_connectivity[:, 0] >= 0) & (b2b_connectivity[:, 0] < num_nodes) & \
                (b2b_connectivity[:, 1] >= 0) & (b2b_connectivity[:, 1] < num_nodes)
        b2b = b2b_connectivity[valid]
        if b2b.shape[0] > 0:
            w = b2b[:, 2].float() if b2b.shape[1] > 2 else torch.ones(b2b.shape[0])
            i_np = b2b[:, 0].long().numpy()
            j_np = b2b[:, 1].long().numpy()
            w_np = w.numpy()
            np.add.at(adj, (i_np, j_np), w_np)
            np.add.at(adj, (j_np, i_np), w_np)

    pe = compute_laplacian_pe(adj, num_nodes, k=2, training=training)
    features[:, 10] = torch.from_numpy(pe[:, 0])
    features[:, 11] = torch.from_numpy(pe[:, 1])

    return features
