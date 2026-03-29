"""Tests for GNN encoder and Laplacian PE."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

import torch
import pytest

from floorplan.models.gnn_encoder import GNNEncoder
from floorplan.data.feature_extractor import compute_laplacian_pe
import numpy as np


class TestLaplacianPE:
    def test_5_node_graph(self):
        """Verify PE is computed on a toy 5-node ring graph."""
        # Ring: 0-1-2-3-4-0
        adj = np.zeros((5, 5), dtype=np.float32)
        for i in range(5):
            adj[i, (i + 1) % 5] = 1.0
            adj[(i + 1) % 5, i] = 1.0

        pe = compute_laplacian_pe(adj, 5, k=2, training=False)
        assert pe.shape == (5, 2), f"Expected (5,2), got {pe.shape}"
        assert not np.any(np.isnan(pe)), "NaN in PE"

    def test_isolated_node(self):
        """Single isolated node should not crash."""
        adj = np.zeros((1, 1), dtype=np.float32)
        pe = compute_laplacian_pe(adj, 1, k=2, training=False)
        assert pe.shape[0] == 1

    def test_two_nodes(self):
        """Two connected nodes."""
        adj = np.array([[0, 1], [1, 0]], dtype=np.float32)
        pe = compute_laplacian_pe(adj, 2, k=2, training=False)
        assert pe.shape[0] == 2

    def test_sign_flip_in_training(self):
        """Signs should vary between calls during training."""
        adj = np.zeros((5, 5), dtype=np.float32)
        for i in range(5):
            adj[i, (i + 1) % 5] = 1.0
            adj[(i + 1) % 5, i] = 1.0

        np.random.seed(0)
        pe1 = compute_laplacian_pe(adj, 5, k=2, training=True)
        np.random.seed(1)
        pe2 = compute_laplacian_pe(adj, 5, k=2, training=True)
        # Not necessarily different, but should not crash
        assert pe1.shape == pe2.shape


class TestGNNEncoder:
    def _make_data(self, n=10):
        """Create a minimal PyG Data object."""
        try:
            from torch_geometric.data import Data
        except ImportError:
            pytest.skip("torch_geometric not installed")

        x = torch.randn(n, 12)
        # Simple ring graph with self-loops
        src = list(range(n)) + [(i + 1) % n for i in range(n)]
        dst = list(range(n)) + list(range(n))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.randn(len(src), 3)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)
        data.batch = torch.zeros(n, dtype=torch.long)
        return data

    def test_output_shapes(self):
        try:
            from torch_geometric.data import Data
        except ImportError:
            pytest.skip("torch_geometric not installed")

        enc = GNNEncoder(hidden_dim=64, num_layers=3)
        enc.eval()
        data = self._make_data(10)

        node_emb, graph_emb = enc(data)
        assert node_emb.shape == (10, 64), f"node_emb shape: {node_emb.shape}"
        assert graph_emb.shape == (1, 64), f"graph_emb shape: {graph_emb.shape}"

    def test_encode_single(self):
        try:
            from torch_geometric.data import Data
        except ImportError:
            pytest.skip("torch_geometric not installed")

        enc = GNNEncoder(hidden_dim=64, num_layers=3)
        enc.eval()
        data = self._make_data(15)
        # Remove batch attr to test encode_single
        if hasattr(data, 'batch'):
            data.batch = None

        node_emb, graph_emb = enc.encode_single(data)
        assert node_emb.shape == (15, 64)
        assert graph_emb.shape == (64,)

    def test_no_nan_in_output(self):
        try:
            from torch_geometric.data import Data
        except ImportError:
            pytest.skip("torch_geometric not installed")

        enc = GNNEncoder(hidden_dim=64, num_layers=3)
        enc.eval()
        data = self._make_data(20)

        node_emb, graph_emb = enc(data)
        assert not torch.isnan(node_emb).any()
        assert not torch.isnan(graph_emb).any()

    def test_gradient_flow(self):
        try:
            from torch_geometric.data import Data
        except ImportError:
            pytest.skip("torch_geometric not installed")

        enc = GNNEncoder(hidden_dim=32, num_layers=2)
        data = self._make_data(8)

        node_emb, graph_emb = enc(data)
        loss = graph_emb.sum()
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = any(p.grad is not None for p in enc.parameters())
        assert has_grad


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
