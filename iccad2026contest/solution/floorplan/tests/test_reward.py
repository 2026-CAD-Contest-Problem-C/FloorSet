"""Tests for reward function and end-to-end episode flow."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

import math
import torch
import pytest

from floorplan.training.losses import compute_reward
from floorplan.env.canvas import Canvas


class TestReward:
    def test_feasible_solution_negative_reward(self):
        """Feasible solution should have negative reward (we minimise cost)."""
        r = compute_reward(
            hpwl=1.5,
            area=1.2,
            overlap=0.0,
            hpwl_base=1.0,
            area_base=1.0,
            alpha=0.5,
        )
        assert r < 0.0, f"Expected negative reward, got {r}"

    def test_overlap_penalty(self):
        """Overlap should return penalty."""
        r = compute_reward(
            hpwl=1.0, area=1.0, overlap=0.1,
            hpwl_base=1.0, area_base=1.0,
            overlap_pen=-10.0,
        )
        assert r == -10.0

    def test_perfect_solution(self):
        """Solution matching baseline should have cost ~1.0, reward ~-1.0."""
        r = compute_reward(
            hpwl=1.0, area=1.0, overlap=0.0,
            hpwl_base=1.0, area_base=1.0,
            alpha=0.5,
        )
        assert abs(r - (-1.0)) < 1e-5, f"Expected -1.0, got {r}"

    def test_better_than_baseline(self):
        """Better HPWL should give reward closer to 0."""
        r_better = compute_reward(0.5, 1.0, 0.0, 1.0, 1.0)
        r_worse = compute_reward(2.0, 1.0, 0.0, 1.0, 1.0)
        assert r_better > r_worse


class TestEndToEndEpisode:
    """Test full episode with random policy."""

    def test_random_policy_episode(self):
        try:
            from torch_geometric.data import Data
            import torch_geometric
        except ImportError:
            pytest.skip("torch_geometric not installed")

        from floorplan.models.full_agent import FloorplanAgent
        from floorplan.env.floorplan_env import FloorplanEnv

        # Build minimal data
        n = 5
        x = torch.randn(n, 12)
        src = list(range(n)) + [(i + 1) % n for i in range(n)]
        dst = list(range(n)) + list(range(n))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.randn(len(src), 3)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)
        data.hpwl_base = 1.0
        data.area_base = 1.0
        data.area_target = torch.ones(n) * 0.04  # 0.2 x 0.2 blocks
        data.b2b_connectivity = torch.tensor([[0, 1, 1.0], [1, 2, 0.5]])
        data.p2b_connectivity = torch.zeros(0, 3)
        data.pins_pos = torch.zeros(0, 2)
        data.block_meta = [
            {'area': 0.04, 'is_fixed_shape': False, 'is_preplaced': False,
             'gt_w': None, 'gt_h': None, 'is_mib': False, 'is_cluster': False, 'is_boundary': False}
            for _ in range(n)
        ]

        agent = FloorplanAgent(hidden_dim=32, num_gnn_layers=2, grid_size=16)
        agent.eval()

        env = FloorplanEnv(agent=agent, grid_size=16, canvas_size=1.0)
        obs = env.reset(data)

        assert obs is not None
        assert 'canvas' in obs
        assert 'legal_mask' in obs
        assert obs['canvas'].shape == (16, 16)

        done = False
        total_steps = 0
        while not done:
            legal = obs['legal_mask'].flatten()
            if legal.any():
                action = int(legal.nonzero(as_tuple=True)[0][0].item())
            else:
                action = 0
            obs, reward, done, info = env.step(action)
            total_steps += 1
            assert total_steps <= n + 1, "Episode took too many steps"

        assert done
        # Reward should be negative (some cost)
        assert reward < 0.0 or reward == 0.0  # 0.0 intermediate, non-zero at end

        sol = env.get_solution()
        assert len(sol) == n
        for x, y, w, h in sol:
            assert w > 0 and h > 0

        # Zero overlap (legal mask guarantee)
        from floorplan.training.metrics import check_overlap
        n_overlaps = check_overlap(sol)
        assert n_overlaps == 0, f"Overlap detected: {n_overlaps} pairs"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
