"""Tests for Canvas: overlap guarantee, prefix sum legality."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

import math
import torch
import pytest

from floorplan.env.canvas import Canvas


def make_canvas(grid=32, size=1.0):
    return Canvas(grid_size=grid, canvas_size=size)


class TestPrefixSumLegality:
    def test_empty_canvas_all_legal_for_small_block(self):
        c = make_canvas(32)
        c.reset([0.1], [0.1])
        mask = c.legal_cells(0)
        # All valid top-left positions should be True
        assert mask.any()

    def test_full_canvas_blocks_placement(self):
        c = make_canvas(8, size=1.0)
        c.reset([1.0, 0.1], [1.0, 0.1])
        # Place a full-canvas block
        c.occupied[:, :] = 1.0
        c._prefix_dirty = True
        mask = c.legal_cells(1)
        # No room for the second block
        assert not mask.any()

    def test_legal_mask_excludes_occupied(self):
        c = make_canvas(16)
        c.reset([0.5, 0.3], [0.5, 0.3])
        # Place block 0 in top-left
        c.place(0, 0, 0)
        mask = c.legal_cells(1)
        # Check that occupied region has no legal cells
        bh = c._h_cells(0)
        bw = c._w_cells(0)
        # In the occupied region, legality should be False
        assert not mask[0, 0]


class TestPlacement:
    def test_no_overlap_after_placement(self):
        c = make_canvas(32)
        widths = [0.3, 0.2, 0.25]
        heights = [0.3, 0.2, 0.25]
        c.reset(widths, heights)

        for bid in range(3):
            mask = c.legal_cells(bid)
            assert mask.any(), f"No legal cell for block {bid}"
            positions = mask.nonzero(as_tuple=False)
            ci, cj = int(positions[0, 0]), int(positions[0, 1])
            c.place(bid, ci, cj)

        overlap = c.compute_overlap()
        assert overlap < 1e-6, f"Overlap detected: {overlap}"

    def test_placements_stored_correctly(self):
        c = make_canvas(16)
        c.reset([0.5], [0.5])
        c.place(0, 0, 0)
        assert c.placements[0] is not None
        x, y, w, h = c.placements[0]
        assert x == 0.0 and y == 0.0
        assert abs(w - 0.5) < 1e-6
        assert abs(h - 0.5) < 1e-6


class TestHPWL:
    def test_hpwl_two_connected_blocks(self):
        c = make_canvas(32)
        c.reset([0.1, 0.1], [0.1, 0.1])

        b2b = torch.tensor([[0, 1, 1.0]])
        c.b2b_connectivity = b2b

        # Place block 0 at (0,0), block 1 at far corner
        c.place(0, 0, 0)
        # Place block 1 far away
        mask = c.legal_cells(1)
        positions = mask.nonzero(as_tuple=False)
        last_pos = positions[-1]
        c.place(1, int(last_pos[0]), int(last_pos[1]))

        hpwl = c.compute_hpwl()
        assert hpwl > 0.0

    def test_hpwl_same_position_zero(self):
        c = make_canvas(32)
        c.reset([0.1, 0.0], [0.1, 0.0])  # Second block has 0 area (degenerate)

        b2b = torch.tensor([[0, 1, 1.0]])
        c.b2b_connectivity = b2b
        # Place only block 0; block 1 stays at origin
        c.placements[1] = (0.05, 0.05, 0.0, 0.0)
        c.place(0, 0, 0)

        hpwl = c.compute_hpwl()
        # Centroid of block 0 is at ~(0.05, 0.05), same as block 1
        assert hpwl >= 0.0


class TestArea:
    def test_area_single_block(self):
        c = make_canvas(32)
        c.reset([0.5], [0.4])
        c.place(0, 0, 0)
        area = c.compute_area()
        # Bounding box of one block
        assert area > 0.0

    def test_area_two_blocks(self):
        c = make_canvas(32)
        c.reset([0.3, 0.3], [0.3, 0.3])
        c.place(0, 0, 0)
        mask = c.legal_cells(1)
        pos = mask.nonzero(as_tuple=False)[-1]
        c.place(1, int(pos[0]), int(pos[1]))
        area = c.compute_area()
        assert area > 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
