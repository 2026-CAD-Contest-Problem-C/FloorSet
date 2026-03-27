#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Hybrid GNN + B*-tree SA Optimizer

Strategy:
  1. FloorplanTransformer predicts initial (x, y, w, h) per block
  2. Sort blocks by predicted (x+y) diagonal to build a smart B*-tree
  3. Simulated Annealing refines from this warm start (fewer iterations needed)

If no trained checkpoint is found, falls back to pure B*-tree SA baseline.

Train the model:
    python iccad2026contest/solution/train.py

Evaluate:
    python iccad2026contest/iccad2026_evaluate.py --evaluate iccad2026contest/my_optimizer.py
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch

CONTEST_DIR = Path(__file__).parent
ROOT_DIR    = CONTEST_DIR.parent
sys.path.insert(0, str(CONTEST_DIR))
sys.path.insert(0, str(ROOT_DIR))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
)
from iccad2026contest.solution.model import FloorplanTransformer

# Path to trained checkpoint (relative to repo root)
DEFAULT_CHECKPOINT = CONTEST_DIR / "solution" / "checkpoints" / "best.pt"


# =============================================================================
# B*-TREE (unchanged from baseline, used for SA refinement)
# =============================================================================

class BStarTree:
    def __init__(self, n_blocks: int, widths: List[float], heights: List[float]):
        self.n = n_blocks
        self.widths = list(widths)
        self.heights = list(heights)
        self.parent = [-1] * n_blocks
        self.left   = [-1] * n_blocks
        self.right  = [-1] * n_blocks
        self.root   = 0
        self._build_random_tree()

    def _build_random_tree(self):
        if self.n == 0:
            return
        self.parent = [-1] * self.n
        self.left   = [-1] * self.n
        self.right  = [-1] * self.n
        order = list(range(self.n))
        random.shuffle(order)
        self.root = order[0]
        for i in range(1, self.n):
            block    = order[i]
            existing = order[random.randint(0, i - 1)]
            if random.random() < 0.5:
                if   self.left[existing]  == -1: self.left[existing]  = block; self.parent[block] = existing
                elif self.right[existing] == -1: self.right[existing] = block; self.parent[block] = existing
                else: self._insert_at_leaf(block, existing)
            else:
                if   self.right[existing] == -1: self.right[existing] = block; self.parent[block] = existing
                elif self.left[existing]  == -1: self.left[existing]  = block; self.parent[block] = existing
                else: self._insert_at_leaf(block, existing)

    def _insert_at_leaf(self, block: int, start: int):
        cur = start
        while True:
            if random.random() < 0.5:
                if self.left[cur]  == -1: self.left[cur]  = block; self.parent[block] = cur; return
                cur = self.left[cur]
            else:
                if self.right[cur] == -1: self.right[cur] = block; self.parent[block] = cur; return
                cur = self.right[cur]

    def build_from_order(self, order: List[int]):
        """Build a left-chain tree from a given block order (leftmost = root)."""
        self.parent = [-1] * self.n
        self.left   = [-1] * self.n
        self.right  = [-1] * self.n
        if not order:
            return
        self.root = order[0]
        for i in range(1, len(order)):
            prev, cur = order[i - 1], order[i]
            self.left[prev] = cur
            self.parent[cur] = prev

    def pack(self) -> List[Tuple[float, float, float, float]]:
        positions = [(0.0, 0.0, self.widths[i], self.heights[i]) for i in range(self.n)]
        if self.n == 0:
            return positions
        contour = [(0.0, 0.0)]

        def get_y(x0, x1):
            max_y = 0.0
            for i, (cx1, cy) in enumerate(contour):
                cx0 = contour[i-1][0] if i > 0 else 0.0
                if x0 < cx1 and x1 > cx0:
                    max_y = max(max_y, cy)
            return max_y

        def update(x0, x1, y1):
            nonlocal contour
            new = []
            for i, (cx1, cy) in enumerate(contour):
                cx0 = contour[i-1][0] if i > 0 else 0.0
                if cx1 <= x0: new.append((cx1, cy))
                elif cx0 >= x1: new.append((cx1, cy))
                else:
                    if cx0 < x0: new.append((x0, cy))
                    if cx1 > x1: new.append((cx1, cy))
            ins = 0
            for i, (cx1, _) in enumerate(new):
                if cx1 <= x0: ins = i + 1
            new.insert(ins, (x1, y1))
            new.sort(key=lambda s: s[0])
            merged = []
            for xe, ye in new:
                if merged and merged[-1][1] == ye: merged[-1] = (xe, ye)
                else: merged.append((xe, ye))
            contour = merged or [(x1, 0.0)]

        def dfs(node, par_right):
            if node == -1: return
            w, h = self.widths[node], self.heights[node]
            x = 0.0 if node == self.root else par_right
            y = get_y(x, x + w)
            positions[node] = (x, y, w, h)
            update(x, x + w, y + h)
            dfs(self.left[node],  x + w)
            dfs(self.right[node], x)

        sys.setrecursionlimit(max(1000, self.n * 4))
        dfs(self.root, 0.0)

        # Safety: push any remaining overlaps up
        for i in range(self.n):
            for j in range(i + 1, self.n):
                x1, y1, w1, h1 = positions[i]
                x2, y2, w2, h2 = positions[j]
                if min(x1+w1, x2+w2) - max(x1, x2) > 1e-6 and \
                   min(y1+h1, y2+h2) - max(y1, y2) > 1e-6:
                    positions[j] = (x2, max(y1+h1, y2), w2, h2)
        return positions

    def copy(self):
        t = BStarTree.__new__(BStarTree)
        t.n       = self.n
        t.widths  = self.widths[:]
        t.heights = self.heights[:]
        t.parent  = self.parent[:]
        t.left    = self.left[:]
        t.right   = self.right[:]
        t.root    = self.root
        return t

    def move_rotate(self, block: int):
        self.widths[block], self.heights[block] = self.heights[block], self.widths[block]

    def move_delete_insert(self, block: int):
        if self.n <= 1: return
        w, h = self.widths[block], self.heights[block]
        self._delete_node(block)
        target = random.randint(0, self.n - 1)
        while target == block: target = random.randint(0, self.n - 1)
        self._insert_node(block, target, random.choice([True, False]))
        self.widths[block], self.heights[block] = w, h

    def _delete_node(self, node: int):
        p, lc, rc = self.parent[node], self.left[node], self.right[node]
        if   lc == -1 and rc == -1: rep = -1
        elif lc == -1:               rep = rc
        elif rc == -1:               rep = lc
        else:
            rep = lc
            rm  = lc
            while self.right[rm] != -1: rm = self.right[rm]
            self.right[rm] = rc; self.parent[rc] = rm
        if   p == -1:              self.root = rep
        elif self.left[p]  == node: self.left[p]  = rep
        else:                       self.right[p] = rep
        if rep != -1: self.parent[rep] = p
        self.parent[node] = self.left[node] = self.right[node] = -1

    def _insert_node(self, node: int, target: int, as_left: bool):
        if as_left:
            old = self.left[target];  self.left[target]  = node
        else:
            old = self.right[target]; self.right[target] = node
        self.parent[node] = target
        if old != -1: self.left[node] = old; self.parent[old] = node


# =============================================================================
# HYBRID OPTIMIZER
# =============================================================================

class MyOptimizer(FloorplanOptimizer):
    """
    GNN-initialised B*-tree Simulated Annealing.

    Phase 1: FloorplanTransformer → predicted (x, y, w, h)
    Phase 2: Build B*-tree ordered by predicted placement diagonal
    Phase 3: SA refinement (warm start → fewer iterations needed)
    """

    # SA hyperparameters
    SA_INITIAL_TEMP  = 100.0
    SA_FINAL_TEMP    = 1.0
    SA_COOLING       = 0.9
    SA_MOVES_PER_T   = 20

    # Reduce SA effort when GNN provides a good warm start
    SA_MOVES_PER_T_WARM = 10

    def __init__(self, verbose: bool = False, checkpoint: str = None):
        super().__init__(verbose)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = None
        ckpt_path   = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
        self._load_model(ckpt_path)

    def _load_model(self, ckpt_path: Path):
        if not ckpt_path.exists():
            if self.verbose:
                print(f"[MyOptimizer] No checkpoint at {ckpt_path}. Using SA-only mode.")
            return
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            args = ckpt.get("args", {})
            self.model = FloorplanTransformer(
                d_model  = args.get("d_model",  128),
                n_heads  = args.get("n_heads",   8),
                n_layers = args.get("n_layers",  6),
            ).to(self.device)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            if self.verbose:
                print(f"[MyOptimizer] Loaded model from {ckpt_path}")
        except Exception as e:
            print(f"[MyOptimizer] Failed to load model: {e}. Using SA-only mode.")
            self.model = None

    @torch.no_grad()
    def _gnn_predict(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
    ) -> List[Tuple[float, float, float, float]]:
        """Run GNN and return [(x, y, w, h)] for valid blocks."""
        area   = area_targets[:block_count].to(self.device)
        b2b    = b2b_connectivity.to(self.device)
        p2b    = p2b_connectivity.to(self.device)
        pins   = pins_pos.to(self.device)
        constr = constraints[:block_count].to(self.device)

        pos = self.model(area, b2b, p2b, pins, constr)  # [N, 4]
        return [(float(pos[i,0]), float(pos[i,1]),
                 float(pos[i,2]), float(pos[i,3]))
                for i in range(block_count)]

    def _gnn_to_tree(
        self,
        gnn_positions: List[Tuple[float, float, float, float]],
        area_targets: torch.Tensor,
        block_count: int,
    ) -> BStarTree:
        """
        Build a B*-tree whose initial shape comes from GNN predictions.

        Ordering: sort blocks by (x + y) diagonal so the B*-tree left-chain
        roughly reproduces the predicted left-to-right placement order.
        """
        widths, heights = [], []
        for i in range(block_count):
            _, _, w, h = gnn_positions[i]
            area = float(area_targets[i])
            # Clamp aspect ratio: ensure w*h ≈ area within 1%
            ratio = max(w, 1e-6) / max(h, 1e-6)
            w = math.sqrt(area * ratio)
            h = math.sqrt(area / ratio)
            widths.append(w)
            heights.append(h)

        tree = BStarTree(block_count, widths, heights)

        # Build left-chain order from predicted diagonal
        order = sorted(range(block_count),
                       key=lambda i: gnn_positions[i][0] + gnn_positions[i][1])
        tree.build_from_order(order)
        return tree

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
    ) -> List[Tuple[float, float, float, float]]:

        warm_start = self.model is not None

        # --- Phase 1: GNN prediction (if available) ---
        if warm_start:
            try:
                gnn_pos = self._gnn_predict(
                    block_count, area_targets, b2b_connectivity,
                    p2b_connectivity, pins_pos, constraints)
                tree = self._gnn_to_tree(gnn_pos, area_targets, block_count)
            except Exception as e:
                if self.verbose:
                    print(f"[MyOptimizer] GNN failed ({e}), falling back to random init.")
                warm_start = False

        # --- Phase 1 fallback: random square init ---
        if not warm_start:
            widths, heights = [], []
            for i in range(block_count):
                area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
                s = math.sqrt(area)
                widths.append(s); heights.append(s)
            tree = BStarTree(block_count, widths, heights)

        # --- Phase 2: SA refinement ---
        current_pos  = tree.pack()
        current_cost = self._cost(current_pos, b2b_connectivity, p2b_connectivity, pins_pos)
        best_tree     = tree.copy()
        best_pos      = current_pos
        best_cost     = current_cost

        moves_per_t = self.SA_MOVES_PER_T_WARM if warm_start else self.SA_MOVES_PER_T
        temp        = self.SA_INITIAL_TEMP

        while temp > self.SA_FINAL_TEMP:
            for _ in range(moves_per_t):
                old_tree = tree.copy()
                if random.randint(0, 1) == 0:
                    tree.move_rotate(random.randint(0, block_count - 1))
                else:
                    tree.move_delete_insert(random.randint(0, block_count - 1))

                new_pos  = tree.pack()
                new_cost = self._cost(new_pos, b2b_connectivity, p2b_connectivity, pins_pos)
                delta    = new_cost - current_cost

                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_pos  = new_pos
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_pos  = new_pos
                        best_tree = tree.copy()
                else:
                    tree = old_tree

            temp *= self.SA_COOLING

        return best_pos

    def _cost(self, positions, b2b_conn, p2b_conn, pins_pos) -> float:
        hpwl_b2b = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        area     = calculate_bbox_area(positions)
        return hpwl_b2b + hpwl_p2b + area * 0.01
