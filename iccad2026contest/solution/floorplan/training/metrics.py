"""Evaluation metrics: HPWL, area, overlap, contest cost."""
import math
from typing import List, Tuple

import torch


def calculate_hpwl_b2b(
    positions: List[Tuple[float, float, float, float]],
    b2b_connectivity: torch.Tensor,
) -> float:
    """Weighted Manhattan distance between block centroids."""
    if b2b_connectivity is None or b2b_connectivity.shape[0] == 0:
        return 0.0

    n = len(positions)
    centroids = torch.tensor(
        [[x + w / 2, y + h / 2] for x, y, w, h in positions],
        dtype=torch.float32,
    )

    valid = (
        (b2b_connectivity[:, 0] >= 0) & (b2b_connectivity[:, 0] < n) &
        (b2b_connectivity[:, 1] >= 0) & (b2b_connectivity[:, 1] < n)
    )
    b2b = b2b_connectivity[valid].float()
    if b2b.shape[0] == 0:
        return 0.0

    i_idx = b2b[:, 0].long()
    j_idx = b2b[:, 1].long()
    w = b2b[:, 2] if b2b.shape[1] > 2 else torch.ones(b2b.shape[0])
    dx = torch.abs(centroids[j_idx, 0] - centroids[i_idx, 0])
    dy = torch.abs(centroids[j_idx, 1] - centroids[i_idx, 1])
    return float(((dx + dy) * w).sum().item())


def calculate_hpwl_p2b(
    positions: List[Tuple[float, float, float, float]],
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
) -> float:
    """Weighted Manhattan distance from pins to block centroids."""
    if p2b_connectivity is None or p2b_connectivity.shape[0] == 0:
        return 0.0
    if pins_pos is None or pins_pos.shape[0] == 0:
        return 0.0

    n = len(positions)
    n_pins = pins_pos.shape[0]
    centroids = torch.tensor(
        [[x + w / 2, y + h / 2] for x, y, w, h in positions],
        dtype=torch.float32,
    )

    valid = (
        (p2b_connectivity[:, 0] >= 0) & (p2b_connectivity[:, 0] < n_pins) &
        (p2b_connectivity[:, 1] >= 0) & (p2b_connectivity[:, 1] < n)
    )
    p2b = p2b_connectivity[valid].float()
    if p2b.shape[0] == 0:
        return 0.0

    pin_idx = p2b[:, 0].long()
    blk_idx = p2b[:, 1].long()
    w = p2b[:, 2] if p2b.shape[1] > 2 else torch.ones(p2b.shape[0])
    px = pins_pos[pin_idx, 0].float()
    py = pins_pos[pin_idx, 1].float()
    dx = torch.abs(centroids[blk_idx, 0] - px)
    dy = torch.abs(centroids[blk_idx, 1] - py)
    return float(((dx + dy) * w).sum().item())


def calculate_bbox_area(positions: List[Tuple[float, float, float, float]]) -> float:
    """Bounding box area of all placed blocks."""
    if not positions:
        return 0.0
    xs = [x for x, y, w, h in positions]
    ys = [y for x, y, w, h in positions]
    x2s = [x + w for x, y, w, h in positions]
    y2s = [y + h for x, y, w, h in positions]
    return max(0.0, (max(x2s) - min(xs)) * (max(y2s) - min(ys)))


def check_overlap(positions: List[Tuple[float, float, float, float]], tol: float = 1e-6) -> int:
    """Count number of overlapping block pairs."""
    count = 0
    n = len(positions)
    for i in range(n):
        xi, yi, wi, hi = positions[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = positions[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            if ox > tol and oy > tol:
                count += 1
    return count


def compute_contest_cost(
    positions: List[Tuple[float, float, float, float]],
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    hpwl_baseline: float,
    area_baseline: float,
    alpha: float = 0.5,
) -> Tuple[float, bool]:
    """
    Compute contest cost (Equation 2).

    Returns:
        cost: float
        feasible: bool (no overlaps, area within 1%)
    """
    # Check overlaps
    n_overlap = check_overlap(positions)
    if n_overlap > 0:
        return 10.0, False

    # Check area constraints (1% tolerance)
    n = len(positions)
    area_violations = 0
    for i, (x, y, w, h) in enumerate(positions):
        if i < area_targets.shape[0]:
            target = float(area_targets[i].item())
            actual = w * h
            if target > 0 and abs(actual - target) / target > 0.01:
                area_violations += 1

    if area_violations > 0:
        return 10.0, False

    # Compute HPWL and area
    hpwl_b2b = calculate_hpwl_b2b(positions, b2b_connectivity)
    hpwl_p2b = calculate_hpwl_p2b(positions, p2b_connectivity, pins_pos)
    hpwl = hpwl_b2b + hpwl_p2b
    area = calculate_bbox_area(positions)

    hpwl_base = max(hpwl_baseline, 1e-8)
    area_base = max(area_baseline, 1e-8)

    hpwl_gap = (hpwl - hpwl_base) / hpwl_base
    area_gap = (area - area_base) / area_base

    cost = 1.0 + alpha * (hpwl_gap + area_gap)
    return cost, True
