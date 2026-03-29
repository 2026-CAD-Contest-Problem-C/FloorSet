"""
Evaluate trained agent on all 100 test cases.
"""
import os
import json
import math
from typing import List, Tuple

import torch
from tqdm import tqdm

from ..models.full_agent import FloorplanAgent
from ..env.floorplan_env import FloorplanEnv
from ..data.dataset import FloorSetDataset
from ..training.metrics import (
    calculate_hpwl_b2b, calculate_hpwl_p2b,
    calculate_bbox_area, check_overlap, compute_contest_cost,
)


def evaluate(
    cfg: dict,
    agent_ckpt: str,
    output_dir: str,
    device_str: str = None,
):
    """
    Run agent on all 100 test cases and save solutions.

    Returns:
        dict of per-test metrics and summary statistics
    """
    if device_str is None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    os.makedirs(output_dir, exist_ok=True)

    # Load agent
    agent = FloorplanAgent.from_config(cfg).to(device)
    agent.load(agent_ckpt, device=device)
    agent.eval()

    # Build env
    env = FloorplanEnv(
        agent=agent,
        grid_size=cfg['placement']['grid_size'],
        alpha=cfg['reward']['alpha'],
        overlap_pen=cfg['reward']['overlap_pen'],
        device=device_str,
    )

    # Load test dataset
    test_ds = FloorSetDataset(
        root=cfg['data']['dataset_path'],
        split='test',
        cache_dir=cfg['data']['cache_dir'],
        training=False,
        test_data_path=cfg['data'].get('test_data_path', None),
    )
    print(f"Test set: {len(test_ds)} cases")

    all_metrics = []
    solutions = {}

    for idx in tqdm(range(len(test_ds)), desc="Evaluating"):
        data = test_ds[idx]
        if data is None:
            print(f"Warning: test case {idx} is None")
            continue

        # Run greedy episode
        sol, metrics = _run_greedy_episode(agent, env, data, device)

        metrics['test_idx'] = idx
        all_metrics.append(metrics)

        # Save solution
        solutions[str(idx)] = {
            'placements': sol,
            'metrics': metrics,
        }

        if idx % 10 == 0:
            _print_metrics(idx, metrics)

    # Summary
    feasible = [m for m in all_metrics if m['feasible']]
    feasibility_rate = len(feasible) / max(len(all_metrics), 1)
    mean_hpwl_gap = sum(m['hpwl_gap'] for m in all_metrics) / max(len(all_metrics), 1)
    mean_area_gap = sum(m['area_gap'] for m in all_metrics) / max(len(all_metrics), 1)
    mean_cost = sum(m['cost'] for m in all_metrics) / max(len(all_metrics), 1)

    summary = {
        'feasibility_rate': feasibility_rate,
        'mean_hpwl_gap': mean_hpwl_gap,
        'mean_area_gap': mean_area_gap,
        'mean_cost': mean_cost,
        'n_feasible': len(feasible),
        'n_total': len(all_metrics),
    }

    print("\n=== Evaluation Summary ===")
    print(f"Feasibility rate: {feasibility_rate:.3f} ({len(feasible)}/{len(all_metrics)})")
    print(f"Mean HPWL gap:    {mean_hpwl_gap:.4f}")
    print(f"Mean area gap:    {mean_area_gap:.4f}")
    print(f"Mean cost:        {mean_cost:.4f}")

    # Save results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump({'summary': summary, 'per_case': all_metrics}, f, indent=2)

    sol_path = os.path.join(output_dir, 'solutions.json')
    _save_solutions_json(solutions, sol_path)

    return summary, all_metrics


def _run_greedy_episode(agent, env, data, device):
    """Run one greedy episode. Returns (solution, metrics dict)."""
    obs = env.reset(data)

    done = False
    while not done and obs is not None:
        obs_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in obs.items()
        }
        with torch.no_grad():
            action, _, _ = agent.placement_policy.get_action(obs_dev, deterministic=True)
        obs, _, done, _ = env.step(action)

    # Get solution
    sol = env.get_solution()

    # Compute metrics
    b2b = getattr(data, 'b2b_connectivity', None)
    p2b = getattr(data, 'p2b_connectivity', None)
    pins = getattr(data, 'pins_pos', None)
    area_targets = getattr(data, 'area_target', None)
    hpwl_base = float(data.hpwl_base) if hasattr(data, 'hpwl_base') else 0.0
    area_base = float(data.area_base) if hasattr(data, 'area_base') else 0.0

    hpwl_b2b = calculate_hpwl_b2b(sol, b2b)
    hpwl_p2b = calculate_hpwl_p2b(sol, p2b, pins)
    hpwl = hpwl_b2b + hpwl_p2b
    area = calculate_bbox_area(sol)
    n_overlaps = check_overlap(sol)

    hpwl_gap = (hpwl - hpwl_base) / max(hpwl_base, 1e-8)
    area_gap = (area - area_base) / max(area_base, 1e-8)
    feasible = (n_overlaps == 0)
    cost = 1.0 + 0.5 * (hpwl_gap + area_gap) if feasible else 10.0

    metrics = {
        'feasible': feasible,
        'n_overlaps': n_overlaps,
        'hpwl': hpwl,
        'hpwl_base': hpwl_base,
        'hpwl_gap': hpwl_gap,
        'area': area,
        'area_base': area_base,
        'area_gap': area_gap,
        'cost': cost,
        'n_blocks': data.num_nodes,
    }

    return sol, metrics


def _print_metrics(idx, metrics):
    status = "OK" if metrics['feasible'] else "INFEASIBLE"
    print(f"  Case {idx:3d} [{status}]: blocks={metrics['n_blocks']:3d} "
          f"cost={metrics['cost']:.3f} hpwl_gap={metrics['hpwl_gap']:.3f} "
          f"area_gap={metrics['area_gap']:.3f}")


def _save_solutions_json(solutions: dict, path: str):
    """Save solutions in FloorSet submission format."""
    output = {}
    for k, v in solutions.items():
        output[k] = {
            'placements': [list(p) for p in v['placements']],
        }
    with open(path, 'w') as f:
        json.dump(output, f)
    print(f"Saved solutions to {path}")
