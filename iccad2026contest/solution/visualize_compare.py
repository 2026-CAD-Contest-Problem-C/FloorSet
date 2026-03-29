"""
比較 Agent 的擺放結果和 GT 的擺放結果。

用法：
    python visualize_compare.py --ckpt checkpoints/best_agent.pt --case 0
    python visualize_compare.py --ckpt checkpoints/best_agent.pt --case 5 --save result.png
"""
import argparse
import os
import sys
import math

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import yaml
from floorplan.models.full_agent import FloorplanAgent
from floorplan.env.floorplan_env import FloorplanEnv
from floorplan.data.dataset import FloorSetDataset


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _block_color(constraints):
    """Return (facecolor, label) based on constraint flags."""
    c = constraints
    if len(c) >= 4 and c[3]:    return 'salmon',      'cluster'
    if len(c) >= 1 and c[0]:    return 'violet',      'fixed'
    if len(c) >= 2 and c[1]:    return 'gray',        'preplaced'
    if len(c) >= 3 and c[2]:    return 'darkgreen',   'mib'
    if len(c) >= 5 and c[4]:    return 'olive',       'boundary'
    return 'steelblue', 'normal'


# ---------------------------------------------------------------------------
# Draw one floorplan onto an axes
# ---------------------------------------------------------------------------

def draw_floorplan(ax, placements, b2b, p2b, pins, constraints, title):
    """
    placements : list of (x, y, w, h)
    b2b        : (E, 3) tensor [i, j, weight]  or None
    p2b        : (E, 3) tensor [pin, block, w] or None
    pins       : (P, 2) tensor                 or None
    constraints: (N, 5) tensor                 or None
    """
    W = H = 0.0
    centroids = {}

    for idx, (x, y, w, h) in enumerate(placements):
        c = constraints[idx].tolist() if constraints is not None else [0]*5
        fc, label = _block_color(c)
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=0.8, edgecolor='black',
            facecolor=fc, alpha=0.4, label=label,
        )
        ax.add_patch(rect)
        ax.annotate(str(idx), (x + w * 0.1, y + h * 0.1), fontsize=5, color='black')
        W = max(W, x + w)
        H = max(H, y + h)
        centroids[idx] = (x + w / 2, y + h / 2)

    # Pins
    if pins is not None:
        for px, py in pins.tolist():
            ax.add_patch(Circle((px, py), radius=W * 0.005, color='limegreen', zorder=5))

    # B2B edges
    if b2b is not None and b2b.shape[0] > 0:
        n = len(placements)
        for row in b2b.tolist():
            i, j = int(row[0]), int(row[1])
            if 0 <= i < n and 0 <= j < n and i != j:
                ax.plot(
                    [centroids[i][0], centroids[j][0]],
                    [centroids[i][1], centroids[j][1]],
                    color='red', linewidth=0.3, alpha=0.6,
                )

    # P2B edges
    if p2b is not None and p2b.shape[0] > 0 and pins is not None:
        n_pins = pins.shape[0]
        n_blk  = len(placements)
        for row in p2b.tolist():
            pi, bi = int(row[0]), int(row[1])
            if 0 <= pi < n_pins and 0 <= bi < n_blk:
                px, py = pins[pi].tolist()
                ax.plot(
                    [px, centroids[bi][0]],
                    [py, centroids[bi][1]],
                    color='blue', linewidth=0.2, alpha=0.5,
                )

    margin = max(W, H) * 0.1
    ax.set_xlim(-margin, W + margin)
    ax.set_ylim(-margin, H + margin)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)

    # Deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=6, loc='upper right')


# ---------------------------------------------------------------------------
# Load GT placement from litelabel file
# ---------------------------------------------------------------------------

def load_gt_placement(config_dir):
    """
    Returns (fp_sol as list of (x,y,w,h), constraints tensor, b2b, p2b, pins)
    """
    data_path  = os.path.join(config_dir, 'litedata_1.pth')
    label_path = os.path.join(config_dir, 'litelabel_1.pth')

    raw   = torch.load(data_path,  weights_only=False)
    label = torch.load(label_path, weights_only=False)

    inner = raw[0]
    block_data   = inner[0]          # (N, 6): area + 5 constraints
    b2b          = inner[1]
    p2b          = inner[2]
    pins         = inner[3]
    constraints  = block_data[:, 1:]  # (N, 5) flags

    # GT solution: label[0] = [metrics(8,), polygons(N,5,2)]
    label_inner = label[0]
    poly_sol = None
    for item in label_inner:
        if isinstance(item, torch.Tensor) and item.ndim == 3 and item.shape[1:] == (5, 2):
            poly_sol = item  # (N, 5, 2) polygon vertices per block
            break

    if poly_sol is None:
        return None, constraints, b2b, p2b, pins

    # Convert polygon (N, 5, 2) → (x, y, w, h) via bounding box
    n = int((block_data[:, 0] > 0).sum().item())
    placements = []
    for i in range(n):
        verts = poly_sol[i]  # (5, 2)
        xs = verts[:, 0]
        ys = verts[:, 1]
        x = float(xs.min().item())
        y = float(ys.min().item())
        w = float((xs.max() - xs.min()).item())
        h = float((ys.max() - ys.min()).item())
        if w > 0 and h > 0:
            placements.append((x, y, w, h))
        else:
            placements.append((0.0, 0.0, math.sqrt(max(block_data[i, 0].item(), 1e-8)),
                                math.sqrt(max(block_data[i, 0].item(), 1e-8))))

    return placements, constraints[:n], b2b, p2b, pins


# ---------------------------------------------------------------------------
# Run agent on one test case
# ---------------------------------------------------------------------------

def run_agent(agent, env, data, device):
    obs = env.reset(data)
    done = False
    while not done and obs is not None:
        obs_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in obs.items()}
        with torch.no_grad():
            action, _, _ = agent.placement_policy.get_action(obs_dev, deterministic=True)
        obs, _, done, _ = env.step(action)
    return env.get_solution()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',      default='checkpoints/best_agent.pt')
    parser.add_argument('--config',    default='floorplan/configs/default.yaml')
    parser.add_argument('--case',      type=int, default=None, help='Test case index (0-99, sorted order)')
    parser.add_argument('--config_id', type=int, default=None, help='Config directory number (e.g. 34 → config_34)')
    parser.add_argument('--save',      default=None, help='Save figure to this path instead of showing')
    parser.add_argument('--no_edges',  action='store_true', help='Hide b2b and p2b connectivity edges')
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load agent
    agent = FloorplanAgent.from_config(cfg).to(device)
    agent.load(args.ckpt, device=device)
    agent.eval()

    env = FloorplanEnv(
        agent=agent,
        grid_size=cfg['placement']['grid_size'],
        alpha=cfg['reward']['alpha'],
        overlap_pen=cfg['reward']['overlap_pen'],
        device=str(device),
    )

    # Load test dataset (full, with block_meta)
    test_ds = FloorSetDataset(
        root=cfg['data']['dataset_path'],
        split='test',
        cache_dir=cfg['data'].get('rl_cache_dir', cfg['data']['cache_dir']),
        training=False,
        test_data_path=cfg['data'].get('test_data_path'),
        lite=False,
    )

    # Resolve case index
    if args.config_id is not None:
        target = f'config_{args.config_id}'
        matches = [i for i, d in enumerate(test_ds.config_dirs) if os.path.basename(d) == target]
        if not matches:
            print(f"config_{args.config_id} not found in test dataset")
            return
        case_idx = matches[0]
    elif args.case is not None:
        case_idx = args.case
    else:
        case_idx = 0

    if case_idx >= len(test_ds):
        print(f"Case {case_idx} out of range (dataset has {len(test_ds)} cases)")
        return

    data = test_ds[case_idx]
    config_name = os.path.basename(test_ds.config_dirs[case_idx])
    print(f"Case {case_idx} ({config_name}): {data.num_nodes} blocks")

    # Run agent
    agent_sol = run_agent(agent, env, data, device)
    print(f"Agent: {len(agent_sol)} blocks placed")

    # Load GT
    config_dir = test_ds.config_dirs[case_idx]
    gt_sol, constraints, b2b, p2b, pins = load_gt_placement(config_dir)

    # Get connectivity from data (for agent plot)
    b2b_data  = getattr(data, 'b2b_connectivity', b2b)
    p2b_data  = getattr(data, 'p2b_connectivity', p2b)
    pins_data = getattr(data, 'pins_pos', pins)

    # Compute quick metrics
    def hpwl(sol, b2b_t, p2b_t, pins_t):
        total = 0.0
        if b2b_t is not None:
            for row in b2b_t.tolist():
                i, j = int(row[0]), int(row[1])
                if 0 <= i < len(sol) and 0 <= j < len(sol):
                    cx1, cy1 = sol[i][0]+sol[i][2]/2, sol[i][1]+sol[i][3]/2
                    cx2, cy2 = sol[j][0]+sol[j][2]/2, sol[j][1]+sol[j][3]/2
                    w = row[2] if len(row) > 2 else 1.0
                    total += abs(cx1-cx2) * w + abs(cy1-cy2) * w
        return total

    def bbox_area(sol):
        xs = [x for x, y, w, h in sol]
        ys = [y for x, y, w, h in sol]
        xe = [x + w for x, y, w, h in sol]
        ye = [y + h for x, y, w, h in sol]
        return (max(xe) - min(xs)) * (max(ye) - min(ys))

    def blocks_area(sol):
        return sum(w * h for x, y, w, h in sol)

    agent_hpwl  = hpwl(agent_sol, b2b_data, p2b_data, pins_data)
    gt_hpwl     = hpwl(gt_sol, b2b, p2b, pins) if gt_sol else 0.0
    agent_bbox  = bbox_area(agent_sol)
    gt_bbox     = bbox_area(gt_sol) if gt_sol else 0.0
    agent_util  = blocks_area(agent_sol) / agent_bbox if agent_bbox > 0 else 0.0
    gt_util     = blocks_area(gt_sol) / gt_bbox if gt_sol and gt_bbox > 0 else 0.0
    print(f"Agent HPWL: {agent_hpwl:.2f}  |  GT HPWL: {gt_hpwl:.2f}")
    print(f"Agent BBox: {agent_bbox:.2f}  |  GT BBox: {gt_bbox:.2f}")
    print(f"Agent Util: {agent_util:.1%}  |  GT Util: {gt_util:.1%}")

    # Plot
    cols = 2 if gt_sol else 1
    fig, axes = plt.subplots(1, cols, figsize=(7 * cols, 7))
    if cols == 1:
        axes = [axes]

    show_b2b = None if args.no_edges else b2b_data
    show_p2b = None if args.no_edges else p2b_data
    show_pins = None if args.no_edges else pins_data

    draw_floorplan(
        axes[0], agent_sol,
        show_b2b, show_p2b, show_pins,
        constraints,
        f'Agent  ({config_name}, {data.num_nodes} blocks)\n'
        f'HPWL={agent_hpwl:.1f}  BBox={agent_bbox:.1f}  Util={agent_util:.1%}',
    )

    if gt_sol:
        draw_floorplan(
            axes[1], gt_sol,
            None if args.no_edges else b2b,
            None if args.no_edges else p2b,
            None if args.no_edges else pins,
            constraints,
            f'GT  ({config_name})\n'
            f'HPWL={gt_hpwl:.1f}  BBox={gt_bbox:.1f}  Util={gt_util:.1%}',
        )

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
