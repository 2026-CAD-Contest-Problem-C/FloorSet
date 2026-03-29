#!/usr/bin/env python3
"""
Entry point for the FloorSet GNN+RL floorplanning agent.

Usage:
    # Phase 1a: GNN pretraining
    python main.py pretrain --config floorplan/configs/default.yaml

    # Phase 1b: RL training (loads pretrained GNN automatically)
    python main.py train --config floorplan/configs/default.yaml --gnn_ckpt gnn_pretrained.pt

    # Evaluation on test set
    python main.py evaluate --config floorplan/configs/default.yaml --ckpt best_agent.pt --output ./solutions

    # Run tests
    python main.py test
"""
import argparse
import os
import sys
import yaml

# Add parent directories to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..', '..')
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def cmd_pretrain(args):
    cfg = load_config(args.config)
    output_dir = args.output or os.path.join(_HERE, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)

    from floorplan.training.pretrain_gnn import pretrain
    ckpt = pretrain(cfg, output_dir=output_dir)
    print(f"Pretrained GNN saved to: {ckpt}")


def cmd_train(args):
    cfg = load_config(args.config)
    output_dir = args.output or os.path.join(_HERE, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)

    gnn_ckpt = args.gnn_ckpt
    if gnn_ckpt is None:
        gnn_ckpt = os.path.join(output_dir, 'gnn_pretrained.pt')

    from floorplan.training.train_rl import train_rl
    train_rl(cfg, gnn_ckpt=gnn_ckpt, output_dir=output_dir)


def cmd_evaluate(args):
    cfg = load_config(args.config)
    output_dir = args.output or os.path.join(_HERE, 'solutions')

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = os.path.join(_HERE, 'checkpoints', 'best_agent.pt')

    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found: {ckpt}")
        sys.exit(1)

    from floorplan.evaluation.evaluate import evaluate
    summary, _ = evaluate(cfg, agent_ckpt=ckpt, output_dir=output_dir)
    print(f"\nFinal Score: feasibility={summary['feasibility_rate']:.3f}, "
          f"mean_cost={summary['mean_cost']:.4f}")


def cmd_test(args):
    """Run all unit tests."""
    import subprocess
    test_dir = os.path.join(_HERE, 'floorplan', 'tests')
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test_dir, '-v'],
        cwd=_HERE,
    )
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description='FloorSet GNN+RL Agent')
    sub = parser.add_subparsers(dest='command')

    # pretrain
    p = sub.add_parser('pretrain', help='Pretrain GNN encoder')
    p.add_argument('--config', default='floorplan/configs/default.yaml')
    p.add_argument('--output', default=None, help='Output directory for checkpoints')

    # train
    p = sub.add_parser('train', help='RL training')
    p.add_argument('--config', default='floorplan/configs/default.yaml')
    p.add_argument('--gnn_ckpt', default=None, help='Pretrained GNN checkpoint')
    p.add_argument('--output', default=None)

    # evaluate
    p = sub.add_parser('evaluate', help='Evaluate on test set')
    p.add_argument('--config', default='floorplan/configs/default.yaml')
    p.add_argument('--ckpt', default=None, help='Agent checkpoint')
    p.add_argument('--output', default=None)

    # test
    p = sub.add_parser('test', help='Run unit tests')

    args = parser.parse_args()

    if args.command == 'pretrain':
        cmd_pretrain(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'test':
        cmd_test(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
