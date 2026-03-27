#!/usr/bin/env python3
"""
Training script for FloorplanTransformer.

Usage:
    python iccad2026contest/solution/train.py
    python iccad2026contest/solution/train.py --epochs 20 --batch-size 4 --lr 1e-4
    python iccad2026contest/solution/train.py --resume checkpoints/best.pt

Checkpoints saved to: iccad2026contest/solution/checkpoints/
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Allow imports from repo root
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "iccad2026contest"))

from iccad2026contest.solution.model import FloorplanTransformer
from iccad2026_evaluate import (
    get_training_dataloader,
    get_validation_dataloader,
    compute_training_loss_differentiable,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=1,
                   help="Keep at 1 for variable-N graphs unless using custom collate")
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--num-samples", type=int,   default=None,
                   help="Limit training samples (None = all 1M)")
    p.add_argument("--d-model",     type=int,   default=128)
    p.add_argument("--n-heads",     type=int,   default=8)
    p.add_argument("--n-layers",    type=int,   default=6)
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--val-every",   type=int,   default=5000,
                   help="Validate every N training steps")
    p.add_argument("--save-dir",    type=str,
                   default=str(Path(__file__).parent / "checkpoints"))
    p.add_argument("--fp16",        action="store_true",
                   help="Mixed precision (fp16) training")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, device):
    model.eval()
    val_loader = get_validation_dataloader(batch_size=1)
    total_loss = 0.0
    component_sums = {}
    count = 0
    for batch in val_loader:
        # validation collate returns [inputs_list, labels_list]
        inputs, labels = batch
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        _fp_sol, metrics = labels

        area_target = area_target.squeeze(0).to(device)
        b2b_conn    = b2b_conn.squeeze(0).to(device)
        p2b_conn    = p2b_conn.squeeze(0).to(device)
        pins_pos    = pins_pos.squeeze(0).to(device)
        constraints = constraints.squeeze(0).to(device)
        metrics     = metrics.squeeze(0).to(device)

        N = int((area_target != -1).sum().item())
        if N == 0:
            continue

        positions = model(
            area_target[:N], b2b_conn, p2b_conn, pins_pos, constraints[:N]
        )
        loss, components = compute_training_loss_differentiable(
            positions, b2b_conn, p2b_conn, pins_pos,
            area_target[:N], metrics
        )
        total_loss += loss.item()
        for k, v in components.items():
            component_sums[k] = component_sums.get(k, 0.0) + v
        count += 1

    avg_components = {k: v / max(count, 1) for k, v in component_sums.items()}
    model.train()
    return total_loss / max(count, 1), avg_components


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = FloorplanTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler() if args.fp16 and device.type == "cuda" else None

    start_epoch = 0
    best_val = math.inf

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_val", math.inf)
        print(f"Resumed from {args.resume} (epoch {start_epoch}, best_val={best_val:.4f})")

    for epoch in range(start_epoch, args.epochs):
        train_loader = get_training_dataloader(
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            shuffle=True,
        )
        total_loss = 0.0
        step = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for batch in pbar:
            area_target, b2b_conn, p2b_conn, pins_pos, constraints, \
                _tree_sol, fp_sol, metrics = batch

            area_target = area_target.squeeze(0).to(device)
            b2b_conn    = b2b_conn.squeeze(0).to(device)
            p2b_conn    = p2b_conn.squeeze(0).to(device)
            pins_pos    = pins_pos.squeeze(0).to(device)
            constraints = constraints.squeeze(0).to(device)
            metrics     = metrics.squeeze(0).to(device)

            N = int((area_target != -1).sum().item())
            if N == 0:
                continue

            optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    positions = model(
                        area_target[:N], b2b_conn, p2b_conn, pins_pos, constraints[:N]
                    )
                    loss, _ = compute_training_loss_differentiable(
                        positions, b2b_conn, p2b_conn, pins_pos,
                        area_target[:N], metrics
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                positions = model(
                    area_target[:N], b2b_conn, p2b_conn, pins_pos, constraints[:N]
                )
                loss, _ = compute_training_loss_differentiable(
                    positions, b2b_conn, p2b_conn, pins_pos,
                    area_target[:N], metrics
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            step += 1

            pbar.set_postfix(loss=f"{total_loss/step:.4f}")
            if step % 500 == 0:
                print(f"  Epoch {epoch+1} step {step}: loss={total_loss/step:.4f}")

            if step % args.val_every == 0:
                val_loss, val_components = validate(model, device)
                print(f"  [Val] step {step}: val_loss={val_loss:.4f} (best={best_val:.4f})")
                print(f"        hpwl_gap={val_components['hpwl_gap']:.4f} "
                      f"area_gap={val_components['area_gap']:.4f} "
                      f"overlap={val_components['overlap_violation']:.4f} "
                      f"V_soft={val_components['V_soft']:.4f} "
                      f"violation_factor={val_components['violation_factor']:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_val": best_val,
                        "args": vars(args),
                    }, save_dir / "best.pt")
                    print(f"  Saved best checkpoint (val={best_val:.4f})")

        avg_loss = total_loss / max(step, 1)
        print(f"Epoch {epoch+1}/{args.epochs}: avg_loss={avg_loss:.4f}")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_val": best_val,
            "args": vars(args),
        }, save_dir / f"epoch_{epoch+1}.pt")

    print("Training complete.")
    print(f"Best val loss: {best_val:.4f}")
    print(f"Best checkpoint: {save_dir}/best.pt")


if __name__ == "__main__":
    main()
