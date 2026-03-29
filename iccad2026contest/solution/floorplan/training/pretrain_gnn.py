"""
Phase 1a: GNN self-supervised pretraining with three tasks:
  A. Masked node feature prediction
  B. Link prediction
  C. HPWL regression
"""
import os
import random
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Batch
except ImportError:
    from torch.utils.data import DataLoader

from ..models.gnn_encoder import GNNEncoder
from ..training.losses import (
    MaskedFeatureHead, LinkPredHead, HPWLHead,
    masked_feature_loss, link_prediction_loss, hpwl_prediction_loss,
)
from ..data.dataset import FloorSetDataset
from ..utils.logger import Logger
from ..utils.seed import set_seed


MASK_RATIO = 0.15


def pretrain(cfg: dict, output_dir: str = '.'):
    """Run GNN pretraining."""
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Pretraining on {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Build model and heads
    gnn = GNNEncoder(
        hidden_dim=cfg['gnn']['hidden_dim'],
        num_layers=cfg['gnn']['num_layers'],
        dropout=cfg['gnn']['dropout'],
    ).to(device)

    hidden_dim = cfg['gnn']['hidden_dim']
    head_masked = MaskedFeatureHead(hidden_dim=hidden_dim).to(device)
    head_link = LinkPredHead(hidden_dim=hidden_dim).to(device)
    head_hpwl = HPWLHead(hidden_dim=hidden_dim).to(device)

    all_params = (
        list(gnn.parameters()) +
        list(head_masked.parameters()) +
        list(head_link.parameters()) +
        list(head_hpwl.parameters())
    )

    optimizer = AdamW(all_params, lr=cfg['gnn']['pretrain_lr'], weight_decay=1e-4)
    n_epochs = cfg['gnn']['pretrain_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    logger = Logger(project='floorset-pretrain', enabled=False)

    # Datasets
    curriculum_stages = cfg['curriculum']['stages']
    data_root = cfg['data']['dataset_path']
    cache_dir = cfg['data']['cache_dir']
    train_data_path = cfg['data'].get('train_data_path', None)

    def make_loader(block_range, shuffle=True):
        ds = FloorSetDataset(
            root=data_root,
            split='train',
            block_count_range=tuple(block_range) if block_range else None,
            cache_dir=cache_dir,
            training=True,
            train_data_path=train_data_path,
        )
        if len(ds) == 0:
            return None
        try:
            from torch_geometric.loader import DataLoader as PyGLoader
            return PyGLoader(
                ds,
                batch_size=cfg['gnn']['pretrain_batch'],
                shuffle=shuffle,
                num_workers=min(cfg['data']['num_workers'], 4),
                follow_batch=[],
                drop_last=True,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=cfg['data']['num_workers'] > 0,
            )
        except Exception:
            return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=shuffle)

    # Curriculum: first 10 epochs on small graphs, then all
    print("Building data loaders...")
    loader_small = make_loader(curriculum_stages[0])  # 21-40
    loader_all = None  # built after epoch 10

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    val_ds = FloorSetDataset(
        root=data_root, split='train',
        block_count_range=(21, 40),
        train_data_path=train_data_path,
        cache_dir=cache_dir, training=False,
    )

    for epoch in range(n_epochs):
        gnn.train()
        head_masked.train()
        head_link.train()
        head_hpwl.train()

        # Curriculum progression
        if epoch == 10 and loader_all is None:
            print("Switching to full curriculum...")
            loader_all = make_loader(None)

        loader = loader_all if (epoch >= 10 and loader_all is not None) else loader_small
        if loader is None:
            print("No data available, skipping epoch")
            continue

        total_loss = 0.0
        total_loss_a = 0.0
        total_loss_b = 0.0
        total_loss_c = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for batch in pbar:
            if batch is None:
                continue
            try:
                batch = batch.to(device)
            except Exception:
                continue

            optimizer.zero_grad()

            # Create node feature mask (15% of nodes)
            n_nodes = batch.x.shape[0]
            mask = torch.rand(n_nodes, device=device) < MASK_RATIO
            if mask.sum() == 0:
                mask[0] = True

            # Save original features, zero out masked ones
            x_orig = batch.x.clone()
            batch_x_masked = batch.x.clone()
            batch_x_masked[mask] = 0.0
            batch.x = batch_x_masked

            # Forward pass
            node_emb, graph_emb = gnn(batch)

            # Task A: masked feature reconstruction
            loss_a = masked_feature_loss(node_emb, x_orig, mask, head_masked)

            # Task B: link prediction
            loss_b = link_prediction_loss(node_emb, batch.edge_index, n_nodes, head_link)

            # Task C: HPWL prediction
            hpwl_targets = torch.tensor(
                [getattr(d, 'hpwl_base', 0.0) for d in batch.to_data_list()],
                device=device, dtype=torch.float32,
            ) if hasattr(batch, 'to_data_list') else torch.zeros(graph_emb.shape[0], device=device)
            loss_c = hpwl_prediction_loss(graph_emb, hpwl_targets, head_hpwl)

            loss = 1.0 * loss_a + 0.5 * loss_b + 1.0 * loss_c
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_loss_a += float(loss_a.item())
            total_loss_b += float(loss_b.item())
            total_loss_c += float(loss_c.item())
            n_batches += 1

            # Restore original x for next iteration (important!)
            batch.x = x_orig

            pbar.set_postfix({
                'loss': f'{total_loss/max(n_batches,1):.4f}',
                'A': f'{total_loss_a/max(n_batches,1):.4f}',
                'C': f'{total_loss_c/max(n_batches,1):.4f}',
            })

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} A={total_loss_a/max(n_batches,1):.4f} "
              f"B={total_loss_b/max(n_batches,1):.4f} C={total_loss_c/max(n_batches,1):.4f}")

        # Early stopping based on val Task-C loss
        val_loss_c = _eval_hpwl_loss(gnn, head_hpwl, val_ds, device, cfg)
        print(f"  Val HPWL MSE: {val_loss_c:.6f}")

        if val_loss_c < best_val_loss:
            best_val_loss = val_loss_c
            patience_counter = 0
            ckpt_path = os.path.join(output_dir, 'gnn_pretrained.pt')
            torch.save({
                'epoch': epoch,
                'gnn_state': gnn.state_dict(),
                'val_loss_c': val_loss_c,
            }, ckpt_path)
            print(f"  Saved best checkpoint (val_loss_c={val_loss_c:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Pretraining complete. Best val HPWL MSE: {best_val_loss:.6f}")
    return os.path.join(output_dir, 'gnn_pretrained.pt')


def _eval_hpwl_loss(gnn, head_hpwl, val_ds, device, cfg):
    """Evaluate Task-C HPWL MSE on a small validation set."""
    gnn.eval()
    head_hpwl.eval()
    total_loss = 0.0
    n = 0

    try:
        from torch_geometric.loader import DataLoader as PyGLoader
        val_loader = PyGLoader(val_ds, batch_size=16, shuffle=False,
                               num_workers=0, drop_last=False)
    except Exception:
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1)

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            try:
                batch = batch.to(device)
            except Exception:
                continue
            _, graph_emb = gnn(batch)
            if hasattr(batch, 'to_data_list'):
                hpwl_targets = torch.tensor(
                    [getattr(d, 'hpwl_base', 0.0) for d in batch.to_data_list()],
                    device=device, dtype=torch.float32,
                )
            else:
                hpwl_targets = torch.zeros(graph_emb.shape[0], device=device)
            loss = hpwl_prediction_loss(graph_emb, hpwl_targets, head_hpwl)
            total_loss += float(loss.item())
            n += 1
            if n >= 20:  # Quick eval
                break

    gnn.train()
    head_hpwl.train()
    return total_loss / max(n, 1)
