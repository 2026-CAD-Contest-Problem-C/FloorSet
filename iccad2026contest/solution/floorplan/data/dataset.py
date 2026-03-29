"""FloorSetDataset: lazy-loading dataset with graph caching and curriculum support."""
import glob
import os
import random
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

from .graph_builder import build_graph


def _make_lite(data):
    """Return a lightweight copy with only the fields needed for GNN training.

    Omits block_meta (Python list of dicts, ~400 KB when serialised) and the
    raw connectivity tensors that are not consumed by the GNN forward pass.
    All items must have an identical attribute set so PyG's Batch collation
    works regardless of whether an item came from cache or was built fresh.
    """
    try:
        lite = data.__class__(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            num_nodes=data.num_nodes,
        )
        lite.hpwl_base = getattr(data, 'hpwl_base', 0.0)
        lite.area_base = getattr(data, 'area_base', 0.0)
        return lite
    except Exception:
        return data


class FloorSetDataset(Dataset):
    """
    Lazy-loading FloorSet-Lite dataset with:
    - On-disk graph caching (PyG Data objects)
    - Curriculum learning via block_count_range filter
    - Support for training and test splits
    """

    LAYOUTS_PER_FILE = 112

    def __init__(
        self,
        root: str,
        split: str = 'train',
        block_count_range: Optional[Tuple[int, int]] = None,
        cache_dir: str = './cache',
        training: bool = True,
        train_data_path: str = None,
        test_data_path: str = None,
        lite: bool = True,
    ):
        """
        Args:
            root: dataset root directory (fallback)
            split: 'train' or 'test'
            block_count_range: (min_k, max_k) filter; None = all sizes
            cache_dir: where to cache processed graphs
            training: passed to graph_builder (affects Laplacian PE sign)
            train_data_path: explicit path to floorset_lite directory
            test_data_path: explicit path to LiteTensorDataTest directory
            lite: if True, strip non-GNN fields (block_meta etc.) before caching.
                  Set False for RL training which needs block_meta and connectivity.
        """
        self.root = root
        self.split = split
        self.block_count_range = block_count_range
        self.cache_dir = cache_dir
        self.training = training
        self.lite = lite

        if split == 'train':
            # Try explicit path first, then common alternatives
            candidates = []
            if train_data_path:
                candidates.append(os.path.join(train_data_path, 'worker_*/layouts*'))
                candidates.append(os.path.join(train_data_path, 'worker_*', 'layouts*'))
            candidates += [
                os.path.join(root, 'floorset_lite', 'worker_*/layouts*'),
                os.path.join(root, 'LiteTensorData', 'worker_*/layouts*'),
            ]
            self.all_files = []
            for pattern in candidates:
                self.all_files = sorted(glob.glob(pattern))
                if self.all_files:
                    break
            self.layouts_per_file = self.LAYOUTS_PER_FILE
            os.makedirs(cache_dir, exist_ok=True)
            self._build_index_train()
        else:
            # Test set: LiteTensorDataTest/config_*/litedata_1.pth
            candidates = []
            if test_data_path:
                candidates.append(os.path.join(test_data_path, 'config_*'))
            candidates += [
                os.path.join(root, 'LiteTensorDataTest', 'config_*'),
                os.path.join(root, 'FloorSet', 'LiteTensorDataTest', 'config_*'),
            ]
            self.config_dirs = []
            for pattern in candidates:
                self.config_dirs = sorted(glob.glob(pattern))
                if self.config_dirs:
                    break
            self._build_index_test()

        os.makedirs(cache_dir, exist_ok=True)  # also covers test split
        self._cached_file_idx = -1
        self._cached_file_data = None

    def _get_file_block_counts(self) -> dict:
        """
        Build or load a mapping {file_idx: block_count} by peeking at the
        first layout of each file.  Cached to disk so it only runs once.
        Since every layout in a file has the same block count, we only need
        to load one layout per file.
        """
        index_cache = os.path.join(self.cache_dir, '_file_block_counts.pt')
        if os.path.exists(index_cache):
            try:
                return torch.load(index_cache, weights_only=True)
            except Exception:
                pass

        print(f"Building file-level block count index for {len(self.all_files)} files "
              f"(one-time, ~{len(self.all_files)/100:.0f}s)...")
        counts = {}
        for fi, fpath in enumerate(self.all_files):
            try:
                data = torch.load(fpath, weights_only=False)
                area = data[0][0][:, 0]        # first layout, area column
                counts[fi] = int((area > 0).sum().item())
            except Exception:
                counts[fi] = -1
            if fi % 500 == 0 and fi > 0:
                print(f"  {fi}/{len(self.all_files)} files scanned...")

        torch.save(counts, index_cache)
        print(f"Block count index saved to {index_cache}")
        return counts

    def _build_index_train(self):
        """Build list of (file_idx, layout_idx) filtered by block_count_range.

        Uses a file-level block count index so we only index files whose
        layouts are guaranteed to pass the filter — no wasted loads at
        __getitem__ time.
        """
        if self.block_count_range is not None:
            file_counts = self._get_file_block_counts()
            lo, hi = self.block_count_range
            valid_files = [fi for fi, cnt in file_counts.items() if lo <= cnt <= hi]
        else:
            valid_files = list(range(len(self.all_files)))

        self.index = []
        for fi in valid_files:
            for li in range(self.layouts_per_file):
                self.index.append((fi, li))
        # Sorted by file_idx so per-worker file cache stays warm.
        # DataLoader handles shuffling via shuffle=True.
        print(f"Index built: {len(valid_files)}/{len(self.all_files)} files, "
              f"{len(self.index)} total items"
              + (f" (block_count_range={self.block_count_range})" if self.block_count_range else ""))

    def _build_index_test(self):
        self.index = list(range(len(self.config_dirs)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.split == 'train':
            return self._get_train_item(idx)
        else:
            return self._get_test_item(idx)

    def _get_train_item(self, idx):
        n = len(self.index)
        for offset in range(n):
            i = (idx + offset) % n
            file_idx, layout_idx = self.index[i]
            cache_key = f"train_{file_idx}_{layout_idx}.pt"
            cache_path = os.path.join(self.cache_dir, cache_key)

            if os.path.exists(cache_path):
                try:
                    return torch.load(cache_path, weights_only=False)
                except Exception:
                    pass

            data = self._load_train_raw(file_idx, layout_idx)
            if data is None:
                continue

            if self.lite:
                # Strip non-GNN fields so all items have identical attribute sets,
                # whether loaded from cache or built fresh.
                data = _make_lite(data)
            try:
                torch.save(data, cache_path)
            except Exception:
                pass
            return data

        raise RuntimeError(
            f"No valid training sample found in dataset "
            f"(block_count_range={self.block_count_range}). "
            "Check that the data path is correct and the block range matches the data."
        )

    def _get_test_item(self, idx):
        config_dir = self.config_dirs[self.index[idx]]
        cache_key = f"test_{os.path.basename(config_dir)}.pt"
        cache_path = os.path.join(self.cache_dir, cache_key)

        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path, weights_only=False)
            except Exception:
                pass

        data_path = os.path.join(config_dir, 'litedata_1.pth')
        label_path = os.path.join(config_dir, 'litelabel_1.pth')

        raw = torch.load(data_path, weights_only=False)
        label = torch.load(label_path, weights_only=False)

        # raw[0] = [area+constraints (n,6), b2b, p2b, pins]
        inner = raw[0]
        block_data = inner[0]  # (n_blocks, 6): area + 5 constraints
        b2b = inner[1]
        p2b = inner[2]
        pins = inner[3]

        area_target = block_data[:, 0]
        constraints = block_data[:, 1:]

        # Label: sol polygons + metrics
        label_inner = label[0]
        metrics = label_inner[0] if len(label_inner) > 0 else None

        graph = build_graph(
            area_target=area_target,
            b2b_connectivity=b2b,
            p2b_connectivity=p2b,
            pins_pos=pins,
            placement_constraints=constraints,
            fp_sol=None,
            metrics_sol=metrics,
            training=False,
        )
        graph.config_dir = config_dir

        try:
            torch.save(graph, cache_path)
        except Exception:
            pass
        return graph

    def _load_train_raw(self, file_idx: int, layout_idx: int):
        """Load one layout from a training file."""
        if file_idx != self._cached_file_idx:
            try:
                self._cached_file_data = torch.load(
                    self.all_files[file_idx], weights_only=False
                )
                self._cached_file_idx = file_idx
            except Exception:
                return None

        contents = self._cached_file_data
        # contents: [area_targets+constraints, b2b, p2b, pins, tree_sol, fp_sol, metrics]
        # Each is a tensor of shape (layouts_per_file, ...)
        try:
            if len(contents) >= 7:
                # Standard format
                block_data = contents[0][layout_idx]  # (n_blocks, 6)
                b2b = contents[1][layout_idx]
                p2b = contents[2][layout_idx]
                pins = contents[3][layout_idx]
                fp_sol = contents[5][layout_idx]
                metrics = contents[6][layout_idx]

                area_target = block_data[:, 0]
                constraints = block_data[:, 1:]
            else:
                # Fallback for different format
                return None

            graph = build_graph(
                area_target=area_target,
                b2b_connectivity=b2b,
                p2b_connectivity=p2b,
                pins_pos=pins,
                placement_constraints=constraints,
                fp_sol=fp_sol,
                metrics_sol=metrics,
                training=self.training,
            )
            return graph

        except Exception as e:
            return None


class CurriculumDataset(Dataset):
    """
    Wraps FloorSetDataset for curriculum learning.
    Supports experience replay from earlier stages.
    """

    def __init__(
        self,
        root: str,
        current_stage: Tuple[int, int],
        past_stages: list = None,
        replay_frac: float = 0.2,
        cache_dir: str = './cache',
    ):
        self.current_ds = FloorSetDataset(
            root, split='train',
            block_count_range=current_stage,
            cache_dir=cache_dir,
        )
        self.past_datasets = []
        if past_stages:
            for stage in past_stages:
                self.past_datasets.append(FloorSetDataset(
                    root, split='train',
                    block_count_range=stage,
                    cache_dir=cache_dir,
                ))
        self.replay_frac = replay_frac

    def __len__(self):
        return len(self.current_ds)

    def __getitem__(self, idx):
        if self.past_datasets and random.random() < self.replay_frac:
            ds = random.choice(self.past_datasets)
            ridx = random.randint(0, len(ds) - 1)
            item = ds[ridx]
            if item is not None:
                return item
        return self.current_ds[idx]
