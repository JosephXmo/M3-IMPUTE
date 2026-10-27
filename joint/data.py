"""Full official UCI-HAR tables, fixed graphs, and auditable provenance."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class HARData:
    features: torch.Tensor
    normalized: torch.Tensor
    offset: torch.Tensor
    scale: torch.Tensor
    labels: np.ndarray
    subjects: np.ndarray
    segments: np.ndarray
    manifest: dict


def load_har(root, strict=True):
    """Concatenate official train then test; all rows participate in training.

    Fit min/max over this explicitly transductive population. The official
    partition is retained only as provenance/segment metadata, not a holdout.
    """
    root = Path(root)
    xs, ys, subjects, boundaries, hashes = [], [], [], [], {}
    for split, expected in [('train', 7352), ('test', 2947)]:
        paths = [root / split / f'{kind}_{split}.txt' for kind in ('X', 'y', 'subject')]
        x = np.loadtxt(paths[0], dtype=np.float32, ndmin=2)
        y = np.loadtxt(paths[1], dtype=np.int64, ndmin=1)
        subject = np.loadtxt(paths[2], dtype=np.int64, ndmin=1)
        if len(y) != len(x) or len(subject) != len(x) or not np.isfinite(x).all():
            raise ValueError(f'Invalid aligned feature/label/subject tables: {split}')
        if strict and x.shape != (expected, 561):
            raise ValueError(f'Expected complete official {split} table, got {x.shape}')
        for path in paths:
            hashes[str(path.relative_to(root))] = sha256(path)
        boundaries.append(np.r_[True, subject[1:] != subject[:-1]])
        xs.append(x)
        ys.append(y)
        subjects.append(subject)
    x = torch.from_numpy(np.concatenate(xs))
    offset, maximum = x.amin(0), x.amax(0)
    scale = maximum - offset
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return HARData(
        x, (x - offset) / scale, offset, scale, np.concatenate(ys),
        np.concatenate(subjects), np.concatenate(boundaries).cumsum() - 1,
        {'files_sha256': hashes, 'rows': len(x), 'features': x.shape[1],
         'official_rows': [len(a) for a in xs], 'row_order': 'official train, then official test',
         'protocol': 'full-population transductive clustering; no held-out rows or cells',
         'normalization_fit': 'all 10299 input rows; labels excluded',
         'm3_input': 'fresh self-supervised cell masking each epoch; no permanent holdout',
         'm3_output': 'decoder reconstruction of every cell'},
    )


def visibility_mask(shape, missing_rate, seed):
    """Epoch-local corruption; independent of labels and the training RNG."""
    if not 0 < missing_rate < 1 or min(shape) < 2:
        raise ValueError('mask_rate must be in (0,1), with at least two rows/columns')
    generator = torch.Generator().manual_seed(seed)
    known = torch.rand(shape, generator=generator) >= missing_rate
    # Keep each node connected even in tiny tests or extreme mask configurations.
    known[~known.any(1), 0] = True
    known[0, ~known.any(0)] = True
    if known.all():
        known[-1, -1] = False
    return known


def bipartite_graph(values, known):
    """Visible sample-feature cells in both directions; hidden values excluded."""
    n, f = values.shape
    rows, columns = known.nonzero(as_tuple=True)
    forward = torch.stack((rows, columns + n))
    return torch.cat((forward, forward.flip(0)), 1), values[known].reshape(-1, 1).repeat(2, 1)


def make_xhar_graph(features, segments, topk=5, threshold=.988, block_size=256, temporal_delta=2):
    """ncos KNN on supplied reconstructions + segment temporal pairs + loops.

    KNN semantics follow XHAR (positive similarities, threshold, index tie break).
    Temporal neighbors never cross subject changes or the official split boundary.
    No class labels participate. Edge dropout is intentionally absent.
    """
    x = features.cpu().numpy().astype(np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n = len(x)
    if not 1 <= topk < n or block_size < 1 or temporal_delta < 0:
        raise ValueError('Invalid graph parameters')
    sources, targets = [], []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        sim = x[start:stop] @ x.T
        sim[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        for source, row in enumerate(sim, start):
            cutoff = np.partition(row, -topk)[-topk]
            better = np.flatnonzero(row > cutoff)
            tied = np.flatnonzero(row == cutoff)[:topk - len(better)]
            neighbors = np.concatenate((better, tied))
            neighbors = neighbors[(row[neighbors] > 0) & (row[neighbors] >= threshold)]
            sources.extend([source] * len(neighbors))
            targets.extend(neighbors.tolist())
    temporal = []
    for gap in range(1, temporal_delta + 1):
        source = np.arange(n - gap)
        source = source[segments[source] == segments[source + gap]]
        temporal.append(np.stack((source, source + gap)))
    pairs = torch.from_numpy(np.concatenate(temporal, 1)) if temporal else torch.empty(2, 0, dtype=torch.long)
    knn = torch.tensor([sources, targets], dtype=torch.long)
    edges = torch.cat((knn, knn.flip(0), pairs, pairs.flip(0), torch.arange(n).repeat(2, 1)), 1)
    # Integer encoding makes coalescing deterministic, without dense adjacency.
    keys = torch.unique(edges[0] * n + edges[1], sorted=True)
    return torch.stack((keys // n, keys % n)), pairs
