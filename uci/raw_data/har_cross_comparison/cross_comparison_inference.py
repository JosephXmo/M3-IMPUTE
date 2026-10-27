"""Checkpoint loading and masked-cell inference for HAR cross comparison.

Only visible S0 values enter the graph. Scaling and checkpoint provenance are
owned by preprocess_cross_comparison; this module has no export CLI.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

# Full-module checkpoints reference classes in the M3 repository.
M3_ROOT = Path(__file__).resolve().parents[3]
if str(M3_ROOT) not in sys.path:
    sys.path.insert(0, str(M3_ROOT))


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_node_features(num_records: int, num_features: int) -> torch.Tensor:
    """Reproduce UCI ``node_mode=0``: all-one samples and one-hot features."""
    sample_nodes = np.ones((num_records, num_features), dtype=np.float32)
    feature_nodes = np.eye(num_features, dtype=np.float32)
    return torch.from_numpy(np.concatenate((sample_nodes, feature_nodes), axis=0))


def build_visible_graph(
    features: np.ndarray,
    visible: np.ndarray,
    epsilon: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the same directed bipartite graph used by ``uci_data.get_data``."""
    num_records, num_features = features.shape
    rows = np.repeat(np.arange(num_records), num_features)
    cols = np.tile(np.arange(num_features), num_records)
    flat_visible = visible.reshape(-1)
    rows = rows[flat_visible]
    cols = cols[flat_visible]
    values = features[visible].astype(np.float32)

    src = np.concatenate((rows, num_records + cols))
    dst = np.concatenate((num_records + cols, rows))
    edge_index = torch.from_numpy(np.stack((src, dst)).astype(np.int64)).to(device)
    edge_attr = torch.from_numpy(np.concatenate((values, values))[:, None]).to(device)

    dense = np.full((num_records, num_features), epsilon, dtype=np.float32)
    dense[visible] = features[visible]
    dense_mask = visible.astype(np.float32)
    dense = torch.from_numpy(dense).to(device)
    dense_mask = torch.from_numpy(dense_mask).to(device)
    node_features = build_node_features(num_records, num_features).to(device)
    return node_features, edge_index, edge_attr, dense, dense_mask


def build_target_edges(
    rows: np.ndarray,
    cols: np.ndarray,
    num_records: int,
    device: torch.device,
    bidirectional: bool = True,
) -> torch.Tensor:
    """Build target edges; decoder batching needs only the forward direction."""
    if not bidirectional:
        return torch.from_numpy(np.stack((rows, num_records + cols)).astype(np.int64)).to(device)
    src = np.concatenate((rows, num_records + cols))
    dst = np.concatenate((num_records + cols, rows))
    return torch.from_numpy(np.stack((src, dst)).astype(np.int64)).to(device)


def move_legacy_tensors(module: torch.nn.Module, device: torch.device) -> None:
    """Move plain Tensor attributes that legacy checkpoints did not register."""
    for child in module.children():
        move_legacy_tensors(child, device)
    registered = set(module._parameters) | set(module._buffers) | set(module._modules)
    for name, value in list(vars(module).items()):
        if name not in registered and isinstance(value, torch.Tensor):
            setattr(module, name, value.to(device))


def load_module(path: Path, device: torch.device) -> torch.nn.Module:
    """Load a full-module checkpoint produced by ``torch.save(model, path)``."""
    try:
        module = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        module = torch.load(path, map_location=device)
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{path} is not a complete torch.nn.Module checkpoint")
    module = module.to(device)
    move_legacy_tensors(module, device)
    module.eval()
    # Registered parameters already moved with module.to(); legacy plain tensors
    # still need explicit movement when loading non-GNN legacy module classes.
    if hasattr(module, "feature_nodes") and not isinstance(module.feature_nodes, torch.nn.Parameter):
        module.feature_nodes = module.feature_nodes.to(device)
    if hasattr(module, "device"):
        module.device = device
    return module


@torch.no_grad()
def impute_missing(
    model: torch.nn.Module,
    impute_model: torch.nn.Module,
    features_normalized: np.ndarray,
    missing_mask: np.ndarray,
    device: torch.device,
    epsilon: float,
    chunk_size: int,
) -> np.ndarray:
    """Predict missing cells in chunks and return a normalized completed matrix."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    node_x, known_edges, known_attr, dense, known_mask = build_visible_graph(
        np.nan_to_num(features_normalized, nan=epsilon),
        ~missing_mask,
        epsilon,
        device,
    )
    embeddings = model(node_x, known_attr, known_edges, dense)
    obs_embeddings = embeddings[: features_normalized.shape[0]]
    feature_embeddings = embeddings[features_normalized.shape[0]:]
    rows, cols = np.nonzero(missing_mask)
    completed = features_normalized.copy()

    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        target_edges = build_target_edges(rows[start:stop], cols[start:stop],
                                          features_normalized.shape[0], device,
                                          bidirectional=False)
        prediction, _ = impute_model(
            obs_nodes_embs=obs_embeddings,
            fea_nodes_embs=feature_embeddings,
            known_edges=known_edges,
            impute_target_edges=target_edges,
            known_mask=known_mask,
        )
        completed[rows[start:stop], cols[start:stop]] = (
            prediction[: stop - start, 0].detach().cpu().numpy()
        )
    return completed
