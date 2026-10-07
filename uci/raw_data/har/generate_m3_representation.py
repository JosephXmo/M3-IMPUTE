"""Impute UCI-HAR with a trained M3 model and export sample representations.

The script deliberately uses the same encoder for both inputs:

* ``data_with_missing.txt`` is first imputed by ``ImputeNet``; the completed
  feature matrix is then encoded with every cell visible.
* ``data.txt`` is encoded directly with every cell visible.

The two exported representation files therefore have the same row order and
dimensionality, which makes them interchangeable as ``X_path`` inputs for a
downstream experiment.  The M3 checkpoints are the complete-module files
written by ``train_mdi.py --save_model``.

Example from the repository root::

    python uci/raw_data/har/generate_m3_representation.py \
        --model-path uci/test/har/har_m3/model.pt \
        --impute-model-path uci/test/har/har_m3/impute_model.pt \
        --device cuda:0

``data.txt`` is complete and remains the reference matrix.  The masked input
must contain NaNs only in feature columns; its final label column is preserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Make the M3 repository importable when this file is launched from any cwd.
M3_ROOT = Path(__file__).resolve().parents[3]
if str(M3_ROOT) not in sys.path:
    sys.path.insert(0, str(M3_ROOT))

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPLETE = SCRIPT_DIR / "data" / "data.txt"
DEFAULT_MISSING = SCRIPT_DIR / "data" / "data_with_missing.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "representation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--impute-model-path", type=Path, required=True)
    parser.add_argument("--complete-input", type=Path, default=DEFAULT_COMPLETE)
    parser.add_argument("--missing-input", type=Path, default=DEFAULT_MISSING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default="har")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="missing target pairs per ImputeNet call")
    parser.add_argument("--epsilon", type=float, default=1e-4,
                        help="placeholder for hidden cells during imputation")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``[features, label]`` and return feature matrix plus labels."""
    table = np.loadtxt(path, dtype=np.float32)
    if table.ndim == 1:
        table = table[None, :]
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f"{path} must be a 2-D table with a final label column")
    return table[:, :-1], table[:, -1]


def fit_minmax(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-feature offsets/scales matching sklearn MinMaxScaler."""
    if not np.isfinite(features).all():
        raise ValueError("the complete reference matrix contains NaN or Inf")
    offset = features.min(axis=0)
    scale = features.max(axis=0) - offset
    # MinMaxScaler maps constant columns to zero; a unit divisor has the same
    # effect while keeping the inverse transform well-defined.
    scale = np.where(scale == 0, 1.0, scale)
    return offset.astype(np.float32), scale.astype(np.float32)


def transform(features: np.ndarray, offset: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply Min-Max normalization while preserving missing NaNs."""
    return ((features - offset) / scale).astype(np.float32)


def inverse_transform(features: np.ndarray, offset: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (features * scale + offset).astype(np.float32)


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
) -> torch.Tensor:
    """Build forward-then-reverse target edges expected by ``ImputeNet``."""
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
    # feature_nodes is a legacy plain Tensor rather than a registered buffer.
    if hasattr(module, "feature_nodes"):
        module.feature_nodes = module.feature_nodes.to(device)
    if hasattr(module, "device"):
        module.device = device
    return module


@torch.no_grad()
def encode_complete(
    model: torch.nn.Module,
    features_normalized: np.ndarray,
    device: torch.device,
    epsilon: float,
) -> np.ndarray:
    """Encode a complete feature matrix and return sample-node embeddings."""
    visible = np.ones(features_normalized.shape, dtype=bool)
    node_x, edge_index, edge_attr, dense, _ = build_visible_graph(
        features_normalized, visible, epsilon, device
    )
    embeddings = model(node_x, edge_attr, edge_index, dense)
    return embeddings[: features_normalized.shape[0]].detach().cpu().numpy()


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
                                          features_normalized.shape[0], device)
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


def save_txt(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, array, fmt="%.8g")


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    if not args.impute_model_path.is_file():
        raise FileNotFoundError(args.impute_model_path)

    device = choose_device(args.device)
    complete_features, complete_labels = load_table(args.complete_input)
    missing_features, missing_labels = load_table(args.missing_input)
    if complete_features.shape != missing_features.shape:
        raise ValueError("complete and missing feature matrices have different shapes")
    if not np.allclose(complete_labels, missing_labels, equal_nan=True):
        raise ValueError("complete and missing files do not have matching labels")

    offset, scale = fit_minmax(complete_features)
    complete_normalized = transform(complete_features, offset, scale)
    missing_mask = np.isnan(missing_features)
    missing_normalized = transform(missing_features, offset, scale)

    model = load_module(args.model_path, device)
    impute_model = load_module(args.impute_model_path, device)

    print(f"[Config] device: {device}")
    print(f"[Config] samples/features: {complete_features.shape}")
    print(f"[Config] missing cells: {int(missing_mask.sum())}")

    completed_normalized = impute_missing(
        model, impute_model, missing_normalized, missing_mask,
        device, args.epsilon, args.chunk_size,
    )
    completed_features = inverse_transform(completed_normalized, offset, scale)

    complete_representation = encode_complete(
        model, complete_normalized, device, args.epsilon
    )
    imputed_representation = encode_complete(
        model, completed_normalized, device, args.epsilon
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    imputed_features_path = args.output_dir / f"{prefix}_missing_imputed_features.txt"
    imputed_data_path = args.output_dir / f"{prefix}_missing_imputed_data.txt"
    complete_rep_path = args.output_dir / f"{prefix}_complete_representation.txt"
    imputed_rep_path = args.output_dir / f"{prefix}_missing_imputed_representation.txt"
    labels_path = args.output_dir / f"{prefix}_labels.txt"

    save_txt(imputed_features_path, completed_features)
    save_txt(imputed_data_path, np.column_stack((completed_features, complete_labels)))
    save_txt(complete_rep_path, complete_representation)
    save_txt(imputed_rep_path, imputed_representation)
    save_txt(labels_path, complete_labels)

    manifest = {
        "samples": int(complete_features.shape[0]),
        "input_features": int(complete_features.shape[1]),
        "representation_dim": int(complete_representation.shape[1]),
        "missing_cells": int(missing_mask.sum()),
        "device": str(device),
        "model_path": str(args.model_path),
        "impute_model_path": str(args.impute_model_path),
        "complete_representation": str(complete_rep_path),
        "imputed_representation": str(imputed_rep_path),
        "imputed_features": str(imputed_features_path),
        "imputed_data_with_label": str(imputed_data_path),
        "labels": str(labels_path),
    }
    with (args.output_dir / f"{prefix}_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[SAVE] complete representation: {complete_rep_path}")
    print(f"[SAVE] imputed representation:  {imputed_rep_path}")
    print(f"[SAVE] imputed features:         {imputed_features_path}")


if __name__ == "__main__":
    main()
