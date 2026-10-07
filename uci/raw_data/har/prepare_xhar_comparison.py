"""Prepare comparable UCI-HAR inputs for an XHAR-SDCN experiment.

The script keeps the original 561-dimensional feature space.  It creates a
fixed missing-value benchmark, a linear-interpolation baseline, and the
row-dropped baseline.  An M3-imputed matrix can be supplied after running the
M3 inference script.  XHAR-SDCN YAML files and a runner script are generated
with identical model settings for every available condition.

Typical row-controlled benchmark (about 10% cell missingness):

    python uci/raw_data/har/prepare_xhar_comparison.py \
        --mask-mode row --rate 0.1 --row-fraction 0.2 \
        --m3-features uci/raw_data/har/representation/har_missing_imputed_features.txt

For ``--mask-mode row``, ``rate`` is the desired global cell-missing ratio.
If ``--cells-per-row`` is omitted, the script derives it from ``rate`` and
``row-fraction``.  For example, 20% selected rows and 280 hidden features per
selected row gives approximately 10% hidden cells while retaining 80% of the
rows in ``S_drop``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
M3_HAR_DATA = SCRIPT_DIR / "data" / "data.txt"
M3_HAR_MISSING = SCRIPT_DIR / "data" / "data_with_missing.txt"
M3_HAR_MASK = SCRIPT_DIR / "data" / "missing_mask.txt"
# ``har/`` -> ``raw_data/`` -> ``uci/`` -> ``M3-IMPUTE/`` -> ``CodingCenter/``.
DEFAULT_XHAR_ROOT = SCRIPT_DIR.parents[3] / "XHAR-SDCN"


def load_feature_table(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load either ``[features, label]`` or a feature-only matrix."""
    table = np.loadtxt(path, dtype=np.float32)
    if table.ndim == 1:
        table = table[None, :]
    if table.ndim != 2:
        raise ValueError(f"{path} must contain a two-dimensional numeric table")
    if table.shape[1] == 562:
        return table[:, :-1], table[:, -1]
    if table.shape[1] == 561:
        return table, None
    raise ValueError(f"{path} must have 561 features or 561 features plus a label")


def load_labels(path: Path | None, n_rows: int, embedded: np.ndarray | None) -> np.ndarray:
    if embedded is not None:
        labels = embedded
    elif path is not None:
        labels = np.loadtxt(path, dtype=np.float32).reshape(-1)
    else:
        raise ValueError("a label file is required when the complete input has 561 columns")
    if labels.shape[0] != n_rows:
        raise ValueError(f"label count {labels.shape[0]} does not match row count {n_rows}")
    return labels


def create_mask(
    shape: tuple[int, int],
    mode: str,
    rate: float,
    seed: int,
    row_fraction: float,
    cells_per_row: int | None,
) -> np.ndarray:
    """Create a reproducible MCAR mask; True means the cell is missing."""
    n_rows, n_features = shape
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0 and 1")
    rng = np.random.default_rng(seed)

    if mode == "cell":
        return rng.random(shape) < rate

    if not 0.0 < row_fraction <= 1.0:
        raise ValueError("row_fraction must be in (0, 1]")
    selected_count = max(1, int(round(n_rows * row_fraction)))
    if cells_per_row is None:
        cells_per_row = int(round(rate * n_features / row_fraction))
    if not 0 < cells_per_row <= n_features:
        raise ValueError(
            f"cells_per_row must be in [1, {n_features}], got {cells_per_row}"
        )

    selected_rows = rng.choice(n_rows, size=selected_count, replace=False)
    mask = np.zeros(shape, dtype=bool)
    for row in selected_rows:
        mask[row, rng.choice(n_features, size=cells_per_row, replace=False)] = True
    return mask


def load_or_create_mask(
    complete: np.ndarray,
    mask_path: Path | None,
    mode: str,
    rate: float,
    seed: int,
    row_fraction: float,
    cells_per_row: int | None,
) -> np.ndarray:
    if mask_path is not None:
        mask = np.loadtxt(mask_path, dtype=np.int8).astype(bool)
        if mask.shape != complete.shape:
            raise ValueError(f"mask shape {mask.shape} does not match {complete.shape}")
        return mask
    return create_mask(
        complete.shape, mode, rate, seed, row_fraction, cells_per_row
    )


def linear_interpolate(features: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Interpolate each feature along the original sample/window order."""
    output = features.copy().astype(np.float32)
    output[missing] = np.nan
    positions = np.arange(features.shape[0])
    for column in range(features.shape[1]):
        observed = np.isfinite(output[:, column])
        if not observed.any():
            output[:, column] = 0.0
        elif not observed.all():
            # np.interp uses the nearest observed endpoint for edge gaps.
            output[:, column] = np.interp(
                positions, positions[observed], output[observed, column]
            )
    return output


def remap_graph(graph_path: Path, keep_rows: np.ndarray) -> np.ndarray:
    """Filter and renumber an XHAR edge list after dropping sample rows."""
    edges = np.loadtxt(graph_path, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    mapping = np.full(keep_rows.shape[0], -1, dtype=np.int64)
    mapping[keep_rows] = np.arange(int(keep_rows.sum()), dtype=np.int64)
    valid = keep_rows[edges[:, 0]] & keep_rows[edges[:, 1]]
    return mapping[edges[valid]]


def save_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.8g")


def write_yaml(
    path: Path,
    x_path: str,
    y_path: str,
    graph_path: str,
    statistics_path: str,
    epochs: int,
    batch_size: int,
    model: str,
) -> None:
    content = f"""# Generated by prepare_xhar_comparison.py
name: hhar
X_path: {x_path!r}
y_path: {y_path!r}
graph_path: {graph_path!r}
pretrain_path: data/hhar.pkl

epochs: {epochs}
lr: 1e-3
window_size: 5
stride: 2
dropout_rate: 0.1
sw_threshold: 0.5
efeat_num: 500
efeat_mode: mean
model: {model}
contrastive_learning: true
device: auto
batch_size: {batch_size}
num_negatives: 2
n_input: 561
n_clusters: 6
k: 5
n_z: 10
sigma: 0.5
heads: 1
edge_mode: 0
rope_warm_epochs: 10
emb_dropout_rate: 0.0
update_interval: 1
statistics_path: {statistics_path!r}
use_wandb: false
wandb_project: ''
debug: false
proceed: true
track_cl: false
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def write_runner(path: Path, configs: Iterable[Path], xhar_root: Path) -> None:
    config_args = "\n".join(
        f'python3 src/main.py -C "{config.relative_to(xhar_root)}"'
        for config in configs
    )
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'cd "{xhar_root}"\n\n'
        "# All conditions use the same XHAR-SDCN model settings.\n"
        f"{config_args}\n"
    )
    path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete", type=Path, default=M3_HAR_DATA)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--mask-file", type=Path, default=None)
    parser.add_argument("--m3-features", type=Path, default=None)
    parser.add_argument("--xhar-root", type=Path, default=DEFAULT_XHAR_ROOT)
    parser.add_argument("--output-tag", default="har_missing")
    parser.add_argument("--mask-mode", choices=["cell", "row"], default="row")
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--row-fraction", type=float, default=0.2)
    parser.add_argument("--cells-per-row", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--model", choices=["GAT", "E-Sage", "E-Sage_RoPE"], default="GAT")
    args = parser.parse_args()

    complete, embedded_labels = load_feature_table(args.complete)
    labels = load_labels(args.labels, complete.shape[0], embedded_labels)
    mask = load_or_create_mask(
        complete, args.mask_file, args.mask_mode, args.rate, args.seed,
        args.row_fraction, args.cells_per_row,
    )

    output_dir = args.xhar_root / "data" / args.output_tag
    config_dir = args.xhar_root / "experiments" / args.output_tag
    graph_path = args.xhar_root / "graph" / "hhar5_graph.txt"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    # Keep the missing input in M3's [features, label] layout for inference.
    missing_table = np.column_stack((np.where(mask, np.nan, complete), labels))
    save_matrix(output_dir / "hhar_sprime_with_missing.txt", missing_table)
    save_matrix(output_dir / "hhar_missing_mask.txt", mask.astype(np.int8))

    conditions: dict[str, tuple[np.ndarray, Path, np.ndarray]] = {}
    conditions["s0"] = (complete, graph_path, labels)
    conditions["lerp"] = (linear_interpolate(complete, mask), graph_path, labels)

    keep_rows = ~mask.any(axis=1)
    if not keep_rows.any():
        raise ValueError(
            "S_drop would contain zero rows. Lower --rate or use --mask-mode row "
            "with --row-fraction before preparing the four-way comparison."
        )
    dropped_graph = args.xhar_root / "graph" / "hhar_sdrop5_graph.txt"
    np.savetxt(dropped_graph, remap_graph(graph_path, keep_rows), fmt="%d")
    conditions["drop"] = (complete[keep_rows], dropped_graph, labels[keep_rows])

    if args.m3_features is not None:
        m3_features, _ = load_feature_table(args.m3_features)
        if m3_features.shape != complete.shape:
            raise ValueError(
                f"M3 feature shape {m3_features.shape} does not match {complete.shape}"
            )
        if not np.isfinite(m3_features).all():
            raise ValueError("--m3-features contains NaN or Inf")
        conditions["m3"] = (m3_features, graph_path, labels)

    config_paths: list[Path] = []
    manifest: dict[str, object] = {
        "complete_shape": list(complete.shape),
        "mask_mode": args.mask_mode,
        "mask_seed": args.seed,
        "missing_cells": int(mask.sum()),
        "missing_rate": float(mask.mean()),
        "rows_with_missing": int(mask.any(axis=1).sum()),
        "s_drop_rows": int(keep_rows.sum()),
        "conditions": {},
        "m3_features": str(args.m3_features) if args.m3_features else None,
    }
    for condition, (features, condition_graph, condition_labels) in conditions.items():
        x_path = output_dir / f"hhar_{condition}.txt"
        y_path = output_dir / f"hhar_{condition}_label.txt"
        save_matrix(x_path, features)
        save_matrix(y_path, condition_labels)
        config_path = config_dir / f"hhar_{condition}.yaml"
        graph_rel = condition_graph.relative_to(args.xhar_root).as_posix()
        x_rel = x_path.relative_to(args.xhar_root).as_posix()
        y_rel = y_path.relative_to(args.xhar_root).as_posix()
        write_yaml(
            config_path, x_rel, y_rel, graph_rel, f"{args.output_tag}_{condition}",
            args.epochs, args.batch_size, args.model,
        )
        config_paths.append(config_path)
        manifest["conditions"][condition] = {
            "rows": int(features.shape[0]),
            "features": int(features.shape[1]),
            "x_path": x_rel,
            "y_path": y_rel,
            "graph_path": graph_rel,
            "config": str(config_path.relative_to(args.xhar_root)),
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    write_runner(config_dir / "run_all.sh", config_paths, args.xhar_root)

    print(f"[MASK] mode={args.mask_mode}, missing={mask.mean():.4%}, seed={args.seed}")
    print(f"[DROP] retained rows: {int(keep_rows.sum())}/{len(keep_rows)}")
    print(f"[SAVE] data/configs: {output_dir} / {config_dir}")
    print(f"[SAVE] manifest: {manifest_path}")
    if args.m3_features is None:
        print("[NOTE] S_M3 config was not written; pass --m3-features after M3 inference.")


if __name__ == "__main__":
    main()
