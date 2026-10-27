"""Random-row HAR imputation benchmark using official 561-dimensional features.

Stages: prepare S_train and S0; train standard M3 on S_train; impute the
explicitly masked S_drop with that checkpoint. Randomized rows have no time
adjacency, so this protocol compares only S0, S_drop and S_m3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


HAR_DIR = Path(__file__).resolve().parent
M3_ROOT = HAR_DIR.parents[2]
XHAR_ROOT = M3_ROOT.parent / "XHAR-SDCN"
DEFAULT_OUTPUT = XHAR_ROOT / "data" / "missing-cross-comparison"
TRAIN_NAME = "har_cross_train"
TRAIN_DATA = M3_ROOT / "uci" / "raw_data" / TRAIN_NAME / "data"
CONFIG_DIR = Path("experiments/har_missing_10rows_102cols")


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def save_txt(path: Path, array: np.ndarray, integer: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, array, fmt="%d" if integer else "%.10g")


def load_source(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, subjects = [], [], []
    for split in ("train", "test"):
        x = np.loadtxt(source / split / f"X_{split}.txt", ndmin=2)
        y = np.loadtxt(source / split / f"y_{split}.txt", dtype=int, ndmin=1)
        subject = np.loadtxt(source / split / f"subject_{split}.txt", dtype=int, ndmin=1)
        if x.shape != (len(y), 561) or len(subject) != len(y):
            raise ValueError(f"invalid HAR feature/label/subject shapes in {split}")
        if not np.isfinite(x).all():
            raise ValueError("HAR source must be finite before masking")
        xs.append(x)
        ys.append(y)
        subjects.append(subject)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(subjects)


def random_row_split(
    subjects: np.ndarray, ratio: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample rows without replacement, retaining the random order on both sides."""
    if not 0 < ratio < 1:
        raise ValueError("train ratio must be between 0 and 1")
    boundary = int(len(subjects) * ratio)
    if not 0 < boundary < len(subjects):
        raise ValueError("row split leaves one side empty")
    order = np.random.default_rng(seed).permutation(len(subjects))
    train_ids, s0_ids = order[:boundary], order[boundary:]
    train_subjects, s0_subjects = np.unique(subjects[train_ids]), np.unique(subjects[s0_ids])
    details = {
        "rule": "random_rows_without_replacement",
        "seed": seed,
        "requested_train_ratio": ratio,
        "train_subjects": train_subjects.tolist(),
        "s0_subjects": s0_subjects.tolist(),
        "shared_subjects": np.intersect1d(train_subjects, s0_subjects).tolist(),
        "train_rows": len(train_ids),
        "s0_rows": len(s0_ids),
        "actual_train_ratio": len(train_ids) / len(subjects),
    }
    return train_ids, s0_ids, details


def training_statistics(features: np.ndarray) -> dict:
    """Match default sklearn MinMaxScaler used by the standard UCI loader."""
    minimum, maximum = features.min(0), features.max(0)
    span = maximum - minimum
    # sklearn handles nearly constant float64 columns with a unit divisor.
    span = np.where(span < 10 * np.finfo(features.dtype).eps, 1.0, span)
    std = features.std(0)
    std = np.where(std < 1e-12, 1.0, std)
    return {
        "mean": features.mean(0).tolist(), "std": std.tolist(),
        "minimum": minimum.tolist(), "maximum": maximum.tolist(),
        "minmax_scale": (1.0 / span).tolist(),
        "minmax_offset": (-minimum / span).tolist(),
    }


def validate_statistics(stats: dict, n_features: int) -> None:
    for key in ("mean", "std", "minimum", "maximum", "minmax_scale", "minmax_offset"):
        values = np.asarray(stats[key])
        if values.shape != (n_features,) or not np.isfinite(values).all():
            raise ValueError(f"invalid training statistics: {key}")
    if np.any(np.asarray(stats["std"]) <= 0) or np.any(np.asarray(stats["minmax_scale"]) <= 0):
        raise ValueError("training standard deviations and scales must be positive")


def verify_artifacts(output: Path, manifest: dict) -> None:
    for filename, key in (("missing_mask.txt", "mask_sha256"),
                          ("hhar_drop.txt", "drop_sha256"),
                          ("train_statistics.json", "statistics_sha256"),
                          ("train_row_ids.txt", "train_row_ids_sha256"),
                          ("s0_row_ids.txt", "s0_row_ids_sha256"),
                          ("hhar_labels.txt", "labels_sha256")):
        if digest(output / filename) != manifest[key]:
            raise ValueError(f"{filename} no longer matches the prepared experiment")


def config_text(original: str, condition: str) -> str:
    """Patch only data paths in the user's existing XHAR experiment config."""
    base = "data/missing-cross-comparison"
    for key, value in (("X_path", f"{base}/hhar_{condition}.txt"),
                       ("y_path", f"{base}/hhar_labels.txt")):
        original, count = re.subn(
            rf"(?m)^{key}:.*$", f'{key}: "{value}"', original,
        )
        if count != 1:
            raise ValueError(f"expected one active {key} in XHAR config")
    return original


def update_configs(xhar: Path) -> None:
    # Read/validate all configs before writing any of them. Retain user's graph,
    # AE heatup, model, device, seed, repeat runner, and statistics settings.
    changes = []
    for condition in ("s0", "drop", "m3"):
        path = xhar / CONFIG_DIR / f"hhar_{condition}.yaml"
        original = path.read_text()
        changes.append((path, original, config_text(original, condition)))
    for path, original, patched in changes:
        if patched != original:
            path.write_text(patched)


def prepare(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    if not 0 <= args.rate <= 1 or args.seed < 0:
        raise ValueError("missing rate must be in [0, 1] and seed nonnegative")
    if args.update_xhar_configs:
        if output != (args.xhar_root / "data/missing-cross-comparison").resolve():
            raise ValueError("config patch expects <xhar-root>/data/missing-cross-comparison")
        for condition in ("s0", "drop", "m3"):
            config_text((args.xhar_root / CONFIG_DIR / f"hhar_{condition}.yaml").read_text(), condition)
    source_x, labels, subjects = load_source(args.source)
    train_ids, ids, split = random_row_split(subjects, args.train_ratio, args.seed)
    # Export the exact serialized training matrix used by uci_data.load_data.
    train_table = np.column_stack((source_x[train_ids], labels[train_ids]))
    save_txt(TRAIN_DATA / "data.txt", train_table)
    save_txt(TRAIN_DATA / "index_features.txt", np.arange(561), integer=True)
    save_txt(TRAIN_DATA / "index_target.txt", np.array([561]), integer=True)
    serialized_train = np.loadtxt(TRAIN_DATA / "data.txt", ndmin=2)[:, :-1]
    stats = training_statistics(serialized_train)
    s0 = source_x[ids]
    mask = np.random.default_rng(args.seed).random(s0.shape) < args.rate
    dropped = np.where(mask, np.asarray(stats["mean"]), s0)
    output.mkdir(parents=True, exist_ok=True)
    save_txt(output / "hhar_s0.txt", s0)
    save_txt(output / "hhar_drop.txt", dropped)
    save_txt(output / "hhar_labels.txt", labels[ids], integer=True)
    save_txt(output / "missing_mask.txt", mask, integer=True)
    save_txt(output / "missing_indices.txt", np.argwhere(mask), integer=True)
    save_txt(output / "s0_row_ids.txt", ids, integer=True)
    save_txt(output / "train_row_ids.txt", train_ids, integer=True)
    save_txt(output / "s0_subjects.txt", subjects[ids], integer=True)
    write_json(output / "train_statistics.json", stats)
    # NaNs explicitly distinguish hidden cells from valid mean-valued cells.
    save_txt(output / "hhar_missing.txt", np.where(mask, np.nan, dropped))
    # Preserve old interpolation artifacts as clearly retired files.
    for name in ("hhar_lerp.txt", "segments.txt"):
        retired = output / name
        if retired.exists():
            retired.replace(output / f"{retired.stem}.retired-{digest(retired)[:12]}.txt")
    # Invalidate the previous M3 artifact when its source/mask changes. Move it
    # aside rather than leave a stale condition looking usable.
    previous_path = output / "manifest.json"
    current_mask_hash = digest(output / "missing_mask.txt")
    current_train_hash = digest(TRAIN_DATA / "data.txt")
    current_s0_hash = digest(output / "hhar_s0.txt")
    previous = read_json(previous_path) if previous_path.exists() else {}
    binding = {"train_sha256": current_train_hash, "mask_sha256": current_mask_hash,
               "s0_sha256": current_s0_hash, "drop_sha256": digest(output / "hhar_drop.txt"),
               "statistics_sha256": digest(output / "train_statistics.json"),
               "train_row_ids_sha256": digest(output / "train_row_ids.txt"),
               "s0_row_ids_sha256": digest(output / "s0_row_ids.txt"),
               "labels_sha256": digest(output / "hhar_labels.txt")}
    unchanged = all(previous.get(key) == value for key, value in binding.items())
    if not unchanged and (output / "hhar_m3.txt").exists():
        stale = output / f"hhar_m3.stale-{digest(output / 'hhar_m3.txt')[:12]}.txt"
        (output / "hhar_m3.txt").replace(stale)
    manifest = {
        "schema": 2, "seed": args.seed, "source_shape": list(source_x.shape),
        "split": split, "family_shape": list(s0.shape), "requested_missing_rate": args.rate,
        "missing_cells": int(mask.sum()), "actual_missing_rate": float(mask.mean()),
        "train_dataset": TRAIN_NAME, "train_data": str(TRAIN_DATA / "data.txt"),
        **binding,
        "m3": previous.get("m3") if unchanged else None,
        "conditions": {key: f"hhar_{key}.txt" for key in ("s0", "drop", "m3")},
        "labels": "hhar_labels.txt",
    }
    write_json(previous_path, manifest)
    if args.update_xhar_configs:
        update_configs(args.xhar_root)
    print(f"S_train={split['train_rows']}x561; S0-family={len(s0)}x561")
    print(f"Random row split; shared subjects: {split['shared_subjects']}")
    print(f"Missing cells={mask.sum()} ({mask.mean():.4%}); seed={args.seed}")
    print(f"Output: {output}")
    print("S_m3 requires the separate train and impute stages.")


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.node_dim < 1 or args.edge_dim < 1 or args.impute_batch_size < 1:
        raise ValueError("epochs, model dimensions and impute batch size must be positive")
    if not 0 < args.known < 1 or not 0 < args.train_edge < 1:
        raise ValueError("known and train-edge probabilities must be between 0 and 1")
    print("[Startup] Verify prepared S_train and checkpoint destination ...", flush=True)
    manifest = read_json(args.output_dir / "manifest.json")
    if digest(TRAIN_DATA / "data.txt") != manifest["train_sha256"]:
        raise ValueError("S_train no longer matches the prepared experiment")
    log_dir = args.log_dir or f"cross_{args.node_dim}_{args.edge_dim}_seed{manifest['seed']}"
    if Path(log_dir).name != log_dir or log_dir in (".", ".."):
        raise ValueError("log-dir must be a directory name, not a path")
    checkpoint_dir = M3_ROOT / "uci/test" / TRAIN_NAME / log_dir
    if (checkpoint_dir / "model.pt").exists() or (checkpoint_dir / "impute_model.pt").exists():
        raise ValueError("checkpoint directory already contains weights; use a new --log-dir")
    command = [sys.executable, "-u", str(M3_ROOT / "train_mdi.py"),
               "--epochs", str(args.epochs), "--gpu", str(args.gpu),
               "--seed", str(manifest["seed"]), "--node_dim", str(args.node_dim),
               "--edge_dim", str(args.edge_dim), "--impute_hiddens", str(args.node_dim),
               "--impute_batch_size", str(args.impute_batch_size),
               "--known", str(args.known), "--valid", "0", "--save_model",
               "--log_dir", log_dir]
    if args.apply_attr:
        command.append("--apply_attr")
    if args.apply_peer:
        command.extend(["--apply_peer", "--sample_strategy", args.sample_strategy,
                        "--very_large_dataset", "--sample_space_size", "20"])
    command.extend(["uci", "--data", TRAIN_NAME, "--train_edge", str(args.train_edge)])
    print(f"[Startup] Launch M3 training: {manifest['split']['train_rows']:,} rows; "
          f"checkpoint directory: {checkpoint_dir}", flush=True)
    subprocess.run(command, cwd=M3_ROOT, check=True)
    write_json(checkpoint_dir / "cross_comparison.json", {
        "train_sha256": manifest["train_sha256"], "training_rows": manifest["split"]["train_rows"],
        "seed": manifest["seed"], "command": command,
        "model_sha256": digest(checkpoint_dir / "model.pt"),
        "impute_model_sha256": digest(checkpoint_dir / "impute_model.pt"),
    })
    print(f"Checkpoint and provenance: {checkpoint_dir}")


def adapt_checkpoint(model, imputer, n_rows: int, n_features: int, dense, device) -> None:
    """Reset inference-only indices/peer caches, leaving learned weights intact."""
    if model.number_of_feature != n_features or imputer.num_feature != n_features:
        raise ValueError("checkpoint feature count differs from S0")
    for root in (model, imputer):
        for module in root.modules():
            if hasattr(module, "device"):
                module.device = device
    model.number_of_obs_node = n_rows
    imputer.num_record = n_rows
    imputer.SRU.num_record = n_rows
    if imputer.apply_peer and imputer.SRU.sample_strategy == "cos-similarity":
        # Never reuse sample indices or probabilities from S_train. Only visible
        # S0 values enter the new cache; hidden truth and mean placeholders don't.
        imputer.SRU.update_cos_prob(model.feature_nodes, dense)
        probabilities = imputer.SRU.sample_prob
        probabilities.clamp_(min=0)
        empty = probabilities.sum(1) == 0
        probabilities[empty] = 1.0
        if not imputer.SRU.running_on_large_dataset:
            probabilities.fill_diagonal_(0)
        available = (probabilities > 0).sum(1)
        if bool((available < imputer.SRU.sample_peer_size).any()):
            raise ValueError("too few positive-similarity peers; use a random-sampling checkpoint")


def impute(args: argparse.Namespace) -> None:
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    # Torch imports are delayed so all preparation works with NumPy.
    if str(HAR_DIR) not in sys.path:
        sys.path.insert(0, str(HAR_DIR))
    from generate_m3_representation import (
        choose_device, load_module, impute_missing,
    )
    import torch

    output = args.output_dir
    manifest = read_json(output / "manifest.json")
    provenance = read_json(args.checkpoint_dir / "cross_comparison.json")
    if provenance["train_sha256"] != manifest["train_sha256"]:
        raise ValueError("checkpoint was trained on a different S_train")
    for name in ("model", "impute_model"):
        if digest(args.checkpoint_dir / f"{name}.pt") != provenance[f"{name}_sha256"]:
            raise ValueError(f"{name}.pt differs from the bound training checkpoint")
    verify_artifacts(output, manifest)
    device = choose_device(args.device)
    np.random.seed(manifest["seed"])
    torch.manual_seed(manifest["seed"])
    dropped = np.loadtxt(output / "hhar_drop.txt", ndmin=2)
    mask = np.loadtxt(output / "missing_mask.txt", dtype=bool, ndmin=2)
    if list(dropped.shape) != manifest["family_shape"] or mask.shape != dropped.shape:
        raise ValueError("S_drop and mask have incorrect shapes")
    if not np.isfinite(dropped).all():
        raise ValueError("S_drop contains nonfinite values")
    # The mean baseline is just an export: its placeholders must be hidden from
    # M3. Use the saved S_train scaler, never fit a scaler on complete S0.
    stats = read_json(output / "train_statistics.json")
    validate_statistics(stats, dropped.shape[1])
    scale, offset = np.asarray(stats["minmax_scale"]), np.asarray(stats["minmax_offset"])
    masked = dropped.copy()
    masked[mask] = np.nan
    normalized = (masked * scale + offset).astype(np.float32)
    save_txt(output / "hhar_missing.txt", masked)
    result = dropped.copy()
    if mask.any():
        model = load_module(args.checkpoint_dir / "model.pt", device)
        imputer = load_module(args.checkpoint_dir / "impute_model.pt", device)
        if int(imputer.num_record) != provenance["training_rows"]:
            raise ValueError("checkpoint row count differs from training provenance")
        epsilon = float(model.EPSILON)
        # Construct only the dense state needed to reset a cosine peer cache.
        # impute_missing builds the graph once; representation export is skipped.
        with torch.no_grad():
            dense = torch.from_numpy(np.where(mask, epsilon, normalized)).to(device)
            adapt_checkpoint(model, imputer, len(dropped), dropped.shape[1], dense, device)
            del dense
            completed = impute_missing(model, imputer, normalized, mask, device,
                                       epsilon, args.chunk_size)
        unscaled = (completed.astype(np.float64) - offset) / scale
        result[mask] = unscaled[mask]
    if not np.isfinite(result).all():
        raise ValueError("M3 produced nonfinite imputations")
    # S0 is read only after inference, for offline missing-cell error metrics.
    if digest(output / "hhar_s0.txt") != manifest["s0_sha256"]:
        raise ValueError("S0 no longer matches the prepared evaluation reference")
    reference = np.loadtxt(output / "hhar_s0.txt", ndmin=2)
    error = result[mask] - reference[mask]
    save_txt(output / "hhar_m3.txt", result)
    manifest["m3"] = {"checkpoint_dir": str(args.checkpoint_dir.resolve()),
                      "device": str(device), "chunk_size": args.chunk_size,
                      "missing_mae": float(np.abs(error).mean()) if error.size else None,
                      "missing_rmse": float(np.sqrt((error ** 2).mean())) if error.size else None}
    write_json(output / "manifest.json", manifest)
    print(f"S_m3={result.shape}; saved {output / 'hhar_m3.txt'}")
    print(f"Missing-cell metrics: {manifest['m3']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare", help="randomly split rows, simulate missingness, export baselines")
    prep.add_argument("--source", type=Path, default=HAR_DIR / "original/UCI HAR Dataset")
    prep.add_argument("--train-ratio", type=float, default=0.7)
    prep.add_argument("--rate", type=float, default=0.1, help="cell-level MCAR missing probability")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--xhar-root", type=Path, default=XHAR_ROOT)
    prep.add_argument("--update-xhar-configs", action="store_true")
    training = sub.add_parser("train", help="run standard M3 with bounded decoder batches")
    training.add_argument("--epochs", type=int, default=200)
    training.add_argument("--gpu", type=int, default=0,
                          help="forward to train_mdi.py: when MPS is available, "
                               "0=MPS and N>=1=cuda:N-1; otherwise N=cuda:N (default: 0)")
    training.add_argument("--node-dim", type=int, default=32)
    training.add_argument("--edge-dim", type=int, default=16)
    training.add_argument("--impute-batch-size", type=int, default=1024)
    training.add_argument("--known", type=float, default=0.7)
    training.add_argument("--train-edge", type=float, default=0.7)
    training.add_argument("--log-dir", default=None)
    training.add_argument("--apply-attr", action="store_true")
    training.add_argument("--apply-peer", action="store_true")
    training.add_argument("--sample-strategy", choices=("random_sample", "cos-similarity"),
                          default="random_sample")
    prediction = sub.add_parser("impute", help="predict only explicit missing cells on S0")
    prediction.add_argument("--checkpoint-dir", type=Path, required=True)
    prediction.add_argument("--device", default="auto")
    prediction.add_argument("--chunk-size", type=int, default=128)
    for command in (prep, training, prediction):
        command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "train":
        train(args)
    elif args.stage == "impute":
        impute(args)


if __name__ == "__main__":
    main()
