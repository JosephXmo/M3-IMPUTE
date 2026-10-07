"""Create row split files for the prepared UCI-HAR data.

By default the original UCI-HAR subject split is preserved: the first 7352
rows are the official training rows and the remaining 2947 rows are the
official test rows.  Use ``--random-splits`` to reproduce the 20 random 90/10
splits used by most legacy datasets in this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent


def write_column_indices(data: np.ndarray) -> None:
    """Write feature columns and the final activity-label column."""
    np.savetxt(DATA_DIR / "index_features.txt", np.arange(data.shape[1] - 1), fmt="%d")
    np.savetxt(DATA_DIR / "index_target.txt", np.array([data.shape[1] - 1]), fmt="%d")


def write_official_split(n_rows: int, train_size: int) -> None:
    if not 0 < train_size < n_rows:
        raise ValueError("official train size must be between 0 and the row count")
    np.savetxt(DATA_DIR / "index_train_0.txt", np.arange(train_size), fmt="%d")
    np.savetxt(DATA_DIR / "index_test_0.txt", np.arange(train_size, n_rows), fmt="%d")
    np.savetxt(DATA_DIR / "n_splits.txt", np.array([1]), fmt="%d")


def write_random_splits(n_rows: int, n_splits: int, test_ratio: float, seed: int) -> None:
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    test_size = max(1, int(round(n_rows * test_ratio)))
    for split in range(n_splits):
        permutation = rng.permutation(n_rows)
        np.savetxt(DATA_DIR / f"index_train_{split}.txt", permutation[:-test_size], fmt="%d")
        np.savetxt(DATA_DIR / f"index_test_{split}.txt", permutation[-test_size:], fmt="%d")
    np.savetxt(DATA_DIR / "n_splits.txt", np.array([n_splits]), fmt="%d")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-splits", action="store_true")
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--official-train-size", type=int, default=7352)
    args = parser.parse_args()

    data = np.loadtxt(DATA_DIR / "data.txt")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("data.txt must be a matrix with features and one target column")

    write_column_indices(data)
    if args.random_splits:
        write_random_splits(data.shape[0], args.n_splits, args.test_ratio, args.seed)
        print(f"wrote {args.n_splits} random splits")
    else:
        write_official_split(data.shape[0], args.official_train_size)
        print("wrote the official UCI-HAR train/test split")


if __name__ == "__main__":
    main()
