"""Prepare the UCI-HAR data for the M3-Impute tabular-data layout.

The source distribution stores train/test feature matrices separately.  This
script concatenates them in their original order, appends the activity label
as the final column, and writes ``data/data.txt``.  A Bernoulli mask controlled
by ``--rate`` is also sampled over the 561 feature cells.  The complete matrix
remains in ``data.txt`` because M3-Impute expects finite values at load time;
the masked view is written to ``data/data_with_missing.txt`` and the selected
positions to ``data/missing_indices.txt``.

Example:

    python preprocess.py --rate 0.1 --seed 0

``rate`` is the probability that an individual sample-feature cell is hidden.
It does not mask the activity-label column or the subject identifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "original" / "UCI HAR Dataset"
OUTPUT_DIR = SCRIPT_DIR / "data"


def load_source() -> tuple[np.ndarray, np.ndarray, int]:
    """Load and concatenate UCI-HAR train/test features and activity labels."""
    train_x = np.loadtxt(SOURCE_DIR / "train" / "X_train.txt")
    test_x = np.loadtxt(SOURCE_DIR / "test" / "X_test.txt")
    train_y = np.loadtxt(SOURCE_DIR / "train" / "y_train.txt").reshape(-1, 1)
    test_y = np.loadtxt(SOURCE_DIR / "test" / "y_test.txt").reshape(-1, 1)

    if train_x.ndim != 2 or test_x.ndim != 2:
        raise ValueError("X_train.txt and X_test.txt must be two-dimensional matrices")
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError("train and test feature counts do not match")
    if train_x.shape[0] != train_y.shape[0] or test_x.shape[0] != test_y.shape[0]:
        raise ValueError("feature rows and label rows do not match")
    if train_x.shape[1] != 561:
        raise ValueError(f"expected 561 UCI-HAR features, got {train_x.shape[1]}")

    features = np.concatenate((train_x, test_x), axis=0)
    labels = np.concatenate((train_y, test_y), axis=0)
    return features, labels, train_x.shape[0]


def write_indices(n_features: int, output_dir: Path) -> None:
    """Write the feature/target column indices used by the GRAPE layout."""
    np.savetxt(output_dir / "index_features.txt", np.arange(n_features, dtype=int), fmt="%d")
    np.savetxt(output_dir / "index_target.txt", np.array([n_features], dtype=int), fmt="%d")


def create_missing_view(
    features: np.ndarray, rate: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hide feature cells independently and return ``(masked, mask)``.

    ``mask`` is True at hidden positions.  Labels are not passed into this
    function, so the target column can never be accidentally masked.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0 and 1")

    rng = np.random.default_rng(seed)
    missing_mask = rng.random(features.shape) < rate
    masked_features = features.copy()
    masked_features[missing_mask] = np.nan
    return masked_features, missing_mask


def write_dataset(rate: float, seed: int) -> None:
    """Create complete, masked, and metadata files under ``har/data``."""
    features, labels, train_size = load_source()
    masked_features, missing_mask = create_missing_view(features, rate, seed)
    complete = np.concatenate((features, labels), axis=1)
    masked = np.concatenate((masked_features, labels), axis=1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Keep data.txt complete: uci.uci_data.load_data() uses MinMaxScaler and
    # M3-Impute creates its train/test edge mask from this finite matrix.
    np.savetxt(OUTPUT_DIR / "data.txt", complete, fmt="%.10g")
    np.savetxt(OUTPUT_DIR / "data_with_missing.txt", masked, fmt="%.10g")

    # A compact row/column list is easier to inspect than a 10299 x 561 mask.
    missing_indices = np.argwhere(missing_mask)
    np.savetxt(
        OUTPUT_DIR / "missing_indices.txt",
        missing_indices,
        fmt="%d",
        header="row feature",
        comments="",
    )
    np.savetxt(
        OUTPUT_DIR / "missing_mask.txt",
        missing_mask.astype(np.int8),
        fmt="%d",
    )

    write_indices(features.shape[1], OUTPUT_DIR)
    print(f"wrote {complete.shape[0]} rows and {features.shape[1]} features")
    print(f"train rows: {train_size}, test rows: {complete.shape[0] - train_size}")
    print(f"missing cells: {missing_indices.shape[0]} ({missing_mask.mean():.4%})")
    print(f"output: {OUTPUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rate",
        type=float,
        default=0.1,
        help="probability of hiding each feature cell (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed for masking")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_dataset(rate=args.rate, seed=args.seed)
