"""Feature-only mHealth input for M3-Impute training.

Activity labels are produced by preprocessing for downstream evaluation. This
module reads only the feature matrix and builds cell-level reconstruction
targets; it never loads activity labels into the M3 graph.
"""

from pathlib import Path

import numpy as np


DATA_ROOT = Path(__file__).resolve().parent.parent / "uci" / "raw_data"
DATASETS = ("mhealth", "mhealth_acc_gyro_only")


def add_mhealth_subparser(subparsers):
    parser = subparsers.add_parser("mhealth")
    parser.set_defaults(domain="mhealth")
    parser.add_argument("--data", choices=DATASETS, default="mhealth")
    parser.add_argument("--train_edge", type=float, default=0.7)


def read_features(dataset: str) -> np.ndarray:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported mHealth dataset: {dataset}")
    data_dir = DATA_ROOT / dataset / "data"
    path = data_dir / "data.txt"
    features = np.loadtxt(path, ndmin=2)
    if features.ndim != 2 or min(features.shape) < 1 or not np.isfinite(features).all():
        raise ValueError(f"{path} must be a finite feature matrix")
    names = (data_dir / "feature_names.txt").read_text().splitlines()
    if len(names) != features.shape[1]:
        raise ValueError(f"{path} has {features.shape[1]} columns but {len(names)} named features")
    return features


def bipartite_arrays(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return directed sample-feature edges and matching cell values."""
    n_samples, n_features = values.shape
    samples = np.repeat(np.arange(n_samples, dtype=np.int64), n_features)
    features = n_samples + np.tile(np.arange(n_features, dtype=np.int64), n_samples)
    edge_index = np.stack((np.concatenate((samples, features)),
                           np.concatenate((features, samples))))
    edge_values = np.tile(values.reshape(-1), 2).reshape(-1, 1)
    return edge_index, edge_values


def load_data(args):
    """Build the graph used by the existing M3 reconstruction training loop."""
    import pandas as pd
    import torch
    from sklearn.preprocessing import MinMaxScaler
    from torch_geometric.data import Data

    from utils.utils import mask_edge, train_test_mask

    if not 0 < args.train_edge < 1:
        raise ValueError("--train_edge must be between 0 and 1")
    raw = read_features(args.data)
    scaled = MinMaxScaler().fit_transform(raw).astype(np.float32)
    n_samples, n_features = scaled.shape
    edge_array, value_array = bipartite_arrays(scaled)
    edge_index = torch.from_numpy(edge_array)
    edge_attr = torch.from_numpy(value_array)
    node_array = np.vstack((np.ones((n_samples, n_features), dtype=np.float32),
                            np.eye(n_features, dtype=np.float32)))
    x = torch.from_numpy(node_array)
    df_X = pd.DataFrame(scaled)

    torch.manual_seed(args.seed)
    train_edge_mask = train_test_mask(args.train_edge, n_samples * n_features,
                                      mode=args.corrupt, mask_dist=args.masking_distribution,
                                      X=df_X, args=args)
    double_mask = torch.cat((train_edge_mask, train_edge_mask))
    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr, double_mask, True)
    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr, ~double_mask, True)
    print(f"[Config] Known Ratio (i.e. Training Edge Ratio): {args.train_edge}, "
          f"Missing Ratio (i.e. Testing Impute Edge Ratio): {1 - args.train_edge}")
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        train_edge_index=train_edge_index, train_edge_attr=train_edge_attr,
        train_edge_mask=train_edge_mask, train_labels=train_edge_attr[:len(train_edge_attr) // 2, 0],
        test_edge_index=test_edge_index, test_edge_attr=test_edge_attr,
        test_edge_mask=~train_edge_mask, test_labels=test_edge_attr[:len(test_edge_attr) // 2, 0],
        df_X=df_X, edge_attr_dim=1, user_num=n_samples,
    )
