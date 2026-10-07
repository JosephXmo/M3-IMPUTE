import argparse

def add_uci_subparser(subparsers):
    """Register the UCI/tabular dataset arguments used by ``train_mdi.py``."""
    subparser = subparsers.add_parser('uci')
    # Dataset and edge-level split settings.  ``train_edge`` controls the
    # fraction of sample-feature cells available as the initial graph input.
    subparser.add_argument('--domain', type=str, default='uci')
    subparser.add_argument('--data', type=str, default='housing')
    subparser.add_argument('--train_edge', type=float, default=0.7)
    subparser.add_argument('--split_sample', type=float, default=0.)
    subparser.add_argument('--split_by', type=str, default='y') # 'y' or 'random'
    subparser.add_argument('--split_train', action='store_true', default=False)
    subparser.add_argument('--split_test', action='store_true', default=False)
    subparser.add_argument('--train_y', type=float, default=0.7)
    subparser.add_argument('--node_mode', type=int, default=0)  # 0: feature onehot/sample all 1; 1: both one-hot
