#!/usr/bin/env python3
"""Train full UCI-HAR with random cell masking, M3 reconstruction and Juicer."""

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits
from joint.runner import DEFAULTS, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON overrides of the compact CPU defaults')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'uci/raw_data/har/original/UCI HAR Dataset')
    parser.add_argument('--output', type=Path, required=True, help='new/empty output directory')
    parser.add_argument('--device', help='cpu or cuda:N')
    parser.add_argument('--epochs', type=int, help='total epoch count, including resumed epochs')
    parser.add_argument('--resume', type=Path, help='trusted checkpoint.pt from this workflow')
    parser.add_argument('--evaluate', action='store_true', help='reload checkpoint and evaluate only')
    args = parser.parse_args()
    config = dict(DEFAULTS)
    if args.config:
        config.update(json.loads(args.config.read_text()))
    if args.resume and not args.config:
        import torch
        config = torch.load(args.resume, map_location='cpu', weights_only=False)['config'].copy()
    if args.device:
        config['device'] = args.device
    if args.epochs is not None:
        config['epochs'] = args.epochs
    with threadpool_limits(limits=config['threads']):
        run(config, args.data_dir, args.output, args.resume, args.evaluate)


if __name__ == '__main__':
    main()
