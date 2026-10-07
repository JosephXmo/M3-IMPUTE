"""Command-line entry point for M^3-Impute training.

The actual optimization loop lives in ``training.gnn_mdi``.  This file keeps
the command-line concerns in one place: argument parsing, reproducibility,
device selection, dataset loading, and output-directory preparation.
"""

import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch


from training.gnn_mdi import train_gnn_mdi
from uci.uci_subparser import add_uci_subparser
from mhealth.mhealth_data import add_mhealth_subparser

def main():
    parser = argparse.ArgumentParser()

    # ----------------------------- Model -----------------------------
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--post_hiddens', type=str, default=None)  # default: one node_dim hidden layer
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None)  # default: normalize every GNN layer
    parser.add_argument('--aggr', type=str, default='mean')
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--edge_mode', type=int, default=1)  # 0: edge attention; 1: edge as MLP input
    parser.add_argument('--gnn_activation', type=str, default='relu')
    # ``--impute_hidden`` was used by the bundled shell scripts.  Keep it as
    # a compatibility alias while retaining the original plural destination.
    parser.add_argument('--impute_hiddens', '--impute_hidden', dest='impute_hiddens',
                        type=str, default='64')
    parser.add_argument('--impute_activation', type=str, default='relu')

    # --------------------------- Optimization -------------------------
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--known', type=float, default=0.7)  # probability that a training cell remains visible
    parser.add_argument('--masking_distribution', type=str, default='uniform')
    parser.add_argument('--auto_known', action='store_true', default=False)
    parser.add_argument('--loss_mode', type=int, default=0)  # 0: all train cells; 1: hidden train cells only
    parser.add_argument('--valid', type=float, default=0.)  # validation ratio among training cells
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='0')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--save_prediction', action='store_true', default=False)
    parser.add_argument('--transfer_dir', type=str, default=None)
    parser.add_argument('--transfer_extra', type=str, default='')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--repeat_exp_num', type=int, default=1)
    parser.add_argument('--display_log', action='store_true', default=False)
    
    # ----------------------- FCU / SCU controls -----------------------
    parser.add_argument('--apply_attr', action='store_true', default=False)
    parser.add_argument('--apply_peer', action='store_true', default=False)
    parser.add_argument('--init_epsilon', type=float, default=1e-4)
    parser.add_argument('--sample_peer_size', type=int, default=5)
    parser.add_argument('--sample_strategy', type=str, default='random_sample')  # random_sample or cos-similarity
    parser.add_argument('--update_cos_sample_prob_every', type=int, default=100)
    parser.add_argument('--impute_nn_dropout', type=float, default=0.1)
    # For large datasets, replace the N x N cosine table with an N x K table.
    # K is controlled by --sample_space_size and should be much smaller than N.
    parser.add_argument('--very_large_dataset', action='store_true', default=False)
    parser.add_argument('--sample_space_size', type=int, default=20)
    
    # ----------------------- Missingness controls ---------------------
    parser.add_argument('--corrupt', type=str, default="mcar")
    parser.add_argument('--mar_rate_obs', type=float, default=0.1)
    parser.add_argument('--mar_rate_missing', type=float, default=0.15)
    parser.add_argument('--mnar_known_mask', type=float, default=0.1)
    
    subparsers = parser.add_subparsers()
    add_uci_subparser(subparsers)
    add_mhealth_subparser(subparsers)
    args = parser.parse_args()
    print(args)
    print('--'*20)
    print(f'[Config] Missing Pattern: {args.corrupt}')

    # --------------------------- Device setup -------------------------
    # Set --gpu explicitly when running on a multi-GPU machine.  The rest of
    # the project receives a normal torch.device object from this block.
    if torch.cuda.is_available():
        # cuda = auto_select_gpu()                      # auto select most suitable GPU
        cuda = args.gpu                                 # manual selection of gpu
        print(f'[Config] Using GPU {args.gpu}')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        print('[Config] Using CPU')
        device = torch.device('cpu')

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --------------------------- Dataset setup ------------------------
    if args.domain == 'uci':
        from uci.uci_data import load_data
        data = load_data(args)
    elif args.domain == 'mhealth':
        from mhealth.mhealth_data import load_data
        data = load_data(args)
    else:
        raise Exception('Unsupported datasets.')


    # Every repeated run writes into the same experiment directory and keeps
    # the exact command line in cmd_input.txt for later reproduction.
    log_path = './{}/test/{}/{}/'.format(args.domain, args.data, args.log_dir)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(osp.join(log_path, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)

    for run_iter_num in range(args.repeat_exp_num):
        args.seed = run_iter_num
        print(f'[Config] seed: {args.seed}')
        train_gnn_mdi(data, args, log_path, run_iter_num, device, print_train_log=args.display_log)


if __name__ == '__main__':
    main()
