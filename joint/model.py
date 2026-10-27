"""Ordinary module composition; optimizer/phase ownership stays in the runner."""

import torch
from torch import nn

from models.gnn_model import GNNStack
from models.prediction_model import ImputeNet
from xhar_juicer import EdgeFeatures, JuicerESageRoPE, JuicerOutput


class JointModel(nn.Module):
    def __init__(self, rows, features, config, device):
        super().__init__()
        hidden = config['m3_hidden']
        layers = config['m3_layers']
        self.encoder = GNNStack(
            node_input_dim=features, edge_input_dim=1, node_dim=hidden, edge_dim=config['m3_edge_dim'],
            edge_mode=1, model_types=['EGSAGE'] * layers, dropout=0., activation='relu',
            concat_states=False, node_post_mlp_hiddens=[hidden], normalize_embs=[True] * layers,
            aggr='mean', EPSILON=1e-4, device=device, number_of_obs_nodes=rows,
        ).to(device)
        self.imputer = ImputeNet(
            hidden_dim=hidden, num_of_record=rows, num_of_feature=features, device=device,
            num_sample_peer=config['peers'], apply_relation=config['fcu'], apply_peer=config['sru'],
            sample_strategy='random_sample', drop_p=config['m3_dropout'],
        ).to(device)
        widths = config['xhar_widths']
        self.juicer = JuicerESageRoPE(
            n_input=features, n_z=config['latent'], n_clusters=config['clusters'],
            n_enc_1=widths[0], n_enc_2=widths[1], n_enc_3=widths[2],
            n_dec_1=widths[2], n_dec_2=widths[1], n_dec_3=widths[0],
            n_einput=config['xhar_edge_dim'] + config['rope_dim'], rope_dim=config['rope_dim'],
            edge_mode=config['xhar_edge_mode'], dropout=0.,
        ).to(device)
        self.edge_features = EdgeFeatures(features, config['xhar_edge_dim'], trainable=True).to(device)

    def encode(self, values, edges, attributes):
        # GNNStack builds nodes from values; its legacy x argument is unused.
        return self.encoder(None, attributes, edges, values)

    def xhar(self, reconstruction, offset, scale, edges, rope):
        # Return from M3's [0,1] scale to the original UCI feature units.
        x = reconstruction * scale + offset
        out = JuicerOutput(*self.juicer(x, edges, self.edge_features(x, edges, rope)))
        return x, out
