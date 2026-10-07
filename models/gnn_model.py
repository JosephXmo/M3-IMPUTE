"""Graph encoder used by M^3-Impute.

The encoder operates on a bidirectional sample-feature graph.  ``train_attr_value``
is the dense sample-feature matrix used to initialize sample nodes; the input
``x`` argument is kept in the public signature for compatibility with the
original GRAPE-style training code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
from models.egcn import EGCNConv
from models.egsage import EGraphSage
from utils.utils import get_activation

def get_gnn(data, args, device):
    """Build a :class:`GNNStack` from dataset metadata and CLI arguments."""
    model_types = args.model_types.split('_')
    if args.norm_embs is None:
        norm_embs = [True,]*len(model_types)
    else:
        norm_embs = list(map(bool,map(int,args.norm_embs.split('_'))))
    if args.post_hiddens is None:
        post_hiddens = [args.node_dim]
    else:
        post_hiddens = list(map(int,args.post_hiddens.split('_')))
        
    print(f'[Config] GNN model: {model_types}')
    
    # ``data.x`` contains sample and feature nodes.  Its second dimension is
    # the node-feature width, which equals the number of data features in the
    # default node mode used by this repository.
    total_number_of_nodes, number_of_feature_nodes = data.x.shape
    number_of_nodes = total_number_of_nodes - number_of_feature_nodes

    model = GNNStack(data.num_node_features, data.edge_attr_dim,
                        args.node_dim, args.edge_dim, args.edge_mode,
                        model_types, args.dropout, args.gnn_activation,
                        args.concat_states, post_hiddens,
                        norm_embs, args.aggr, EPSILON=args.init_epsilon, device=device, number_of_obs_nodes = number_of_nodes)
    return model

class GNNStack(torch.nn.Module):
    """Stack edge-aware message-passing layers and edge update MLPs.

    Each layer first updates node embeddings and then transforms edge
    attributes so the next layer can use a learned edge representation.
    """
    def __init__(self, 
                node_input_dim, edge_input_dim,
                node_dim, edge_dim, edge_mode,
                model_types, dropout, activation,
                concat_states, node_post_mlp_hiddens,
                normalize_embs, aggr,
                EPSILON=None,
                device=None,
                number_of_obs_nodes=None,
                ):
        super(GNNStack, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.concat_states = concat_states
        self.model_types = model_types
        self.gnn_layer_num = len(model_types)
        
        self.convs = self.build_convs(node_dim, edge_input_dim,
                                    node_dim, edge_dim, edge_mode,
                                    model_types, normalize_embs, activation, aggr)

        # The post MLP keeps the output width at ``node_dim`` unless the
        # caller explicitly requests another hidden layout.
        if concat_states:
            self.node_post_mlp = self.build_node_post_mlp(int(node_dim*len(model_types)), int(node_dim*len(model_types)), node_post_mlp_hiddens, dropout, activation)
        else:
            self.node_post_mlp = self.build_node_post_mlp(node_dim, node_dim, node_post_mlp_hiddens, dropout, activation)

        self.edge_update_mlps = self.build_edge_update_mlps(node_dim, edge_input_dim, edge_dim, self.gnn_layer_num, activation)
        

        # ----------------------- M^3-Impute state -----------------------
        self.device = device if device is not None else torch.device('cpu')
        self.EPSILON = EPSILON
        self.number_of_obs_node = number_of_obs_nodes
        self.number_of_feature = node_input_dim
        self.hidden_dim = node_dim

        # Feature-node embeddings are initialized once and reused to derive
        # the initial sample-node embeddings from the observed-value matrix.
        self.__init_node(node_input_dim, self.hidden_dim)

        self.init_obs_mlp = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.init_feature_mlp = nn.Linear(self.hidden_dim, self.hidden_dim)

    def __init_node(self, feature_input_dim, hidden_dim):
        """Initialize one trainable-width vector per input feature."""
        self.feature_nodes = torch.rand((feature_input_dim, hidden_dim), requires_grad=True, dtype=torch.float32, device=self.device)

    def node_init(self, attn_score):
        """Create ``[sample nodes; feature nodes]`` before message passing."""
        obs_node_embs = attn_score @ self.feature_nodes        # (n_of_sample, n_of_feature) x (n_of_feature, d) => (number of sample, d)
        
        # feed into mlp
        obs_node_embs     = F.sigmoid(self.init_obs_mlp(obs_node_embs))
        feature_node_embs = self.init_feature_mlp(self.feature_nodes)

        return torch.cat((obs_node_embs, feature_node_embs))

    def build_node_post_mlp(self, input_dim, output_dim, hidden_dims, dropout, activation):
        if 0 in hidden_dims:
            return get_activation('none')
        else:
            layers = []
            for hidden_dim in hidden_dims:
                layer = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            get_activation(activation),
                            nn.Dropout(dropout),
                            )
                layers.append(layer)
                input_dim = hidden_dim
            layer = nn.Linear(input_dim, output_dim)
            layers.append(layer)
            return nn.Sequential(*layers)

    def build_convs(self, node_input_dim, edge_input_dim,
                     node_dim, edge_dim, edge_mode,
                     model_types, normalize_embs, activation, aggr):
        """Instantiate the requested GNN layers with compatible edge widths."""
        convs = nn.ModuleList()
        conv = self.build_conv_model(model_types[0],node_input_dim,node_dim,
                                    edge_input_dim, edge_mode, normalize_embs[0], activation, aggr)
        convs.append(conv)
        for l in range(1,len(model_types)):
            conv = self.build_conv_model(model_types[l],node_dim, node_dim,
                                    edge_dim, edge_mode, normalize_embs[l], activation, aggr)
            convs.append(conv)
        return convs

    def build_conv_model(self, model_type, node_in_dim, node_out_dim, edge_dim, edge_mode, normalize_emb, activation, aggr):
        """Map the short CLI name to a PyG or repository-local layer."""
        if model_type == 'GCN':
            return pyg_nn.GCNConv(node_in_dim,node_out_dim)
        elif model_type == 'GraphSage':
            return pyg_nn.SAGEConv(node_in_dim,node_out_dim)
        elif model_type == 'GAT':
            return pyg_nn.GATConv(node_in_dim,node_out_dim)
        elif model_type == 'EGCN':
            return EGCNConv(node_in_dim,node_out_dim,edge_dim,edge_mode)
        elif model_type == 'EGSAGE':
            return EGraphSage(node_in_dim,node_out_dim,edge_dim,activation,edge_mode,normalize_emb, aggr)

    def build_edge_update_mlps(self, node_dim, edge_input_dim, edge_dim, gnn_layer_num, activation):
        """Build one edge-attribute projection for every GNN layer."""
        edge_update_mlps = nn.ModuleList()
        edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+node_dim+edge_input_dim,edge_dim),
                get_activation(activation),
                )
        edge_update_mlps.append(edge_update_mlp)
        for l in range(1,gnn_layer_num):
            edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+node_dim+edge_dim,edge_dim),
                get_activation(activation),
                )
            edge_update_mlps.append(edge_update_mlp)
        return edge_update_mlps

    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        """Update each edge from its two endpoint embeddings and old attribute."""
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        edge_attr = mlp(torch.cat((x_i,x_j,edge_attr),dim=-1))
        return edge_attr

    def forward(self, x, edge_attr, edge_index, train_attr_value):
        """Encode a masked matrix and the currently visible graph edges."""
        # ``train_attr_value`` contains observed values and epsilon markers at
        # hidden positions.  It is the mask-aware initialization signal.
        x = self.node_init(train_attr_value) # (Number of Node, Hidden Dim)

        concat_x = []
        for l,(conv_name,conv) in enumerate(zip(self.model_types,self.convs)):
            # Standard PyG layers ignore edge_attr; EGCN/EGSAGE consume it.
            if conv_name == 'EGCN' or conv_name == 'EGSAGE':
                x = conv(x, edge_attr, edge_index)
            else:
                x = conv(x, edge_index)
            
            concat_x.append(x)
            edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])

        # Return the final node representation.  The first N rows are sample
        # nodes and the remaining rows are feature nodes.
        x = self.node_post_mlp(x)

        return x

    def check_input(self, xs, edge_attr, edge_index):
        Os = {}
        for indx in range(128):
            i=edge_index[0,indx].detach().numpy()
            j=edge_index[1,indx].detach().numpy()
            xi=xs[i].detach().numpy()
            xj=list(xs[j].detach().numpy())
            eij=list(edge_attr[indx].detach().numpy())
            if str(i) not in Os.keys():
                Os[str(i)] = {'x_j':[],'e_ij':[]}
            Os[str(i)]['x_i'] = xi
            Os[str(i)]['x_j'] += xj
            Os[str(i)]['e_ij'] += eij

        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1,3,1)
        for i in Os.keys():
            plt.plot(Os[str(i)]['x_i'],label=str(i))
            plt.title('x_i')
        plt.legend()
        plt.subplot(1,3,2)
        for i in Os.keys():
            plt.plot(Os[str(i)]['e_ij'],label=str(i))
            plt.title('e_ij')
        plt.legend()
        plt.subplot(1,3,3)
        for i in Os.keys():
            plt.plot(Os[str(i)]['x_j'],label=str(i))
            plt.title('x_j')
        plt.legend()
        plt.show()

