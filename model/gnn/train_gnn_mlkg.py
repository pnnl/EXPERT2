import os.path as osp

import torch
from torch import Tensor
print(torch.__version__)
import torch.nn.functional as F

import torch_geometric
print(torch_geometric.__version__)

import torch_geometric.transforms as T
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.transforms import AddMetaPaths

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear
import numpy as np
import torch_geometric.transforms as T
from s2orc import S2ORC
from torch_geometric.nn import SAGEConv, to_hetero

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset
)

from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F

from torch_geometric.loader import LinkNeighborLoader

import tqdm
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser(
                    prog='EXPERT HGT',
                    description='HGT Model Pretraining with EXPERT MLKG',
                    epilog='Text at the bottom of help')
parser.add_argument('--data_dir',help="Location of the Pytorch Geometric data objects")           # positional argument
parser.add_argument('--batch_size',type=int,help="Batch size") 
args = parser.parse_args()

#path = osp.join('/rcfs/projects/expert/data/S2ORC_gensci/S2ORC_gensci_contextGraph_v2/', 'S2ORC_sample')
path = args.data_dir

# We initialize conference node features with a single one-vector as feature:
dataset = S2ORC(path)

#transform=T.Constant(node_types='conference')
data = dataset[0].to(device)
print(data)

# edges used for message passing (edge_index)
# edges used for supervision (edge_label_index)

_GLOBAL_METADATA=(['paper'],
 [('paper', 'outbound_cites', 'paper'),
#   ('paper', 'inbound_cites', 'paper'),
  ('paper', 'metapath_0', 'paper'),
  ('paper', 'metapath_1', 'paper'),
#   ('paper', 'metapath_2', 'paper')
  ]) 


_SEED_EDGE_TYPE=('paper', 'outbound_cites', 'paper')
_SEED_EDGE_TYPE_REV=None#('paper', 'inbound_cites', 'paper')

_PRED_EDGE_TYPE=('paper', 'outbound_cites', 'paper')




@functional_transform('two_hop_metapaths')
class Two_Hop_Metapaths(BaseTransform):
    r"""Add Metapaths to create MLKG from context graphs
    """

    def __init__(self):
        from torch_geometric.nn.aggr.fused import FusedAggregation
        self.aggr = FusedAggregation(['sum', 'min', 'max', 'mean', 'std'])

    def __call__(self, data: HeteroData) -> HeteroData:
        data = self.get_metapaths(data)

        return self.get_paper_layers(data)
    
    def get_metapaths(self, data: HeteroData):
        metapaths = [
            [("paper", "author"), ("author", "paper")],
            [("paper", "venue"), ("venue", "paper")],
            # [("paper", "topic"), ("topic", "paper")]
                    ]
        data = AddMetaPaths(metapaths)(data)
        ##print(data)

        return data
    
    def get_paper_layers(self, data: HeteroData):
        del data['author', 'writes', 'paper'] 
        del data['paper', 'written_by', 'author'] 

        del data['paper', 'published_in', 'venue'] 
        del data['venue', 'publishes', 'paper'] 

        del data['topic', 'covered by', 'paper']
        del data['paper', 'covers', 'topic']
        
        del data['paper', 'inbound_cites', 'paper']

        del data['author']
        del data['venue']
        del data['topic']
        
        return data

# For this, we first split the set of edges into
# training (80%), validation (10%), and testing edges (10%).
# Across the training edges, we use 70% of edges for message passing,
# and 30% of edges for supervision.
# We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
# Negative edges during training will be generated on-the-fly.
# We can leverage the `RandomLinkSplit()` transform for this from PyG:
train_test_split_transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=[_SEED_EDGE_TYPE],
    rev_edge_types=[_SEED_EDGE_TYPE_REV],
)
train_data, val_data, test_data = train_test_split_transform(data)

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
    
model = Model(hidden_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data['user', 'movie'].edge_label_index)
    target = train_data['user', 'movie'].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['user', 'movie'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['user', 'movie'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)