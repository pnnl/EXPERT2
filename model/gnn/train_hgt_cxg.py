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
from s2orc_local import S2ORC
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

# _GLOBAL_METADATA=(['paper'],
#  [('paper', 'outbound_cites', 'paper'),
# #   ('paper', 'inbound_cites', 'paper'),
#   ('paper', 'metapath_0', 'paper'),
#   ('paper', 'metapath_1', 'paper'),
# #   ('paper', 'metapath_2', 'paper')
#   ]) 


_SEED_EDGE_TYPE=('paper', 'outbound_cites', 'paper')
_SEED_EDGE_TYPE_REV=('paper', 'inbound_cites', 'paper')

_PRED_EDGE_TYPE=('paper', 'outbound_cites', 'paper')




# @functional_transform('two_hop_metapaths')
# class Two_Hop_Metapaths(BaseTransform):
#     r"""Add Metapaths to create MLKG from context graphs
#     """

#     def __init__(self):
#         from torch_geometric.nn.aggr.fused import FusedAggregation
#         self.aggr = FusedAggregation(['sum', 'min', 'max', 'mean', 'std'])

#     def __call__(self, data: HeteroData) -> HeteroData:
#         data = self.get_metapaths(data)
#         return data
#         # return self.get_paper_layers(data)
    
#     def get_metapaths(self, data: HeteroData):
#         metapaths = [
#             [("paper", "author"), ("author", "paper")],
#             [("paper", "venue"), ("venue", "paper")],
#             # [("paper", "topic"), ("topic", "paper")]
#                     ]
#         data = AddMetaPaths(metapaths)(data)
#         ##print(data)

#         return data
    
    # def get_paper_layers(self, data: HeteroData):
    #     del data['author', 'writes', 'paper'] 
    #     del data['paper', 'written_by', 'author'] 

    #     del data['paper', 'published_in', 'venue'] 
    #     del data['venue', 'publishes', 'paper'] 

    #     del data['topic', 'covered by', 'paper']
    #     del data['paper', 'covers', 'topic']
        
    #     del data['paper', 'inbound_cites', 'paper']

    #     del data['author']
    #     del data['venue']
    #     del data['topic']
        
    #     return data




# class GNN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         self.conv1 = SAGEConv(hidden_channels, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#     def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x
 
    
## HGT
class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(hidden_channels,out_channels)

        self.convs = torch.nn.ModuleList()
        self.data_metadata = None
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),#_GLOBAL_METADATA,
                           num_heads, group='sum')
            self.convs.append(conv)

#          self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        ##print(x_dict, edge_index_dict)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict
    
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_author: Tensor, x_paper: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_author = x_author[edge_label_index[0]]
        edge_feat_paper = x_paper[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_author * edge_feat_paper).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.paper_lin = torch.nn.Linear(20, hidden_channels)
        self.paper_emb = torch.nn.Embedding(data["paper"].num_nodes, hidden_channels)
        self.author_emb = torch.nn.Embedding(data["author"].num_nodes, hidden_channels)
        self.venue_emb = torch.nn.Embedding(data["venue"].num_nodes, hidden_channels)
        
        
        # # Instantiate homogeneous GNN:
        # self.gnn = GNN(hidden_channels)
        # # Convert GNN model into a heterogeneous variant:
        # self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        # Build heterogeneous GNN models 
        self.gnn = HGT(hidden_channels=hidden_channels, out_channels=hidden_channels, num_heads=2, num_layers=2)
        
        
        
        self.classifier = Classifier()


    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "paper": self.paper_emb(data["paper"].y_index),
          "author": self.author_emb(data["author"].y_index),
          "venue": self.venue_emb(data["venue"].y_index),
        } 
        ##print(data)
        ##"paper": self.paper_lin(data["paper"].x) + self.paper_emb(data["paper"].y_index),
        
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        ##print(x_dict)
        pred = self.classifier(
            x_dict["paper"],
            x_dict["paper"],
#             data["paper", "outbound_cites", "paper"].edge_label_index,
            data[_PRED_EDGE_TYPE[0], _PRED_EDGE_TYPE[1], _PRED_EDGE_TYPE[2]].edge_label_index,
        )
        return pred



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

# edges used for message passing (edge_index)
# edges used for supervision (edge_label_index)

# In the first hop, we sample at most 20 neighbors.
# In the second hop, we sample at most 10 neighbors.
# In addition, during training, we want to sample negative edges on-the-fly with
# a ratio of 2:1.
# We can make use of the `loader.LinkNeighborLoader` from PyG:
# Define seed edges:
edge_label_index = train_data[_SEED_EDGE_TYPE[0], _SEED_EDGE_TYPE[1], _SEED_EDGE_TYPE[2]].edge_label_index
edge_label = train_data[_SEED_EDGE_TYPE[0], _SEED_EDGE_TYPE[1], _SEED_EDGE_TYPE[2]].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20,10],
    neg_sampling_ratio=2.0,
    edge_label_index=(_SEED_EDGE_TYPE, edge_label_index),
    edge_label=edge_label,
    batch_size=args.batch_size,
    shuffle=True,
    # transform=T.Compose([Two_Hop_Metapaths()]),
)

#T.ToUndirected()

model = Model(hidden_channels=64)
print(model)

print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 6):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        print("Minibatch",sampled_data)
        sampled_data.to(device)
        pred = model(sampled_data)
        
        ground_truth = sampled_data[_PRED_EDGE_TYPE[0], _PRED_EDGE_TYPE[1], _PRED_EDGE_TYPE[2]].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

local_file_path=f'{args.data_dir}/model/cxg_model_state_dict.pth'
torch.save(model.state_dict(), local_file_path)
# torch.save(model,local_file_path)

# Define the validation seed edges:
edge_label_index = val_data[_PRED_EDGE_TYPE[0], _PRED_EDGE_TYPE[1], _PRED_EDGE_TYPE[2]].edge_label_index
edge_label = val_data[_PRED_EDGE_TYPE[0], _PRED_EDGE_TYPE[1], _PRED_EDGE_TYPE[2]].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(_PRED_EDGE_TYPE, edge_label_index),
    edge_label=edge_label,
    batch_size=args.batch_size,
    shuffle=False,
    # transform=Two_Hop_Metapaths(),
)
sampled_data = next(iter(val_loader))

preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(val_loader):
    with torch.no_grad():
        sampled_data.to(device)
        preds.append(model(sampled_data))
        ground_truths.append(sampled_data[_PRED_EDGE_TYPE[0], _PRED_EDGE_TYPE[1], _PRED_EDGE_TYPE[2]].edge_label)
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth, pred)
print()
print(f"Validation AUC: {auc:.4f}")
        