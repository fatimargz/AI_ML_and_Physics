import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(parent_dir)
from helpers import get_event
from torch_geometric.nn import GCNConv, global_add_pool
import torch.nn.functional as F
class EdgeClassifier(torch.nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # this is a perceptron
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # generating node embeddings
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        source, target = edge_index
        edge_emb = torch.cat([x[source], x[target], edge_attr], dim=1)
        out = self.edge_mlp(edge_emb).squeeze(1)
        return torch.sigmoid(out)



model = EdgeClassifier(node_feat_dim = 5, edge_feat_dim = 4, hidden_dim=64)
model.load_state_dict(torch.load('/data/ac.frodriguez/best_gnn_model.pth'))
model.eval()
#print("loaded fine")


