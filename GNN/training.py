import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import sys 
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(parent_dir)
from helpers import get_event

# radial distance function
def radial(f):
    r = np.sqrt(f[0]**2 + f[1]**2 + f[2]**2)
    return r

# global 
all_node_features = []
all_source_nodes = []
all_target_nodes = []
all_deltas = []
all_distances = []
all_edge_y = []

for i in range(10,20):
    event = 'event0000010%02d'%i
    hits, cells, truth, particles = get_event(event)
    hit_cells = cells.groupby(['hit_id']).value.count().values
    hit_value = cells.groupby(['hit_id']).value.sum().values

    # node features
    features = np.hstack((hits[['x','y','z']]/1000, hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)))
    all_node_features.append(features)

    particle_ids = truth.particle_id.unique()
    particle_ids = particle_ids[np.where(particle_ids!=0)[0]]
    for particle_id in particle_ids:
        hit_ids = truth[truth.particle_id == particle_id].hit_id.values-1
        for i in hit_ids:
            for j in hit_ids:
                if i != j:
                     # to determine source vs target. source should have smaller radial distance from origin than target 
                    ri = radial(features[i,:3]) 
                    rj = radial(features[j,:3])
                    if ri < rj:
                        all_source_nodes.append(i)
                        all_target_nodes.append(j)
                        delta = features[j,:3] - features[i,:3]
                    if rj < ri:
                        all_source_nodes.append(j)
                        all_target_nodes.append(i)
                        delta = features[i,:3] - features[j,:3]

                    # getting edge features
                    all_deltas.append(delta)
                    all_distances.append(np.linalg.norm(delta))

all_pos_edges = list(zip(all_source_nodes, all_target_nodes))
#print(all_pos_edges)
#print(len(all_pos_edges))

# generating hard negatives
from sklearn.neighbors import NearestNeighbors
def generate_hard_negatives(features_all, pos_edges, k=5):
    """Sample negatives from k-NN of each node, excluding true edges."""
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(features_all[:, :3])
    distances, indices = nbrs.kneighbors(features_all[:, :3])

    neg_edges = []
    pos_set = set(pos_edges)  # For O(1) lookups
    for u in range(len(features_all)):
        for v in indices[u]:
            if u != v and (u, v) not in pos_set:
                neg_edges.append((u, v))
    return neg_edges

# Combine features from all events
features_all = np.vstack([event_features for event_features in all_node_features])
all_neg_edges = np.array(generate_hard_negatives(features_all, all_pos_edges, k=20))
num_pos = len(all_pos_edges)
# 1:1 neg to pos, before I did not do this and got poor model
sampled_indices = np.random.choice(len(all_neg_edges), size = num_pos, replace = False)
all_neg_edges = all_neg_edges[sampled_indices]
num_neg = len(all_neg_edges)
print("num positives:", num_pos)
print("num negatives:", num_neg)
all_neg_deltas = []
all_neg_distances = []
for u, v in all_neg_edges:
    delta = features_all[v, :3] - features_all[u, :3]
    all_neg_deltas.append(delta)
    all_neg_distances.append(np.linalg.norm(delta))

# combining
# edge indices
edges = np.concatenate([all_pos_edges,all_neg_edges]) 
# edge labels
labels = np.concatenate([np.ones(num_pos), np.zeros(num_neg)])
# edge features
edge_features = np.vstack([
    np.hstack([all_deltas, np.array(all_distances).reshape(-1, 1)]),
    np.hstack([all_neg_deltas, np.array(all_neg_distances).reshape(-1, 1)])
])

# need to shuffle in unison
shuffle_idx = np.random.permutation(len(edges))
edges = edges[shuffle_idx]
labels = labels[shuffle_idx]
edge_features = edge_features[shuffle_idx]

data = Data(
    x=torch.tensor(features_all, dtype=torch.float),
    edge_index=torch.tensor(edges.T, dtype=torch.long),
    edge_attr=torch.tensor(edge_features, dtype=torch.float),
    edge_y=torch.tensor(labels, dtype=torch.float)
)

## GNN MODEL
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

# training loop 
model = EdgeClassifier(node_feat_dim=5, edge_feat_dim=4, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()

def train(model, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.edge_y)
    loss.backward()
    optimizer.step()
    return loss.item()

# going to add validation check and split data 80/20
from torch_geometric.transforms import RandomLinkSplit
edge_transform = RandomLinkSplit(num_val = .2, num_test = .8, is_undirected=False, split_labels=True)
train_data, val_data, test_data = edge_transform(data)

def validate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        val_loss = criterion(out, data.edge_y)
    return val_loss

patience = 10
epochs_no_improve = 0
best_val_loss = float('inf')
for epoch in range(100):
    loss = train(model, train_data)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
    total_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters()))
    print(f"Gradient norm: {total_norm:.6f}")
    val_loss = validate(model, val_data)
    print("val_loss", val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0 
        torch.save(model.state_dict(), '/data/ac.frodriguez/best_gnn_model2.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("early stopping!")
            break

torch.save(model.state_dict(), 'final_gcn_model.pth')

