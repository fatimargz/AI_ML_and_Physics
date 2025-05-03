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

    # edges
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
                        delta = features[i,:3] - features[j:3]

                    # getting edge features
                    all_deltas.append(delta)
                    all_distances.append(np.linalg.norm(delta))


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
hard_neg_edges = generate_hard_negatives(features_all, all_pos_edges, k=5)
# Subsample negatives
num_pos = len(all_pos_edges)
hard_neg_edges = np.random.choice(hard_neg_edges, size=num_pos, replace=False)
all_neg_deltas = []
all_neg_distances = []
for u, v in hard_neg_edges:
    delta = features_all[v, :3] - features_all[u, :3]
    all_neg_deltas.append(delta)
    all_neg_distances.append(np.linalg.norm(delta))

# combining
# edge indices
edge_index = np.array(all_pos_edges + hard_neg_edges).T 
# edge labels
edge_y = np.concatenate([np.ones(num_pos), np.zeros(num_pos)])
# edge features
edge_attr = np.vstack([
    np.hstack([all_pos_deltas, np.array(all_pos_distances).reshape(-1, 1)]),  # Positives
    np.hstack([all_neg_deltas, np.array(all_neg_distances).reshape(-1, 1)])   # Negatives
])

# need to shuffle in unison
shuffle_idx = np.random.permutation(len(edges))
edges = edges[shuffle_idx]
labels = labels[shuffle_idx]
edge_features = edge_features[shuffle_idx]

data = Data(
    x=torch.tensor(features_all, dtype=torch.float),
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    edge_y=torch.tensor(edge_y, dtype=torch.float)
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss()

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.edge_y)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(30):
    loss = train(data)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

torch.save(model.state_dict(), 'gcn_model.pth')

