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

for i in range(10,40):
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

#all_pos_edges = list(zip(all_source_nodes, all_target_nodes))
#print(all_pos_edges)
#print(len(all_pos_edges))
all_pos_edges = torch.stack([
    torch.tensor(all_source_nodes, dtype=torch.long),
    torch.tensor(all_target_nodes, dtype=torch.long)
]) 

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

# going to try random sampling
import random
def generate_rand_negatives(num_nodes, truth_edges, num_neg_samples):
    neg_edges = []
    truth_set = set((u.item(), v.item()) for u, v in truth_edges.T)
    while len(neg_edges) < num_neg_samples:
        u,v = random.randint(0, num_nodes-1), random.randint(0, num_nodes-1)
        if u != v and (u,v) not in truth_set:
            neg_edges.append([u,v])
    return torch.tensor(neg_edges, dtype = torch.long).t()


# Combine features from all events
all_features = np.vstack([event_features for event_features in all_node_features])
#all_neg_edges = np.array(generate_hard_negatives(all_features, all_pos_edges, k=20))
num_pos = all_pos_edges.size(1)
all_neg_edges = generate_rand_negatives(3, all_pos_edges, num_pos)
num_neg = all_neg_edges.shape[1]
## 1:1 neg to pos, before I did not do this and got poor model
#sampled_indices = np.random.choice(len(all_neg_edges), size = num_pos, replace = False)
#all_neg_edges = all_neg_edges[sampled_indices]

#all_neg_deltas = []
#all_neg_distances = []
#for u, v in all_neg_edges:
#    delta = all_features[v, :3] - all_features[u, :3]
#    all_neg_deltas.append(delta)
#    all_neg_distances.append(np.linalg.norm(delta))

# combining
# edge indices
edges = torch.cat([all_pos_edges,all_neg_edges], dim = 1) 
# edge labels
#labels = np.concatenate([np.ones(num_pos), np.zeros(num_neg)])
labels = torch.cat([
    torch.ones(num_pos, dtype=torch.float),
    torch.zeros(num_neg, dtype=torch.float)
])
# edge features
#edge_features = np.vstack([
#    np.hstack([all_deltas, np.array(all_distances).reshape(-1, 1)]),
#    np.hstack([all_neg_deltas, np.array(all_neg_distances).reshape(-1, 1)])
#])
# shuffle in unison
perm = torch.randperm(edges.size(1))
edges = edges[:, perm]
labels = labels[perm]

all_features = torch.tensor(all_features, dtype=torch.float)

src, dst = edges
delta_pos = all_features[dst, :3] - all_features[src, :3]
edge_features = torch.cat([
    delta_pos,
    torch.norm(delta_pos, dim=1, keepdim=True),
    (all_features[src, 3] - all_features[dst, 3]).abs().unsqueeze(1)], dim=1)


data = Data(
    x=all_features,
    edge_index=edges,
    edge_attr=edge_features,
    edge_y=labels
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
            torch.nn.Linear(hidden_dim, hidden_dim),
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
loss_vals = []
val_vals = []
for epoch in range(100):
    loss = train(model, train_data)
    loass_vals.append(loss_vals)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
    total_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters()))
    print(f"Gradient norm: {total_norm:.6f}")
    val_loss = validate(model, val_data)
    val_vals.append(val_vals)
    print("val_loss", val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0 
        torch.save(model.state_dict(), '/data/ac.frodriguez/best_gnn_model3.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("early stopping!")
            break
np.save('loss_values.npy',loss_vals)
np.save('validationloss_values.npy', val_vals)
torch.save(model.state_dict(), 'final_gcn_model.pth')

