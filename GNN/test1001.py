import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(parent_dir)
from helpers import get_event
from torch_geometric.nn import GCNConv, global_add_pool
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

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
model.load_state_dict(torch.load('/data/ac.frodriguez/best_gnn_model2.pth'))

event = 'event000001001'
hits, cells, truth, particles = get_event(event)
hit_cells = cells.groupby(['hit_id']).value.count().values
hit_value = cells.groupby(['hit_id']).value.sum().values
features = np.hstack((hits[['x','y','z']]/1000, hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)))
node_features = torch.tensor(features, dtype = torch.float)

particle_ids = truth.particle_id.unique()
particle_ids = particle_ids[np.where(particle_ids!=0)[0]]
source_nodes = []
target_nodes = []
deltas = []
distances = []
for particle_id in particle_ids:
        hit_ids = truth[truth.particle_id == particle_id].hit_id.values-1
        hit_ids = sorted(hit_ids, key=lambda x: np.linalg.norm(features[x,:3])) 
        for i in range(len(hit_ids)-1):
            u = hit_ids[i]
            v = hit_ids[i+1]
            source_nodes.append(u)
            target_nodes.append(v)
            delta = features[v,:3] - features[u,3]
            deltas.append(delta)
            distances.append(np.linalg.norm(delta))

edges = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
edge_features = torch.tensor(np.hstack([deltas, np.array(distances).reshape(-1, 1)]), dtype=torch.float)
data = Data(x = node_features, edge_index = edges, edge_attr = edge_features)

model.eval()
with torch.no_grad():
    logits = model(data)
    edge_prob = torch.sigmoid(logits)

candidate_edges = edges[:, edge_prob>0.50].numpy().T

def is_valid_edge(u, v, features, min_r=0.1, max_angle=30):
    """Check if edge follows detector physics"""
    r_u = np.linalg.norm(features[u,:2])  # Radial distance (xy-plane)
    r_v = np.linalg.norm(features[v,:2])
    delta_r = r_v - r_u
    
    theta_u = np.arctan2(features[u,1], features[u,0])
    theta_v = np.arctan2(features[v,1], features[v,0])
    delta_theta = np.abs(theta_v - theta_u)
    
    # Must move outward with limited angular change
    return (delta_r > min_r) and (delta_theta < np.radians(max_angle))

# Convert features to numpy for faster processing
features_np = data.x[:,:3].cpu().numpy()  # Only need x,y,z

# Filter edges
valid_edges = []
for u, v in candidate_edges:
    if is_valid_edge(u, v, features_np):
        valid_edges.append([u, v])

pred_edges = torch.tensor(valid_edges, dtype=torch.long).T

def plot_track_comparison(hits, pred_edges, gt_indices, hit):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot predicted path
    if len(pred_edges) > 0:
        for u, v in pred_edges.T:
            ax.plot(*zip(hits.iloc[int(u)][['x', 'y', 'z']], hits.iloc[int(v)][['x', 'y', 'z']]),
                    'r-', linewidth=2, alpha=0.7, label='Predicted' if (u == pred_edges[0,0]) else "")

    # Plot ground truth
    gt_hits = hits.iloc[gt_indices]
    ax.plot(gt_hits['x'], gt_hits['y'], gt_hits['z'], label='Ground Truth', color='g', marker='x')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title(f"Reconstructed path for hit num {hit}")
    plt.savefig(f"ReconstructedPath{event}Hit{hit}")
    plt.close()



def compute_metrics(predicted_indices, true_indices, n_hits):
    predicted = np.unique(pred_edges.flatten())
    true = np.array(true_indices)

    y_true = np.zeros(n_hits)
    y_pred = np.zeros(n_hits)

    y_pred[predicted] = 1
    y_true[true] = 1

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1

for i, hit in enumerate([0, 1, 2]):
    gt = np.where(truth.particle_id==truth.particle_id[hit])[0] 
    plot_track_comparison(hits, pred_edges, gt.tolist(), i+1)
   # precision, recall, f1 = compute_metrics(pred_edges, gt.tolist(), len(truth))
   # print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
