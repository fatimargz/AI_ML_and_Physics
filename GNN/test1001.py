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
            delta = features[v,:3] - features[u,:3]
            deltas.append(delta)
            distances.append(np.linalg.norm(delta))

edges = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
edge_features = torch.tensor(np.hstack([deltas, np.array(distances).reshape(-1, 1)]), dtype=torch.float)
data = Data(x = node_features, edge_index = edges, edge_attr = edge_features)

def plot_prob(scores, hit):
    plt.hist(scores, bins=50)
    plt.xlabel("Edge Probabilities for Hit%d"%hit)
    plt.ylabel("Count")
    plt.title("Distribution of Edge Prediction")
    title = 'probability_distribution_hit%d'%hit
    plt.savefig(title)


def get_predict(data, hit, edge_threshold = 0.5):
    model.eval()
    with torch.no_grad():
        probs = model(data)

    mask = (data.edge_index[0] ==hit)
    edges_hit = data.edge_index[:,mask]
    scores = probs[mask]
    plot_prob(scores, hit)

    pred = torch.zeros(data.num_nodes)
    pred[edges_hit[1]] = scores

    return pred.numpy()

def get_path(data, mask, thr, hit):
    path = [hit]
    a = 0
    while True:
        c = get_predict(data, path[-1], thr/2)
        mask = (c>thr)*mask
        mask[path[-1]]=0

        cand = np.where(c>thr)[0]
        if len(cand)>0:
            mask[cand[np.isin(module_id[cand], module_id[path])]] = 0

        a = (c+a) * mask

        if a.max() <thr * len(path):
            break
        
        next_hit = a.argmax()
        
        if features[next_hit][2] <= features[path[-1]][2]:
            break

        path.append(next_hit)
    return path



def plot_track_comparison(hits, path_indices, gt_indices,hit):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot predicted path
    pred_hits = hits.iloc[path_indices]
    ax.plot(pred_hits['x'], pred_hits['y'], pred_hits['z'], label='Predicted Path', color='r', marker='o')

    # Plot ground truth
    gt_hits = hits.iloc[gt_indices]
    ax.plot(gt_hits['x'], gt_hits['y'], gt_hits['z'], label='Ground Truth', color='g', marker='x')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title(f"Reconstructed path for hit num {hit}")
    plt.savefig(f"ReconstructedPath{event}Hit{hit}")


def compute_metrics(predicted_indices, true_indices, n_hits_total):
    # Create binary labels for all hits (1 if in path, 0 otherwise)
    pred_binary = np.zeros(n_hits_total, dtype=int)
    true_binary = np.zeros(n_hits_total, dtype=int)

    pred_binary[predicted_indices] = 1
    true_binary[true_indices] = 1

    precision = precision_score(true_binary, pred_binary)
    recall = recall_score(true_binary, pred_binary)
    f1 = f1_score(true_binary, pred_binary)

    return precision, recall, f1



for hit in range(3):
    gt = np.where(truth.particle_id==truth.particle_id[hit])[0] 
    path = get_path(data, np.ones(len(truth)), 0.4, hit)
    plot_track_comparison(hits, path, gt.tolist(), hit+1)
    precision, recall, f1 = compute_metrics(path, gt.tolist(), len(truth))
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
