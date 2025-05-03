import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from helpers import get_event
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import precision_score, recall_score, f1_score

model = load_model('model_1/KERAS_check_best_model.h5')

event = 'event000001001'
hits, cells, truth, particles = get_event(event)
hit_cells = cells.groupby(['hit_id']).value.count().values
hit_value = cells.groupby(['hit_id']).value.sum().values
features = np.hstack((hits[['x','y','z']]/1000, hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)))
count = hits.groupby(['volume_id','layer_id','module_id'])['hit_id'].count().values
module_id = np.zeros(len(hits), dtype='int32')

for i in range(len(count)):
    si = np.sum(count[:i])
    module_id[si:si+count[i]] = i

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


def radial(f): 
    return np.sqrt(f[0]**2 + f[1]**2)

def get_path(hit, mask, thr):
    path = [hit]
    a = 0
    while True:
        c = get_predict(path[-1], thr/2)
        mask = (c > thr)*mask
        mask[path[-1]] = 0
        
        # dont add hits in the same module as previous
        cand = np.where(c >thr)[0]
        if len(cand)>0:
            mask[cand[np.isin(module_id[cand], module_id[path])]]=0
                
        a = (c + a)*mask
        
        # check for other valid candidates
        if a.max() < thr*len(path):
            break

        # move forward in z
        next_hit = a.argmax()
        if features[next_hit][2] <= features[path[-1]][2]:
            break
        
        path.append(a.argmax())
    return path

def get_predict(hit, thr=0.5):
    Tx = np.zeros((len(truth),10))
    Tx[:,5:] = features
    Tx[:,:5] = np.tile(features[hit], (len(Tx), 1))
    pred = model.predict(Tx, batch_size=len(Tx))[:,0]
    # TTA
    idx = np.where(pred > thr)[0]
    Tx2 = np.zeros((len(idx),10))
    Tx2[:,5:] = Tx[idx,:5]
    Tx2[:,:5] = Tx[idx,5:]
    pred1 = model.predict(Tx2, batch_size=len(idx))[:,0]
    pred[idx] = (pred[idx] + pred1)/2
    return pred


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

for hit in range(3):
    path = get_path(hit, np.ones(len(truth)), 0.80)
    gt = np.where(truth.particle_id==truth.particle_id[hit])[0]
    print('hit_id = ', hit+1)
    print('reconstruct :', path)
    print('ground truth:', gt.tolist())
    
    precision, recall, f1 = compute_metrics(path, gt.tolist(), len(truth))
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    plot_track_comparison(hits, path, gt.tolist(), hit+1)

