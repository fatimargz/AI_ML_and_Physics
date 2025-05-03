import zipfile
import pandas as pd
import numpy as np

def get_event(event):
    zf = zipfile.ZipFile('/data/ac.frodriguez/train_1.zip')
    hits= pd.read_csv(zf.open('train_1/%s-hits.csv'%event))
    cells= pd.read_csv(zf.open('train_1/%s-cells.csv'%event))
    truth= pd.read_csv(zf.open('train_1/%s-truth.csv'%event))
    particles = pd.read_csv(zf.open('train_1/%s-particles.csv'%event))
    return hits, cells, truth, particles

def get_path(hit, mask, thr, truth):
    path = [hit]
    a = 0
    while True:
        c = get_predict(path[-1], truth, thr/2)
        mask = (c > thr)*mask
        mask[path[-1]] = 0
        
        if 1:
            cand = np.where(c>thr)[0]
            if len(cand)>0:
                mask[cand[np.isin(module_id[cand], module_id[path])]]=0
                
        a = (c + a)*mask
        if a.max() < thr*len(path):
            break
        path.append(a.argmax())
    return path

def get_predict(hit, truth, thr=0.5):
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

