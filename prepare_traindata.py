import numpy as np
import pandas as pd
import zipfile
from helpers import get_event


## Preprocess Data
Train = []
for i in range(10,20):
    event = 'event0000010%02d'%i
    hits, cells, truth, particles = get_event(event)
    hit_cells = cells.groupby(['hit_id']).value.count().values
    hit_value = cells.groupby(['hit_id']).value.sum().values
    
    features = np.hstack((hits[['x','y','z']]/1000, hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)))

    particle_ids = truth.particle_id.unique()
    particle_ids = particle_ids[np.where(particle_ids!=0)[0]]
    # positive pairs
    pos_pairs = []
    for particle_id in particle_ids:
        hit_ids = truth[truth.particle_id == particle_id].hit_id.values-1
        for i in hit_ids:
            for j in hit_ids:
                if i != j:
                    pair = np.concatenate([features[i],features[j],[1]])
                    pos_pairs.append(pair)
    pos_pairs = np.array(pos_pairs)
    #print("pos_pair length:", len(pos_pairs))
    # negative pairs
    n_hits = len(hits)
    size = len(pos_pairs) 
    i = np.random.randint(n_hits, size = size)
    j = np.random.randint(n_hits, size = size)
    
    neg_pairs = []
    for idx in range(size):
        if truth.particle_id.values[i[idx]] != truth.particle_id.values[j[idx]]:
            pair = np.concatenate([features[i[idx]],features[j[idx]],[0]])
            neg_pairs.append(pair)

    neg_pairs = np.array(neg_pairs)
    #print("neg_pair:", neg_pairs)
    #print("neg_pair length:", len(neg_pairs))
    event_train = np.vstack((pos_pairs, neg_pairs))
    Train.append(event_train)

Train = np.vstack(Train)
np.random.shuffle(Train)
np.save('/data/ac.frodriguez/train_data2.npy',Train)
print("finished saving file")
