import matplotlib.pyplot as plt
import numpy as np

#train = np.load('/data/ac.frodriguez/train_data2.npy')

#print(len(train), len(np.unique(train, axis =0)))

from tensorflow.keras.models import load_model

model = load_model('model_1/KERAS_check_best_model.h5')

y_pred = model.predict(X_test,y_pred)
