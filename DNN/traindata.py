import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from callbacks import all_callbacks

Train = np.load('/data/ac.frodriguez/train_data2.npy')

model = Sequential()
model.add(Dense(800, activation='selu', input_shape=(10,)))
model.add(Dense(400, activation='selu'))
model.add(Dense(400, activation='selu'))
model.add(Dense(400, activation='selu'))
model.add(Dense(200, activation='selu'))
model.add(Dense(1, activation='sigmoid'))

output_folder = '/data/ac.frodriguez/model_2'

adam = Adam(learning_rate=1e-4)
model.compile(optimizer=adam, loss=['binary_crossentropy'], metrics=['accuracy'])
callbacks = all_callbacks(
    stop_patience=10,
    lr_factor=0.1,
    lr_patience=2,
    lr_epsilon=0.000001,
    lr_cooldown=0,
    lr_minimum=0.000001,
    outputDir=output_folder,
)
model.fit(
    Train[:,:-1],
    Train[:,-1],
    batch_size=8000,
    epochs=30,
    validation_split=0.05,
    shuffle=True,
    callbacks=callbacks.callbacks,
)
