import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

trainl = np.load('loss_values.npy')
vall = np.load('validationloss_values.npy')

epoch = np.arange(35)
plt.plot(epoch, trainl, label='training')
plt.plot(epoch, vall, label='validation')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.savefig('losscurve.png')

print(trainl[-1])
print(vall[-1])
