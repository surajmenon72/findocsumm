import numpy as np
import matplotlib.pyplot as plt

file = 'eval_results/eval_metrics_eval_metrics-EASTER-sm3-255-full.npy'

vec = np.load(file)

#off by 1 for EASTER-sm3-215, first one, no training loss
epoch_index = 0
train_loss_index = 1
test_loss_index = 2
precision_index = 3
recall_index = 4
variance_index = 5

epochs = vec[epoch_index, :]
train_losses = vec[train_loss_index, :]
test_losses = vec[test_loss_index, :]
precisions = vec[precision_index, :]
recalls = vec[recall_index, :]
variances = vec[variance_index, :]

plt.figure(1)
plt.plot(epochs, variances)
plt.show()