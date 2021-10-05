import numpy as np
import matplotlib.pyplot as plt
import torch

file1 = 'eval_results2/eval_metrics2_eval_metrics-EAST-llr-.3.npy'
file2 = 'eval_results2/eval_metrics2_eval_metrics-EAST-llr-.45.npy'
file3 = 'eval_results2/eval_metrics2_eval_metrics-EAST-llr-.6.npy'
file4 = 'eval_results2/eval_metrics2_eval_metrics-EAST-llr-.8.npy'
file5 = 'eval_results2/eval_metrics2_eval_metrics-EAST-llr-1.npy'

vec1 = np.load(file1)
vec2 = np.load(file2)
vec3 = np.load(file3)
vec4 = np.load(file4)
vec5 = np.load(file5)

epoch_index = 0
train_loss_index = 1
test_loss_index = 2
precision_index = 3
recall_index = 4
variance_index = 5

num_samples = 6
epoch_offset = 450

epochs = vec1[epoch_index, :num_samples] - epoch_offset
variances1 = vec1[variance_index, :num_samples]
variances2 = vec2[variance_index, :num_samples]
variances3 = vec3[variance_index, :num_samples]
variances4 = vec4[variance_index, :num_samples]
variances5 = vec5[variance_index, :num_samples]

plt.figure(1)
plt.plot(epochs, variances1)
plt.plot(epochs, variances2)
plt.plot(epochs, variances3)
plt.plot(epochs, variances4)
plt.plot(epochs, variances5)
#plt.show()
#plt.ylim(20, 30)
plt.xlabel('Epoch')
plt.ylabel('Variance')
plt.legend(['.3', '.45', '.6', '.8', '1'])
plt.title('Comparing Variances')
plt.show()