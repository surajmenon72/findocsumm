import numpy as np
import matplotlib.pyplot as plt
import torch

file = 'eval_results2/eval_metrics2_eval_metrics-EASTER-sm1-aug3-no_ignore-fs-1--600.npy'
# file2 = 'eval_results/eval_metrics_eval_metrics-EASTER-sm3-aug2250-150--425.npy'
# file3 = 'eval_results/eval_metrics_eval_metrics-EASTER-sm3-aug3250-150--410.npy'

vec = np.load(file)
# vec2 = np.load(file2)
# vec3 = np.load(file3)

epoch_index = 0
train_loss_index = 1
test_loss_index = 2
precision_index = 3
recall_index = 4
variance_index = 5

num_samples = vec.shape[1]

epochs = vec[epoch_index, :]
train_losses = vec[train_loss_index, :]
test_losses = vec[test_loss_index, :]
precisions = vec[precision_index, :]
recalls = vec[recall_index, :]
variances = vec[variance_index, :]

plt.figure(1)
plt.plot(epochs, train_losses)
plt.savefig('train_loss.png')

plt.figure(2)
plt.plot(epochs, test_losses)
plt.savefig('test_loss.png')

plt.figure(3)
plt.plot(epochs, precisions)
plt.savefig('precisions.png')

#some recalls should be scaled by 5
recalls = recalls*5

plt.figure(4)
plt.plot(epochs, recalls)
plt.savefig('recalls.png')

plt.figure(5)
plt.plot(epochs, variances)
plt.savefig('variances.png')


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

mv_avg_n = 10
begin_epoch = epochs[0]
end_epoch = epochs[num_samples-1]

results = variances
mv_avg_results = moving_average(results, mv_avg_n)
mv_avg_shape = mv_avg_results.shape[0]
x = np.linspace(begin_epoch, end_epoch, num=mv_avg_shape)

plt.figure(6)
plt.plot(x, mv_avg_results)
plt.savefig('smooth.png')

recalls += 1e-6
precisions += 1e-6

f1 = (2*recalls*precisions)/(recalls+precisions)

plt.figure(7)
plt.plot(epochs, f1)
plt.savefig('f1.png')

c1 = np.corrcoef(variances, recalls)
print (c1)

# results2 = vec2[variance_index, :]
# results3 = vec3[variance_index, :]

# mv_avg_results2 = moving_average(results2, mv_avg_n)
# mv_avg_results3 = moving_average(results3, mv_avg_n)

# mv_avg_results2_f = mv_avg_results2[:mv_avg_shape]
# mv_avg_results3_f = mv_avg_results3[:mv_avg_shape]

# plt.figure(2)
# plt.plot(x, mv_avg_results, color='red')
# plt.plot(x, mv_avg_results2_f, color='blue')
# plt.plot(x, mv_avg_results3_f, color='green')
# #plt.ylim(20, 30)
# plt.xlabel('Epoch')
# plt.ylabel('Spread')
# plt.legend(['Lowest', 'Middle', 'Highest'])
# plt.title('Comparing Variances')
# plt.show()