import numpy as np

file1 = 'eval_results/eval_metrics_eval_metrics-EASTER-sm3-255-full.npy'
file2 = 'eval_results/eval_metrics_eval_metrics-EASTER-sm3-aug250-315-full.npy'

out_file = 'eval_results/eval_metrics_eval_metrics-EASTER-sm3-aug250-150--315.npy'

vec1 = np.load(file1)
vec2 = np.load(file2)

# full_vec_1 = np.zeros((vec2.shape[0], vec1.shape[1]))

# full_vec_1[0, :] = vec1[0, :]
# full_vec_1[2, :] = vec1[1, :]
# full_vec_1[3, :] = vec1[2, :]
# full_vec_1[4, :] = vec1[3, :]
# full_vec_1[5, :] = vec1[4, :]

combined_vec = np.concatenate((vec1,vec2), axis=1)

np.save(out_file, combined_vec)

