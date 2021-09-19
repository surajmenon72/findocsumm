import numpy as np

file1 = 'eval_results2/eval_metrics2_eval_metrics-EASTER-sm1-aug3-1--185.npy'
file2 = 'eval_results2/eval_metrics2_eval_metrics-EASTER-sm1-aug3-295.npy'

out_file = 'eval_results2/eval_metrics2_eval_metrics-EASTER-sm1-aug3-1--295.npy'

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

