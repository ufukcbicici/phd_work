import tensorflow as tf
import numpy as np

from data_handling.mnist_data_set import MnistDataSet


def check_result(x, slice_arrays, branch_arrays):
    for i in range(len(slice_arrays)):
        slice_array = slice_arrays[i]
        branch_array = branch_arrays[i]
        nonzero_indices = np.nonzero(slice_array)
        print("Branch {0} Sample Count:{1}".format(i, nonzero_indices[0].shape[0]))
        for j in range(nonzero_indices[0].shape[0]):
            index = nonzero_indices[0][j]
            x_0 = x[index, :]
            x_1 = branch_array[j, :]
            dif = x_0 - x_1
            nz = np.flatnonzero(dif)
            if len(nz) != 0:
                raise Exception("!!!ERROR!!!")
    print("Correct Result.")


k = 3
D = MnistDataSet.MNIST_SIZE * MnistDataSet.MNIST_SIZE
threshold = 0.3
feature_count = 32
epsilon = 0.000001
batch_size = 100

dataset = MnistDataSet(validation_sample_count=5000)
dataset.load_dataset()

samples, labels, indices_list = dataset.get_next_batch(batch_size=batch_size)
index_list = np.arange(0, batch_size)
initializer = tf.contrib.layers.xavier_initializer()
x = tf.placeholder(tf.float32, name="x")
indices = tf.placeholder(tf.int64, name="indices")
# Convolution
x_image = tf.reshape(x, [-1, MnistDataSet.MNIST_SIZE, MnistDataSet.MNIST_SIZE, 1])
C = tf.get_variable(name="C", shape=[5, 5, 1, feature_count], initializer=initializer,
                    dtype=tf.float32)
b_c = tf.get_variable(name="b_c", shape=(feature_count,), initializer=initializer, dtype=tf.float32)
conv_without_bias = tf.nn.conv2d(x_image, C, strides=[1, 1, 1, 1], padding="SAME")
conv = conv_without_bias + b_c
# Branching
flat_x = tf.reshape(x, [-1, D])
W = tf.get_variable(name="W", shape=(D, k), initializer=initializer,
                    dtype=tf.float32)
b = tf.get_variable(name="b", shape=(k,), initializer=initializer, dtype=tf.float32)
activations = tf.matmul(flat_x, W) + b
h = tf.nn.softmax(logits=activations)
pass_check = tf.greater_equal(x=h, y=threshold)
branched_conv_list = []
branched_fc_list = []
branched_final_feature_list = []
branched_indices_list = []
sample_counts_list = []
slices = []
summation = None
for n in range(k):
    pre_mask = tf.slice(pass_check, [0, n], [-1, 1])
    mask = tf.reshape(pre_mask, [-1])
    branched_conv = tf.boolean_mask(tensor=conv, mask=mask)
    branched_activations = tf.boolean_mask(tensor=activations, mask=mask)
    branched_indices = tf.boolean_mask(tensor=indices, mask=mask)
    branched_conv_flat = tf.reshape(branched_conv, [-1, D * feature_count])
    branched_final_feature = tf.concat(values=[branched_conv_flat, branched_activations], axis=1)
    branched_final_feature_list.append(branched_final_feature)
    branched_conv_list.append(branched_conv)
    branched_indices_list.append(branched_indices)
    sample_count = tf.size(branched_indices)
    sample_counts_list.append(sample_count)
    weights = tf.get_variable(name="weights{0}".format(n), shape=(D * feature_count, 25), initializer=initializer,
                              dtype=tf.float32)
    biases = tf.get_variable(name="biases{0}".format(n), shape=(25,), initializer=initializer, dtype=tf.float32)
    branched_fc = tf.matmul(branched_conv_flat, weights) + biases
    branched_fc_list.append(branched_fc)
    slices.append(mask)
    if n == 0:
        summation = tf.reduce_sum(branched_final_feature)
    else:
        summation = summation + tf.reduce_sum(branched_final_feature)
grads = tf.gradients(summation, [C, b_c, W, b])

# Init
init = tf.global_variables_initializer()
config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
sess.run(init)
initial_values = sess.run([W, b])
initial_values = sess.run([W, b])
initial_values = sess.run([W, b])
initial_values = sess.run([W, b])

# Gradient Test with finite differences
initial_C, initial_b_c, initial_W, initial_b = sess.run([C, b_c, W, b], {x: samples})
value_dict = {"C": initial_C, "b_c": initial_b_c, "W": initial_W, "b": initial_b}
tensor_dict = {"C": C, "b_c": b_c, "W": W, "b": b}
eval_list = [summation, grads]
eval_list.extend(branched_indices_list)
eval_list.extend(sample_counts_list)
eval_list.extend(branched_conv_list)
eval_list.extend(branched_fc_list)
results = sess.run(eval_list, {x: samples, indices: index_list})
original_loss = results[0]
tf_grads_dict = {"C": results[1][0], "b_c": results[1][1], "W": results[1][2], "b": results[1][3]}
print("Tf Mean={0}".format(results[0]))
tf_grads = results[1]

sample_histograms_per_branch = {}
for i in range(k):
    i_list = results[2 + i]
    for index in i_list:
        if index not in sample_histograms_per_branch:
            sample_histograms_per_branch[index] = 0
        sample_histograms_per_branch[index] += 1
# Calculate the gradients of convolution filters (C), strides assumed to be [1, 1, 1, 1]
# filter_size = 5
# manual_grad_C = np.zeros(shape=(filter_size, filter_size, 1, feature_count))
# for index in range(batch_size):
#     count = sample_histograms_per_branch[index]
#     image = samples[index]
#     print("index={0}".format(index))
#     for r in range(MnistDataSet.MNIST_SIZE):
#         for c in range(MnistDataSet.MNIST_SIZE):
#             for feature_index in range(feature_count):
#                 for ii in range(-int(filter_size / 2), int(filter_size / 2) + 1):
#                     for jj in range(-int(filter_size / 2), int(filter_size / 2) + 1):
#                         r_index = r + ii
#                         c_index = c + jj
#                         if (r_index < 0 or r_index >= MnistDataSet.MNIST_SIZE) or (
#                                 c_index < 0 or c_index >= MnistDataSet.MNIST_SIZE):
#                             manual_grad_C[ii + int(filter_size / 2), jj + int(
#                                 filter_size / 2), 0, feature_index] += 0.0
#                         else:
#                             manual_grad_C[ii + int(filter_size / 2), jj + int(
#                                 filter_size / 2), 0, feature_index] += float(count) * image[r_index][c_index]
# Calculate the gradients of hyperplanes (W)
manual_grad_W = np.zeros(shape=(D, k))
for index in range(batch_size):
    count = sample_histograms_per_branch[index]
    image = samples[index].reshape(D, 1)
    manual_grad_W += (float(count) * image)



# for arr_name, arr in value_dict.items():
#     print("Array Name:{0}".format(arr_name))
#     it = np.nditer(arr, flags=['multi_index'])
#     while not it.finished:
#         real_grad = tf_grads_dict[arr_name][it.multi_index]
#         calc_grad_list = []
#         original_value = arr[it.multi_index]
#         print("********************************")
#         # loss_list = []
#         # for trial1 in range(10000):
#         #     res_m = sess.run(eval_list, {x: samples, indices: index_list})
#         #     loss_list.append(res_m[0])
#         for trial2 in range(100):
#             tensor = tensor_dict[arr_name]
#             # Delta Plus
#             arr[it.multi_index] = original_value + epsilon
#             res_p = sess.run(eval_list, {x: samples, indices: index_list, tensor: arr})
#             val_p = res_p[0]
#             # Delta Minus
#             arr[it.multi_index] = original_value - epsilon
#             res_m = sess.run(eval_list, {x: samples, indices: index_list, tensor: arr})
#             val_m = res_m[0]
#             # Check if the indices are the same.
#             for i in range(k):
#                 if len(res_p[i + 1]) != len(res_m[i + 1]):
#                     raise Exception("Index lengths are different")
#                 for j in range(len(res_p[i + 1])):
#                     if res_p[i + 1][j] != res_m[i + 1][j]:
#                         raise Exception("Different indices")
#             # print("value:{0}".format(original_loss))
#             # print("val_p:{0}".format(val_p))
#             # print("val_m:{0}".format(val_m))
#             # print("real_grad:{0}".format(real_grad))
#             calc_grad = (val_p - val_m) / (2.0 * epsilon)
#             calc_grad_list.append(calc_grad)
#             # Restore back
#             arr[it.multi_index] = original_value
#         mean_calc_grad = np.array(calc_grad_list).mean()
#         print("Array:{0} Entry:<{1}>".format(arr_name, it.multi_index))
#         print("value:{0}".format(original_value))
#         print("real_grad:{0}".format(real_grad))
#         print("mean_calc_grad:{0}".format(mean_calc_grad))
#         relative_difference = np.abs(real_grad - mean_calc_grad) / max(np.abs(mean_calc_grad), np.abs(real_grad))
#         print("relative_difference={0}".format(relative_difference))
#         print("********************************")
#         it.iternext()
print("X")










# Tests without branching
# conv_flat = tf.reshape(conv, [-1, D * feature_count])
# final_feature = tf.concat(values=[conv_flat, h], axis=1)
# mean = tf.reduce_mean(final_feature)
# grads = tf.gradients(mean, [C, b_c, W, b])
#
#
# eval_list = [final_feature, h, grads]
# results = sess.run(eval_list, {x: samples})
# print("X")

# eval_list = [conv]
# eval_list.extend(slices)
# eval_list.extend(branched_conv_list)
# eval_list.extend(branched_final_feature_list)
# eval_list.append(mean)
# eval_list.append(grads)

# Run
# sess = tf.Session()
# sess.run(init)
# tf_results = []
# for i in range(1000):
#     results = sess.run(eval_list, {x: samples})
#     check_result(x=results[0], slice_arrays=results[1:k + 1], branch_arrays=results[k + 1:2 * k + 1])
#     branched_arrays = results[2*k + 1:3*k + 1]
#     real_mean32 = 0.0
#     real_mean64 = 0.0
#     for arr in branched_arrays:
#         real_mean32 += arr.astype(np.float32).mean()
#         real_mean64 += arr.astype(np.float64).mean()
#     tf_results.append(results[1 + 3*k])
#     print("Tf Mean={0}".format(results[1 + 3*k]))
#     print("Real Mean32={0}".format(real_mean32))
#     print("Real Mean64={0}".format(real_mean64))
# print("X")
