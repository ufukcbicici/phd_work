import tensorflow as tf
import numpy as np
import time

from algorithms.custom_batch_norm_algorithms import CustomBatchNormAlgorithms
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
batch_size = 125
child_count = 4
num_of_input_channels = 32
num_of_output_channels = 32
filter_size = 5
momentum = 0.9
layer_count = 100
t0 = time.time()

activation_arr = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, child_count))
x = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 28, 28, num_of_input_channels))

input_tensor = tf.placeholder(name="input_tensor", dtype=tf.float32, shape=x.shape)
is_train = tf.placeholder(name="is_train", dtype=tf.bool)
iteration_holder = tf.placeholder(name="iteration_holder", dtype=tf.int32)
net = input_tensor

# *********************** Layers - Batch Norm with Tensorflow layers op ***********************
for l in range(layer_count):
    net = tf.layers.batch_normalization(inputs=net, name="layer_{0}_bn".format(l), momentum=momentum, training=is_train)
    net = tf.nn.relu(net)
    conv_weights = tf.Variable(
        tf.truncated_normal([filter_size, filter_size, num_of_input_channels, num_of_output_channels],
                            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
    conv_biases = tf.Variable(tf.constant(0.1, shape=[num_of_output_channels], dtype=GlobalConstants.DATA_TYPE))
    net = tf.nn.conv2d(net, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    net = CustomBatchNormAlgorithms.masked_batch_norm_multi_gpu(x=net, masked_x=net, network=None, node=None,
                                                                momentum=momentum,
                                                                is_training_phase=tf.cast(is_train, tf.int32),
                                                                iteration=iteration_holder, counter=l)
    net = tf.nn.relu(net)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
custom_batch_norm_averages = tf.get_collection(CustomBatchNormAlgorithms.CUSTOM_BATCH_NORM_OPS)
moving_avg_update_pairs = {}
manuel_update_dict = {}
# with tf.control_dependencies(update_ops):
moving_avgs = [(v.name, v) for v in tf.global_variables() if "moving_" in v.name]
assign_sub_ops = []
for tpl in moving_avgs:
    var_name = tpl[0]
    moving_avg_var = tpl[1]
    matched_pairs = [op.inputs for op in update_ops if op.inputs[0].name == var_name]
    delta_value = matched_pairs[0][1]
    assert len(matched_pairs) == 1
    moving_avg_update_pairs[var_name] = (moving_avg_var, delta_value)
    if "mean" in var_name:
        manuel_update_dict[var_name] = np.zeros(shape=moving_avg_var.shape)
    elif "var" in var_name:
        manuel_update_dict[var_name] = np.ones(shape=moving_avg_var.shape)
    assign_sub_op = tf.assign_sub(moving_avg_var, delta_value)
    assign_sub_ops.append(assign_sub_op)
with tf.control_dependencies(assign_sub_ops):
    network = tf.identity(net)
# for tpl in moving_avgs:
#     var_name = tpl[0]
#     moving_avg_var = tpl[1]
#     matched_pairs = [op.inputs for op in update_ops if op.inputs[0].name == var_name]
#     assert len(matched_pairs) == 1
#     moving_avg_update_pairs[var_name] = (moving_avg_var, matched_pairs[0][1])
#     if "mean" in var_name:
#         manuel_update_dict[var_name] = np.zeros(shape=moving_avg_var.shape)
#     elif "var" in var_name:
#         manuel_update_dict[var_name] = np.ones(shape=moving_avg_var.shape)
#     else:
#         raise NotImplementedError()
# print("X")

# ************************ Layers - Batch Norm with custom op ***********************
# for l in range(layer_count):
#     net = CustomBatchNormAlgorithms.batch_norm_multi_gpu_v2(input_tensor=net, is_training=tf.cast(is_train, tf.int32),
#                                                             momentum=momentum, counter=l)
#     net = tf.nn.relu(net)
#     conv_weights = tf.Variable(
#         tf.truncated_normal([filter_size, filter_size, num_of_input_channels, num_of_output_channels],
#                             stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
#     conv_biases = tf.Variable(tf.constant(0.1, shape=[num_of_output_channels], dtype=GlobalConstants.DATA_TYPE))
#     net = tf.nn.conv2d(net, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
# batch_norm_moving_averages = tf.get_collection(CustomBatchNormAlgorithms.CUSTOM_BATCH_NORM_OPS)
# # Assert that for every (moving_average, new_value) tuple, we have exactly #tower_count tuples with a specific
# # moving_average entry.
# batch_norm_ops_dict = {}
# batchNormMovingAvgAssignOps = []
# for moving_average, new_value in batch_norm_moving_averages:
#     if moving_average not in batch_norm_ops_dict:
#         batch_norm_ops_dict[moving_average] = []
#     expanded_new_value = tf.expand_dims(new_value, 0)
#     batch_norm_ops_dict[moving_average].append(expanded_new_value)
# assert all([len(v) == 1 for k, v in batch_norm_ops_dict.items()])
# # Take the mean of all values for every moving average and update the moving average value.
# for moving_average, values_list in batch_norm_ops_dict.items():
#     values_concat = tf.concat(axis=0, values=values_list)
#     mean_new_value = tf.reduce_mean(values_concat, 0)
#     momentum = GlobalConstants.BATCH_NORM_DECAY
#     new_moving_average_value = (momentum * moving_average + (1.0 - momentum) * mean_new_value)
#     new_moving_average_value_assign_op = tf.assign(moving_average, new_moving_average_value)
#     batchNormMovingAvgAssignOps.append(new_moving_average_value_assign_op)
# # update_ops = tf.get_collection(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
# with tf.control_dependencies(batchNormMovingAvgAssignOps):
#     network = tf.identity(net)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

total_time = 0
for exp_index in range(1000):
    t0 = time.time()
    activation_arr = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, child_count))
    results = sess.run([network, moving_avg_update_pairs],
                       feed_dict={input_tensor: x, is_train: True, iteration_holder: exp_index})
    update_results_dict = results[1]
    for k, v in update_results_dict.items():
        manuel_update_dict[k] -= v[1]
        assert np.allclose(manuel_update_dict[k], v[0])
    t1 = time.time()
    total_time += t1 - t0
    print("Exp:{0} Time:{1}".format(exp_index, t1 - t0))
print("Total Time:{0}".format(total_time))
