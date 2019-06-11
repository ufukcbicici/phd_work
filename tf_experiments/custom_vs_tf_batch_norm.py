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

net_tf = tf.layers.batch_normalization(inputs=net, name="tf_bn", momentum=momentum, epsilon=1e-3, training=is_train)
net_cs = CustomBatchNormAlgorithms.batch_norm_multi_gpu_v2(input_tensor=net, epsilon=1e-3,
                                                           is_training=tf.cast(is_train, tf.int32),
                                                           momentum=momentum)
batch_norm_moving_averages = tf.get_collection(CustomBatchNormAlgorithms.BATCH_NORM_OPS)
batch_norm_ops_dict = {}
batchNormMovingAvgAssignOps = []
for moving_average, new_value in batch_norm_moving_averages:
    if moving_average not in batch_norm_ops_dict:
        batch_norm_ops_dict[moving_average] = []
    expanded_new_value = tf.expand_dims(new_value, 0)
    batch_norm_ops_dict[moving_average].append(expanded_new_value)
assert all([len(v) == 1 for k, v in batch_norm_ops_dict.items()])
# Take the mean of all values for every moving average and update the moving average value.
for moving_average, values_list in batch_norm_ops_dict.items():
    values_concat = tf.concat(axis=0, values=values_list)
    mean_new_value = tf.reduce_mean(values_concat, 0)
    momentum = GlobalConstants.BATCH_NORM_DECAY
    new_moving_average_value = tf.where(iteration_holder > 0,
                                        (momentum * moving_average + (1.0 - momentum) * mean_new_value),
                                        mean_new_value)
    new_moving_average_value_assign_op = tf.assign(moving_average, new_moving_average_value)
    batchNormMovingAvgAssignOps.append(new_moving_average_value_assign_op)

all_update_ops = []
all_update_ops.extend(batchNormMovingAvgAssignOps)
all_update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
with tf.control_dependencies(all_update_ops):
    net_tf = tf.identity(net_tf)
    net_cs = tf.identity(net_cs)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

all_vars = tf.global_variables()
results = sess.run([net_tf, net_cs],  feed_dict={input_tensor: x, is_train: True, iteration_holder: 0})
var_values = sess.run([all_vars])
print("X")


# total_time = 0
# for exp_index in range(1000):
#     t0 = time.time()
#     activation_arr = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, child_count))
#     results = sess.run([network, moving_avgs, assign_inputs],
#                        feed_dict={input_tensor: x, is_train: True, iteration_holder: exp_index})
#     t1 = time.time()
#     total_time += t1 - t0
#     print("Exp:{0} Time:{1}".format(exp_index, t1 - t0))
# print("Total Time:{0}".format(total_time))
