import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
from tensorflow.contrib.nccl.ops import gen_nccl_ops

from algorithms.custom_batch_norm import CustomBatchNorm
from auxillary.general_utility_funcs import UtilityFuncs
#
# def experiment():
#     available_devices = UtilityFuncs.get_available_devices()
#     print(available_devices)
#     batch_size = 250
#     gpu_count = 1
#     input_tensors = []
#     means = []
#     sync_means = []
#     with tf.device("/cpu:0"):
#         with tf.variable_scope("Shared"):
#             for device_id in range(gpu_count):
#                 with tf.device("/gpu:{0}".format(device_id)):
#                     # Input Tensors
#                     input_tensor = tf.placeholder(name="input_{0}".format(device_id),
#                                                   dtype=tf.float32, shape=(batch_size, 32, 32, 20))
#                     input_tensors.append(input_tensor)
#                     # Mean (per GPU)
#                     mean = tf.reduce_mean(input_tensor, axis=[0, 1, 2])
#                     means.append(mean)
#                     # Mean (Over all GPUs)
#                     sync_mean = gen_nccl_ops.nccl_all_reduce(
#                               input=mean,
#                               reduction='sum',
#                               num_devices=gpu_count,
#                               shared_name="Shared") * (1.0 / gpu_count)
#                     sync_means.append(sync_mean)
#
#     input_tensors_numpy = \
#         [np.random.uniform(0, 1.0, (batch_size, 32, 32, 20)) for _ in range(gpu_count)]
#     sess = tf.Session()
#     res = sess.run([means, sync_means],
#                    feed_dict={input_tensors[device_id]: input_arr
#                               for device_id, input_arr in enumerate(input_tensors_numpy)})
#     print("X")
#
#
#     # input_tensors_numpy = [np.random.uniform(0, 1.0, (batch_size, 32, 32, 20)) for _ in range(len(available_devices))]
#     # # gen_nccl_ops.nccl_all_reduce
#
#     print("X")
from simple_tf.global_params import GlobalConstants
from simple_tf.resnet_experiments.resnet_generator import ResnetGenerator


def experiment_with_towers():
    # Conv layer
    batch_size = 250
    width = 16
    height = 16
    channels = 64
    _x = tf.placeholder(name="input", dtype=tf.float32, shape=(batch_size, width, height, channels))
    is_train = tf.placeholder(name="is_train", dtype=tf.int32)
    np_x = np.random.uniform(0, 1.0, (batch_size, width, height, channels))

    tower_count = 4
    strides = GlobalConstants.RESNET_HYPERPARAMS.strides
    activate_before_residual = GlobalConstants.RESNET_HYPERPARAMS.activate_before_residual
    filters = GlobalConstants.RESNET_HYPERPARAMS.num_of_features_per_block
    num_of_units_per_block = GlobalConstants.RESNET_HYPERPARAMS.num_residual_units
    relu_leakiness = GlobalConstants.RESNET_HYPERPARAMS.relu_leakiness
    first_conv_filter_size = GlobalConstants.RESNET_HYPERPARAMS.first_conv_filter_size

    for tower_id in range(tower_count):
        with tf.device("/cpu:0"):
            with tf.name_scope("tower_{0}".format(tower_id)):
                net = ResnetGenerator.get_input(input=_x, out_filters=filters[0],
                                                first_conv_filter_size=first_conv_filter_size)
                with tf.variable_scope("block_1_0"):
                    net = ResnetGenerator.bottleneck_residual(x=net, in_filter=filters[0], out_filter=filters[1],
                                                              stride=ResnetGenerator.stride_arr(strides[0]),
                                                              activate_before_residual=activate_before_residual[0],
                                                              relu_leakiness=relu_leakiness, is_train=is_train,
                                                              bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
                for i in range(num_of_units_per_block - 1):
                    with tf.variable_scope("block_1_{0}".format(i + 1)):
                        net = ResnetGenerator.bottleneck_residual(x=net, in_filter=filters[1],
                                                                  out_filter=filters[1],
                                                                  stride=ResnetGenerator.stride_arr(1),
                                                                  activate_before_residual=False,
                                                                  relu_leakiness=relu_leakiness, is_train=is_train,
                                                                  bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
                tf.get_variable_scope().reuse_variables()

    batch_norm_ops = tf.get_collection(key=CustomBatchNorm.BATCH_NORM_OPS)
    op_name = "block_1_0/sub2/pop_var:0"
    selected_ops = [tpl[0] for tpl in batch_norm_ops if tpl[0].name == op_name]
    temp = tf.placeholder(name="temp", dtype=tf.float32, shape=selected_ops[0].get_shape())
    assign_op = tf.assign(selected_ops[0], temp)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    res = sess.run(selected_ops, feed_dict={_x: np_x, is_train: 1})
    res2 = sess.run([assign_op], feed_dict={temp: 5 * np.ones(shape=selected_ops[0].get_shape())})
    res3 = sess.run(selected_ops, feed_dict={})
    print("X")

# mu, sigma, normalized_x = CustomBatchNorm.batch_norm(input_tensor=_x,
#                                                      momentum=GlobalConstants.BATCH_NORM_DECAY,
#                                                      epsilon=1e-3,
#                                                      is_training=is_train)
# tf_normalized_x = tf.layers.batch_normalization(inputs=_x,
#                                                 momentum=GlobalConstants.BATCH_NORM_DECAY,
#                                                 epsilon=1e-3,
#                                                 training=tf.cast(is_train, tf.bool))
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
#
# x = np.random.uniform(size=(batch_size, width, height, channels))
#
# res = sess.run([mu, sigma, normalized_x, tf_normalized_x], feed_dict={_x: x, is_train: 1})
# print("X")
