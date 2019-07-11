import numpy as np
import tensorflow as tf

from algorithms.custom_batch_norm_algorithms import CustomBatchNormAlgorithms
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants


class ResnetGenerator:
    # BottleneckGroup = namedtuple('BottleneckGroup', ['num_blocks', 'num_filters', 'bottleneck_size', 'down_sample'])

    # MultiGpu OK
    @staticmethod
    def conv(name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            shape = [filter_size, filter_size, in_filters, out_filters]
            initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n))
            kernel = UtilityFuncs.create_variable(name="conv_kernel", shape=shape, dtype=tf.float32,
                                                  initializer=initializer)
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    # MultiGpu OK
    @staticmethod
    def batch_norm(name, x, is_train, momentum):
        # return tf.identity(x)
        if GlobalConstants.USE_MULTI_GPU:
            normalized_x = CustomBatchNormAlgorithms.batch_norm_multi_gpu_v2(x=x, is_training=is_train,
                                                                             momentum=momentum)
            # with tf.device(GlobalConstants.GLOBAL_PINNING_DEVICE):
            #     normalized_x = tf.layers.batch_normalization(inputs=x, name=name, momentum=momentum,
            #                                                  training=tf.cast(is_train, tf.bool))
        else:
            normalized_x = tf.layers.batch_normalization(inputs=x, name=name, momentum=momentum,
                                                         training=tf.cast(is_train, tf.bool))
        return normalized_x

    # MultiGpu OK
    @staticmethod
    def stride_arr(stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    # MultiGpu OK
    @staticmethod
    def relu(x, leakiness=0.0):
        """Relu, with optional leaky support."""
        if leakiness <= 0.0:
            return tf.nn.relu(features=x, name="relu")
        else:
            return tf.nn.leaky_relu(features=x, alpha=leakiness, name="leaky_relu")

    # MultiGpu OK
    @staticmethod
    def global_avg_pool(x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    # MultiGpu OK
    @staticmethod
    def bottleneck_residual(x, is_train, in_filter, out_filter, stride, relu_leakiness, activate_before_residual,
                            bn_momentum):
        """Bottleneck residual unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope("common_bn_relu"):
                x = ResnetGenerator.batch_norm("init_bn", x, is_train, bn_momentum)
                x = ResnetGenerator.relu(x, relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope("residual_bn_relu"):
                orig_x = x
                x = ResnetGenerator.batch_norm("init_bn", x, is_train, bn_momentum)
                x = ResnetGenerator.relu(x, relu_leakiness)

        with tf.variable_scope("sub1"):
            x = ResnetGenerator.conv("conv_1", x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope("sub2"):
            x = ResnetGenerator.batch_norm("bn2", x, is_train, bn_momentum)
            x = ResnetGenerator.relu(x, relu_leakiness)
            x = ResnetGenerator.conv("conv2", x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        with tf.variable_scope("sub3"):
            x = ResnetGenerator.batch_norm("bn3", x, is_train, bn_momentum)
            x = ResnetGenerator.relu(x, relu_leakiness)
            x = ResnetGenerator.conv("conv3", x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope("sub_add"):
            if in_filter != out_filter or not all([d == 1 for d in stride]):
                orig_x = ResnetGenerator.conv("project", orig_x, 1, in_filter, out_filter, stride)
            x += orig_x
        return x

    # MultiGpu OK
    @staticmethod
    def get_input(input, out_filters, first_conv_filter_size):
        assert input.get_shape().ndims == 4
        input_filters = input.get_shape().as_list()[-1]
        x = ResnetGenerator.conv("init_conv", input, first_conv_filter_size, input_filters, out_filters,
                                 ResnetGenerator.stride_arr(1))
        return x

    # MultiGpu OK
    @staticmethod
    def get_output(x, is_train, leakiness, bn_momentum):
        x = ResnetGenerator.batch_norm("final_bn", x, is_train, bn_momentum)
        x = ResnetGenerator.relu(x, leakiness)
        x = ResnetGenerator.global_avg_pool(x)
        return x