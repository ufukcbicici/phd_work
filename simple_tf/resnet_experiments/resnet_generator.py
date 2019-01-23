import tensorflow as tf
import numpy as np
from collections import namedtuple


class ResnetGenerator:
    # BottleneckGroup = namedtuple('BottleneckGroup', ['num_blocks', 'num_filters', 'bottleneck_size', 'down_sample'])
    ResnetHParams = namedtuple('ResnetHParams',
                               'num_residual_units, use_bottleneck, '
                               'num_of_features_per_block, relu_leakiness, first_conv_filter_size, strides, '
                               'activate_before_residual')

    @staticmethod
    def conv(name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                "conv_kernel", [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    @staticmethod
    def batch_norm(name, x, is_train, momentum):
        normalized_x = tf.layers.batch_normalization(inputs=x, name=name, momentum=momentum,
                                                     training=tf.cast(is_train, tf.bool))
        return normalized_x

    @staticmethod
    def stride_arr(stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    @staticmethod
    def relu(x, leakiness=0.0):
        """Relu, with optional leaky support."""
        if leakiness <= 0.0:
            return tf.nn.relu(features=x, name="relu")
        else:
            return tf.nn.leaky_relu(features=x, alpha=leakiness, name="leaky_relu")

    @staticmethod
    def global_avg_pool(x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

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

    @staticmethod
    def get_input(input, out_filters, first_conv_filter_size):
        assert input.get_shape().ndims == 4
        input_filters = input.get_shape().as_list()[-1]
        x = ResnetGenerator.conv("init_conv", input, first_conv_filter_size, input_filters, out_filters,
                                 ResnetGenerator.stride_arr(1))
        return x

    @staticmethod
    def get_output(x, is_train, leakiness, bn_momentum):
        x = ResnetGenerator.batch_norm("final_bn", x, is_train, bn_momentum)
        x = ResnetGenerator.relu(x, leakiness)
        x = ResnetGenerator.global_avg_pool(x)
        return x
