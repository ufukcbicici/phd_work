import numpy as np
import tensorflow as tf


class ResidualNetworkGenerator:
    # BottleneckGroup = namedtuple('BottleneckGroup', ['num_blocks', 'num_filters', 'bottleneck_size', 'down_sample'])

    # MultiGpu OK
    @staticmethod
    def conv(name, x, filter_size, in_filters, out_filters, strides, padding='SAME', bias=None):
        """Convolution."""
        with tf.variable_scope(name):
            assert len(x.get_shape().as_list()) == 4
            assert x.get_shape().as_list()[3] == in_filters
            assert strides[1] == strides[2]
            n = filter_size * filter_size * out_filters
            shape = [filter_size, filter_size, in_filters, out_filters]
            initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n))
            W = tf.get_variable("{0}_kernel".format(name), shape, initializer=initializer, dtype=tf.float32,
                                trainable=True)
            x_hat = tf.nn.conv2d(x, W, strides, padding=padding)
            if bias is not None:
                b = tf.get_variable("{0}_bias".format(name), [out_filters], initializer=initializer, dtype=tf.float32,
                                    trainable=True)
                x_hat = tf.nn.bias_add(x_hat, bias)
            return x_hat

    # MultiGpu OK
    @staticmethod
    def batch_norm(name, x, is_train, momentum):
        # return tf.identity(x)
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
                x = ResidualNetworkGenerator.batch_norm("init_bn", x, is_train, bn_momentum)
                x = ResidualNetworkGenerator.relu(x, relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope("residual_bn_relu"):
                orig_x = x
                x = ResidualNetworkGenerator.batch_norm("init_bn", x, is_train, bn_momentum)
                x = ResidualNetworkGenerator.relu(x, relu_leakiness)

        with tf.variable_scope("sub1"):
            x = ResidualNetworkGenerator.conv("conv_1", x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope("sub2"):
            x = ResidualNetworkGenerator.batch_norm("bn2", x, is_train, bn_momentum)
            x = ResidualNetworkGenerator.relu(x, relu_leakiness)
            x = ResidualNetworkGenerator.conv("conv2", x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        with tf.variable_scope("sub3"):
            x = ResidualNetworkGenerator.batch_norm("bn3", x, is_train, bn_momentum)
            x = ResidualNetworkGenerator.relu(x, relu_leakiness)
            x = ResidualNetworkGenerator.conv("conv3", x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope("sub_add"):
            if in_filter != out_filter or not all([d == 1 for d in stride]):
                orig_x = ResidualNetworkGenerator.conv("project", orig_x, 1, in_filter, out_filter, stride)
            x += orig_x
        return x

    # MultiGpu OK
    @staticmethod
    def get_input(input, out_filters, first_conv_filter_size):
        assert input.get_shape().ndims == 4
        input_filters = input.get_shape().as_list()[-1]
        x = ResidualNetworkGenerator.conv("init_conv", input, first_conv_filter_size, input_filters, out_filters,
                                          ResidualNetworkGenerator.stride_arr(1))
        return x

    # MultiGpu OK
    @staticmethod
    def get_output(x, is_train, leakiness, bn_momentum):
        x = ResidualNetworkGenerator.batch_norm("final_bn", x, is_train, bn_momentum)
        x = ResidualNetworkGenerator.relu(x, leakiness)
        x = ResidualNetworkGenerator.global_avg_pool(x)
        return x

    @staticmethod
    def generate_resnet_blocks(
            input_net,
            num_of_units_per_block,
            num_of_feature_maps_per_block,
            first_conv_filter_size,
            relu_leakiness,
            stride_list,
            active_before_residuals,
            is_train_tensor,
            batch_norm_decay):
        assert len(num_of_feature_maps_per_block) == len(stride_list) + 1 and \
               len(num_of_feature_maps_per_block) == len(active_before_residuals) + 1
        x = ResidualNetworkGenerator.get_input(input=input_net, out_filters=num_of_feature_maps_per_block[0],
                                               first_conv_filter_size=first_conv_filter_size)
        # Loop over blocks, the resnet trunk
        for block_id in range(len(num_of_feature_maps_per_block) - 1):
            with tf.variable_scope("block_{0}_0".format(block_id)):
                x = ResidualNetworkGenerator.bottleneck_residual(
                    x=x,
                    in_filter=num_of_feature_maps_per_block[block_id],
                    out_filter=num_of_feature_maps_per_block[block_id + 1],
                    stride=ResidualNetworkGenerator.stride_arr(stride_list[block_id]),
                    activate_before_residual=active_before_residuals[block_id],
                    relu_leakiness=relu_leakiness,
                    is_train=is_train_tensor,
                    bn_momentum=batch_norm_decay)
            for i in range(num_of_units_per_block - 1):
                with tf.variable_scope("block_{0}_{1}".format(block_id, i + 1)):
                    x = ResidualNetworkGenerator.bottleneck_residual(
                        x=x,
                        in_filter=num_of_feature_maps_per_block[block_id + 1],
                        out_filter=num_of_feature_maps_per_block[block_id + 1],
                        stride=ResidualNetworkGenerator.stride_arr(1),
                        activate_before_residual=False,
                        relu_leakiness=relu_leakiness,
                        is_train=is_train_tensor,
                        bn_momentum=batch_norm_decay)
        return x
