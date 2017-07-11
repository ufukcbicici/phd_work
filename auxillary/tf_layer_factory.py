import tensorflow as tf

from auxillary.constants import OperationNames, InitType, OperationTypes, ActivationType, PoolingType, ArgumentTypes


class TfLayerFactory:
    @staticmethod
    def create_convolutional_layer(node, channel, input_tensor, conv_filter_shape, conv_stride_shape, pooling_shape,
                                   conv_padding,
                                   pooling_stride_shape, pooling_padding,
                                   init_type, activation_type, pooling_type, post_fix):
        # Initializers
        if init_type == InitType.xavier:
            conv_filter_initializer = channel.add_operation(op=tf.contrib.layers.xavier_initializer())
            conv_bias_initializer = channel.add_operation(op=tf.contrib.layers.xavier_initializer())
        else:
            raise NotImplementedError()
        # Filters and bias
        W = node.create_variable(name="W{0}".format(post_fix), initializer=conv_filter_initializer,
                                 shape=conv_filter_shape, needs_gradient=True, dtype=tf.float32,
                                 arg_type=ArgumentTypes.learnable_parameter, channel=channel)
        b = node.create_variable(name="b{0}".format(post_fix), initializer=conv_bias_initializer,
                                 shape=conv_filter_shape[3], needs_gradient=True, dtype=tf.float32,
                                 arg_type=ArgumentTypes.learnable_parameter, channel=channel)
        # Operations
        conv_intermediate = channel.add_operation(
            op=tf.nn.conv2d(input_tensor, W, strides=conv_stride_shape, padding=conv_padding,
                            name="conv{0}_intermediate".format(post_fix)))
        conv = channel.add_operation(op=conv_intermediate + b)
        if activation_type == ActivationType.relu:
            relu = channel.add_operation(op=tf.nn.relu(conv, name="relu{0}".format(post_fix)))
        else:
            raise NotImplementedError()
        if pooling_type == PoolingType.max:
            pool = channel.add_operation(
                op=tf.nn.max_pool(relu, ksize=pooling_shape, strides=pooling_stride_shape, padding=pooling_padding,
                                  name="maxpool{0}".format(post_fix)))
        else:
            raise NotImplementedError()
        return pool

    @staticmethod
    def create_fc_layer(node, channel, input_tensor, fc_shape, init_type, activation_type, post_fix):
        # Initializers
        if init_type == InitType.xavier:
            fc_filter_initializer = channel.add_operation(op=tf.contrib.layers.xavier_initializer())
            fc_bias_initializer = channel.add_operation(op=tf.contrib.layers.xavier_initializer())
        else:
            raise NotImplementedError()
        # Filters and bias
        W = node.create_variable(name="A{0}".format(post_fix), initializer=fc_filter_initializer,
                                 shape=fc_shape, needs_gradient=True, dtype=tf.float32,
                                 arg_type=ArgumentTypes.learnable_parameter, channel=channel)
        b = node.create_variable(name="b{0}".format(post_fix), initializer=fc_bias_initializer,
                                 shape=fc_shape[1], needs_gradient=True, dtype=tf.float32,
                                 arg_type=ArgumentTypes.learnable_parameter, channel=channel)
        # Operations
        matmul = channel.add_operation(op=tf.matmul(input_tensor, W, name="matmul{0}".format(post_fix)))
        fc = channel.add_operation(op=matmul + b)
        if activation_type == ActivationType.relu:
            relu = channel.add_operation(op=tf.nn.relu(fc, name="relu{0}".format(post_fix)))
        else:
            raise NotImplementedError()
        return relu
