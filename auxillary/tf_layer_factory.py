import tensorflow as tf

from auxillary.constants import GlobalInputNames, InitType, ChannelTypes, ActivationType, PoolingType, parameterTypes


class TfLayerFactory:
    @staticmethod
    def get_initializer(init_type):
        if init_type == InitType.xavier:
            initializer =  tf.contrib.layers.xavier_initializer()
        else:
            raise NotImplementedError()
        return initializer

    @staticmethod
    def create_convolutional_layer(node, channel, input_tensor, conv_filter_shape, conv_stride_shape, pooling_shape,
                                   conv_padding,
                                   pooling_stride_shape, pooling_padding,
                                   init_type, activation_type, pooling_type, post_fix):
        # Initializers
        conv_filter_initializer = channel.add_operation(op=TfLayerFactory.get_initializer(init_type=init_type))
        conv_bias_initializer = channel.add_operation(op=TfLayerFactory.get_initializer(init_type=init_type))
        # Filters and bias
        W = node.create_variable(name="Convolution_Filter_{0}".format(post_fix), initializer=conv_filter_initializer,
                                 shape=conv_filter_shape, dtype=tf.float32,
                                 arg_type=parameterTypes.learnable_parameter, channel=channel)
        b = node.create_variable(name="Convolution_Bias_{0}".format(post_fix), initializer=conv_bias_initializer,
                                 shape=conv_filter_shape[3], dtype=tf.float32,
                                 arg_type=parameterTypes.learnable_parameter, channel=channel)
        # Operations
        conv_intermediate = channel.add_operation(
            op=tf.nn.conv2d(input_tensor, W, strides=conv_stride_shape, padding=conv_padding))
        conv = channel.add_operation(op=conv_intermediate + b)
        # Nonlinearity
        nonlinear = TfLayerFactory.apply_nonlinearity(channel=channel, input_tensor=conv,
                                                      activation_type=activation_type, post_fix=post_fix)
        # Pooling
        if pooling_type == PoolingType.max:
            pool = channel.add_operation(
                op=tf.nn.max_pool(nonlinear, ksize=pooling_shape, strides=pooling_stride_shape,
                                  padding=pooling_padding))
        else:
            raise NotImplementedError()
        return pool

    @staticmethod
    def create_fc_layer(node, channel, input_tensor, fc_shape, init_type, activation_type, post_fix):
        # Initializers
        fc_filter_initializer = channel.add_operation(op=TfLayerFactory.get_initializer(init_type=init_type))
        fc_bias_initializer = channel.add_operation(op=TfLayerFactory.get_initializer(init_type=init_type))
        # Filters and bias
        W = node.create_variable(name="FullyConnected_Weight_{0}".format(post_fix), initializer=fc_filter_initializer,
                                 shape=fc_shape, dtype=tf.float32,
                                 arg_type=parameterTypes.learnable_parameter, channel=channel)
        b = node.create_variable(name="FullyConnected_Bias_{0}".format(post_fix), initializer=fc_bias_initializer,
                                 shape=fc_shape[1], dtype=tf.float32,
                                 arg_type=parameterTypes.learnable_parameter, channel=channel)
        # Operations
        matmul = channel.add_operation(op=tf.matmul(input_tensor, W))
        fc = channel.add_operation(op=matmul + b)
        # Nonlinearity
        nonlinear = TfLayerFactory.apply_nonlinearity(channel=channel, input_tensor=fc,
                                                      activation_type=activation_type, post_fix=post_fix)
        return nonlinear

    @staticmethod
    def apply_nonlinearity(channel, input_tensor, activation_type, post_fix):
        if activation_type == ActivationType.relu:
            output_tensor = channel.add_operation(op=tf.nn.relu(input_tensor))
        elif activation_type == ActivationType.tanh:
            output_tensor = channel.add_operation(op=tf.nn.tanh(input_tensor))
        elif activation_type == ActivationType.no_activation:
            output_tensor = input_tensor
        else:
            raise NotImplementedError()
        return output_tensor
