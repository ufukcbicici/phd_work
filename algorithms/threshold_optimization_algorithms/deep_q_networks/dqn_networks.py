import numpy as np
import tensorflow as tf


class DeepQNetworks:
    def __init__(self):
        pass

    @staticmethod
    def get_conv_layers(net_input, is_train, conv_features, filter_sizes, strides, pools):
        net = net_input
        conv_layer_id = 0
        for conv_feature, filter_size, stride, max_pool in zip(conv_features, filter_sizes, strides, pools):
            in_filters = net.get_shape().as_list()[-1]
            out_filters = conv_feature
            kernel = [filter_size, filter_size, in_filters, out_filters]
            strides = [1, stride, stride, 1]
            W = tf.get_variable("conv_layer_kernel_{0}".format(conv_layer_id), kernel, trainable=True)
            b = tf.get_variable("conv_layer_bias_{0}".format(conv_layer_id), [kernel[-1]], trainable=True)
            net = tf.nn.conv2d(net, W, strides, padding='SAME')
            net = tf.nn.bias_add(net, b)
            net = tf.nn.relu(net)
            if max_pool is not None:
                net = tf.nn.max_pool(net, ksize=[1, max_pool, max_pool, 1], strides=[1, max_pool, max_pool, 1],
                                     padding='SAME')
            conv_layer_id += 1
        return net

    @staticmethod
    def global_average_pooling(net_input):
        net = net_input
        net_shape = net.get_shape().as_list()
        net = tf.nn.avg_pool(net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
        net_shape = net.get_shape().as_list()
        net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
        return net

    @staticmethod
    def get_dense_layers(net_input, is_train, hidden_layers):
        net = net_input
        for layer_id, layer_dim in enumerate(hidden_layers):
            net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
        return net

    @staticmethod
    def get_lenet_network(net_input, is_train, level, class_obj):
        hidden_layers = class_obj.HIDDEN_LAYERS[level]
        conv_features = class_obj.CONV_FEATURES[level]
        filter_sizes = class_obj.FILTER_SIZES[level]
        strides = class_obj.STRIDES[level]
        pools = class_obj.MAX_POOL[level]
        net = net_input
        net = tf.layers.batch_normalization(inputs=net, momentum=0.9, training=is_train)
        net = DeepQNetworks.get_conv_layers(net_input=net, is_train=is_train,
                                            conv_features=conv_features, filter_sizes=filter_sizes,
                                            strides=strides, pools=pools)
        net = DeepQNetworks.global_average_pooling(net_input=net)
        net = DeepQNetworks.get_dense_layers(net_input=net, is_train=is_train, hidden_layers=hidden_layers)
        class_obj.qFuncs[level] = class_obj.get_q_net_output(net=net, level=level)

    @staticmethod
    def get_squeeze_and_excitation_block(net_input, is_train, level, class_obj):
        hidden_layers = class_obj.HIDDEN_LAYERS[level]
        conv_features = class_obj.CONV_FEATURES[level]
        filter_sizes = class_obj.FILTER_SIZES[level]
        strides = class_obj.STRIDES[level]
        pools = class_obj.MAX_POOL[level]
        net = net_input
        net = tf.layers.batch_normalization(inputs=net, momentum=0.9, training=is_train)
        reduction_ratio = class_obj.SE_REDUCTION_RATIO[level]
        conv_input = DeepQNetworks.get_conv_layers(net_input=net, is_train=is_train,
                                                   conv_features=conv_features, filter_sizes=filter_sizes,
                                                   strides=strides, pools=pools)
        net = conv_input
        # Global Average Pooling
        net = DeepQNetworks.global_average_pooling(net_input=net)
        net_shape = net.get_shape().as_list()
        # Dimensionality reduction and nonlinearity
        net = tf.layers.dense(inputs=net, units=net_shape[-1] // reduction_ratio, activation=tf.nn.relu,
                              name="dim_reduction_and_nonlinearity")
        # Increase dimensionality and apply sigmoid
        net = tf.layers.dense(inputs=net, units=net_shape[-1], activation=tf.nn.sigmoid, name="sigmoid_layer")
        net = tf.expand_dims(tf.expand_dims(net, axis=1), axis=1)
        # Multiply with the input
        net = net * conv_input
        net = DeepQNetworks.global_average_pooling(net_input=net)
        net = DeepQNetworks.get_dense_layers(net_input=net, is_train=is_train, hidden_layers=hidden_layers)
        class_obj.qFuncs[level] = class_obj.get_q_net_output(net=net, level=level)

        # x = GlobalAveragePooling2D()(in_block)
        # x = Dense(ch // ratio, activation='relu')(x)
        # x = Dense(ch, activation='sigmoid')(x)
        # return multiply()([in_block, x])

        # conv_layer_id = 0
        # for conv_feature, filter_size, stride, max_pool in zip(conv_features, filter_sizes, strides, pools):
        #     in_filters = net.get_shape().as_list()[-1]
        #     out_filters = conv_feature
        #     kernel = [filter_size, filter_size, in_filters, out_filters]
        #     strides = [1, stride, stride, 1]
        #     W = tf.get_variable("conv_layer_kernel_{0}".format(conv_layer_id), kernel, trainable=True)
        #     b = tf.get_variable("conv_layer_bias_{0}".format(conv_layer_id), [kernel[-1]], trainable=True)
        #     net = tf.nn.conv2d(net, W, strides, padding='SAME')
        #     net = tf.nn.bias_add(net, b)
        #     net = tf.nn.relu(net)
        #     if max_pool is not None:
        #         net = tf.nn.max_pool(net, ksize=[1, max_pool, max_pool, 1], strides=[1, max_pool, max_pool, 1],
        #                              padding='SAME')
        #     conv_layer_id += 1
        # # net = tf.contrib.layers.flatten(net)
        # net_shape = net.get_shape().as_list()
        # net = tf.nn.avg_pool(net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
        # net_shape = net.get_shape().as_list()
        # net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
        # for layer_id, layer_dim in enumerate(hidden_layers):
        #     net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
        # class_obj.qFuncs[level] = class_obj.get_q_net_output(net=net, level=level)

    # def build_cnn_q_network(self, level):
    #     hidden_layers = DqnWithRegression.HIDDEN_LAYERS[level]
    #     # hidden_layers.append(self.actionSpaces[level].shape[0])
    #     conv_features = DqnWithRegression.CONV_FEATURES[level]
    #     filter_sizes = DqnWithRegression.FILTER_SIZES[level]
    #     strides = DqnWithRegression.STRIDES[level]
    #     pools = DqnWithRegression.MAX_POOL[level]
    #     net = self.stateInputs[level]
    #     net = tf.layers.batch_normalization(inputs=net,
    #                                         momentum=0.9,
    #                                         training=self.isTrain)
    #     conv_layer_id = 0
    #     for conv_feature, filter_size, stride, max_pool in zip(conv_features, filter_sizes, strides, pools):
    #         in_filters = net.get_shape().as_list()[-1]
    #         out_filters = conv_feature
    #         kernel = [filter_size, filter_size, in_filters, out_filters]
    #         strides = [1, stride, stride, 1]
    #         W = tf.get_variable("conv_layer_kernel_{0}".format(conv_layer_id), kernel, trainable=True)
    #         b = tf.get_variable("conv_layer_bias_{0}".format(conv_layer_id), [kernel[-1]], trainable=True)
    #         net = tf.nn.conv2d(net, W, strides, padding='SAME')
    #         net = tf.nn.bias_add(net, b)
    #         net = tf.nn.relu(net)
    #         if max_pool is not None:
    #             net = tf.nn.max_pool(net, ksize=[1, max_pool, max_pool, 1], strides=[1, max_pool, max_pool, 1],
    #                                  padding='SAME')
    #         conv_layer_id += 1
    #     # net = tf.contrib.layers.flatten(net)
    #     net_shape = net.get_shape().as_list()
    #     net = tf.nn.avg_pool(net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
    #     net_shape = net.get_shape().as_list()
    #     net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
    #     for layer_id, layer_dim in enumerate(hidden_layers):
    #         net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
    #     self.qFuncs[level] = self.get_q_net_output(net=net, level=level)
