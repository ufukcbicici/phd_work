import numpy as np
import tensorflow as tf


class DeepQNetworks:
    def __init__(self):
        pass

    @staticmethod
    def get_lenet_network(net_input, is_train, level, class_obj):
        hidden_layers = class_obj.HIDDEN_LAYERS[level]
        conv_features = class_obj.CONV_FEATURES[level]
        filter_sizes = class_obj.FILTER_SIZES[level]
        strides = class_obj.STRIDES[level]
        pools = class_obj.MAX_POOL[level]
        net = net_input
        net = tf.layers.batch_normalization(inputs=net, momentum=0.9, training=is_train)
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
        # net = tf.contrib.layers.flatten(net)
        net_shape = net.get_shape().as_list()
        net = tf.nn.avg_pool(net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
        net_shape = net.get_shape().as_list()
        net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
        for layer_id, layer_dim in enumerate(hidden_layers):
            net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
        class_obj.qFuncs[level] = class_obj.get_q_net_output(net=net, level=level)




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