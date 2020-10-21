import numpy as np
import tensorflow as tf

from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.fashion_net.fashion_cign_lite import FashionCignLite


class DeepQNetworks:
    def __init__(self):
        pass

    @staticmethod
    def get_conv_layers(net_input, node, is_train, conv_features, filter_sizes, strides, pools):
        net = net_input
        conv_layer_id = 0
        for conv_feature, filter_size, stride, max_pool in zip(conv_features, filter_sizes, strides, pools):
            in_filters = net.get_shape().as_list()[-1]
            out_filters = conv_feature
            kernel = [filter_size, filter_size, in_filters, out_filters]
            strides = [1, stride, stride, 1]
            W = tf.get_variable("conv_layer_kernel_{0}".format(conv_layer_id), kernel, trainable=True)
            b = tf.get_variable("conv_layer_bias_{0}".format(conv_layer_id), [kernel[-1]], trainable=True)
            net = FastTreeNetwork.conv_layer(x=net, kernel=W, strides=strides, padding='SAME', bias=b, node=node)
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
    def get_dense_layers(net_input, node, is_train, hidden_layers, name="fc_layer"):
        net = net_input
        for layer_id, layer_dim in enumerate(hidden_layers):
            curr_dim = net.get_shape().as_list[-1]
            W, b = FashionCignLite.get_affine_layer_params(
                layer_shape=[curr_dim, layer_dim],
                W_name="{0}_W_{1}".format(name, layer_id),
                b_name="{0}_b_{1}".format(name, layer_id))
            net = FastTreeNetwork.fc_layer(x=net, W=W, b=b, node=node)
            net = tf.nn.relu(net)
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
                                            strides=strides, pools=pools, node=class_obj.nodes[level])
        net = DeepQNetworks.global_average_pooling(net_input=net)
        net = DeepQNetworks.get_dense_layers(net_input=net, is_train=is_train, hidden_layers=hidden_layers,
                                             node=class_obj.nodes[level])
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
                                                   strides=strides, pools=pools, node=class_obj.nodes[level])
        net = conv_input
        # Global Average Pooling
        net = DeepQNetworks.global_average_pooling(net_input=net)
        net_shape = net.get_shape().as_list()
        # Dimensionality reduction and nonlinearity
        net = DeepQNetworks.get_dense_layers(net_input=net, is_train=is_train,
                                             hidden_layers=[net_shape[-1] // reduction_ratio],
                                             node=class_obj.nodes[level],
                                             name="dim_reduction_and_nonlinearity")
        net = tf.nn.relu(net)
        # Increase dimensionality and apply sigmoid
        net = DeepQNetworks.get_dense_layers(net_input=net, is_train=is_train,
                                             hidden_layers=[net_shape[-1]],
                                             node=class_obj.nodes[level],
                                             name="sigmoid_layer")
        net = tf.nn.sigmoid(net)
        net = tf.expand_dims(tf.expand_dims(net, axis=1), axis=1)
        # Multiply with the input
        net = net * conv_input
        net = DeepQNetworks.global_average_pooling(net_input=net)
        net = DeepQNetworks.get_dense_layers(net_input=net, is_train=is_train, hidden_layers=hidden_layers,
                                             node=class_obj.nodes[level])
        class_obj.qFuncs[level] = class_obj.get_q_net_output(net=net, level=level)
