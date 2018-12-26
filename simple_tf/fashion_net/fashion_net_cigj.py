import tensorflow as tf

from auxillary.constants import DatasetTypes
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DecayingParameter, FixedParameter
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.global_params import GlobalConstants


class FashionNetCigj:
    LAYER_SHAPES = []

    @staticmethod
    def build_conv_layer(input, node, filter_size, num_of_input_channels, num_of_output_channels, name_suffix=""):
        # OK
        conv_weights = tf.Variable(
            tf.truncated_normal([filter_size, filter_size, num_of_input_channels, num_of_output_channels],
                                stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=UtilityFuncs.get_variable_name(name="conv_weight{0}".format(name_suffix), node=node))
        # OK
        conv_biases = tf.Variable(
            tf.constant(0.1, shape=[num_of_output_channels], dtype=GlobalConstants.DATA_TYPE),
            name=UtilityFuncs.get_variable_name(name="conv_bias{0}".format(name_suffix), node=node))
        conv = tf.nn.conv2d(input, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool

    @staticmethod
    def build_fc_layer(input, node, input_dim, output_dim, dropout_prob_tensor, name_suffix=""):
        fc_weights = tf.Variable(tf.truncated_normal(
            [input_dim, output_dim],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=UtilityFuncs.get_variable_name(name="fc_weights{0}".format(name_suffix), node=node))
        # OK
        fc_biases = tf.Variable(tf.constant(0.1, shape=[output_dim], dtype=GlobalConstants.DATA_TYPE),
                                name=UtilityFuncs.get_variable_name(name="fc_biases{0}".format(name_suffix), node=node))
        hidden_layer = tf.nn.relu(tf.matmul(input, fc_weights) + fc_biases)
        dropped_layer = tf.nn.dropout(hidden_layer, dropout_prob_tensor)
        return dropped_layer

    @staticmethod
    def h_transform(input, node, network, h_feature_size, pool_size):
        h_net = input
        # Parametric Average Pooling if the input layer is convolutional
        assert len(h_net.get_shape().as_list()) == 2 or len(h_net.get_shape().as_list()) == 4
        if len(h_net.get_shape().as_list()) == 4:
            h_net = tf.nn.avg_pool(h_net, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1],
                                   padding='SAME')
            h_net = UtilityFuncs.tf_safe_flatten(input_tensor=h_net)
            # h_net = tf.contrib.layers.flatten(h_net)
        feature_size = h_net.get_shape().as_list()[-1]
        fc_h_weights = tf.Variable(tf.truncated_normal(
            [feature_size, h_feature_size],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_decision_weights", node=node))
        fc_h_bias = tf.Variable(
            tf.constant(0.1, shape=[h_feature_size], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_decision_bias", node=node))
        h_net = tf.matmul(h_net, fc_h_weights) + fc_h_bias
        h_net = tf.nn.relu(h_net)
        h_net = tf.nn.dropout(h_net, keep_prob=network.decisionDropoutKeepProb)
        ig_feature = h_net
        return ig_feature

    @staticmethod
    def f_conv_layer_func(node, network):
        network.mask_input_nodes(node=node)
        filter_size = GlobalConstants.CIGJ_FASHION_NET_CONV_FILTER_SIZES[node.depth]
        num_of_input_channels = GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[node.depth]
        num_of_output_channels = GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[node.depth + 1]
        node.F_output = FashionNetCigj.build_conv_layer(input=node.F_input,
                                                        node=node,
                                                        filter_size=filter_size,
                                                        num_of_input_channels=num_of_input_channels,
                                                        num_of_output_channels=num_of_output_channels)

    @staticmethod
    def f_l3_func(node, network):
        network.mask_input_nodes(node=node)
        # net = tf.contrib.layers.flatten(node.F_input)
        net = UtilityFuncs.tf_safe_flatten(input_tensor=node.F_input)
        flattened_F_feature_size = net.get_shape().as_list()[-1]
        dimensions = [flattened_F_feature_size]
        dimensions.extend(GlobalConstants.CIGJ_FASHION_NET_FC_DIMS)
        for layer in range(len(dimensions) - 1):
            net = FashionNetCigj.build_fc_layer(input=net, node=node,
                                                input_dim=dimensions[layer],
                                                output_dim=dimensions[layer + 1],
                                                dropout_prob_tensor=network.classificationDropoutKeepProb,
                                                name_suffix="{0}".format(layer))
        node.F_output = net

    @staticmethod
    def f_leaf_func(node, network):
        network.mask_input_nodes(node=node)
        final_feature = node.F_input
        network.apply_loss_jungle(node=node, final_feature=final_feature)

    @staticmethod
    def h_func(node, network):
        network.stitch_samples(node=node)
        node_degree = network.degreeList[node.depth + 1]
        if node_degree > 1:
            h_feature_size = GlobalConstants.CIGJ_FASHION_NET_H_FEATURES[node.depth]
            pool_size = GlobalConstants.CIGJ_FASHION_NET_H_POOL_SIZES[node.depth]
            node.H_output = FashionNetCigj.h_transform(input=node.F_input, network=network, node=node,
                                                       h_feature_size=h_feature_size,
                                                       pool_size=pool_size)
        else:
            node.H_output = node.H_input
        network.apply_decision(node=node, branching_feature=node.H_output)

    @staticmethod
    def threshold_calculator_func(network):
        network.decisionLossCoefficientCalculator = FixedParameter(name="decision_loss_coefficient_calculator",
                                                                   value=1.0)
        for node in network.topologicalSortedNodes:
            if node.nodeType == NodeType.h_node:
                # Softmax Decay
                decay_name = network.get_variable_name(name="softmax_decay", node=node)
                node.softmaxDecayCalculator = DecayingParameter(name=decay_name,
                                                                value=GlobalConstants.SOFTMAX_DECAY_INITIAL,
                                                                decay=GlobalConstants.SOFTMAX_DECAY_COEFFICIENT,
                                                                decay_period=GlobalConstants.SOFTMAX_DECAY_PERIOD,
                                                                min_limit=GlobalConstants.SOFTMAX_DECAY_MIN_LIMIT)
