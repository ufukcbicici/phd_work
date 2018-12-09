import tensorflow as tf

from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DecayingParameter, FixedParameter
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.global_params import GlobalConstants


def build_conv_layer(input, node, filter_size, num_of_input_channels, num_of_output_channels):
    # OK
    conv_weights = tf.Variable(
        tf.truncated_normal([filter_size, filter_size, num_of_input_channels, num_of_output_channels],
                            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=UtilityFuncs.get_variable_name(name="conv_weight", node=node))
    # OK
    conv_biases = tf.Variable(
        tf.constant(0.1, shape=[num_of_output_channels], dtype=GlobalConstants.DATA_TYPE),
        name=UtilityFuncs.get_variable_name(name="conv_bias", node=node))
    conv = tf.nn.conv2d(input, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pool


def h_transform(input, node, network, h_feature_size):
    h_net = input
    net_shape = h_net.get_shape().as_list()
    # Global Average Pooling
    h_net = tf.nn.avg_pool(h_net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
    net_shape = h_net.get_shape().as_list()
    h_net = tf.reshape(h_net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
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


def f_root_func(node, network):
    network.mask_input_nodes(node=node)
    node.F_output = build_conv_layer(input=network.dataTensor, node=node,
                                     filter_size=GlobalConstants.FASHION_FILTERS_1_SIZE,
                                     num_of_input_channels=GlobalConstants.NUM_CHANNELS,
                                     num_of_output_channels=GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[0])


def f_l1_func(node, network):
    parent_F, parent_H = network.mask_input_nodes(node=node)
    node.F_output = build_conv_layer(input=parent_F, node=node,
                                     filter_size=GlobalConstants.FASHION_FILTERS_2_SIZE,
                                     num_of_input_channels=GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[0],
                                     num_of_output_channels=GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[1])


def f_l2_func(node, network):
    parent_F, parent_H = network.mask_input_nodes(node=node)
    node.F_output = build_conv_layer(input=parent_F, node=node,
                                     filter_size=GlobalConstants.FASHION_FILTERS_3_SIZE,
                                     num_of_input_channels=GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[1],
                                     num_of_output_channels=GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[2])


def f_l3_func(node, network):
    parent_F, parent_H = network.mask_input_nodes(node=node)
    net = tf.contrib.layers.flatten(parent_F)
    flattened_F_feature_size = net.get_shape().as_list()[-1]
    # Parameters
    # OK
    fc_weights_1 = tf.Variable(tf.truncated_normal(
        [flattened_F_feature_size,
         GlobalConstants.CIGJ_FASHION_NET_FC_DIMS[0]],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=UtilityFuncs.get_variable_name(name="fc_weights_1", node=node))
    # OK
    fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.CIGJ_FASHION_NET_FC_DIMS[0]],
                                          dtype=GlobalConstants.DATA_TYPE),
                              name=UtilityFuncs.get_variable_name(name="fc_biases_1", node=node))
    # OK
    fc_weights_2 = tf.Variable(tf.truncated_normal(
        [GlobalConstants.CIGJ_FASHION_NET_FC_DIMS[0],
         GlobalConstants.CIGJ_FASHION_NET_FC_DIMS[1]],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=UtilityFuncs.get_variable_name(name="fc_weights_2", node=node))
    # OK
    fc_biases_2 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.CIGJ_FASHION_NET_FC_DIMS[1]],
                                          dtype=GlobalConstants.DATA_TYPE),
                              name=UtilityFuncs.get_variable_name(name="fc_biases_2", node=node))
    # Layers
    hidden_layer_1 = tf.nn.relu(tf.matmul(net, fc_weights_1) + fc_biases_1)
    dropped_layer_1 = tf.nn.dropout(hidden_layer_1, network.classificationDropoutKeepProb)
    hidden_layer_2 = tf.nn.relu(tf.matmul(dropped_layer_1, fc_weights_2) + fc_biases_2)
    dropped_layer_2 = tf.nn.dropout(hidden_layer_2, network.classificationDropoutKeepProb)
    node.F_output = dropped_layer_2


def f_leaf_func(node, network):
    parent_F, parent_H = network.mask_input_nodes(node=node)


def h_l1_func(node, network):
    h_net, _ = network.stitch_samples(node=node)
    node.H_output = h_transform(input=h_net, network=network, node=node,
                                h_feature_size=GlobalConstants.CIGJ_FASHION_NET_H_FEATURES[0])
    network.apply_decision(node=node, branching_feature=node.H_output)


def h_l2_func(node, network):
    h_net, _ = network.stitch_samples(node=node)
    node.H_output = h_transform(input=h_net, network=network, node=node,
                                h_feature_size=GlobalConstants.CIGJ_FASHION_NET_H_FEATURES[1])
    network.apply_decision(node=node, branching_feature=node.H_output)


def h_l3_func(node, network):
    h_net, _ = network.stitch_samples(node=node)
    node.H_output = h_transform(input=h_net, network=network, node=node,
                                h_feature_size=GlobalConstants.CIGJ_FASHION_NET_H_FEATURES[2])
    network.apply_decision(node=node, branching_feature=node.H_output)


def threshold_calculator_func(network):
    network.decisionLossCoefficientCalculator = FixedParameter(name="decision_loss_coefficient_calculator", value=1.0)
    for node in network.topologicalSortedNodes:
        if node.nodeType == NodeType.h_node:
            # Softmax Decay
            decay_name = network.get_variable_name(name="softmax_decay", node=node)
            node.softmaxDecayCalculator = DecayingParameter(name=decay_name,
                                                            value=GlobalConstants.SOFTMAX_DECAY_INITIAL,
                                                            decay=GlobalConstants.SOFTMAX_DECAY_COEFFICIENT,
                                                            decay_period=GlobalConstants.SOFTMAX_DECAY_PERIOD,
                                                            min_limit=GlobalConstants.SOFTMAX_DECAY_MIN_LIMIT)
