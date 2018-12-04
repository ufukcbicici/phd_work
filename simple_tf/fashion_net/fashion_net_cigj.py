import tensorflow as tf

from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DecayingParameter, FixedParameter
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.global_params import GlobalConstants


def f_root_func(node, network):
    # Convolution 1
    # OK
    conv1_weights = tf.Variable(
        tf.truncated_normal([GlobalConstants.FASHION_FILTERS_1_SIZE, GlobalConstants.FASHION_FILTERS_1_SIZE,
                             GlobalConstants.NUM_CHANNELS, GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[0]],
                            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=UtilityFuncs.get_variable_name(name="conv1_weight", node=node))
    # OK
    conv1_biases = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[0]], dtype=GlobalConstants.DATA_TYPE),
        name=UtilityFuncs.get_variable_name(name="conv1_bias", node=node))
    # Shared Conv Layer
    network.mask_input_nodes(node=node)
    conv1 = tf.nn.conv2d(network.dataTensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([pool1])


def f_l1_func(node, network):
    # Convolution 2
    # OK
    conv2_weights = tf.Variable(
        tf.truncated_normal([GlobalConstants.FASHION_FILTERS_2_SIZE, GlobalConstants.FASHION_FILTERS_2_SIZE,
                             GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[0],
                             GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[1]],
                            stddev=0.1,
                            seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=UtilityFuncs.get_variable_name(name="conv2_weight", node=node))
    # OK
    conv2_biases = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[1]], dtype=GlobalConstants.DATA_TYPE),
        name=UtilityFuncs.get_variable_name(name="conv2_bias",
                                       node=node))
    # L1 Conv Layer
    parent_F, parent_H = network.mask_input_nodes(node=node)
    conv = tf.nn.conv2d(parent_F, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([pool])


def f_l2_func(node, network):
    # Convolution 2
    # OK
    conv3_weights = tf.Variable(
        tf.truncated_normal([GlobalConstants.FASHION_FILTERS_3_SIZE, GlobalConstants.FASHION_FILTERS_3_SIZE,
                             GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[1],
                             GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[2]],
                            stddev=0.1,
                            seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=UtilityFuncs.get_variable_name(name="conv3_weight", node=node))
    # OK
    conv3_biases = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.CIGJ_FASHION_NET_CONV_CHANNELS[2]], dtype=GlobalConstants.DATA_TYPE),
        name=UtilityFuncs.get_variable_name(name="conv3_bias",
                                       node=node))
    # L1 Conv Layer
    parent_F, parent_H = network.mask_input_nodes(node=node)
    conv = tf.nn.conv2d(parent_F, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([pool])


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
    node.fOpsList.extend([dropped_layer_2])


def f_leaf_func(node, network):
    parent_F, parent_H = network.mask_input_nodes(node=node)


def h_l1_func(node, network):
    h_net = network.stitch_samples(node=node)
    net_shape = h_net.get_shape().as_list()
    # Global Average Pooling
    h_net = tf.nn.avg_pool(h_net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
    net_shape = h_net.get_shape().as_list()
    h_net = tf.reshape(h_net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
    feature_size = h_net.get_shape().as_list()[-1]
    fc_h_weights = tf.Variable(tf.truncated_normal(
        [feature_size, GlobalConstants.CIGJ_FASHION_NET_H_FEATURES[0]],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_decision_weights", node=node))
    fc_h_bias = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.CIGJ_FASHION_NET_H_FEATURES[0]], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_decision_bias", node=node))
    h_net = tf.matmul(h_net, fc_h_weights) + fc_h_bias
    h_net = tf.nn.relu(h_net)
    h_net = tf.nn.dropout(h_net, keep_prob=network.decisionDropoutKeepProb)
    ig_feature = h_net
    node.hOpsList.extend([ig_feature])


def threshold_calculator_func(network):
    # Noise Coefficient
    network.noiseCoefficientCalculator = DecayingParameter(name="noise_coefficient_calculator", value=0.0,
                                                           decay=0.0,
                                                           decay_period=1,
                                                           min_limit=0.0)
    # Decision Loss Coefficient
    # network.decisionLossCoefficientCalculator = DiscreteParameter(name="decision_loss_coefficient_calculator",
    #                                                               value=0.0,
    #                                                               schedule=[(12000, 1.0)])
    network.decisionLossCoefficientCalculator = FixedParameter(name="decision_loss_coefficient_calculator", value=1.0)
    for node in network.topologicalSortedNodes:
        if node.nodeType == NodeType.h_node:


        # if node.isLeaf:
        #     continue
        # # Probability Threshold
        # node_degree = GlobalConstants.TREE_DEGREE_LIST[node.depth]
        # initial_value = 1.0 / float(node_degree)
        # threshold_name = network.get_variable_name(name="prob_threshold_calculator", node=node)
        # node.probThresholdCalculator = DecayingParameter(name=threshold_name, value=initial_value, decay=0.8,
        #                                                  decay_period=12000,
        #                                                  min_limit=0.4)
        # # Softmax Decay
        # decay_name = network.get_variable_name(name="softmax_decay", node=node)
        # node.softmaxDecayCalculator = DecayingParameter(name=decay_name, value=GlobalConstants.SOFTMAX_DECAY_INITIAL,
        #                                                 decay=GlobalConstants.SOFTMAX_DECAY_COEFFICIENT,
        #                                                 decay_period=GlobalConstants.SOFTMAX_DECAY_PERIOD,
        #                                                 min_limit=GlobalConstants.SOFTMAX_DECAY_MIN_LIMIT)