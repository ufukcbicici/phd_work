import numpy as np
import tensorflow as tf
from collections import deque, Counter

from algorithms.threshold_optimization_algorithms.deep_q_networks.q_learning_threshold_optimizer import \
    QLearningThresholdOptimizer
from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.policy_gradients_network import \
    RoutingDataForMDP
from auxillary.general_utility_funcs import UtilityFuncs


class DeepQThresholdOptimizer(QLearningThresholdOptimizer):
    CONV_FEATURES = [[], []]
    HIDDEN_LAYERS = [[128, 64], [128, 64]]
    FILTER_SIZES = [[], []]
    STRIDES = [[], []]
    MAX_POOL = [[None], [None]]

    def __init__(self, validation_data, test_data, network, network_name, run_id, used_feature_names, q_learning_func,
                 lambda_mac_cost):
        super().__init__(validation_data, test_data, network, network_name, run_id, used_feature_names, q_learning_func,
                         lambda_mac_cost)
        self.experienceReplayTable = deque(maxlen=1000000)
        self.stateInputs = []
        self.qFuncs = []
        for level in range(self.get_max_trajectory_length()):
            self.build_q_function(level=level)

    def build_q_function(self, level):
        if level != self.get_max_trajectory_length() - 1:
            self.stateInputs.append(None)
            self.qFuncs.append(None)
        else:
            if self.qLearningFunc == "cnn":
                nodes_at_level = self.network.orderedNodesPerLevel[level]
                entry_shape = list(self.validationFeaturesDict[nodes_at_level[0].index].shape)
                entry_shape[0] = None
                entry_shape[-1] = len(nodes_at_level) * entry_shape[-1]
                tf_state_input = tf.placeholder(dtype=tf.float32, shape=entry_shape, name="inputs_{0}".format(level))
                self.stateInputs.append(tf_state_input)

    def build_cnn_q_network(self, level):
        hidden_layers = DeepQThresholdOptimizer.HIDDEN_LAYERS[level]
        hidden_layers.append(self.actionSpaces[level].shape[0])
        conv_features = DeepQThresholdOptimizer.CONV_FEATURES[level]
        filter_sizes = DeepQThresholdOptimizer.FILTER_SIZES[level]
        strides = DeepQThresholdOptimizer.STRIDES[level]
        pools = DeepQThresholdOptimizer.MAX_POOL[level]

        net = self.stateInputs[level]
        conv_layer_id = 0
        for conv_feature, filter_size, stride, max_pool in zip(conv_features, filter_sizes, strides, pools):
            in_filters = net.get_shape().as_list()[-1]
            out_filters = conv_feature
            kernel = [filter_size, filter_size, in_filters, out_filters]
            strides = [1, stride, stride, 1]
            W = tf.get_variable("conv_layer_kernel_{0}_t{1}".format(conv_layer_id, level), kernel,
                                trainable=True)
            b = tf.get_variable("conv_layer_bias_{0}_t{1}".format(conv_layer_id, level), [kernel[-1]],
                                trainable=True)
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
            if layer_id < len(hidden_layers) - 1:
                net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
            else:
                net = tf.layers.dense(inputs=net, units=layer_dim, activation=None)
        q_values = net
        self.qFuncs.append(q_values)
