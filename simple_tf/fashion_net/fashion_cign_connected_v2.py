import tensorflow as tf

from auxillary.parameters import DiscreteParameter, DecayingParameter, FixedParameter
from simple_tf.global_params import GlobalConstants
from simple_tf.cign.fast_tree import FastTreeNetwork


class FashionCignV2(FastTreeNetwork):
    def __init__(self, degree_list, dataset):
        node_build_funcs = [self.root_func, self.l1_func, self.leaf_func]
        super().__init__(node_build_funcs, None, None, None, None, degree_list, dataset)

    def apply_heavy_router_transform(self, net, node, decision_feature_size):
        flat_pool = tf.contrib.layers.flatten(net)
        feature_size = flat_pool.get_shape().as_list()[-1]
        fc_h_weights = tf.Variable(tf.truncated_normal(
            [feature_size, decision_feature_size],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="fc_decision_weights", node=node))
        fc_h_bias = tf.Variable(
            tf.constant(0.1, shape=[decision_feature_size], dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="fc_decision_bias", node=node))
        node.variablesSet.add(fc_h_weights)
        node.variablesSet.add(fc_h_bias)
        raw_ig_feature = tf.matmul(flat_pool, fc_h_weights) + fc_h_bias
        # ***************** Dropout *****************
        relu_ig_feature = tf.nn.relu(raw_ig_feature)
        dropped_ig_feature = tf.nn.dropout(relu_ig_feature, keep_prob=self.decisionDropoutKeepProb)
        ig_feature = dropped_ig_feature
        # ***************** Dropout *****************
        # node.hOpsList.extend([pool_h, flat_pool, raw_ig_feature, relu_ig_feature, drooped_ig_feature, ig_feature])
        node.hOpsList.extend([flat_pool, raw_ig_feature, relu_ig_feature, dropped_ig_feature, ig_feature])
        # node.hOpsList.extend([flat_pool, raw_ig_feature, relu_ig_feature, ig_feature])
        # Decisions
        if GlobalConstants.USE_UNIFIED_BATCH_NORM:
            self.apply_decision_with_unified_batch_norm(node=node, branching_feature=ig_feature)
        else:
            self.apply_decision(node=node, branching_feature=ig_feature)

    def apply_router_transformation(self, net, node, decision_feature_size):
        h_net = net
        net_shape = h_net.get_shape().as_list()
        # Global Average Pooling
        h_net = tf.nn.avg_pool(h_net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
        net_shape = h_net.get_shape().as_list()
        h_net = tf.reshape(h_net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
        feature_size = h_net.get_shape().as_list()[-1]
        fc_h_weights = tf.Variable(tf.truncated_normal(
            [feature_size, decision_feature_size],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="fc_decision_weights", node=node))
        fc_h_bias = tf.Variable(
            tf.constant(0.1, shape=[decision_feature_size], dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="fc_decision_bias", node=node))
        node.variablesSet.add(fc_h_weights)
        node.variablesSet.add(fc_h_bias)
        h_net = tf.matmul(h_net, fc_h_weights) + fc_h_bias
        h_net = tf.nn.relu(h_net)
        h_net = tf.nn.dropout(h_net, keep_prob=self.decisionDropoutKeepProb)
        ig_feature = h_net
        node.hOpsList.extend([ig_feature])
        # Decisions
        if GlobalConstants.USE_UNIFIED_BATCH_NORM:
            self.apply_decision_with_unified_batch_norm(node=node, branching_feature=ig_feature)
        else:
            self.apply_decision(node=node, branching_feature=ig_feature)

    def root_func(self, node):
        # Parameters
        # Convolution 1
        # OK
        conv1_weights = tf.Variable(
            tf.truncated_normal([GlobalConstants.FASHION_FILTERS_1_SIZE, GlobalConstants.FASHION_FILTERS_1_SIZE,
                                 GlobalConstants.NUM_CHANNELS, GlobalConstants.FASHION_F_NUM_FILTERS_1], stddev=0.1,
                                seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE), name=self.get_variable_name(name="conv1_weight",
                                                                                              node=node))
        # OK
        conv1_biases = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.FASHION_F_NUM_FILTERS_1], dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="conv1_bias",
                                        node=node))

        node.variablesSet = {conv1_weights, conv1_biases}
        # ***************** F: Convolution Layers *****************
        # First Conv Layer
        self.mask_input_nodes(node=node)
        net = tf.nn.conv2d(self.dataTensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv1_biases))
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        node.fOpsList.extend([net])
        # ***************** F: Convolution Layers *****************

        # ***************** H: Connected to F *****************
        self.apply_router_transformation(net=net, node=node,
                                         decision_feature_size=GlobalConstants.FASHION_NO_H_FROM_F_UNITS_1)
        # ***************** H: Connected to F *****************

    def l1_func(self, node):
        # Parameters
        # Convolution 2
        # OK
        conv_weights = tf.Variable(
            tf.truncated_normal([GlobalConstants.FASHION_FILTERS_2_SIZE, GlobalConstants.FASHION_FILTERS_2_SIZE,
                                 GlobalConstants.FASHION_F_NUM_FILTERS_1, GlobalConstants.FASHION_F_NUM_FILTERS_2],
                                stddev=0.1,
                                seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="conv2_weight", node=node))
        # OK
        conv_biases = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.FASHION_F_NUM_FILTERS_2], dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="conv2_bias",
                                        node=node))
        node.variablesSet = {conv_weights, conv_biases}
        # ***************** F: Convolution Layer *****************
        parent_F, parent_H = self.mask_input_nodes(node=node)
        net = tf.nn.conv2d(parent_F, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv_biases))
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        node.fOpsList.extend([net])
        # ***************** F: Convolution Layer *****************

        # ***************** H: Connected to F *****************
        self.apply_router_transformation(net=net, node=node,
                                         decision_feature_size=GlobalConstants.FASHION_NO_H_FROM_F_UNITS_2)
        # ***************** H: Connected to F *****************

    def leaf_func(self, node):
        softmax_input_dim = GlobalConstants.FASHION_F_FC_2
        conv_weights = tf.Variable(
            tf.truncated_normal([GlobalConstants.FASHION_FILTERS_3_SIZE, GlobalConstants.FASHION_FILTERS_3_SIZE,
                                 GlobalConstants.FASHION_F_NUM_FILTERS_2, GlobalConstants.FASHION_F_NUM_FILTERS_3],
                                stddev=0.1,
                                seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="conv3_weight", node=node))
        # OK
        conv_biases = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.FASHION_F_NUM_FILTERS_3], dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="conv3_bias",
                                        node=node))
        node.variablesSet = {conv_weights, conv_biases}
        # ***************** F: Convolution Layer *****************
        # Conv Layer
        parent_F, parent_H = self.mask_input_nodes(node=node)
        net = tf.nn.conv2d(parent_F, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv_biases))
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # FC Layers
        net = tf.contrib.layers.flatten(net)
        flattened_F_feature_size = net.get_shape().as_list()[-1]
        # Parameters
        # OK
        fc_weights_1 = tf.Variable(tf.truncated_normal(
            [flattened_F_feature_size,
             GlobalConstants.FASHION_F_FC_1],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="fc_weights_1", node=node))
        # OK
        fc_biases_1 = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.FASHION_F_FC_1], dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="fc_biases_1", node=node))
        # OK
        fc_weights_2 = tf.Variable(tf.truncated_normal(
            [GlobalConstants.FASHION_F_FC_1,
             GlobalConstants.FASHION_F_FC_2],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="fc_weights_2", node=node))
        # OK
        fc_biases_2 = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.FASHION_F_FC_2], dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="fc_biases_2", node=node))
        # OK
        fc_softmax_weights = tf.Variable(
            tf.truncated_normal([softmax_input_dim, GlobalConstants.NUM_LABELS],
                                stddev=0.1,
                                seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="fc_softmax_weights", node=node))
        # OK
        fc_softmax_biases = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS], dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="fc_softmax_biases", node=node))
        node.variablesSet = {fc_weights_1, fc_biases_1, fc_weights_2, fc_biases_2, fc_softmax_weights,
                             fc_softmax_biases}
        # OPS
        hidden_layer_1 = tf.nn.relu(tf.matmul(net, fc_weights_1) + fc_biases_1)
        dropped_layer_1 = tf.nn.dropout(hidden_layer_1, self.classificationDropoutKeepProb)
        hidden_layer_2 = tf.nn.relu(tf.matmul(dropped_layer_1, fc_weights_2) + fc_biases_2)
        dropped_layer_2 = tf.nn.dropout(hidden_layer_2, self.classificationDropoutKeepProb)
        final_feature, logits = self.apply_loss(node=node, final_feature=dropped_layer_2,
                                                softmax_weights=fc_softmax_weights, softmax_biases=fc_softmax_biases)
        node.fOpsList.extend([net])
        # Evaluation
        node.evalDict[self.get_variable_name(name="final_eval_feature", node=node)] = final_feature
        node.evalDict[self.get_variable_name(name="logits", node=node)] = logits
        node.evalDict[self.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
        node.evalDict[self.get_variable_name(name="fc_softmax_weights", node=node)] = fc_softmax_weights
        node.evalDict[self.get_variable_name(name="fc_softmax_biases", node=node)] = fc_softmax_biases
        # ***************** F: Convolution Layer *****************

    def residue_network_func(self):
        pass
        # all_residue_features, input_labels, input_indices = self.prepare_residue_input_tensors()
        # self.residueInputTensor = all_residue_features  # tf.stop_gradient(all_residue_features)
        # # Residue Network Parameters
        # variable_list = []
        # curr_layer = self.residueInputTensor
        # for layer_index in range(GlobalConstants.FASHION_F_RESIDUE_LAYER_COUNT):
        #     input_dim = curr_layer.get_shape().as_list()[-1]
        #     fc_residue_weights = tf.Variable(
        #         tf.truncated_normal([input_dim, GlobalConstants.FASHION_F_RESIDUE], stddev=0.1,
        #                             seed=GlobalConstants.SEED,
        #                             dtype=GlobalConstants.DATA_TYPE), name="fc_residue_weights_{0}".format(layer_index))
        #     fc_residue_bias = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.FASHION_F_RESIDUE],
        #                                               dtype=GlobalConstants.DATA_TYPE),
        #                                   name="fc_residue_bias_{0}".format(layer_index))
        #     variable_list.extend([fc_residue_weights, fc_residue_bias])
        #     curr_layer = tf.nn.relu(tf.matmul(curr_layer, fc_residue_weights) + fc_residue_bias)
        #     if GlobalConstants.FASHION_F_RESIDUE_USE_DROPOUT:
        #         curr_layer = tf.nn.dropout(curr_layer, keep_prob=self.classificationDropoutKeepProb)
        # # Loss layer
        # input_dim = curr_layer.get_shape().as_list()[-1]
        # fc_residue_softmax_weights = tf.Variable(
        #     tf.truncated_normal([input_dim, GlobalConstants.NUM_LABELS], stddev=0.1, seed=GlobalConstants.SEED,
        #                         dtype=GlobalConstants.DATA_TYPE), name="fc_residue_final_weights")
        # fc_residue_softmax_bias = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS],
        #                                                   dtype=GlobalConstants.DATA_TYPE),
        #                                       name="fc_residue_final_bias")
        # variable_list.extend([fc_residue_softmax_weights, fc_residue_softmax_bias])
        # self.variableManager.add_variables_to_node(node=None, tf_variables=variable_list)
        # curr_layer = tf.nn.dropout(curr_layer, keep_prob=self.classificationDropoutKeepProb)
        # residue_logits = tf.matmul(curr_layer, fc_residue_softmax_weights) + fc_residue_softmax_bias
        # cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_labels,
        #                                                                            logits=residue_logits)
        # loss = tf.reduce_mean(cross_entropy_loss_tensor)
        # self.evalDict["residue_probabilities"] = tf.nn.softmax(residue_logits)
        # self.evalDict["residue_labels"] = input_labels
        # self.evalDict["residue_indices"] = input_indices
        # self.evalDict["residue_features"] = self.residueInputTensor
        # return loss
        # return tf.constant(value=0.0)

    def set_hyperparameters(self, **kwargs):
        # Training Parameters
        GlobalConstants.TOTAL_EPOCH_COUNT = 100
        GlobalConstants.EPOCH_COUNT = 100
        GlobalConstants.EPOCH_REPORT_PERIOD = 1
        GlobalConstants.BATCH_SIZE = 125
        GlobalConstants.EVAL_BATCH_SIZE = 250
        GlobalConstants.USE_MULTI_GPU = False
        GlobalConstants.INITIAL_LR = 0.01
        GlobalConstants.LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator",
                                                                     value=GlobalConstants.INITIAL_LR,
                                                                     schedule=[(15000, 0.005),
                                                                               (30000, 0.0025),
                                                                               (40000, 0.00025)])
        # Noise Coefficient
        self.noiseCoefficientCalculator = DecayingParameter(name="noise_coefficient_calculator", value=0.0,
                                                            decay=0.0,
                                                            decay_period=1,
                                                            min_limit=0.0)
        # Decision Loss Coefficient
        # network.decisionLossCoefficientCalculator = DiscreteParameter(name="decision_loss_coefficient_calculator",
        #                                                               value=0.0,
        #                                                               schedule=[(12000, 1.0)])
        self.decisionLossCoefficientCalculator = FixedParameter(name="decision_loss_coefficient_calculator",
                                                                value=1.0)
        for node in self.topologicalSortedNodes:
            if node.isLeaf:
                continue
            # Probability Threshold
            node_degree = GlobalConstants.TREE_DEGREE_LIST[node.depth]
            initial_value = 1.0 / float(node_degree)
            threshold_name = self.get_variable_name(name="prob_threshold_calculator", node=node)
            node.probThresholdCalculator = DecayingParameter(name=threshold_name, value=initial_value, decay=0.8,
                                                             decay_period=12000,
                                                             min_limit=0.4)
            # Softmax Decay
            decay_name = self.get_variable_name(name="softmax_decay", node=node)
            node.softmaxDecayCalculator = DecayingParameter(name=decay_name,
                                                            value=GlobalConstants.SOFTMAX_DECAY_INITIAL,
                                                            decay=GlobalConstants.SOFTMAX_DECAY_COEFFICIENT,
                                                            decay_period=GlobalConstants.SOFTMAX_DECAY_PERIOD,
                                                            min_limit=GlobalConstants.SOFTMAX_DECAY_MIN_LIMIT)
        GlobalConstants.SOFTMAX_TEST_TEMPERATURE = 50.0
