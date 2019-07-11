import tensorflow as tf
import numpy as np
from auxillary.parameters import DiscreteParameter, DecayingParameter, FixedParameter
from simple_tf.global_params import GlobalConstants
from simple_tf.cign.fast_tree import FastTreeNetwork


class FashionCignLite(FastTreeNetwork):
    def __init__(self, degree_list, dataset):
        node_build_funcs = [FashionCignLite.root_func, FashionCignLite.l1_func, FashionCignLite.leaf_func]
        super().__init__(node_build_funcs, None, None, None, None, degree_list, dataset)

    @staticmethod
    def apply_heavy_router_transform(network, net, node, decision_feature_size):
        flat_pool = tf.contrib.layers.flatten(net)
        feature_size = flat_pool.get_shape().as_list()[-1]
        fc_h_weights = tf.Variable(tf.truncated_normal(
            [feature_size, decision_feature_size],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_decision_weights", node=node))
        fc_h_bias = tf.Variable(
            tf.constant(0.1, shape=[decision_feature_size], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_decision_bias", node=node))
        node.variablesSet.add(fc_h_weights)
        node.variablesSet.add(fc_h_bias)
        raw_ig_feature = tf.matmul(flat_pool, fc_h_weights) + fc_h_bias
        # ***************** Dropout *****************
        relu_ig_feature = tf.nn.relu(raw_ig_feature)
        dropped_ig_feature = tf.nn.dropout(relu_ig_feature, keep_prob=network.decisionDropoutKeepProb)
        ig_feature = dropped_ig_feature
        # ***************** Dropout *****************
        # node.hOpsList.extend([pool_h, flat_pool, raw_ig_feature, relu_ig_feature, drooped_ig_feature, ig_feature])
        node.hOpsList.extend([flat_pool, raw_ig_feature, relu_ig_feature, dropped_ig_feature, ig_feature])
        # node.hOpsList.extend([flat_pool, raw_ig_feature, relu_ig_feature, ig_feature])
        # Decisions
        if GlobalConstants.USE_UNIFIED_BATCH_NORM:
            network.apply_decision_with_unified_batch_norm(node=node, branching_feature=ig_feature)
        else:
            network.apply_decision(node=node, branching_feature=ig_feature)

    @staticmethod
    def apply_router_transformation(network, net, node, decision_feature_size):
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
            name=network.get_variable_name(name="fc_decision_weights", node=node))
        fc_h_bias = tf.Variable(
            tf.constant(0.1, shape=[decision_feature_size], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_decision_bias", node=node))
        node.variablesSet.add(fc_h_weights)
        node.variablesSet.add(fc_h_bias)
        h_net = tf.matmul(h_net, fc_h_weights) + fc_h_bias
        h_net = tf.nn.relu(h_net)
        h_net = tf.nn.dropout(h_net, keep_prob=network.decisionDropoutKeepProb)
        ig_feature = h_net
        node.hOpsList.extend([ig_feature])
        # Decisions
        if GlobalConstants.USE_UNIFIED_BATCH_NORM:
            network.apply_decision_with_unified_batch_norm(node=node, branching_feature=ig_feature)
        else:
            network.apply_decision(node=node, branching_feature=ig_feature)

    @staticmethod
    def root_func(network, node):
        # Parameters
        # Convolution 1
        # OK
        conv1_weights = tf.Variable(
            tf.truncated_normal([GlobalConstants.FASHION_FILTERS_1_SIZE, GlobalConstants.FASHION_FILTERS_1_SIZE,
                                 GlobalConstants.NUM_CHANNELS, GlobalConstants.FASHION_F_NUM_FILTERS_1], stddev=0.1,
                                seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE), name=network.get_variable_name(name="conv1_weight",
                                                                                                 node=node))
        # OK
        conv1_biases = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.FASHION_F_NUM_FILTERS_1], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv1_bias",
                                           node=node))

        node.variablesSet = {conv1_weights, conv1_biases}
        # ***************** F: Convolution Layers *****************
        # First Conv Layer
        network.mask_input_nodes(node=node)
        net = tf.nn.conv2d(network.dataTensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv1_biases))
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        node.fOpsList.extend([net])
        # ***************** F: Convolution Layers *****************

        # ***************** H: Connected to F *****************
        FashionCignLite.apply_router_transformation(network=network, net=net, node=node,
                                                    decision_feature_size=GlobalConstants.FASHION_NO_H_FROM_F_UNITS_1)
        # ***************** H: Connected to F *****************

    @staticmethod
    def l1_func(network, node):
        # Parameters
        # Convolution 2
        # OK
        conv_weights = tf.Variable(
            tf.truncated_normal([GlobalConstants.FASHION_FILTERS_2_SIZE, GlobalConstants.FASHION_FILTERS_2_SIZE,
                                 GlobalConstants.FASHION_F_NUM_FILTERS_1, GlobalConstants.FASHION_F_NUM_FILTERS_2],
                                stddev=0.1,
                                seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv2_weight", node=node))
        # OK
        conv_biases = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.FASHION_F_NUM_FILTERS_2], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv2_bias",
                                           node=node))
        node.variablesSet = {conv_weights, conv_biases}
        # ***************** F: Convolution Layer *****************
        parent_F, parent_H = network.mask_input_nodes(node=node)
        net = tf.nn.conv2d(parent_F, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv_biases))
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        node.fOpsList.extend([net])
        # ***************** F: Convolution Layer *****************

        # ***************** H: Connected to F *****************
        network.apply_router_transformation(network=network, net=net, node=node,
                                            decision_feature_size=GlobalConstants.FASHION_NO_H_FROM_F_UNITS_2)
        # ***************** H: Connected to F *****************

    @staticmethod
    def leaf_func(network, node):
        softmax_input_dim = GlobalConstants.FASHION_F_FC_2
        conv_weights = tf.Variable(
            tf.truncated_normal([GlobalConstants.FASHION_FILTERS_3_SIZE, GlobalConstants.FASHION_FILTERS_3_SIZE,
                                 GlobalConstants.FASHION_F_NUM_FILTERS_2, GlobalConstants.FASHION_F_NUM_FILTERS_3],
                                stddev=0.1,
                                seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv3_weight", node=node))
        # OK
        conv_biases = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.FASHION_F_NUM_FILTERS_3], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv3_bias",
                                           node=node))
        node.variablesSet = {conv_weights, conv_biases}
        # ***************** F: Convolution Layer *****************
        # Conv Layer
        parent_F, parent_H = network.mask_input_nodes(node=node)
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
            name=network.get_variable_name(name="fc_weights_1", node=node))
        # OK
        fc_biases_1 = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.FASHION_F_FC_1], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_biases_1", node=node))
        # OK
        fc_weights_2 = tf.Variable(tf.truncated_normal(
            [GlobalConstants.FASHION_F_FC_1,
             GlobalConstants.FASHION_F_FC_2],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_weights_2", node=node))
        # OK
        fc_biases_2 = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.FASHION_F_FC_2], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_biases_2", node=node))
        # OK
        fc_softmax_weights = tf.Variable(
            tf.truncated_normal([softmax_input_dim, GlobalConstants.NUM_LABELS],
                                stddev=0.1,
                                seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_softmax_weights", node=node))
        # OK
        fc_softmax_biases = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_softmax_biases", node=node))
        node.variablesSet = {fc_weights_1, fc_biases_1, fc_weights_2, fc_biases_2, fc_softmax_weights,
                             fc_softmax_biases}
        # OPS
        hidden_layer_1 = tf.nn.relu(tf.matmul(net, fc_weights_1) + fc_biases_1)
        dropped_layer_1 = tf.nn.dropout(hidden_layer_1, network.classificationDropoutKeepProb)
        hidden_layer_2 = tf.nn.relu(tf.matmul(dropped_layer_1, fc_weights_2) + fc_biases_2)
        dropped_layer_2 = tf.nn.dropout(hidden_layer_2, network.classificationDropoutKeepProb)
        final_feature, logits = network.apply_loss(node=node, final_feature=dropped_layer_2,
                                                   softmax_weights=fc_softmax_weights, softmax_biases=fc_softmax_biases)
        node.fOpsList.extend([net])
        # Evaluation
        node.evalDict[network.get_variable_name(name="final_eval_feature", node=node)] = final_feature
        node.evalDict[network.get_variable_name(name="logits", node=node)] = logits
        node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
        node.evalDict[network.get_variable_name(name="fc_softmax_weights", node=node)] = fc_softmax_weights
        node.evalDict[network.get_variable_name(name="fc_softmax_biases", node=node)] = fc_softmax_biases
        # ***************** F: Convolution Layer *****************

    def get_explanation_string(self):
        total_param_count = 0
        for v in tf.trainable_variables():
            total_param_count += np.prod(v.get_shape().as_list())
        # Tree
        explanation = "Fashion Net - Multipath Optimization\n"
        # "(Lr=0.01, - Decay 1/(1 + i*0.0001) at each i. iteration)\n"
        explanation += "Using Fast Tree Version:{0}\n".format(GlobalConstants.USE_FAST_TREE_MODE)
        explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
        explanation += "Tree Degree:{0}\n".format(GlobalConstants.TREE_DEGREE_LIST)
        explanation += "Concat Trick:{0}\n".format(GlobalConstants.USE_CONCAT_TRICK)
        explanation += "Info Gain:{0}\n".format(GlobalConstants.USE_INFO_GAIN_DECISION)
        explanation += "Using Effective Sample Counts:{0}\n".format(GlobalConstants.USE_EFFECTIVE_SAMPLE_COUNTS)
        explanation += "Gradient Type:{0}\n".format(GlobalConstants.GRADIENT_TYPE)
        explanation += "Probability Threshold:{0}\n".format(GlobalConstants.USE_PROBABILITY_THRESHOLD)
        explanation += "********Lr Settings********\n"
        explanation += GlobalConstants.LEARNING_RATE_CALCULATOR.get_explanation()
        explanation += "********Lr Settings********\n"
        if not self.isBaseline:
            explanation += "********Decision Loss Weight Settings********\n"
            explanation += self.decisionLossCoefficientCalculator.get_explanation()
            explanation += "********Decision Loss Weight Settings********\n"
        explanation += "Use Unified Batch Norm:{0}\n".format(GlobalConstants.USE_UNIFIED_BATCH_NORM)
        explanation += "Batch Norm Decay:{0}\n".format(GlobalConstants.BATCH_NORM_DECAY)
        explanation += "Param Count:{0}\n".format(total_param_count)
        explanation += "Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
        explanation += "Decision Wd:{0}\n".format(GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT)
        explanation += "Residue Loss Coefficient:{0}\n".format(GlobalConstants.RESIDUE_LOSS_COEFFICIENT)
        explanation += "Residue Affects All Network:{0}\n".format(GlobalConstants.RESIDE_AFFECTS_WHOLE_NETWORK)
        explanation += "Using Info Gain:{0}\n".format(GlobalConstants.USE_INFO_GAIN_DECISION)
        explanation += "Info Gain Loss Lambda:{0}\n".format(GlobalConstants.DECISION_LOSS_COEFFICIENT)
        explanation += "Use Batch Norm Before Decisions:{0}\n".format(GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING)
        explanation += "Use Trainable Batch Norm Parameters:{0}\n".format(
            GlobalConstants.USE_TRAINABLE_PARAMS_WITH_BATCH_NORM)
        explanation += "Hyperplane bias at 0.0\n"
        explanation += "Using Convolutional Routing Networks:{0}\n".format(GlobalConstants.USE_CONVOLUTIONAL_H_PIPELINE)
        explanation += "Softmax Decay Initial:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_INITIAL)
        explanation += "Softmax Decay Coefficient:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_COEFFICIENT)
        explanation += "Softmax Decay Period:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_PERIOD)
        explanation += "Softmax Min Limit:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_MIN_LIMIT)
        explanation += "Softmax Test Temperature:{0}\n".format(GlobalConstants.SOFTMAX_TEST_TEMPERATURE)
        explanation += "Reparametrized Noise:{0}\n".format(GlobalConstants.USE_REPARAMETRIZATION_TRICK)
        # for node in network.topologicalSortedNodes:
        #     if node.isLeaf:
        #         continue
        #     explanation += "Node {0} Info Gain Balance Coefficient:{1}\n".format(node.index,
        #                                                                          node.infoGainBalanceCoefficient)
        explanation += "Info Gain Balance Coefficient:{0}\n".format(GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT)
        explanation += "Adaptive Weight Decay:{0}\n".format(GlobalConstants.USE_ADAPTIVE_WEIGHT_DECAY)
        if GlobalConstants.USE_REPARAMETRIZATION_TRICK:
            explanation += "********Reparametrized Noise Settings********\n"
            explanation += "Noise Coefficient Initial Value:{0}\n".format(self.noiseCoefficientCalculator.value)
            explanation += "Noise Coefficient Decay Step:{0}\n".format(self.noiseCoefficientCalculator.decayPeriod)
            explanation += "Noise Coefficient Decay Ratio:{0}\n".format(self.noiseCoefficientCalculator.decay)
            explanation += "********Reparametrized Noise Settings********\n"
        explanation += "Use Decision Dropout:{0}\n".format(GlobalConstants.USE_DROPOUT_FOR_DECISION)
        explanation += "Use Decision Augmentation:{0}\n".format(GlobalConstants.USE_DECISION_AUGMENTATION)
        if GlobalConstants.USE_DROPOUT_FOR_DECISION:
            explanation += "********Decision Dropout Schedule********\n"
            explanation += "Iteration:{0} Probability:{1}\n".format(0, GlobalConstants.DROPOUT_INITIAL_PROB)
            for tpl in GlobalConstants.DROPOUT_SCHEDULE:
                explanation += "Iteration:{0} Probability:{1}\n".format(tpl[0], tpl[1])
            explanation += "********Decision Dropout Schedule********\n"
        explanation += "Use Classification Dropout:{0}\n".format(GlobalConstants.USE_DROPOUT_FOR_CLASSIFICATION)
        explanation += "Classification Dropout Probability:{0}\n".format(
            GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB)
        explanation += "Decision Dropout Probability:{0}\n".format(self.decisionDropoutKeepProbCalculator.value)
        if GlobalConstants.USE_PROBABILITY_THRESHOLD:
            for node in self.topologicalSortedNodes:
                if node.isLeaf:
                    continue
                explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
                explanation += node.probThresholdCalculator.get_explanation()
                explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
        explanation += "Use Softmax Compression:{0}\n".format(GlobalConstants.USE_SOFTMAX_DISTILLATION)
        explanation += "Waiting Epochs for Softmax Compression:{0}\n".format(GlobalConstants.MODE_WAIT_EPOCHS)
        explanation += "Mode Percentile:{0}\n".format(GlobalConstants.PERCENTILE_THRESHOLD)
        explanation += "Mode Tracking Strategy:{0}\n".format(GlobalConstants.MODE_TRACKING_STRATEGY)
        explanation += "Mode Max Class Count:{0}\n".format(GlobalConstants.MAX_MODE_CLASSES)
        explanation += "Mode Computation Strategy:{0}\n".format(GlobalConstants.MODE_COMPUTATION_STRATEGY)
        explanation += "Constrain Softmax Compression With Label Count:{0}\n".format(GlobalConstants.
                                                                                     CONSTRAIN_WITH_COMPRESSION_LABEL_COUNT)
        explanation += "Softmax Distillation Cross Validation Count:{0}\n". \
            format(GlobalConstants.SOFTMAX_DISTILLATION_CROSS_VALIDATION_COUNT)
        explanation += "Softmax Distillation Strategy:{0}\n". \
            format(GlobalConstants.SOFTMAX_COMPRESSION_STRATEGY)
        explanation += "F Conv1:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_1_SIZE,
                                                               GlobalConstants.FASHION_F_NUM_FILTERS_1)
        explanation += "F Conv2:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_2_SIZE,
                                                               GlobalConstants.FASHION_F_NUM_FILTERS_2)
        explanation += "F Conv3:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_3_SIZE,
                                                               GlobalConstants.FASHION_F_NUM_FILTERS_3)
        explanation += "F FC1:{0} Units\n".format(GlobalConstants.FASHION_F_FC_1)
        explanation += "F FC2:{0} Units\n".format(GlobalConstants.FASHION_F_FC_2)
        explanation += "F Residue FC:{0} Units\n".format(GlobalConstants.FASHION_F_RESIDUE)
        explanation += "Residue Hidden Layer Count:{0}\n".format(GlobalConstants.FASHION_F_RESIDUE_LAYER_COUNT)
        explanation += "Residue Use Dropout:{0}\n".format(GlobalConstants.FASHION_F_RESIDUE_USE_DROPOUT)
        explanation += "H Conv1:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_H_FILTERS_1_SIZE,
                                                               GlobalConstants.FASHION_H_NUM_FILTERS_1)
        explanation += "H Conv2:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_H_FILTERS_2_SIZE,
                                                               GlobalConstants.FASHION_H_NUM_FILTERS_2)
        explanation += "FASHION_NO_H_FROM_F_UNITS_1:{0} Units\n".format(GlobalConstants.FASHION_NO_H_FROM_F_UNITS_1)
        explanation += "FASHION_NO_H_FROM_F_UNITS_2:{0} Units\n".format(GlobalConstants.FASHION_NO_H_FROM_F_UNITS_2)
        explanation += "Use Class Weighting:{0}\n".format(GlobalConstants.USE_CLASS_WEIGHTING)
        explanation += "Class Weight Running Average:{0}\n".format(GlobalConstants.CLASS_WEIGHT_RUNNING_AVERAGE)
        explanation += "Zero Label Count Epsilon:{0}\n".format(GlobalConstants.LABEL_EPSILON)
        # Baseline
        # explanation = "Fashion Mnist Baseline. Double Dropout, Discrete learning rate\n"
        # explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
        # explanation += "Gradient Type:{0}\n".format(GlobalConstants.GRADIENT_TYPE)
        # explanation += "Initial Lr:{0}\n".format(GlobalConstants.INITIAL_LR)
        # explanation += "Decay Steps:{0}\n".format(GlobalConstants.DECAY_STEP)
        # explanation += "Decay Rate:{0}\n".format(GlobalConstants.DECAY_RATE)
        # explanation += "Param Count:{0}\n".format(total_param_count)
        # explanation += "Model: {0}Conv - {1}Conv - {2}Conv - {3}FC - {4}FC\n".\
        #     format(GlobalConstants.FASHION_NUM_FILTERS_1, GlobalConstants.FASHION_NUM_FILTERS_2,
        #            GlobalConstants.FASHION_NUM_FILTERS_3, GlobalConstants.FASHION_FC_1, GlobalConstants.FASHION_FC_2)
        # explanation += "Conv1 Filters:{0} Conv2 Filters:{1} Conv3 Filters:{2}".\
        #     format(GlobalConstants.FASHION_FILTERS_1_SIZE, GlobalConstants.FASHION_FILTERS_2_SIZE,
        #            GlobalConstants.FASHION_FILTERS_3_SIZE)
        # explanation += "Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
        # explanation += "Dropout Prob:{0}\n".format(GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB)
        return explanation

    def set_training_parameters(self):
        # Training Parameters
        GlobalConstants.TOTAL_EPOCH_COUNT = 100
        GlobalConstants.EPOCH_COUNT = 100
        GlobalConstants.EPOCH_REPORT_PERIOD = 1
        GlobalConstants.BATCH_SIZE = 125
        GlobalConstants.EVAL_BATCH_SIZE = 250
        GlobalConstants.USE_MULTI_GPU = False
        GlobalConstants.USE_SAMPLING_CIGN = False
        GlobalConstants.USE_RANDOM_SAMPLING = False
        GlobalConstants.INITIAL_LR = 0.01
        GlobalConstants.LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator",
                                                                     value=GlobalConstants.INITIAL_LR,
                                                                     schedule=[(15000, 0.005),
                                                                               (30000, 0.0025),
                                                                               (40000, 0.00025)])

    def set_hyperparameters(self, **kwargs):
        # Regularization Parameters
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = kwargs["weight_decay_coefficient"]
        GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB = kwargs["classification_keep_probability"]
        GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = kwargs["decision_weight_decay_coefficient"]
        GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT = kwargs["info_gain_balance_coefficient"]
        self.decisionDropoutKeepProbCalculator = FixedParameter(name="decision_dropout_prob",
                                                                value=kwargs["decision_keep_probability"])
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
        # Thresholding and Softmax Decay
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
