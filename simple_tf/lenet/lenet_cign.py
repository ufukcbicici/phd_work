import tensorflow as tf
import numpy as np

from auxillary.parameters import DecayingParameter, FixedParameter, DiscreteParameter
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants


class Lenet_Cign(FastTreeNetwork):
    def __init__(self, degree_list, dataset):
        node_build_funcs = [None, None, None]
        super().__init__(node_build_funcs, None, None, None, None, degree_list, dataset)

    @staticmethod
    def apply_router_transformation(network, net, node, decision_feature_size):
        pool_h = tf.nn.max_pool(net, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
        flat_pool = tf.contrib.layers.flatten(pool_h)
        feature_size = flat_pool.get_shape().as_list()[-1]
        fc_h_weights = tf.Variable(tf.truncated_normal(
            [feature_size, decision_feature_size],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_decision_weights", node=node))
        fc_h_bias = tf.Variable(
            tf.constant(0.1, shape=[decision_feature_size], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_decision_bias", node=node))
        raw_ig_feature = FastTreeNetwork.fc_layer(x=flat_pool, W=fc_h_weights, b=fc_h_bias, node=node)
        # ***************** Dropout *****************
        relu_ig_feature = tf.nn.relu(raw_ig_feature)
        ig_feature = tf.nn.dropout(relu_ig_feature, keep_prob=network.decisionDropoutKeepProb)
        # ***************** Dropout *****************
        node.hOpsList.extend([ig_feature])
        # Decisions
        if GlobalConstants.USE_UNIFIED_BATCH_NORM:
            network.apply_decision_with_unified_batch_norm(node=node, branching_feature=ig_feature)
        else:
            network.apply_decision(node=node, branching_feature=ig_feature)

    @staticmethod
    def root_func(node, network):
        # Parameters
        node_degree = network.degreeList[node.depth]
        conv_weights = tf.Variable(
            tf.truncated_normal([5, 5, GlobalConstants.NUM_CHANNELS, GlobalConstants.NO_FILTERS_1], stddev=0.1,
                                seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE), name=network.get_variable_name(name="conv_weight",
                                                                                                 node=node))
        conv_biases = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.NO_FILTERS_1], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv_bias",
                                           node=node))
        node.variablesSet = {conv_weights, conv_biases}
        # Operations
        network.mask_input_nodes(node=node)
        # F
        conv = FastTreeNetwork.conv_layer(x=network.dataTensor, kernel=conv_weights, strides=[1, 1, 1, 1],
                                          padding='SAME', bias=conv_biases, node=node)
        relu = tf.nn.relu(conv)
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        node.fOpsList.extend([conv, relu, pool])
        # ***************** H: Connected to F *****************
        Lenet_Cign.apply_router_transformation(network=network, net=relu, node=node,
                                               decision_feature_size=GlobalConstants.LENET_H_FEATURE_SIZE_1)
        # ***************** H: Connected to F *****************

    @staticmethod
    def l1_func(node, network):
        # Parameters
        conv_weights = tf.Variable(
            tf.truncated_normal([5, 5, GlobalConstants.NO_FILTERS_1, GlobalConstants.NO_FILTERS_2], stddev=0.1,
                                seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv_weight", node=node))
        conv_biases = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.NO_FILTERS_2], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv_bias",
                                           node=node))
        node.variablesSet = {conv_weights, conv_biases}
        # Operations
        parent_F, parent_H = network.mask_input_nodes(node=node)
        # F
        conv = FastTreeNetwork.conv_layer(x=parent_F, kernel=conv_weights, strides=[1, 1, 1, 1],
                                          padding='SAME', bias=conv_biases, node=node)
        relu = tf.nn.relu(conv)
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        node.fOpsList.extend([conv, relu, pool])
        # H
        # ***************** H: Connected to F *****************
        Lenet_Cign.apply_router_transformation(network=network, net=relu, node=node,
                                               decision_feature_size=GlobalConstants.LENET_H_FEATURE_SIZE_2)
        # ***************** H: Connected to F *****************

    @staticmethod
    def leaf_func(node, network):
        # Parameters
        fc_weights_1 = tf.Variable(tf.truncated_normal(
            [GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.NO_FILTERS_2,
             GlobalConstants.NO_HIDDEN],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_weights_1", node=node))
        fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_HIDDEN], dtype=GlobalConstants.DATA_TYPE),
                                  name=network.get_variable_name(name="fc_biases_1", node=node))
        softmax_input_dim = GlobalConstants.NO_HIDDEN
        fc_softmax_weights = tf.Variable(
            tf.truncated_normal([softmax_input_dim, GlobalConstants.NUM_LABELS],
                                stddev=0.1,
                                seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_softmax_weights", node=node))
        fc_softmax_biases = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_softmax_biases", node=node))
        node.variablesSet = {fc_weights_1, fc_biases_1, fc_softmax_weights, fc_softmax_biases}
        # Operations
        # Mask inputs
        parent_F, parent_H = network.mask_input_nodes(node=node)
        flattened = tf.contrib.layers.flatten(parent_F)
        x_hat = FastTreeNetwork.fc_layer(x=flattened, W=fc_weights_1, b=fc_biases_1, node=node)
        hidden_layer = tf.nn.relu(x_hat)
        final_feature, logits = network.apply_loss(node=node, final_feature=hidden_layer,
                                                   softmax_weights=fc_softmax_weights, softmax_biases=fc_softmax_biases)
        node.fOpsList.extend([flattened])
        # Evaluation
        node.evalDict[network.get_variable_name(name="final_eval_feature", node=node)] = final_feature
        node.evalDict[network.get_variable_name(name="logits", node=node)] = logits
        node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
        node.evalDict[network.get_variable_name(name="fc_softmax_weights", node=node)] = fc_softmax_weights
        node.evalDict[network.get_variable_name(name="fc_softmax_biases", node=node)] = fc_softmax_biases

    def get_explanation_string(self):
        total_param_count = 0
        for v in tf.trainable_variables():
            total_param_count += np.prod(v.get_shape().as_list())
        # Tree
        explanation = "Lenet CIGN\n"
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
        explanation += "NO_FILTERS_1:{0}\n".format(GlobalConstants.NO_FILTERS_1)
        explanation += "LENET_H_FEATURE_SIZE_1:{0}\n".format(GlobalConstants.LENET_H_FEATURE_SIZE_1)
        explanation += "NO_FILTERS_2:{0}\n".format(GlobalConstants.NO_FILTERS_2)
        explanation += "LENET_H_FEATURE_SIZE_2:{0}\n".format(GlobalConstants.LENET_H_FEATURE_SIZE_2)
        explanation += "NO_HIDDEN:{0}\n".format(GlobalConstants.NO_HIDDEN)
        explanation += "TRAINING PARAMETERS:\n"
        explanation += super().get_explanation_string()
        return explanation

    def set_training_parameters(self):
        # Training Parameters
        GlobalConstants.TOTAL_EPOCH_COUNT = 100
        GlobalConstants.EPOCH_COUNT = 100
        GlobalConstants.EPOCH_REPORT_PERIOD = 5
        GlobalConstants.BATCH_SIZE = 125
        GlobalConstants.EVAL_BATCH_SIZE = 1000
        GlobalConstants.USE_MULTI_GPU = False
        GlobalConstants.USE_SAMPLING_CIGN = False
        GlobalConstants.USE_RANDOM_SAMPLING = False
        GlobalConstants.INITIAL_LR = 0.025
        GlobalConstants.LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator",
                                                                     value=GlobalConstants.INITIAL_LR,
                                                                     schedule=[(15000, 0.0125),
                                                                               (30000, 0.00625),
                                                                               (45000, 0.003125)])
        GlobalConstants.GLOBAL_PINNING_DEVICE = "/device:GPU:0"
        self.networkName = "Lenet_CIGN"

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
