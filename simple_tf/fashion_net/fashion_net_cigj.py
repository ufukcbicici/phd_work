import tensorflow as tf
import numpy as np
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DecayingParameter, FixedParameter, DiscreteParameter
from simple_tf.cigj.jungle_gumbel_softmax import JungleGumbelSoftmax
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants


class FashionNetCigj(JungleGumbelSoftmax):
    def __init__(self, node_build_funcs, h_funcs, grad_func, hyperparameter_func, residue_func, summary_func,
                 degree_list, dataset, network_name):
        super().__init__(node_build_funcs, h_funcs, grad_func, hyperparameter_func, residue_func, summary_func,
                         degree_list, dataset, network_name)

    @staticmethod
    def build_conv_layer(input, node, filter_size, num_of_input_channels, num_of_output_channels, name_suffix=""):
        conv_weights = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="conv_weight{0}".format(name_suffix), node=node),
            shape=[filter_size, filter_size, num_of_input_channels, num_of_output_channels],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.truncated_normal([filter_size, filter_size, num_of_input_channels, num_of_output_channels],
                                            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
        conv_biases = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="conv_bias{0}".format(name_suffix), node=node),
            shape=[num_of_output_channels],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.constant(0.1, shape=[num_of_output_channels], dtype=GlobalConstants.DATA_TYPE))
        net = FastTreeNetwork.conv_layer(x=input, kernel=conv_weights, strides=[1, 1, 1, 1],
                                         padding='SAME', bias=conv_biases, node=node)
        relu = tf.nn.relu(net)
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool

    @staticmethod
    def build_fc_layer(input, node, input_dim, output_dim, dropout_prob_tensor, name_suffix=""):
        fc_weights = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="fc_weights{0}".format(name_suffix), node=node),
            shape=[input_dim, output_dim],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.truncated_normal(
                [input_dim, output_dim],
                stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
        fc_biases = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="fc_biases{0}".format(name_suffix), node=node),
            shape=[output_dim],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.constant(0.1, shape=[output_dim], dtype=GlobalConstants.DATA_TYPE))
        x_hat = FastTreeNetwork.fc_layer(x=input, W=fc_weights, b=fc_biases, node=node)
        hidden_layer = tf.nn.relu(x_hat)
        dropped_layer = tf.nn.dropout(hidden_layer, dropout_prob_tensor)
        return dropped_layer

    def get_explanation_string(self):
        total_param_count = 0
        for v in tf.trainable_variables():
            total_param_count += np.prod(v.get_shape().as_list())
        # Tree
        explanation = "CIGJ Fashion MNIST Gumbel-Softmax Tests: 128 Sized H Features\n"
        explanation += "Network Type:{0}\n".format(self.__class__)
        # "(Lr=0.01, - Decay 1/(1 + i*0.0001) at each i. iteration)\n"
        explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
        explanation += "Jungle Degree Degree:{0}\n".format(GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST)
        explanation += "Optimizer:{0}\n".format(GlobalConstants.OPTIMIZER_TYPE)
        explanation += "********Lr Settings********\n"
        explanation += GlobalConstants.LEARNING_RATE_CALCULATOR.get_explanation()
        explanation += "********Lr Settings********\n"
        if not self.isBaseline:
            explanation += "********Decision Loss Weight Settings********\n"
            explanation += self.decisionLossCoefficientCalculator.get_explanation()
            explanation += "********Decision Loss Weight Settings********\n"
        explanation += "Batch Norm Decay:{0}\n".format(GlobalConstants.BATCH_NORM_DECAY)
        explanation += "Param Count:{0}\n".format(total_param_count)
        explanation += "Classification Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
        explanation += "Decision Wd:{0}\n".format(GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT)
        explanation += "Info Gain Loss Lambda:{0}\n".format(GlobalConstants.DECISION_LOSS_COEFFICIENT)
        explanation += "Use Batch Norm Before Decisions:{0}\n".format(GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING)
        # explanation += "Softmax Decay Initial:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_INITIAL)
        # explanation += "Softmax Decay Coefficient:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_COEFFICIENT)
        # explanation += "Softmax Decay Period:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_PERIOD)
        # explanation += "Softmax Min Limit:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_MIN_LIMIT)
        # explanation += "Softmax Test Temperature:{0}\n".format(GlobalConstants.SOFTMAX_TEST_TEMPERATURE)

        explanation += "********Softmax Decay Settings********\n"
        for node in self.topologicalSortedNodes:
            if node.nodeType == NodeType.h_node:
                explanation += "********Node{0} Softmax Decay********\n".format(node.index)
                explanation += node.softmaxDecayCalculator.get_explanation()
                explanation += "********Node{0} Softmax Decay********\n".format(node.index)
        explanation += "********Softmax Decay Settings********\n"

        explanation += "********Gumbel Softmax Temperature Settings********\n"
        for node in self.topologicalSortedNodes:
            if node.nodeType == NodeType.h_node:
                explanation += "********Node{0} Gumbel Softmax Temperature********\n".format(node.index)
                explanation += node.gumbelSoftmaxTemperatureCalculator.get_explanation()
                explanation += "********Node{0} Gumbel Softmax Temperature********\n".format(node.index)
        explanation += "********Gumbel Softmax Temperature Settings********\n"

        explanation += "Info Gain Balance Coefficient:{0}\n".format(GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT)
        explanation += "Classification Dropout Probability:{0}\n".format(
            GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB)
        explanation += "Decision Dropout Probability:{0}\n".format(GlobalConstants.DECISION_DROPOUT_KEEP_PROB)
        explanation += "H Feature Sizes:{0}\n".format(GlobalConstants.CIGJ_FASHION_NET_H_FEATURES)
        explanation += "H Pooling Sizes:{0}\n".format(GlobalConstants.CIGJ_FASHION_NET_H_POOL_SIZES)
        # explanation += "Decision Dropout Probability:{0}\n".format(network.decisionDropoutKeepProbCalculator.value)
        # if GlobalConstants.USE_PROBABILITY_THRESHOLD:
        #     for node in network.topologicalSortedNodes:
        #         if node.isLeaf:
        #             continue
        #         explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
        #         explanation += node.probThresholdCalculator.get_explanation()
        #         explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
        return explanation

    @staticmethod
    def h_transform(input_net, node, network, h_feature_size, pool_size):
        h_net = input_net
        # Parametric Average Pooling if the input layer is convolutional
        assert len(h_net.get_shape().as_list()) == 2 or len(h_net.get_shape().as_list()) == 4
        if len(h_net.get_shape().as_list()) == 4:
            h_net = tf.nn.avg_pool(h_net, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1],
                                   padding='SAME')
            # h_net = UtilityFuncs.tf_safe_flatten(input_tensor=h_net)
            h_net = tf.contrib.layers.flatten(h_net)

        feature_size = h_net.get_shape().as_list()[-1]
        print("h pre input size:{0}".format(feature_size))
        fc_h_weights = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="fc_decision_weights", node=node),
            shape=[feature_size, h_feature_size],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.truncated_normal(
                [feature_size, h_feature_size],
                stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
        fc_h_bias = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="fc_decision_bias", node=node),
            shape=[h_feature_size],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.constant(0.1, shape=[h_feature_size], dtype=GlobalConstants.DATA_TYPE))
        h_net = FastTreeNetwork.fc_layer(x=h_net, W=fc_h_weights, b=fc_h_bias, node=node)
        h_net = tf.nn.relu(h_net)
        h_net = tf.nn.dropout(h_net, keep_prob=network.decisionDropoutKeepProb)
        ig_feature = h_net
        return ig_feature

    @staticmethod
    def f_conv_layer_func(node, network):
        network.mask_input_nodes(node=node)
        filter_size = GlobalConstants.CIGJ_FASHION_NET_CONV_FILTER_SIZES[node.depth]
        num_of_input_channels = 1 if node.depth == 0 else GlobalConstants.CIGJ_FASHION_NET_OUTPUT_DIMS[node.depth - 1]
        num_of_output_channels = GlobalConstants.CIGJ_FASHION_NET_OUTPUT_DIMS[node.depth]
        node.F_output = FashionNetCigj.build_conv_layer(input=node.F_input,
                                                        node=node,
                                                        filter_size=filter_size,
                                                        num_of_input_channels=num_of_input_channels,
                                                        num_of_output_channels=num_of_output_channels)

    @staticmethod
    def f_fc_layer_func(node, network):
        network.mask_input_nodes(node=node)
        net = node.F_input
        if len(net.get_shape().as_list()) == 4:
            net = tf.contrib.layers.flatten(node.F_input)
        input_dim = net.get_shape().as_list()[-1]
        dimensions = [input_dim]
        dimensions.extend(GlobalConstants.CIGJ_FASHION_NET_OUTPUT_DIMS[node.depth])
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
        if node.depth + 1 <= len(network.degreeList) - 1:
            node_degree = network.degreeList[node.depth + 1]
            if node_degree > 1:
                h_feature_size = GlobalConstants.CIGJ_FASHION_NET_H_FEATURES[node.depth]
                pool_size = GlobalConstants.CIGJ_FASHION_NET_H_POOL_SIZES[node.depth]
                node.H_output = FashionNetCigj.h_transform(input_net=node.F_input, network=network, node=node,
                                                           h_feature_size=h_feature_size,
                                                           pool_size=pool_size)
            else:
                node.H_output = tf.constant(0)
        network.apply_decision(node=node, branching_feature=node.H_output)

    def set_training_parameters(self):
        # Training Parameters
        GlobalConstants.TOTAL_EPOCH_COUNT = 100
        GlobalConstants.EPOCH_COUNT = 100
        GlobalConstants.EPOCH_REPORT_PERIOD = 1
        GlobalConstants.BATCH_SIZE = 125
        GlobalConstants.EVAL_BATCH_SIZE = 125
        GlobalConstants.USE_MULTI_GPU = False
        GlobalConstants.USE_SAMPLING_CIGN = False
        GlobalConstants.USE_RANDOM_SAMPLING = False
        GlobalConstants.INITIAL_LR = 0.01
        GlobalConstants.LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator",
                                                                     value=GlobalConstants.INITIAL_LR,
                                                                     schedule=[(15000, 0.005),
                                                                               (30000, 0.0025),
                                                                               (40000, 0.00025)])
        GlobalConstants.GLOBAL_PINNING_DEVICE = "/device:GPU:0"
        self.networkName = "FashionNet_Lite"

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
        for node in self.topologicalSortedNodes:
            if node.nodeType == NodeType.h_node:
                # Softmax Decay
                decay_name = self.get_variable_name(name="softmax_decay", node=node)
                node.softmaxDecayCalculator = DecayingParameter(name=decay_name,
                                                                value=GlobalConstants.SOFTMAX_DECAY_INITIAL,
                                                                decay=GlobalConstants.SOFTMAX_DECAY_COEFFICIENT,
                                                                decay_period=GlobalConstants.SOFTMAX_DECAY_PERIOD,
                                                                min_limit=GlobalConstants.SOFTMAX_DECAY_MIN_LIMIT)
                # Gm Temperature
                temperature_name = self.get_variable_name(name="gm_temperature", node=node)
                node.gumbelSoftmaxTemperatureCalculator = \
                    DecayingParameter(name=temperature_name,
                                      value=GlobalConstants.CIGJ_GUMBEL_SOFTMAX_TEMPERATURE_INITIAL,
                                      decay=GlobalConstants.CIGJ_GUMBEL_SOFTMAX_DECAY_COEFFICIENT,
                                      decay_period=GlobalConstants.CIGJ_GUMBEL_SOFTMAX_DECAY_PERIOD,
                                      min_limit=GlobalConstants.CIGJ_GUMBEL_SOFTMAX_DECAY_MIN_LIMIT)

