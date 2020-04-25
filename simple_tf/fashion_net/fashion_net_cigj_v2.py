import tensorflow as tf
import numpy as np

from auxillary.parameters import DiscreteParameter, FixedParameter, DecayingParameter
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.cigj_v2.junglev2 import JungleV2
from simple_tf.fashion_net.fashion_net_cigj import FashionNetCigj
from simple_tf.global_params import GlobalConstants


class FashionNetCigjV2(JungleV2):
    # self, node_build_funcs, h_dimensions, dataset, network_name
    def __init__(self, node_build_funcs, h_dimensions, dataset, network_name, level_params):
        super().__init__(node_build_funcs, h_dimensions, dataset, network_name)
        self.levelParams = level_params
        assert len(node_build_funcs) == len(h_dimensions) + 1 and len(node_build_funcs) == len(self.levelParams)

    # CIGJ_V2_PARAMS = [[("conv", 32, 5, True)],
    #                   [("conv", 64, 5, True), ("conv", 64, 1, True)],
    #                   [("fc", 512), ("fc", 256)]]

    def build_lenet_node(self, node, input_x, depth):
        node_params = self.levelParams[depth]
        net = input_x
        for layer_id, params in enumerate(node_params):
            layer_type = params[0]
            assert layer_type == "conv" or layer_type == "fc"
            if layer_type == "conv":
                assert len(net.get_shape().as_list()) == 4
                input_feature_map_count = net.get_shape().as_list()[-1]
                output_feature_map_count = params[1]
                filter_size = params[2]
                use_pooling = params[3]
                net = FashionNetCigj.build_conv_layer(input=net, node=node, filter_size=filter_size,
                                                      num_of_input_channels=input_feature_map_count,
                                                      num_of_output_channels=output_feature_map_count,
                                                      use_pooling=use_pooling, name_suffix="{0}".format(layer_id))
            else:
                assert len(net.get_shape().as_list()) == 4 or len(net.get_shape().as_list()) == 2
                if len(net.get_shape().as_list()) == 4:
                    net = tf.contrib.layers.flatten(net)
                input_dim = net.get_shape().as_list()[-1]
                output_dim = params[1]
                net = FashionNetCigj.build_fc_layer(input=net, node=node, input_dim=input_dim, output_dim=output_dim,
                                                    dropout_prob_tensor=self.classificationDropoutKeepProb,
                                                    name_suffix="{0}".format(layer_id))
        return net

    def get_explanation_string(self):
        total_param_count = 0
        for v in tf.trainable_variables():
            total_param_count += np.prod(v.get_shape().as_list())
        # Tree
        explanation = "CIGJ - Sparse Layers\n"
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
        explanation += "Decision Dropout Probability:{0}\n".format(
            self.decisionDropoutKeepProbCalculator.get_explanation())
        explanation += "CIGJ_FASHION_NET_CONV_FILTER_SIZES:{0}\n"\
            .format(GlobalConstants.CIGJ_FASHION_NET_CONV_FILTER_SIZES)
        explanation += "CIGJ_FASHION_NET_OUTPUT_DIMS:{0}\n"\
            .format(GlobalConstants.CIGJ_FASHION_NET_OUTPUT_DIMS)
        explanation += "CIGJ_FASHION_NET_DEGREE_LIST:{0}\n"\
            .format(GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST)
        explanation += "CIGJ_FASHION_NET_H_FEATURES:{0}\n"\
            .format(GlobalConstants.CIGJ_FASHION_NET_H_FEATURES)
        explanation += "CIGJ_FASHION_NET_H_POOL_SIZES:{0}\n"\
            .format(GlobalConstants.CIGJ_FASHION_NET_H_POOL_SIZES)
        explanation += "CIGJ_GUMBEL_SOFTMAX_SAMPLE_COUNT:{0}\n"\
            .format(GlobalConstants.CIGJ_GUMBEL_SOFTMAX_SAMPLE_COUNT)
        explanation += "CIGJ_GUMBEL_SOFTMAX_TEMPERATURE_INITIAL:{0}\n"\
            .format(GlobalConstants.CIGJ_GUMBEL_SOFTMAX_TEMPERATURE_INITIAL)
        explanation += "CIGJ_GUMBEL_SOFTMAX_DECAY_COEFFICIENT:{0}\n"\
            .format(GlobalConstants.CIGJ_GUMBEL_SOFTMAX_DECAY_COEFFICIENT)
        explanation += "CIGJ_GUMBEL_SOFTMAX_DECAY_PERIOD:{0}\n"\
            .format(GlobalConstants.CIGJ_GUMBEL_SOFTMAX_DECAY_PERIOD)
        explanation += "CIGJ_GUMBEL_SOFTMAX_DECAY_MIN_LIMIT:{0}\n"\
            .format(GlobalConstants.CIGJ_GUMBEL_SOFTMAX_DECAY_MIN_LIMIT)
        explanation += "CIGJ_GUMBEL_SOFTMAX_TEST_TEMPERATURE:{0}\n" \
            .format(GlobalConstants.CIGJ_GUMBEL_SOFTMAX_TEST_TEMPERATURE)
        explanation += super().get_explanation_string()
        # explanation += "Decision Dropout Probability:{0}\n".format(network.decisionDropoutKeepProbCalculator.value)
        # if GlobalConstants.USE_PROBABILITY_THRESHOLD:
        #     for node in network.topologicalSortedNodes:
        #         if node.isLeaf:
        #             continue
        #         explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
        #         explanation += node.probThresholdCalculator.get_explanation()
        #         explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
        return explanation

    def set_training_parameters(self):
        # Training Parameters
        GlobalConstants.TOTAL_EPOCH_COUNT = 100
        GlobalConstants.EPOCH_COUNT = 100
        GlobalConstants.EPOCH_REPORT_PERIOD = 1
        GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING = 10
        GlobalConstants.BATCH_SIZE = 125
        GlobalConstants.EVAL_BATCH_SIZE = 5000
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
        self.networkName = "FashionNetCigjV2"

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

        GlobalConstants.SOFTMAX_TEST_TEMPERATURE = 50.0
