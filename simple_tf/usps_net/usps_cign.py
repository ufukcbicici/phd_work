import tensorflow as tf
import numpy as np
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DiscreteParameter, FixedParameter, DecayingParameter
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.fashion_net.fashion_cign_lite import FashionCignLite
from simple_tf.global_params import GlobalConstants
from algorithms.resnet.resnet_generator import ResnetGenerator
from simple_tf.usps_net.usps_baseline import UspsBaseline


class UspsCIGN(FastTreeNetwork):
    FC_LAYERS = [32, 24, 16]
    DECISION_DIMS = [8, 4, 4]
    SOFTMAX_DECAY_INITIAL = 10.0
    SOFTMAX_DECAY_PERIOD = 1000
    THRESHOLD_LOWER_LIMIT = 0.4
    THRESHOLD_PERIOD = 2000

    def __init__(self, degree_list, dataset, network_name):
        node_build_funcs = len(UspsCIGN.FC_LAYERS) * [UspsCIGN.node_func]
        super().__init__(node_build_funcs, None, None, None, None, degree_list, dataset, network_name)
        self.dataTensor = tf.placeholder(GlobalConstants.DATA_TYPE,
                                         shape=(None, dataset.trainingSamples.shape[-1]),
                                         name="dataTensor_USPS")

    @staticmethod
    def apply_router_transformation(network, net, node):
        node.evalDict[network.get_variable_name(name="pre_branch_feature", node=node)] = net
        decision_dim = UspsCIGN.DECISION_DIMS[node.depth]
        ig_feature = UspsBaseline.get_mlp_layers(net_input=net, node=node, network=network, layers=[decision_dim],
                                                 op_pre_fix="decision")
        node.hOpsList.extend([ig_feature])
        if GlobalConstants.USE_UNIFIED_BATCH_NORM:
            network.apply_decision_with_unified_batch_norm(node=node, branching_feature=ig_feature)
        else:
            network.apply_decision(node=node, branching_feature=ig_feature)

    @staticmethod
    def node_func(network, node):
        fc_dim = UspsCIGN.FC_LAYERS[node.depth]
        parent_F, parent_H = network.mask_input_nodes(node=node)
        net = network.dataTensor if node.isRoot else parent_F
        net = UspsBaseline.get_mlp_layers(net_input=net, node=node, network=network, layers=[fc_dim])
        node.fOpsList.extend([net])
        if not node.isLeaf:
            UspsCIGN.apply_router_transformation(network=network, net=net, node=node)
        else:
            input_dim = net.get_shape().as_list()[-1]
            output_dim = network.labelCount
            softmax_weights, softmax_biases = FashionCignLite.get_affine_layer_params(
                layer_shape=[input_dim, output_dim],
                W_name=network.get_variable_name(name="fc_softmax_weights", node=node),
                b_name=network.get_variable_name(name="fc_softmax_b", node=node))
            final_feature, logits = network.apply_loss(node=node, final_feature=net,
                                                       softmax_weights=softmax_weights, softmax_biases=softmax_biases)
            # Evaluation
            node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
            node.evalDict[network.get_variable_name(name="labels", node=node)] = node.labelTensor
        print("X")

    def get_explanation_string(self):
        total_param_count = 0
        for v in tf.trainable_variables():
            total_param_count += np.prod(v.get_shape().as_list())
        # Tree
        explanation = "USPS - CIGN - All Samples Routed\n"
        # "(Lr=0.01, - Decay 1/(1 + i*0.0001) at each i. iteration)\n"
        explanation += "Using Fast Tree Version:{0}\n".format(GlobalConstants.USE_FAST_TREE_MODE)
        explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
        explanation += "Tree Degree:{0}\n".format(GlobalConstants.TREE_DEGREE_LIST)
        explanation += "Concat Trick:{0}\n".format(GlobalConstants.USE_CONCAT_TRICK)
        explanation += "Info Gain:{0}\n".format(GlobalConstants.USE_INFO_GAIN_DECISION)
        explanation += "Using Effective Sample Counts:{0}\n".format(GlobalConstants.USE_EFFECTIVE_SAMPLE_COUNTS)
        explanation += "Gradient Type:{0}\n".format(GlobalConstants.GRADIENT_TYPE)
        explanation += "Probability Threshold:{0}\n".format(GlobalConstants.USE_PROBABILITY_THRESHOLD)

        # USPS
        explanation += "USPS FC_LAYERS:{0}\n".format(UspsCIGN.FC_LAYERS)
        explanation += "USPS DECISION_DIMS:{0}\n".format(UspsCIGN.DECISION_DIMS)
        explanation += "USPS SOFTMAX_DECAY_INITIAL:{0}\n".format(UspsCIGN.SOFTMAX_DECAY_INITIAL)
        explanation += "USPS SOFTMAX_DECAY_PERIOD:{0}\n".format(UspsCIGN.SOFTMAX_DECAY_PERIOD)
        explanation += "USPS THRESHOLD_LOWER_LIMIT:{0}\n".format(UspsCIGN.THRESHOLD_LOWER_LIMIT)
        explanation += "USPS THRESHOLD_PERIOD:{0}\n".format(UspsCIGN.THRESHOLD_PERIOD)
        # USPS

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
        explanation += "Constrain Softmax Compression With Label Count:{0}\n".\
            format(GlobalConstants.CONSTRAIN_WITH_COMPRESSION_LABEL_COUNT)
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
        GlobalConstants.TOTAL_EPOCH_COUNT = 200
        GlobalConstants.EPOCH_COUNT_INVALID = 200
        GlobalConstants.EPOCH_REPORT_PERIOD = 10
        GlobalConstants.BATCH_SIZE = 125
        GlobalConstants.EVAL_BATCH_SIZE = 1000
        GlobalConstants.USE_MULTI_GPU = False
        GlobalConstants.USE_SAMPLING_CIGN = False
        GlobalConstants.USE_RANDOM_SAMPLING = False
        # GlobalConstants.INITIAL_LR = 0.001
        GlobalConstants.LEARNING_RATE_CALCULATOR = \
            DiscreteParameter(name="lr_calculator",
                              value=GlobalConstants.INITIAL_LR,
                              schedule=[(2500, GlobalConstants.INITIAL_LR / 2.0),
                                        (5000, GlobalConstants.INITIAL_LR / 4.0),
                                        (7500, GlobalConstants.INITIAL_LR / 40.0)])
        GlobalConstants.GLOBAL_PINNING_DEVICE = "/device:GPU:0"
        self.networkName = "USPS_CIGN"

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
            # node.probThresholdCalculator = DecayingParameter(name=threshold_name,
            #                                                  value=initial_value,
            #                                                  decay=0.5,
            #                                                  decay_period=UspsCIGN.THRESHOLD_PERIOD,
            #                                                  min_limit=UspsCIGN.THRESHOLD_LOWER_LIMIT)
            node.probThresholdCalculator = FixedParameter(name=threshold_name, value=initial_value)
            # Softmax Decay
            decay_name = self.get_variable_name(name="softmax_decay", node=node)
            node.softmaxDecayCalculator = DecayingParameter(name=decay_name,
                                                            value=UspsCIGN.SOFTMAX_DECAY_INITIAL,
                                                            decay=0.5,
                                                            decay_period=UspsCIGN.SOFTMAX_DECAY_PERIOD,
                                                            min_limit=1.0)
        GlobalConstants.SOFTMAX_TEST_TEMPERATURE = 50.0
