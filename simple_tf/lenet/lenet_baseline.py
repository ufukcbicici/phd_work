import tensorflow as tf
import numpy as np
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DiscreteParameter, FixedParameter, DecayingParameter
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.fashion_net.fashion_cign_lite import FashionCignLite
from simple_tf.global_params import GlobalConstants
from algorithms.resnet.resnet_generator import ResnetGenerator


class LeNetBaseline(FastTreeNetwork):
    CONV_LAYERS = [20, 50]
    CONV_FILTER_SIZES = [5, 5]
    CONV_POOL_LAYERS = [True, True]
    FC_LAYERS = [500]

    def __init__(self, dataset, network_name):
        node_build_funcs = [LeNetBaseline.baseline]
        super().__init__(node_build_funcs, None, None, None, None, [], dataset, network_name)

    @staticmethod
    def build_lenet_structure(network, node, parent_F, conv_layers, conv_filters, fc_layers, pool_layers,
                              conv_name, fc_name):
        conv_weights = []
        conv_biases = []
        assert len(parent_F.get_shape().as_list()) == 4
        conv_layers = list(conv_layers)
        conv_layers.insert(0, parent_F.get_shape().as_list()[-1])
        assert len(conv_layers) == len(conv_filters) + 1
        net = parent_F
        # Conv Layers
        for idx in range(len(conv_layers) - 1):
            is_last_layer = idx == len(conv_layers) - 2
            f_map_count_0 = conv_layers[idx]
            f_map_count_1 = conv_layers[idx + 1]
            filter_sizes = conv_filters[idx]
            conv_W, conv_b = FashionCignLite.get_affine_layer_params(
                layer_shape=[filter_sizes, filter_sizes, f_map_count_0, f_map_count_1],
                W_name=network.get_variable_name(name="conv{0}_weight".format(idx), node=node),
                b_name=network.get_variable_name(name="conv{0}_biases".format(idx), node=node))
            conv_weights.append(conv_W)
            conv_biases.append(conv_b)
            # Apply conv layers
            net = FastTreeNetwork.conv_layer(x=net, kernel=conv_W, strides=[1, 1, 1, 1], padding='SAME', bias=conv_b,
                                             node=node, name=conv_name)
            net = tf.nn.relu(net)
            if pool_layers[idx]:
                net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            node.fOpsList.extend([net])
        # FC Layers
        # if is_late_exit and FashionCignLiteEarlyExit.LATE_EXIT_USE_GAP:
        #     net = ResnetGenerator.global_avg_pool(x=net)
        #     print("GAP commited.")
        # else:
        #     net = tf.contrib.layers.flatten(net)
        net = tf.contrib.layers.flatten(net)
        fc_weights = []
        fc_biases = []
        fc_dimensions = [net.get_shape().as_list()[-1]]
        fc_dimensions.extend(fc_layers)
        fc_dimensions.append(network.labelCount)
        for idx in range(len(fc_dimensions) - 1):
            is_last_layer = idx == len(fc_dimensions) - 2
            fc_W_name = "fc{0}_weights".format(idx) if not is_last_layer else "fc_softmax_weights"
            fc_b_name = "fc{0}_b".format(idx) if not is_last_layer else "fc_softmax_b"
            input_dim = fc_dimensions[idx]
            output_dim = fc_dimensions[idx + 1]
            fc_W, fc_b = FashionCignLite.get_affine_layer_params(
                layer_shape=[input_dim, output_dim],
                W_name=network.get_variable_name(name=fc_W_name, node=node),
                b_name=network.get_variable_name(name=fc_b_name, node=node))
            fc_weights.append(fc_W)
            fc_biases.append(fc_b)
            # Apply FC layer
            node.fOpsList.extend([net])
            if not is_last_layer:
                net = FastTreeNetwork.fc_layer(x=net, W=fc_W, b=fc_b, node=node, name=fc_name)
                net = tf.nn.relu(net)
                net = tf.nn.dropout(net, network.classificationDropoutKeepProb)
        return net, fc_weights[-1], fc_biases[-1]

    @staticmethod
    def baseline(network, node):
        network.mask_input_nodes(node=node)
        net = network.dataTensor

        net, softmax_weights, softmax_biases = LeNetBaseline.build_lenet_structure(
            network=network, node=node, parent_F=net,
            conv_layers=LeNetBaseline.CONV_LAYERS,
            conv_filters=LeNetBaseline.CONV_FILTER_SIZES,
            fc_layers=LeNetBaseline.FC_LAYERS,
            pool_layers=LeNetBaseline.CONV_POOL_LAYERS,
            conv_name="conv_op",
            fc_name="fc_op")

        # Loss
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
        explanation = "Lenet CIGN - Baseline\n"
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