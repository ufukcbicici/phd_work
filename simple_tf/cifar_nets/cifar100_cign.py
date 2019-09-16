import tensorflow as tf
import numpy as np
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DecayingParameter, FixedParameter, DiscreteParameter
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants
from algorithms.resnet.resnet_generator import ResnetGenerator

strides = GlobalConstants.RESNET_HYPERPARAMS.strides
activate_before_residual = GlobalConstants.RESNET_HYPERPARAMS.activate_before_residual
filters = GlobalConstants.RESNET_HYPERPARAMS.num_of_features_per_block
num_of_units_per_block = GlobalConstants.RESNET_HYPERPARAMS.num_residual_units
relu_leakiness = GlobalConstants.RESNET_HYPERPARAMS.relu_leakiness
first_conv_filter_size = GlobalConstants.RESNET_HYPERPARAMS.first_conv_filter_size

assert len(num_of_units_per_block) + 1 == len(filters)


class Cifar100_Cign(FastTreeNetwork):
    def __init__(self, degree_list, dataset):
        node_build_funcs = [Cifar100_Cign.cign_block_func] * (len(degree_list) + 1)
        super().__init__(node_build_funcs, None, None, None, None, degree_list, dataset)

    @staticmethod
    def apply_router_transformation(network, net, node, decision_feature_size):
        h_net = net
        net_shape = h_net.get_shape().as_list()
        # Global Average Pooling
        h_net = tf.nn.avg_pool(h_net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
        net_shape = h_net.get_shape().as_list()
        h_net = tf.reshape(h_net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
        feature_size = h_net.get_shape().as_list()[-1]
        # MultiGPU OK
        fc_h_weights = UtilityFuncs.create_variable(
            name=network.get_variable_name(name="fc_decision_weights", node=node),
            shape=[feature_size, decision_feature_size],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.truncated_normal(
                [feature_size, decision_feature_size], stddev=0.1, seed=GlobalConstants.SEED,
                dtype=GlobalConstants.DATA_TYPE))
        # MultiGPU OK
        fc_h_bias = UtilityFuncs.create_variable(
            name=network.get_variable_name(name="fc_decision_bias", node=node),
            shape=[decision_feature_size],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.constant(0.1, shape=[decision_feature_size], dtype=GlobalConstants.DATA_TYPE))
        h_net = FastTreeNetwork.fc_layer(x=h_net, W=fc_h_weights, b=fc_h_bias, node=node)
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
    def apply_resnet_loss(x, network, node):
        # Logit Layers
        with tf.variable_scope(UtilityFuncs.get_variable_name(name="unit_last", node=node)):
            x = ResnetGenerator.get_output(x=x, is_train=network.isTrain, leakiness=relu_leakiness,
                                           bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
        # assert len(net_shape) == 4
        # x = tf.reshape(x, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
        output = x
        out_dim = network.labelCount
        # MultiGPU OK
        weight = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="fc_softmax_weights", node=node),
            shape=[output.get_shape()[1], out_dim],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        # MultiGPU OK
        bias = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="fc_softmax_biases", node=node),
            shape=[out_dim],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.constant_initializer())
        # Loss
        # MultiGPU OK
        final_feature, logits = network.apply_loss(node=node, final_feature=output,
                                                   softmax_weights=weight, softmax_biases=bias)
        # Evaluation
        node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
        node.evalDict[network.get_variable_name(name="labels", node=node)] = node.labelTensor

    @staticmethod
    def cign_block_func(network, node):
        # Block Parameters
        in_filter = filters[node.depth]
        out_filter = filters[node.depth + 1]
        stride = strides[node.depth]
        _activate_before_residual = activate_before_residual[node.depth]
        num_of_units_in_this_node = num_of_units_per_block[node.depth]
        # Input to the node
        parent_F, parent_H = network.mask_input_nodes(node=node)
        if node.isRoot:
            x = ResnetGenerator.get_input(input=network.dataTensor, out_filters=in_filter,
                                          first_conv_filter_size=first_conv_filter_size, node=node)
        else:
            x = parent_F
        # Block body
        if num_of_units_in_this_node > 0:
            # MultiGPU OK
            with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_{0}_0".format(node.depth + 1), node=node)):
                x = ResnetGenerator.bottleneck_residual(x=x, in_filter=in_filter, out_filter=out_filter,
                                                        stride=ResnetGenerator.stride_arr(stride),
                                                        activate_before_residual=_activate_before_residual,
                                                        relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                        bn_momentum=GlobalConstants.BATCH_NORM_DECAY, node=node)
            # MultiGPU OK
            for i in range(num_of_units_in_this_node - 1):
                with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_{0}_{1}".format(node.depth + 1, i + 1),
                                                                      node=node)):
                    x = ResnetGenerator.bottleneck_residual(x=x, in_filter=out_filter,
                                                            out_filter=out_filter,
                                                            stride=ResnetGenerator.stride_arr(1),
                                                            activate_before_residual=False,
                                                            relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                            bn_momentum=GlobalConstants.BATCH_NORM_DECAY, node=node)
        node.fOpsList.extend([x])
        if not node.isLeaf:
            # ***************** H: Connected to F *****************
            with tf.variable_scope(UtilityFuncs.get_variable_name(name="decision_variables", node=node)):
                Cifar100_Cign.apply_router_transformation(net=x, network=network, node=node,
                                                          decision_feature_size=GlobalConstants.RESNET_DECISION_DIMENSION)
            # ***************** H: Connected to F *****************
        else:
            Cifar100_Cign.apply_resnet_loss(x=x, network=network, node=node)

    @staticmethod
    def residue_network_func(network):
        return tf.constant(0.0)

    @staticmethod
    def grad_func(network):
        pass

    @staticmethod
    def tensorboard_func(network):
        pass

    def get_explanation_string(self):
        total_param_count = 0
        for v in tf.trainable_variables():
            total_param_count += np.prod(v.get_shape().as_list())
        explanation = "Resnet-50 CIGN Tests\n"
        # explanation = "Resnet-50 CIGN Random Sampling Routing Tests\n"
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
        explanation += "Classification Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
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
        explanation += "Softmax Decay Initial:{0}\n".format(GlobalConstants.RESNET_SOFTMAX_DECAY_INITIAL)
        explanation += "Softmax Decay Coefficient:{0}\n".format(GlobalConstants.RESNET_SOFTMAX_DECAY_COEFFICIENT)
        explanation += "Softmax Decay Period:{0}\n".format(GlobalConstants.RESNET_SOFTMAX_DECAY_PERIOD)
        explanation += "Softmax Min Limit:{0}\n".format(GlobalConstants.RESNET_SOFTMAX_DECAY_MIN_LIMIT)
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
        explanation += "***** ResNet Parameters *****\n"
        explanation += str(GlobalConstants.RESNET_HYPERPARAMS)
        explanation += "\nUse Sampling CIGN:{0}".format(GlobalConstants.USE_SAMPLING_CIGN)
        explanation += "\nUse Random Sampling CIGN:{0}".format(GlobalConstants.USE_RANDOM_SAMPLING)
        explanation += "\nPinning Device:{0}".format(GlobalConstants.GLOBAL_PINNING_DEVICE)
        explanation += "TRAINING PARAMETERS:\n"
        explanation += super().get_explanation_string()
        return explanation

    def set_training_parameters(self):
        # Training Parameters
        GlobalConstants.TOTAL_EPOCH_COUNT = 600
        GlobalConstants.EPOCH_COUNT = 600
        GlobalConstants.EPOCH_REPORT_PERIOD = 5
        GlobalConstants.BATCH_SIZE = 250
        GlobalConstants.EVAL_BATCH_SIZE = 250
        GlobalConstants.USE_MULTI_GPU = False
        GlobalConstants.USE_SAMPLING_CIGN = False
        GlobalConstants.USE_RANDOM_SAMPLING = False
        GlobalConstants.INITIAL_LR = 0.1
        GlobalConstants.LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator",
                                                                     value=GlobalConstants.INITIAL_LR,
                                                                     schedule=[(40000, 0.01),
                                                                               (70000, 0.001),
                                                                               (100000, 0.0001)])
        GlobalConstants.GLOBAL_PINNING_DEVICE = "/device:GPU:0"
        self.networkName = "Cifar100_CIGN_Single_GPU"

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
                                                             decay_period=35000,
                                                             min_limit=0.4)
            # Softmax Decay
            decay_name = self.get_variable_name(name="softmax_decay", node=node)
            node.softmaxDecayCalculator = DecayingParameter(name=decay_name,
                                                            value=GlobalConstants.RESNET_SOFTMAX_DECAY_INITIAL,
                                                            decay=GlobalConstants.RESNET_SOFTMAX_DECAY_COEFFICIENT,
                                                            decay_period=GlobalConstants.RESNET_SOFTMAX_DECAY_PERIOD,
                                                            min_limit=GlobalConstants.RESNET_SOFTMAX_DECAY_MIN_LIMIT)
        GlobalConstants.SOFTMAX_TEST_TEMPERATURE = 50.0
