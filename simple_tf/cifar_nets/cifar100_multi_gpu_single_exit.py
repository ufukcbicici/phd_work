import tensorflow as tf
import numpy as np

from algorithms.resnet.resnet_generator import ResnetGenerator
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DiscreteParameter, FixedParameter, DecayingParameter
from simple_tf.cifar_nets.cifar100_cign import Cifar100_Cign
from simple_tf.cign.cign_multi_gpu import CignMultiGpu
from simple_tf.cign.cign_multi_gpu_single_late_exit import CignMultiGpuSingleLateExit
from simple_tf.global_params import GlobalConstants

strides = GlobalConstants.RESNET_HYPERPARAMS.strides
activate_before_residual = GlobalConstants.RESNET_HYPERPARAMS.activate_before_residual
filters = GlobalConstants.RESNET_HYPERPARAMS.num_of_features_per_block
num_of_units_per_block = GlobalConstants.RESNET_HYPERPARAMS.num_residual_units
relu_leakiness = GlobalConstants.RESNET_HYPERPARAMS.relu_leakiness
first_conv_filter_size = GlobalConstants.RESNET_HYPERPARAMS.first_conv_filter_size


class Cifar100_MultiGpuCignSingleLateExit(CignMultiGpuSingleLateExit):
    # Late Exit
    LATE_EXIT_NUM_OF_CONV_LAYERS = 16
    LATE_EXIT_CONV_LAYER_FEATURE_COUNT = 64
    LATE_EXIT_STRIDE = 2
    LATE_EXIT_FIRST_KERNEL_SIZE = 1

    def __init__(self, degree_list, dataset, network_name):
        node_build_funcs = [Cifar100_Cign.cign_block_func] * (len(degree_list))
        node_build_funcs.append(Cifar100_MultiGpuCignSingleLateExit.leaf_func)
        super().__init__(node_build_funcs, None, None, None, None, degree_list, dataset, network_name,
                         late_exit_func=Cifar100_MultiGpuCignSingleLateExit.late_exit_func)

    @staticmethod
    def leaf_func(network, node):
        Cifar100_Cign.cign_block_func(network=network, node=node)
        network.leafNodeOutputsToLateExit[node.index] = \
            node.fOpsList[Cifar100_MultiGpuCignSingleLateExit.LATE_EXIT_NUM_OF_CONV_LAYERS]

    @staticmethod
    def late_exit_func(network, node, x):
        num_of_layers = Cifar100_MultiGpuCignSingleLateExit.LATE_EXIT_NUM_OF_CONV_LAYERS
        num_of_features = Cifar100_MultiGpuCignSingleLateExit.LATE_EXIT_CONV_LAYER_FEATURE_COUNT
        first_layer_stride = Cifar100_MultiGpuCignSingleLateExit.LATE_EXIT_STRIDE
        first_kernel_size = Cifar100_MultiGpuCignSingleLateExit.LATE_EXIT_FIRST_KERNEL_SIZE
        input_shape = x.get_shape().as_list()
        # First: Convert the concatenated input to a ResNet block.
        x = ResnetGenerator.get_input(input=x, out_filters=num_of_features, first_conv_filter_size=first_kernel_size,
                                      node=node)
        for layer_id in range(num_of_layers):
            with tf.variable_scope(
                    UtilityFuncs.get_variable_name(name="block_{0}_{1}".format(node.depth + 1, layer_id + 1),
                                                   node=node)):
                x = ResnetGenerator.bottleneck_residual(
                    x=x,
                    in_filter=num_of_features,
                    out_filter=num_of_features,
                    stride=ResnetGenerator.stride_arr(first_layer_stride)
                    if layer_id == 0 else ResnetGenerator.stride_arr(1),
                    activate_before_residual=layer_id == 0,
                    relu_leakiness=relu_leakiness,
                    is_train=network.isTrain,
                    bn_momentum=GlobalConstants.BATCH_NORM_DECAY,
                    node=node)
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
        return output, weight, bias

    def get_explanation_string(self):
        total_param_count = 0
        for v in tf.trainable_variables():
            total_param_count += np.prod(v.get_shape().as_list())
        explanation = "Resnet-50 Multi Gpu Single Late Exit \n"
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
        explanation += "\nLATE_EXIT_NUM_OF_CONV_LAYERS:{0}".format(
            Cifar100_MultiGpuCignSingleLateExit.LATE_EXIT_NUM_OF_CONV_LAYERS)
        explanation += "\nLATE_EXIT_CONV_LAYER_FEATURE_COUNT:{0}".format(
            Cifar100_MultiGpuCignSingleLateExit.LATE_EXIT_CONV_LAYER_FEATURE_COUNT)
        explanation += "\nLATE_EXIT_STRIDE:{0}".format(Cifar100_MultiGpuCignSingleLateExit.LATE_EXIT_STRIDE)
        return explanation

    def set_training_parameters(self):
        # Training Parameters
        # GlobalConstants.TOTAL_EPOCH_COUNT = 1800
        # GlobalConstants.EPOCH_COUNT = 1800
        # GlobalConstants.EPOCH_REPORT_PERIOD = 30
        # GlobalConstants.BATCH_SIZE = 750
        # GlobalConstants.EVAL_BATCH_SIZE = 250
        # GlobalConstants.USE_MULTI_GPU = True
        # GlobalConstants.USE_SAMPLING_CIGN = False
        # GlobalConstants.USE_RANDOM_SAMPLING = False
        # GlobalConstants.INITIAL_LR = 0.1
        # GlobalConstants.LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator",
        #                                                              value=GlobalConstants.INITIAL_LR,
        #                                                              schedule=[(40000, 0.01),
        #                                                                        (70000, 0.001),
        #                                                                        (100000, 0.0001)])
        GlobalConstants.TOTAL_EPOCH_COUNT = 1200
        GlobalConstants.EPOCH_COUNT_INVALID = 1200
        GlobalConstants.EPOCH_REPORT_PERIOD = 20
        GlobalConstants.BATCH_SIZE = 500
        GlobalConstants.EVAL_BATCH_SIZE = 250
        GlobalConstants.USE_MULTI_GPU = True
        GlobalConstants.USE_SAMPLING_CIGN = False
        GlobalConstants.USE_RANDOM_SAMPLING = False
        GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING = 10
        GlobalConstants.INITIAL_LR = 0.1
        GlobalConstants.LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator",
                                                                     value=GlobalConstants.INITIAL_LR,
                                                                     schedule=[(40000, 0.01),
                                                                               (70000, 0.001),
                                                                               (100000, 0.0001)])
        # GlobalConstants.TOTAL_EPOCH_COUNT = 1800
        # GlobalConstants.EPOCH_COUNT = 1800
        # GlobalConstants.EPOCH_REPORT_PERIOD = 1
        # GlobalConstants.BATCH_SIZE = 250
        # GlobalConstants.EVAL_BATCH_SIZE = 125
        # GlobalConstants.USE_MULTI_GPU = True
        # GlobalConstants.USE_SAMPLING_CIGN = False
        # GlobalConstants.USE_RANDOM_SAMPLING = False
        # GlobalConstants.INITIAL_LR = 0.1
        # GlobalConstants.LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator",
        #                                                              value=GlobalConstants.INITIAL_LR,
        #                                                              schedule=[(40000, 0.01),
        #                                                                        (70000, 0.001),
        #                                                                        (100000, 0.0001)])

        GlobalConstants.GLOBAL_PINNING_DEVICE = "/device:CPU:0"
        self.networkName = "Cifar100_CIGN_MultiGpuSingleLateExit"

    def set_hyperparameters(self, **kwargs):
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = kwargs["weight_decay_coefficient"]
        GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB = kwargs["classification_keep_probability"]
        GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = kwargs["decision_weight_decay_coefficient"]
        GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT = kwargs["info_gain_balance_coefficient"]
        GlobalConstants.EARLY_EXIT_WEIGHT = kwargs["early_exit_weight"]
        GlobalConstants.LATE_EXIT_WEIGHT = kwargs["late_exit_weight"]
        for tower_id, tpl in enumerate(self.towerNetworks):
            network = tpl[1]
            network.decisionDropoutKeepProbCalculator = FixedParameter(name="decision_dropout_prob",
                                                                       value=kwargs["decision_keep_probability"])

            # Noise Coefficient
            network.noiseCoefficientCalculator = DecayingParameter(name="noise_coefficient_calculator", value=0.0,
                                                                   decay=0.0,
                                                                   decay_period=1,
                                                                   min_limit=0.0)
            # Decision Loss Coefficient
            network.decisionLossCoefficientCalculator = FixedParameter(name="decision_loss_coefficient_calculator",
                                                                       value=1.0)
            for node in network.topologicalSortedNodes:
                if node.isLeaf:
                    continue
                # Probability Threshold
                node_degree = GlobalConstants.TREE_DEGREE_LIST[node.depth]
                initial_value = 1.0 / float(node_degree)
                threshold_name = network.get_variable_name(name="prob_threshold_calculator", node=node)
                node.probThresholdCalculator = DecayingParameter(name=threshold_name, value=initial_value,
                                                                 decay=0.8,
                                                                 decay_period=35000,
                                                                 min_limit=initial_value * 0.8)
                # Softmax Decay
                decay_name = self.get_variable_name(name="softmax_decay", node=node)
                node.softmaxDecayCalculator = DecayingParameter(name=decay_name,
                                                                value=GlobalConstants.RESNET_SOFTMAX_DECAY_INITIAL,
                                                                decay=GlobalConstants.RESNET_SOFTMAX_DECAY_COEFFICIENT,
                                                                decay_period=GlobalConstants.RESNET_SOFTMAX_DECAY_PERIOD,
                                                                min_limit=GlobalConstants.RESNET_SOFTMAX_DECAY_MIN_LIMIT)

        GlobalConstants.SOFTMAX_TEST_TEMPERATURE = 50.0
        self.decisionDropoutKeepProbCalculator = self.towerNetworks[0][1].decisionDropoutKeepProbCalculator
        self.noiseCoefficientCalculator = self.towerNetworks[0][1].noiseCoefficientCalculator
        self.decisionLossCoefficientCalculator = self.towerNetworks[0][1].decisionLossCoefficientCalculator
