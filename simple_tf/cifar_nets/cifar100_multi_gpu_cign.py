import tensorflow as tf
import numpy as np

from auxillary.parameters import DiscreteParameter, FixedParameter, DecayingParameter
from simple_tf.cifar_nets.cifar100_cign import Cifar100_Cign
from simple_tf.cign.cign_multi_gpu import CignMultiGpu
from simple_tf.global_params import GlobalConstants


class Cifar100_MultiGpuCign(CignMultiGpu):
    def __init__(self, degree_list, dataset):
        node_build_funcs = [Cifar100_Cign.cign_block_func] * (len(degree_list) + 1)
        super().__init__(node_build_funcs, None, None, None, None, degree_list, dataset)

    def get_explanation_string(self):
        total_param_count = 0
        for v in tf.trainable_variables():
            total_param_count += np.prod(v.get_shape().as_list())
        explanation = "Resnet-50 Multi Gpu CIGN\n"
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
        GlobalConstants.TOTAL_EPOCH_COUNT = 1800
        GlobalConstants.EPOCH_COUNT = 1800
        GlobalConstants.EPOCH_REPORT_PERIOD = 30
        GlobalConstants.BATCH_SIZE = 750
        GlobalConstants.EVAL_BATCH_SIZE = 250
        GlobalConstants.USE_MULTI_GPU = True
        GlobalConstants.USE_SAMPLING_CIGN = False
        GlobalConstants.USE_RANDOM_SAMPLING = False
        GlobalConstants.INITIAL_LR = 0.1
        GlobalConstants.LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator",
                                                                     value=GlobalConstants.INITIAL_LR,
                                                                     schedule=[(40000, 0.01),
                                                                               (70000, 0.001),
                                                                               (100000, 0.0001)])
        # GlobalConstants.TOTAL_EPOCH_COUNT = 1200
        # GlobalConstants.EPOCH_COUNT = 1200
        # GlobalConstants.EPOCH_REPORT_PERIOD = 10
        # GlobalConstants.BATCH_SIZE = 500
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

        GlobalConstants.GLOBAL_PINNING_DEVICE = "/device:CPU:0"
        self.networkName = "Cifar100_CIGN_MultiGpu"

    def set_hyperparameters(self, **kwargs):
        self.nodeCosts = {node.index: 1 for node in self.topologicalSortedNodes}
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = kwargs["weight_decay_coefficient"]
        GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB = kwargs["classification_keep_probability"]
        GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = kwargs["decision_weight_decay_coefficient"]
        GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT = kwargs["info_gain_balance_coefficient"]
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
                                                                 min_limit=0.4)
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
        super().set_hyperparameters()
