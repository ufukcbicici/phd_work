from tf_2_cign.cign import Cign
import tensorflow as tf

from tf_2_cign.cign_no_mask import CignNoMask
from tf_2_cign.fashion_net.fashion_net_inner_node_func import FashionNetInnerNodeFunc
from tf_2_cign.fashion_net.fashion_net_leaf_node_func import FashionNetLeafNodeFunc
from tf_2_cign.utilities import Utilities


class FashionCign(CignNoMask):
    def __init__(self,
                 batch_size,
                 input_dims,
                 node_degrees,
                 filter_counts,
                 kernel_sizes,
                 hidden_layers,
                 decision_drop_probability,
                 classification_drop_probability,
                 decision_wd,
                 classification_wd,
                 decision_dimensions,
                 class_count,
                 information_gain_balance_coeff,
                 softmax_decay_controller,
                 learning_rate_schedule,
                 decision_loss_coeff):
        super().__init__(batch_size,
                         input_dims,
                         class_count,
                         node_degrees,
                         decision_drop_probability,
                         classification_drop_probability,
                         decision_wd,
                         classification_wd,
                         information_gain_balance_coeff,
                         softmax_decay_controller,
                         learning_rate_schedule,
                         decision_loss_coeff)
        self.filterCounts = filter_counts
        self.kernelSizes = kernel_sizes
        self.hiddenLayers = hidden_layers
        self.decisionDimensions = decision_dimensions

    def node_build_func(self, node):
        if not node.isLeaf:
            node_func = FashionNetInnerNodeFunc(node=node,
                                                network=self,
                                                kernel_size=self.kernelSizes[node.depth],
                                                num_of_filters=self.filterCounts[node.depth],
                                                strides=(1, 1),
                                                activation="relu",
                                                decision_dim=self.decisionDimensions[node.depth],
                                                decision_drop_probability=self.decisionDropProbability,
                                                use_bias=True)
        else:
            node_func = FashionNetLeafNodeFunc(node=node,
                                               network=self,
                                               kernel_size=self.kernelSizes[node.depth],
                                               num_of_filters=self.filterCounts[node.depth],
                                               strides=(1, 1),
                                               activation="relu",
                                               hidden_layer_dims=self.hiddenLayers,
                                               classification_dropout_prob=self.classificationDropProbability,
                                               use_bias=True)

        return node_func

    def get_explanation_string(self):
        # total_param_count = 0
        # for v in tf.trainable_variables():
        #     total_param_count += np.prod(v.get_shape().as_list())
        # # Tree
        # explanation = "Fashion CIGN; Non Masked\n"
        # # "(Lr=0.01, - Decay 1/(1 + i*0.0001) at each i. iteration)\n"
        # explanation += "Using Fast Tree Version:{0}\n".format(GlobalConstants.USE_FAST_TREE_MODE)
        # explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
        # explanation += "Tree Degree:{0}\n".format(GlobalConstants.TREE_DEGREE_LIST)
        # explanation += "Concat Trick:{0}\n".format(GlobalConstants.USE_CONCAT_TRICK)
        # explanation += "Info Gain:{0}\n".format(GlobalConstants.USE_INFO_GAIN_DECISION)
        # explanation += "Using Effective Sample Counts:{0}\n".format(GlobalConstants.USE_EFFECTIVE_SAMPLE_COUNTS)
        # explanation += "Gradient Type:{0}\n".format(GlobalConstants.GRADIENT_TYPE)
        # explanation += "Probability Threshold:{0}\n".format(GlobalConstants.USE_PROBABILITY_THRESHOLD)
        # explanation += "********Lr Settings********\n"
        # explanation += GlobalConstants.LEARNING_RATE_CALCULATOR.get_explanation()
        # explanation += "********Lr Settings********\n"
        # if not self.isBaseline:
        #     explanation += "********Decision Loss Weight Settings********\n"
        #     explanation += self.decisionLossCoefficientCalculator.get_explanation()
        #     explanation += "********Decision Loss Weight Settings********\n"
        # explanation += "Use Unified Batch Norm:{0}\n".format(GlobalConstants.USE_UNIFIED_BATCH_NORM)
        # explanation += "Batch Norm Decay:{0}\n".format(GlobalConstants.BATCH_NORM_DECAY)
        # explanation += "Param Count:{0}\n".format(total_param_count)
        # explanation += "Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
        # explanation += "Decision Wd:{0}\n".format(GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT)
        # explanation += "Residue Loss Coefficient:{0}\n".format(GlobalConstants.RESIDUE_LOSS_COEFFICIENT)
        # explanation += "Residue Affects All Network:{0}\n".format(GlobalConstants.RESIDE_AFFECTS_WHOLE_NETWORK)
        # explanation += "Using Info Gain:{0}\n".format(GlobalConstants.USE_INFO_GAIN_DECISION)
        # explanation += "Info Gain Loss Lambda:{0}\n".format(GlobalConstants.DECISION_LOSS_COEFFICIENT)
        # explanation += "Use Batch Norm Before Decisions:{0}\n".format(GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING)
        # explanation += "Use Trainable Batch Norm Parameters:{0}\n".format(
        #     GlobalConstants.USE_TRAINABLE_PARAMS_WITH_BATCH_NORM)
        # explanation += "Hyperplane bias at 0.0\n"
        # explanation += "Using Convolutional Routing Networks:{0}\n".format(GlobalConstants.USE_CONVOLUTIONAL_H_PIPELINE)
        # explanation += "Softmax Decay Initial:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_INITIAL)
        # explanation += "Softmax Decay Coefficient:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_COEFFICIENT)
        # explanation += "Softmax Decay Period:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_PERIOD)
        # explanation += "Softmax Min Limit:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_MIN_LIMIT)
        # explanation += "Softmax Test Temperature:{0}\n".format(GlobalConstants.SOFTMAX_TEST_TEMPERATURE)
        # explanation += "Reparametrized Noise:{0}\n".format(GlobalConstants.USE_REPARAMETRIZATION_TRICK)
        # # for node in network.topologicalSortedNodes:
        # #     if node.isLeaf:
        # #         continue
        # #     explanation += "Node {0} Info Gain Balance Coefficient:{1}\n".format(node.index,
        # #                                                                          node.infoGainBalanceCoefficient)
        # explanation += "Info Gain Balance Coefficient:{0}\n".format(GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT)
        # explanation += "Adaptive Weight Decay:{0}\n".format(GlobalConstants.USE_ADAPTIVE_WEIGHT_DECAY)
        # if GlobalConstants.USE_REPARAMETRIZATION_TRICK:
        #     explanation += "********Reparametrized Noise Settings********\n"
        #     explanation += "Noise Coefficient Initial Value:{0}\n".format(self.noiseCoefficientCalculator.value)
        #     explanation += "Noise Coefficient Decay Step:{0}\n".format(self.noiseCoefficientCalculator.decayPeriod)
        #     explanation += "Noise Coefficient Decay Ratio:{0}\n".format(self.noiseCoefficientCalculator.decay)
        #     explanation += "********Reparametrized Noise Settings********\n"
        # explanation += "Use Decision Dropout:{0}\n".format(GlobalConstants.USE_DROPOUT_FOR_DECISION)
        # explanation += "Use Decision Augmentation:{0}\n".format(GlobalConstants.USE_DECISION_AUGMENTATION)
        # if GlobalConstants.USE_DROPOUT_FOR_DECISION:
        #     explanation += "********Decision Dropout Schedule********\n"
        #     explanation += "Iteration:{0} Probability:{1}\n".format(0, GlobalConstants.DROPOUT_INITIAL_PROB)
        #     for tpl in GlobalConstants.DROPOUT_SCHEDULE:
        #         explanation += "Iteration:{0} Probability:{1}\n".format(tpl[0], tpl[1])
        #     explanation += "********Decision Dropout Schedule********\n"
        # explanation += "Use Classification Dropout:{0}\n".format(GlobalConstants.USE_DROPOUT_FOR_CLASSIFICATION)
        # explanation += "Classification Dropout Probability:{0}\n".format(
        #     GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB)
        # explanation += "Decision Dropout Probability:{0}\n".format(self.decisionDropoutKeepProbCalculator.value)
        # if GlobalConstants.USE_PROBABILITY_THRESHOLD:
        #     for node in self.topologicalSortedNodes:
        #         if node.isLeaf:
        #             continue
        #         explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
        #         explanation += node.probThresholdCalculator.get_explanation()
        #         explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
        # explanation += "Use Softmax Compression:{0}\n".format(GlobalConstants.USE_SOFTMAX_DISTILLATION)
        # explanation += "Waiting Epochs for Softmax Compression:{0}\n".format(GlobalConstants.MODE_WAIT_EPOCHS)
        # explanation += "Mode Percentile:{0}\n".format(GlobalConstants.PERCENTILE_THRESHOLD)
        # explanation += "Mode Tracking Strategy:{0}\n".format(GlobalConstants.MODE_TRACKING_STRATEGY)
        # explanation += "Mode Max Class Count:{0}\n".format(GlobalConstants.MAX_MODE_CLASSES)
        # explanation += "Mode Computation Strategy:{0}\n".format(GlobalConstants.MODE_COMPUTATION_STRATEGY)
        # explanation += "Constrain Softmax Compression With Label Count:{0}\n".format(GlobalConstants.
        #                                                                              CONSTRAIN_WITH_COMPRESSION_LABEL_COUNT)
        # explanation += "Softmax Distillation Cross Validation Count:{0}\n". \
        #     format(GlobalConstants.SOFTMAX_DISTILLATION_CROSS_VALIDATION_COUNT)
        # explanation += "Softmax Distillation Strategy:{0}\n". \
        #     format(GlobalConstants.SOFTMAX_COMPRESSION_STRATEGY)
        # explanation += "F Conv1:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_1_SIZE,
        #                                                        GlobalConstants.FASHION_F_NUM_FILTERS_1)
        # explanation += "F Conv2:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_2_SIZE,
        #                                                        GlobalConstants.FASHION_F_NUM_FILTERS_2)
        # explanation += "F Conv3:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_3_SIZE,
        #                                                        GlobalConstants.FASHION_F_NUM_FILTERS_3)
        # explanation += "F FC1:{0} Units\n".format(GlobalConstants.FASHION_F_FC_1)
        # explanation += "F FC2:{0} Units\n".format(GlobalConstants.FASHION_F_FC_2)
        # explanation += "F Residue FC:{0} Units\n".format(GlobalConstants.FASHION_F_RESIDUE)
        # explanation += "Residue Hidden Layer Count:{0}\n".format(GlobalConstants.FASHION_F_RESIDUE_LAYER_COUNT)
        # explanation += "Residue Use Dropout:{0}\n".format(GlobalConstants.FASHION_F_RESIDUE_USE_DROPOUT)
        # explanation += "H Conv1:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_H_FILTERS_1_SIZE,
        #                                                        GlobalConstants.FASHION_H_NUM_FILTERS_1)
        # explanation += "H Conv2:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_H_FILTERS_2_SIZE,
        #                                                        GlobalConstants.FASHION_H_NUM_FILTERS_2)
        # explanation += "FASHION_NO_H_FROM_F_UNITS_1:{0} Units\n".format(GlobalConstants.FASHION_NO_H_FROM_F_UNITS_1)
        # explanation += "FASHION_NO_H_FROM_F_UNITS_2:{0} Units\n".format(GlobalConstants.FASHION_NO_H_FROM_F_UNITS_2)
        # explanation += "Use Class Weighting:{0}\n".format(GlobalConstants.USE_CLASS_WEIGHTING)
        # explanation += "Class Weight Running Average:{0}\n".format(GlobalConstants.CLASS_WEIGHT_RUNNING_AVERAGE)
        # explanation += "Zero Label Count Epsilon:{0}\n".format(GlobalConstants.LABEL_EPSILON)
        # explanation += "TRAINING PARAMETERS:\n"
        explanation = ""
        explanation += super().get_explanation_string()
        return explanation
