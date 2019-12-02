import tensorflow as tf
import numpy as np
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DiscreteParameter, FixedParameter
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants
from algorithms.resnet.resnet_generator import ResnetGenerator


class FashionNetBaseline(FastTreeNetwork):
    def __init__(self, dataset, network_name):
        node_build_funcs = [FashionNetBaseline.baseline]
        super().__init__(node_build_funcs, None, None, None, None, [], dataset, network_name)

    @staticmethod
    def baseline(network, node):
        network.mask_input_nodes(node=node)
        net = network.dataTensor
        # Convolution Layers
        conv_weights = []
        conv_biases = []
        # Conv Layers
        for idx in range(len(GlobalConstants.FASHION_NET_BASELINE_CONV_FEATURE_MAPS) - 1):
            f_map_count_0 = GlobalConstants.FASHION_NET_BASELINE_CONV_FEATURE_MAPS[idx]
            f_map_count_1 = GlobalConstants.FASHION_NET_BASELINE_CONV_FEATURE_MAPS[idx + 1]
            filter_sizes = GlobalConstants.FASHION_NET_BASELINE_CONV_FILTER_SIZES[idx]
            conv_W = tf.Variable(
                tf.truncated_normal([filter_sizes, filter_sizes, f_map_count_0, f_map_count_1], stddev=0.1,
                                    seed=GlobalConstants.SEED,
                                    dtype=GlobalConstants.DATA_TYPE),
                name=network.get_variable_name(name="conv{0}_weight".format(idx), node=node))
            conv_b = tf.Variable(
                tf.constant(0.1, shape=[f_map_count_1], dtype=GlobalConstants.DATA_TYPE),
                name=network.get_variable_name(name="conv{0}_biases".format(idx), node=node))
            conv_weights.append(conv_W)
            conv_biases.append(conv_b)
            # Apply conv layers
            net = FastTreeNetwork.conv_layer(x=net, kernel=conv_W, strides=[1, 1, 1, 1], padding='SAME', bias=conv_b,
                                             node=node)
            net = tf.nn.relu(net)
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            node.fOpsList.extend([net])
        # FC Layers
        net = tf.contrib.layers.flatten(net)
        flat_dimension_size = net.get_shape().as_list()[-1]
        fc_weights = []
        fc_biases = []
        fc_dimensions = [net.get_shape().as_list()[-1]]
        fc_dimensions.extend(GlobalConstants.FASHION_NET_BASELINE_FC_DIMENSIONS)
        fc_dimensions.append(network.labelCount)
        for idx in range(len(fc_dimensions) - 1):
            is_last_layer = idx == len(fc_dimensions) - 2
            fc_W_name = "fc{0}_weights".format(idx) if not is_last_layer else "fc_softmax_weights"
            fc_b_name = "fc{0}_b".format(idx) if not is_last_layer else "fc_softmax_b"
            input_dim = fc_dimensions[idx]
            output_dim = fc_dimensions[idx + 1]
            fc_W = tf.Variable(tf.truncated_normal([input_dim, output_dim],
                                                   stddev=0.1, seed=GlobalConstants.SEED,
                                                   dtype=GlobalConstants.DATA_TYPE),
                               name=network.get_variable_name(name=fc_W_name, node=node))
            fc_b = tf.Variable(tf.constant(0.1, shape=[output_dim], dtype=GlobalConstants.DATA_TYPE),
                               name=network.get_variable_name(name=fc_b_name, node=node))
            fc_weights.append(fc_W)
            fc_biases.append(fc_b)
            if not is_last_layer:
                net = tf.nn.relu(tf.matmul(net, fc_W) + fc_b)
                net = tf.nn.dropout(net, network.classificationDropoutKeepProb)
        # Loss
        final_feature, logits = network.apply_loss(node=node, final_feature=net,
                                                   softmax_weights=fc_weights[-1], softmax_biases=fc_biases[-1])
        # Evaluation
        node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
        node.evalDict[network.get_variable_name(name="labels", node=node)] = node.labelTensor

    def get_explanation_string(self):
        total_param_count = 0
        for v in tf.trainable_variables():
            total_param_count += np.prod(v.get_shape().as_list())
        # Tree
        explanation = "Fashion Net - Baseline\n"
        # "(Lr=0.01, - Decay 1/(1 + i*0.0001) at each i. iteration)\n"
        explanation += "FASHION_NET_BASELINE_CONV_FEATURE_MAPS:{0}\n"\
            .format(GlobalConstants.FASHION_NET_BASELINE_CONV_FEATURE_MAPS)
        explanation += "FASHION_NET_BASELINE_CONV_FILTER_SIZES:{0}\n"\
            .format(GlobalConstants.FASHION_NET_BASELINE_CONV_FILTER_SIZES)
        explanation += "FASHION_NET_BASELINE_FC_DIMENSIONS:{0}\n"\
            .format(GlobalConstants.FASHION_NET_BASELINE_FC_DIMENSIONS)
        explanation += "Using Fast Tree Version:{0}\n".format(GlobalConstants.USE_FAST_TREE_MODE)
        explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
        explanation += "********Lr Settings********\n"
        explanation += GlobalConstants.LEARNING_RATE_CALCULATOR.get_explanation()
        explanation += "********Lr Settings********\n"
        explanation += "Param Count:{0}\n".format(total_param_count)
        explanation += "Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
        explanation += "Decision Wd:{0}\n".format(GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT)
        explanation += "Use Decision Dropout:{0}\n".format(GlobalConstants.USE_DROPOUT_FOR_DECISION)
        explanation += "Use Decision Augmentation:{0}\n".format(GlobalConstants.USE_DECISION_AUGMENTATION)
        explanation += "Use Classification Dropout:{0}\n".format(GlobalConstants.USE_DROPOUT_FOR_CLASSIFICATION)
        explanation += "Classification Dropout Probability:{0}\n".format(
            GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB)
        explanation += "TRAINING PARAMETERS:\n"
        explanation += super().get_explanation_string()
        return explanation

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
        self.networkName = "FashionNet_Baseline"

    def set_hyperparameters(self, **kwargs):
        # Regularization Parameters
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = kwargs["weight_decay_coefficient"]
        GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB = kwargs["classification_keep_probability"]
        GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = kwargs["decision_weight_decay_coefficient"]
        GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT = kwargs["info_gain_balance_coefficient"]

        # Decision Loss Coefficient
        self.decisionLossCoefficientCalculator = FixedParameter(name="decision_loss_coefficient_calculator",
                                                                value=0.0)
