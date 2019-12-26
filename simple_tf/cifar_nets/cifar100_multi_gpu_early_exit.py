import tensorflow as tf
import numpy as np

from algorithms.resnet.resnet_generator import ResnetGenerator
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DiscreteParameter, FixedParameter, DecayingParameter
from simple_tf.cifar_nets.cifar100_cign import Cifar100_Cign
from simple_tf.cign.cign_multi_gpu import CignMultiGpu
from simple_tf.cign.cign_multi_gpu_early_exit import CignMultiGpuEarlyExit
from simple_tf.cign.cign_multi_gpu_single_late_exit import CignMultiGpuSingleLateExit
from simple_tf.global_params import GlobalConstants

strides = GlobalConstants.RESNET_HYPERPARAMS.strides
activate_before_residual = GlobalConstants.RESNET_HYPERPARAMS.activate_before_residual
filters = GlobalConstants.RESNET_HYPERPARAMS.num_of_features_per_block
num_of_units_per_block = GlobalConstants.RESNET_HYPERPARAMS.num_residual_units
relu_leakiness = GlobalConstants.RESNET_HYPERPARAMS.relu_leakiness
first_conv_filter_size = GlobalConstants.RESNET_HYPERPARAMS.first_conv_filter_size


class Cifar100_MultiGpuCignEarlyExit(CignMultiGpuEarlyExit):
    # Late Exit
    LATE_EXIT_NUM_OF_CONV_LAYERS = 4
    LATE_EXIT_CONV_LAYER_FEATURE_COUNT = 64
    LATE_EXIT_STRIDE = 1
    LATE_EXIT_FIRST_KERNEL_SIZE = 1

    def __init__(self, degree_list, dataset, network_name):
        node_build_funcs = [Cifar100_Cign.cign_block_func] * (len(degree_list))
        node_build_funcs.append(Cifar100_MultiGpuCignEarlyExit.leaf_func)
        super().__init__(node_build_funcs, None, None, None, None, degree_list, dataset, network_name)

    @staticmethod
    def apply_resnet_multi_exit_losses(x, network, node):
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
        return output, weight, bias

    @staticmethod
    def leaf_func(network, node):
        parent_F, parent_H = network.mask_input_nodes(node=node)
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
        node.fOpsList.append(x)
        # Block body
        if num_of_units_in_this_node > 0:
            # MultiGPU OK
            with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_{0}_0".format(node.depth + 1), node=node)):
                x = ResnetGenerator.bottleneck_residual(x=x, in_filter=in_filter, out_filter=out_filter,
                                                        stride=ResnetGenerator.stride_arr(stride),
                                                        activate_before_residual=_activate_before_residual,
                                                        relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                        bn_momentum=GlobalConstants.BATCH_NORM_DECAY, node=node)
                node.fOpsList.append(x)
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
                    node.fOpsList.append(x)
        assert node.fOpsList[-1] == x
        # Early Exit
        with tf.variable_scope("early_exit"):
            early_exit_features, early_exit_softmax_weights, early_exit_softmax_biases = \
                Cifar100_MultiGpuCignEarlyExit.apply_resnet_multi_exit_losses(x=x, network=network, node=node)
            final_feature_early, logits_early = \
                network.apply_loss(node=node, final_feature=early_exit_features,
                                   softmax_weights=early_exit_softmax_weights,
                                   softmax_biases=early_exit_softmax_biases)
            node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits_early)
        # Late Exit
        assert node.fOpsList[-1] == x
        with tf.variable_scope("late_exit"):
            late_exit_num_of_layers = Cifar100_MultiGpuCignEarlyExit.LATE_EXIT_NUM_OF_CONV_LAYERS
            # MultiGPU OK
            for i in range(late_exit_num_of_layers):
                with tf.variable_scope(UtilityFuncs.get_variable_name(name="late_block_{0}_{1}".format(node.depth + 1, i + 1),
                                                                      node=node)):
                    x = ResnetGenerator.bottleneck_residual(x=x, in_filter=out_filter,
                                                            out_filter=out_filter,
                                                            stride=ResnetGenerator.stride_arr(1),
                                                            activate_before_residual=False,
                                                            relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                            bn_momentum=GlobalConstants.BATCH_NORM_DECAY, node=node)
                    node.fOpsList.append(x)
            late_exit_features, late_exit_softmax_weights, late_exit_softmax_biases = \
                Cifar100_MultiGpuCignEarlyExit.apply_resnet_multi_exit_losses(x=x, network=network, node=node)
            final_feature_late, logits_late = network.apply_late_loss(node=node, final_feature=late_exit_features,
                                                                      softmax_weights=late_exit_softmax_weights,
                                                                      softmax_biases=late_exit_softmax_biases)
            node.evalDict[network.get_variable_name(name="posterior_probs_late", node=node)] = tf.nn.softmax(logits_late)
