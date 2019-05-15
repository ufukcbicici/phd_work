import tensorflow as tf
import numpy as np

from collections import deque

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants
from simple_tf.node import Node


class CignMultiGpu(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset):
        super().__init__(node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset)
        self.currentTowerId = None
        self.towerCount = None
        self.towerBatchSize = None

    def build_network(self):
        # Build the tree topologically and create the Tensorflow placeholders
        self.build_tree()
        # Build symbolic networks
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        self.isBaseline = len(self.topologicalSortedNodes) == 1
        # Disable some properties if we are using a baseline
        if self.isBaseline:
            GlobalConstants.USE_INFO_GAIN_DECISION = False
            GlobalConstants.USE_CONCAT_TRICK = False
            GlobalConstants.USE_PROBABILITY_THRESHOLD = False
        # Build all symbolic networks in each node, using multi gpu support
        gpu_names = UtilityFuncs.get_available_devices()
        self.towerCount = len(gpu_names)
        assert GlobalConstants.BATCH_SIZE % self.towerCount == 0
        self.towerBatchSize = GlobalConstants.BATCH_SIZE / self.towerCount
        for tower_id in range(self.towerCount):
            self.currentTowerId = tower_id
            with tf.device('/gpu:%d' % tower_id):
                with tf.name_scope("tower_{0}".format(tower_id)):
                    # Build all symbolic networks in each node
                    for node in self.topologicalSortedNodes:
                        self.nodeBuildFuncs[node.depth](node=node, network=self)
        # Build main classification loss
        self.build_main_loss()



    # # Build information gain loss
    # self.build_decision_loss()
    # # Build regularization loss
    # self.build_regularization_loss()
    # # Final Loss
    # self.finalLoss = self.mainLoss + self.regularizationLoss + self.decisionLoss + self.residueLoss
    # # Build optimizer
    # self.globalCounter = tf.Variable(0, trainable=False)
    # boundaries = [tpl[0] for tpl in GlobalConstants.LEARNING_RATE_CALCULATOR.schedule]
    # values = [GlobalConstants.INITIAL_LR]
    # values.extend([tpl[1] for tpl in GlobalConstants.LEARNING_RATE_CALCULATOR.schedule])
    # self.learningRate = tf.train.piecewise_constant(self.globalCounter, boundaries, values)
    # self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # # pop_var = tf.Variable(name="pop_var", initial_value=tf.constant(0.0, shape=(16, )), trainable=False)
    # # pop_var_assign_op = tf.assign(pop_var, tf.constant(45.0, shape=(16, )))
    # with tf.control_dependencies(self.extra_update_ops):
    #     self.optimizer = tf.train.MomentumOptimizer(self.learningRate, 0.9).minimize(self.finalLoss,
    #                                                                                  global_step=self.globalCounter)
    # # Prepare tensors to evaluate
    # for node in self.topologicalSortedNodes:
    #     # if node.isLeaf:
    #     #     continue
    #     # F
    #     f_output = node.fOpsList[-1]
    #     self.evalDict["Node{0}_F".format(node.index)] = f_output
    #     # H
    #     if len(node.hOpsList) > 0:
    #         h_output = node.hOpsList[-1]
    #         self.evalDict["Node{0}_H".format(node.index)] = h_output
    #     # Activations
    #     for k, v in node.activationsDict.items():
    #         self.evalDict["Node{0}_activation_from_{1}".format(node.index, k)] = v
    #     # Decision masks
    #     for k, v in node.maskTensors.items():
    #         self.evalDict["Node{0}_{1}".format(node.index, v.name)] = v
    #     # Evaluation outputs
    #     for k, v in node.evalDict.items():
    #         self.evalDict[k] = v
    #     # Label outputs
    #     if node.labelTensor is not None:
    #         self.evalDict["Node{0}_label_tensor".format(node.index)] = node.labelTensor
    #         # Sample indices
    #         self.evalDict["Node{0}_indices_tensor".format(node.index)] = node.indicesTensor
    #     # One Hot Label outputs
    #     if node.oneHotLabelTensor is not None:
    #         self.evalDict["Node{0}_one_hot_label_tensor".format(node.index)] = node.oneHotLabelTensor
    #     if node.filteredMask is not None:
    #         self.evalDict["Node{0}_filteredMask".format(node.index)] = node.filteredMask
    #
    # self.sampleCountTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "sample_count" in k}
    # self.isOpenTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "is_open" in k}
    # self.infoGainDicts = {k: v for k, v in self.evalDict.items() if "info_gain" in k}


    # MultiGPU OK
    def mask_input_nodes(self, node):
        if node.isRoot:
            lower_bound = int(self.currentTowerId*self.towerBatchSize)
            upper_bound = int((self.currentTowerId+1)*self.towerBatchSize)
            node.labelTensor = self.labelTensor[lower_bound:upper_bound]
            node.indicesTensor = self.indicesTensor[lower_bound:upper_bound]
            node.oneHotLabelTensor = self.oneHotLabelTensor[lower_bound:upper_bound]
            # node.filteredMask = tf.constant(value=True, dtype=tf.bool, shape=(GlobalConstants.BATCH_SIZE, ))
            node.filteredMask = self.filteredMask[lower_bound:upper_bound]
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = tf.size(node.labelTensor)
            node.isOpenIndicatorTensor = tf.constant(value=1.0, dtype=tf.float32)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
        else:
            # Obtain the mask vector, sample counts and determine if this node receives samples.
            parent_node = self.dagObject.parents(node=node)[0]
            mask_tensor = parent_node.maskTensors[node.index]
            if GlobalConstants.USE_UNIFIED_BATCH_NORM:
                mask_without_threshold = parent_node.masksWithoutThreshold[node.index]
            mask_tensor = tf.where(self.useMasking > 0, mask_tensor,
                                   tf.logical_or(x=tf.constant(value=True, dtype=tf.bool), y=mask_tensor))
            sample_count_tensor = tf.reduce_sum(tf.cast(mask_tensor, tf.float32))
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = sample_count_tensor
            node.isOpenIndicatorTensor = tf.where(sample_count_tensor > 0.0, 1.0, 0.0)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            # Mask all inputs: F channel, H  channel, activations from ancestors, labels
            parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
            parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
            for k, v in parent_node.activationsDict.items():
                node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
            node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
            node.indicesTensor = tf.boolean_mask(parent_node.indicesTensor, mask_tensor)
            node.oneHotLabelTensor = tf.boolean_mask(parent_node.oneHotLabelTensor, mask_tensor)
            if GlobalConstants.USE_UNIFIED_BATCH_NORM:
                node.filteredMask = tf.boolean_mask(mask_without_threshold, mask_tensor)
            return parent_F, parent_H
