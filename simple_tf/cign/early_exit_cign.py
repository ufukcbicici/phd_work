from collections import deque

import numpy as np
import tensorflow as tf
import time
import os

from algorithms.custom_batch_norm_algorithms import CustomBatchNormAlgorithms
from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from algorithms.threshold_optimization_algorithms.threshold_optimization_helpers import RoutingDataset
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.cign.tree import TreeNetwork
from simple_tf.global_params import GlobalConstants, AccuracyCalcType, TrainingUpdateResult
from algorithms.info_gain import InfoGainLoss
from simple_tf.uncategorized.node import Node
from auxillary.constants import DatasetTypes
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.framework.python.framework import checkpoint_utils


class EarlyExitTree(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset, network_name):

        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                         dataset, network_name)
        self.earlyExitFeatures = {}
        self.earlyExitLogits = {}
        self.earlyExitLosses = {}
        self.earlyExitWeight = None

        self.lateExitFeatures = {}
        self.lateExitLogits = {}
        self.lateExitLosses = {}
        self.lateExitWeight = None

        self.sumEarlyExits = None
        self.sumLateExits = None

    def make_loss(self, node, logits):
        cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node.labelTensor,
                                                                                   logits=logits)
        pre_loss = tf.reduce_mean(cross_entropy_loss_tensor)
        loss = tf.where(tf.is_nan(pre_loss), 0.0, pre_loss)
        node.fOpsList.extend([cross_entropy_loss_tensor, pre_loss, loss])
        node.lossList.append(loss)
        return loss

    def apply_loss(self, node, final_feature, softmax_weights, softmax_biases):
        node.residueOutputTensor = final_feature
        node.finalFeatures = final_feature
        self.earlyExitFeatures[node.index] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_final", node=node)] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_mag", node=node)] = tf.nn.l2_loss(final_feature)
        logits = FastTreeNetwork.fc_layer(x=final_feature, W=softmax_weights, b=softmax_biases, node=node)
        node.evalDict[self.get_variable_name(name="logits", node=node)] = logits
        self.earlyExitLogits[node.index] = logits
        loss = self.make_loss(node=node, logits=logits)
        self.earlyExitLosses[node.index] = loss
        return final_feature, logits

    def apply_late_loss(self, node, final_feature, softmax_weights, softmax_biases):
        self.lateExitFeatures[node.index] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_late_exit", node=node)] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_late_exit_mag", node=node)] = \
            tf.nn.l2_loss(final_feature)
        logits = FastTreeNetwork.fc_layer(x=final_feature, W=softmax_weights, b=softmax_biases, node=node)
        node.evalDict[self.get_variable_name(name="logits_late_exit", node=node)] = logits
        self.lateExitLogits[node.index] = logits
        loss = self.make_loss(node=node, logits=logits)
        self.lateExitLosses[node.index] = loss
        return final_feature, logits

    def build_main_loss(self):
        assert self.earlyExitWeight is not None and self.lateExitWeight is not None
        self.sumEarlyExits = tf.add_n(list(self.earlyExitLosses.values()))
        self.sumLateExits = tf.add_n(list(self.lateExitLosses.values()))
        self.mainLoss = (self.earlyExitWeight * self.sumEarlyExits) + (self.lateExitWeight * self.sumLateExits)
