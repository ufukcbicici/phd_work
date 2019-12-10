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


class CignSingleLateExit(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset, network_name, late_exit_train_func, late_exit_test_func):
        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                         dataset, network_name)
        self.lateExitTrainFunc = late_exit_train_func
        self.lateExitTestFunc = late_exit_test_func
        self.lateExitNode = None
        self.leafNodes = sorted([node for node in self.topologicalSortedNodes if node.isLeaf], key=lambda n: n.index)
        self.routingCombinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(self.leafNodes))
        self.routingCombinations = [np.array(route_vec) for route_vec in self.routingCombinations]
        self.leafOutputTraining = None
        self.leafOutputsTest = {}
        self.lateEvaluationExits = {}
        self.leafNodeOutputs = {}
        self.lateExitLoss = None

        # self.earlyExitFeatures = {}
        # self.earlyExitLogits = {}
        # self.earlyExitLosses = {}
        # self.earlyExitWeight = tf.placeholder(name="early_exit_loss_weight", dtype=tf.float32)
        #
        # self.leafNodeOutputs = {}

        # self.lateExitFeatures = {}
        # self.lateExitLogits = {}
        # self.lateExitLosses = {}
        # self.lateExitWeight = tf.placeholder(name="late_exit_loss_weight", dtype=tf.float32)
        #
        # self.sumEarlyExits = None
        # self.sumLateExits = None

    def unify_leaf_outputs_train(self):
        leaf_exit_features = []
        leaf_node_exit_shape = list(self.leafNodeOutputs.values())[0].get_shape().as_list()
        leaf_node_exit_shape[0] = self.batchSizeTf
        leaf_node_exit_shape = tuple(leaf_node_exit_shape)
        for leaf_node in self.leafNodes:
            indices = tf.expand_dims(leaf_node.batchIndicesTensor, -1)
            sparse_output = tf.scatter_nd(indices, self.leafNodeOutputs[leaf_node.index], leaf_node_exit_shape)
            self.evalDict[UtilityFuncs.get_variable_name(name="dense_output", node=leaf_node)] = \
                self.leafNodeOutputs[leaf_node.index]
            self.evalDict[UtilityFuncs.get_variable_name(name="sparse_output", node=leaf_node)] = sparse_output
            leaf_exit_features.append(sparse_output)
        self.leafOutputTraining = tf.concat(leaf_exit_features, axis=-1)
        self.evalDict["leafOutputTraining"] = self.leafOutputTraining

    def unify_leaf_outputs_test(self, route_vector):
        leaf_exit_features = []
        for idx, leaf_node in enumerate(self.leafNodes):
            scaled_output = route_vector[idx] * self.leafNodeOutputs[leaf_node.index]
            leaf_exit_features.append(scaled_output)
        leaf_test_output_for_route = tf.concat(leaf_exit_features, axis=-1)
        self.leafOutputsTest[route_vector] = leaf_test_output_for_route
        self.evalDict["leafOutputTest_for_route_{0}".format(route_vector)] = leaf_test_output_for_route

    def apply_late_loss(self, node, final_feature, softmax_weights, softmax_biases):
        node.evalDict[self.get_variable_name(name="final_feature_late_exit", node=node)] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_late_exit_mag", node=node)] \
            = tf.nn.l2_loss(final_feature)
        logits = FastTreeNetwork.fc_layer(x=final_feature, W=softmax_weights, b=softmax_biases, node=node,
                                          name="late_exit_fc_op")
        node.evalDict[self.get_variable_name(name="logits_late_exit", node=node)] = logits
        self.lateExitLoss = self.make_loss(node=node, logits=logits)

    def apply_late_softmax(self, node, final_feature, softmax_weights, softmax_biases, route_vector):
        logits = FastTreeNetwork.fc_layer(x=final_feature, W=softmax_weights, b=softmax_biases, node=node,
                                          name="late_exit_fc_op_test_route_{0}".format(route_vector))
        posteriors = tf.nn.softmax(logits)
        node.evalDict[self.get_variable_name(name="logits_late_exit_test_route_{0}".format(route_vector), node=node)] \
            = logits
        node.evalDict[self.get_variable_name(name="posteriors_late_exit_test_route_{0}".format(route_vector),
                                             node=node)] = posteriors

    def build_network(self):
        # Add late exit node explicitly as a separate node.
        self.build_tree()
        self.lateExitNode = Node(index=max([node.index for node in self.topologicalSortedNodes]) + 1,
                                 depth=max([node.depth for node in self.topologicalSortedNodes]) + 1,
                                 is_root=False,
                                 is_leaf=False)
        self.nodes[self.lateExitNode.index] = self.lateExitNode
        for leaf_node in self.leafNodes:
            self.dagObject.add_edge(parent=leaf_node, child=self.lateExitNode)
        # Build symbolic networks
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        self.isBaseline = len(self.topologicalSortedNodes) == 1
        # Disable some properties if we are using a baseline
        if self.isBaseline:
            GlobalConstants.USE_INFO_GAIN_DECISION = False
            GlobalConstants.USE_CONCAT_TRICK = False
            GlobalConstants.USE_PROBABILITY_THRESHOLD = False
        # Build all symbolic networks in each node
        for node in self.topologicalSortedNodes:
            print("Building Node {0}".format(node.index))
            self.nodeBuildFuncs[node.depth](network=self, node=node)
        # Build late exit for training and evaluation outputs
        with tf.variable_scope("late_exit"):
            self.unify_leaf_outputs_train()
            # Training Output
            late_exit_output, late_exit_sm_W, late_exit_sm_b = \
                self.lateExitTrainFunc(network=self, node=self.lateExitNode, x=self.leafOutputTraining)
            # Apply Loss for Late Exit
            self.apply_late_loss(node=self.lateExitNode, final_feature=late_exit_output,
                                 softmax_weights=late_exit_sm_W, softmax_biases=late_exit_sm_b)
            var_scope = tf.get_variable_scope()
            var_scope.reuse_variables()
            # Evaluation (Test) Outputs
            for routing_vector in self.routingCombinations:
                route_vector_as_tuple = tuple(routing_vector.tolist())
                self.unify_leaf_outputs_test(route_vector=route_vector_as_tuple)
                self.lateEvaluationExits[route_vector_as_tuple] = \
                    self.lateExitTestFunc(network=self,  node=self.lateExitNode,
                                          x=self.leafOutputsTest[route_vector_as_tuple])
                self.apply_late_softmax(node=self.lateExitNode,
                                        final_feature=self.lateEvaluationExits[route_vector_as_tuple],
                                        softmax_weights=late_exit_sm_W, softmax_biases=late_exit_sm_b,
                                        route_vector=route_vector_as_tuple,)
        self.dbName = DbLogger.log_db_path[DbLogger.log_db_path.rindex("/") + 1:]
        print(self.dbName)
        self.nodeCosts = {node.index: node.macCost for node in self.topologicalSortedNodes}
        # Build main classification loss
        self.build_main_loss()
        # Build information gain loss
        self.build_decision_loss()
        # Build regularization loss
        self.build_regularization_loss()
        # Final Loss
        self.finalLoss = self.mainLoss + self.regularizationLoss + self.decisionLoss
        if not GlobalConstants.USE_MULTI_GPU:
            self.build_optimizer()
        self.prepare_evaluation_dictionary()

    # def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking):
    #     feed_dict = super().prepare_feed_dict(minibatch=minibatch, iteration=iteration,
    #                                           use_threshold=use_threshold, is_train=is_train,
    #                                           use_masking=use_masking)
    #     feed_dict[self.earlyExitWeight] = GlobalConstants.EARLY_EXIT_WEIGHT
    #     feed_dict[self.lateExitWeight] = GlobalConstants.LATE_EXIT_WEIGHT
    #     return feed_dict
    #
    # def make_loss(self, node, logits):
    #     cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node.labelTensor,
    #                                                                                logits=logits)
    #     pre_loss = tf.reduce_mean(cross_entropy_loss_tensor)
    #     loss = tf.where(tf.is_nan(pre_loss), 0.0, pre_loss)
    #     node.fOpsList.extend([cross_entropy_loss_tensor, pre_loss, loss])
    #     node.lossList.append(loss)
    #     return loss
    #
    # def apply_loss(self, node, final_feature, softmax_weights, softmax_biases):
    #     node.residueOutputTensor = final_feature
    #     node.finalFeatures = final_feature
    #     self.earlyExitFeatures[node.index] = final_feature
    #     node.evalDict[self.get_variable_name(name="final_feature_final", node=node)] = final_feature
    #     node.evalDict[self.get_variable_name(name="final_feature_mag", node=node)] = tf.nn.l2_loss(final_feature)
    #     logits = FastTreeNetwork.fc_layer(x=final_feature, W=softmax_weights, b=softmax_biases, node=node,
    #                                       name="early_exit_fc_op")
    #     node.evalDict[self.get_variable_name(name="logits", node=node)] = logits
    #     self.earlyExitLogits[node.index] = logits
    #     loss = self.make_loss(node=node, logits=logits)
    #     self.earlyExitLosses[node.index] = loss
    #     return final_feature, logits
    #
    # def apply_late_loss(self, node, final_feature, softmax_weights, softmax_biases):
    #     self.lateExitFeatures[node.index] = final_feature
    #     node.evalDict[self.get_variable_name(name="final_feature_late_exit", node=node)] = final_feature
    #     node.evalDict[self.get_variable_name(name="final_feature_late_exit_mag", node=node)] = \
    #         tf.nn.l2_loss(final_feature)
    #     logits = FastTreeNetwork.fc_layer(x=final_feature, W=softmax_weights, b=softmax_biases, node=node,
    #                                       name="late_exit_fc_op")
    #     node.evalDict[self.get_variable_name(name="logits_late_exit", node=node)] = logits
    #     self.lateExitLogits[node.index] = logits
    #     loss = self.make_loss(node=node, logits=logits)
    #     self.lateExitLosses[node.index] = loss
    #     return final_feature, logits
    #
    # def build_main_loss(self):
    #     assert self.earlyExitWeight is not None and self.lateExitWeight is not None
    #     self.sumEarlyExits = tf.add_n(list(self.earlyExitLosses.values()))
    #     self.sumLateExits = tf.add_n(list(self.lateExitLosses.values()))
    #     self.mainLoss = (self.earlyExitWeight * self.sumEarlyExits) + (self.lateExitWeight * self.sumLateExits)
    #
    # def calculate_model_performance(self, sess, dataset, run_id, epoch_id, iteration):
    #     # moving_results_1 = sess.run(moving_stat_vars)
    #     is_evaluation_epoch_at_report_period = \
    #         epoch_id < GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING \
    #         and (epoch_id + 1) % GlobalConstants.EPOCH_REPORT_PERIOD == 0
    #     is_evaluation_epoch_before_ending = \
    #         epoch_id >= GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING
    #     if is_evaluation_epoch_at_report_period or is_evaluation_epoch_before_ending:
    #         training_accuracy, training_confusion = \
    #             self.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training,
    #                                     run_id=run_id,
    #                                     iteration=iteration)
    #         validation_accuracy, validation_confusion = \
    #             self.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test,
    #                                     run_id=run_id,
    #                                     iteration=iteration)
    #         validation_accuracy_late = 0.0
    #         if not self.isBaseline:
    #             validation_accuracy_late, validation_confusion_late = \
    #                 self.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test,
    #                                         run_id=run_id,
    #                                         iteration=iteration, posterior_entry_name="posterior_probs_late")
    #             if is_evaluation_epoch_before_ending:
    #                 # self.save_model(sess=sess, run_id=run_id, iteration=iteration)
    #                 t0 = time.time()
    #                 self.save_routing_info(sess=sess, run_id=run_id, iteration=iteration,
    #                                        dataset=dataset, dataset_type=DatasetTypes.training)
    #                 t1 = time.time()
    #                 self.save_routing_info(sess=sess, run_id=run_id, iteration=iteration,
    #                                        dataset=dataset, dataset_type=DatasetTypes.test)
    #                 t2 = time.time()
    #                 print("t1-t0={0}".format(t1 - t0))
    #                 print("t2-t1={0}".format(t2 - t1))
    #         DbLogger.write_into_table(
    #             rows=[(run_id, iteration, epoch_id, training_accuracy,
    #                    validation_accuracy, validation_accuracy_late,
    #                    0.0, 0.0, "XXX")], table=DbLogger.logsTable, col_count=9)
