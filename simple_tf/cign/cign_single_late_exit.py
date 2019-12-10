import numpy as np
import tensorflow as tf

from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants
from simple_tf.uncategorized.node import Node


class CignSingleLateExit(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset, network_name, late_exit_train_func, late_exit_test_func):
        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                         dataset, network_name)
        self.routingCombinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(self.leafNodes))
        self.routingCombinations = [np.array(route_vec) for route_vec in self.routingCombinations]
        self.lateExitTrainFunc = late_exit_train_func
        self.lateExitTestFunc = late_exit_test_func
        self.leafNodeOutputsToLateExit = {}
        self.leafNodes = sorted([node for node in self.topologicalSortedNodes if node.isLeaf], key=lambda n: n.index)
        self.lateExitTrainingInput = None
        self.lateExitTestInputsDict = {}
        self.lateExitLoss = None
        self.earlyExitWeight = tf.placeholder(name="early_exit_loss_weight", dtype=tf.float32)
        self.lateExitWeight = tf.placeholder(name="late_exit_loss_weight", dtype=tf.float32)
        self.lateExitNode = None
        self.lateExitTrainingOutput = None
        self.lateExitTestOutputsDict = {}

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking):
        feed_dict = super().prepare_feed_dict(minibatch=minibatch, iteration=iteration,
                                              use_threshold=use_threshold, is_train=is_train,
                                              use_masking=use_masking)
        feed_dict[self.earlyExitWeight] = GlobalConstants.EARLY_EXIT_WEIGHT
        feed_dict[self.lateExitWeight] = GlobalConstants.LATE_EXIT_WEIGHT
        return feed_dict

    def unify_leaf_outputs_train(self):
        leaf_exit_features = []
        leaf_node_exit_shape = list(self.leafNodeOutputsToLateExit.values())[0].get_shape().as_list()
        leaf_node_exit_shape[0] = self.batchSizeTf
        leaf_node_exit_shape = tuple(leaf_node_exit_shape)
        for leaf_node in self.leafNodes:
            indices = tf.expand_dims(leaf_node.batchIndicesTensor, -1)
            sparse_output = tf.scatter_nd(indices, self.leafNodeOutputsToLateExit[leaf_node.index], leaf_node_exit_shape)
            self.evalDict[UtilityFuncs.get_variable_name(name="dense_output", node=leaf_node)] = \
                self.leafNodeOutputsToLateExit[leaf_node.index]
            self.evalDict[UtilityFuncs.get_variable_name(name="sparse_output", node=leaf_node)] = sparse_output
            leaf_exit_features.append(sparse_output)
        self.lateExitTrainingInput = tf.concat(leaf_exit_features, axis=-1)
        self.evalDict["earlyExitTrainingInput"] = self.lateExitTrainingInput

    def unify_leaf_outputs_test(self, route_vector):
        leaf_exit_features = []
        for idx, leaf_node in enumerate(self.leafNodes):
            scaled_output = route_vector[idx] * self.leafNodeOutputsToLateExit[leaf_node.index]
            leaf_exit_features.append(scaled_output)
        leaf_test_output_for_route = tf.concat(leaf_exit_features, axis=-1)
        self.lateExitTestInputsDict[route_vector] = leaf_test_output_for_route
        self.evalDict["lateExitTestInput_for_route_{0}".format(route_vector)] = leaf_test_output_for_route

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
            self.lateExitTrainingOutput, late_exit_sm_W, late_exit_sm_b = \
                self.lateExitTrainFunc(network=self, node=self.lateExitNode, x=self.lateExitTrainingInput)
            # Apply Loss for Late Exit
            self.apply_late_loss(node=self.lateExitNode, final_feature=self.lateExitTrainingOutput,
                                 softmax_weights=late_exit_sm_W, softmax_biases=late_exit_sm_b)
            var_scope = tf.get_variable_scope()
            var_scope.reuse_variables()
            # Evaluation (Test) Outputs
            for routing_vector in self.routingCombinations:
                route_vector_as_tuple = tuple(routing_vector.tolist())
                self.unify_leaf_outputs_test(route_vector=route_vector_as_tuple)
                self.lateExitTestOutputsDict[route_vector_as_tuple] = \
                    self.lateExitTestFunc(network=self,  node=self.lateExitNode,
                                          x=self.lateExitTestInputsDict[route_vector_as_tuple])
                self.apply_late_softmax(node=self.lateExitNode,
                                        final_feature=self.lateExitTestOutputsDict[route_vector_as_tuple],
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
        self.finalLoss = (self.earlyExitWeight * self.mainLoss + self.lateExitWeight * self.lateExitLoss) + \
                            self.regularizationLoss + self.decisionLoss
        if not GlobalConstants.USE_MULTI_GPU:
            self.build_optimizer()
        self.prepare_evaluation_dictionary()
