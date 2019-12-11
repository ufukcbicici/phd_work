import numpy as np
import tensorflow as tf
import time

from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants
from simple_tf.uncategorized.node import Node
from sklearn.metrics import confusion_matrix


class CignSingleLateExit(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset, network_name, late_exit_func):
        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                         dataset, network_name)
        self.leafNodes = None
        self.routingCombinations = None
        self.lateExitFunc = late_exit_func
        self.earlyExitWeight = tf.placeholder(name="early_exit_loss_weight", dtype=tf.float32)
        self.lateExitWeight = tf.placeholder(name="late_exit_loss_weight", dtype=tf.float32)
        self.leafNodeOutputsToLateExit = {}
        self.lateExitTrainingInput = None
        self.lateExitLabelInput = None
        self.lateExitTestInput = None
        self.lateExitNode = None
        self.lateExitOutput = None
        self.lateExitLoss = None
        self.lateExitTestLogits = None
        self.lateExitTestPosteriors = None

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking):
        feed_dict = super().prepare_feed_dict(minibatch=minibatch, iteration=iteration,
                                              use_threshold=use_threshold, is_train=is_train,
                                              use_masking=use_masking)
        feed_dict[self.earlyExitWeight] = GlobalConstants.EARLY_EXIT_WEIGHT
        feed_dict[self.lateExitWeight] = GlobalConstants.LATE_EXIT_WEIGHT
        return feed_dict

    def build_late_exit_inputs(self):
        leaf_exit_features = []
        # leaf_exit_labels = []
        leaf_node_exit_shape = list(self.leafNodeOutputsToLateExit.values())[0].get_shape().as_list()
        leaf_node_exit_shape[0] = self.batchSizeTf
        leaf_node_exit_shape = tuple(leaf_node_exit_shape)
        for leaf_node in self.leafNodes:
            indices = tf.expand_dims(leaf_node.batchIndicesTensor, -1)
            # Features
            sparse_output = tf.scatter_nd(indices, self.leafNodeOutputsToLateExit[leaf_node.index],
                                          leaf_node_exit_shape)
            self.evalDict[UtilityFuncs.get_variable_name(name="dense_output", node=leaf_node)] = \
                self.leafNodeOutputsToLateExit[leaf_node.index]
            self.evalDict[UtilityFuncs.get_variable_name(name="sparse_output", node=leaf_node)] = sparse_output
            leaf_exit_features.append(sparse_output)
        self.lateExitTrainingInput = tf.concat(leaf_exit_features, axis=-1)
        self.evalDict["lateExitTrainingInput"] = self.lateExitTrainingInput
        input_shape = self.lateExitTrainingInput.get_shape().as_list()
        input_shape[0] = None
        self.lateExitTestInput = tf.placeholder(name="lateExitTestInput", dtype=tf.float32, shape=input_shape)

    def apply_late_loss(self, node, final_feature, softmax_weights, softmax_biases):
        node.evalDict[self.get_variable_name(name="final_feature_late_exit", node=node)] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_late_exit_mag", node=node)] \
            = tf.nn.l2_loss(final_feature)
        logits = FastTreeNetwork.fc_layer(x=final_feature, W=softmax_weights, b=softmax_biases, node=node,
                                          name="late_exit_fc_op")
        node.evalDict[self.get_variable_name(name="logits_late_exit", node=node)] = logits
        self.lateExitLoss = self.make_loss(node=node, logits=logits)
        posteriors = tf.nn.softmax(logits)
        node.evalDict[self.get_variable_name(name="posteriors_late_exit", node=node)] = posteriors

    def build_network(self):
        # Add late exit node explicitly as a separate node.
        self.build_tree()
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        self.lateExitNode = Node(index=max([node.index for node in self.topologicalSortedNodes]) + 1,
                                 depth=max([node.depth for node in self.topologicalSortedNodes]) + 1,
                                 is_root=False,
                                 is_leaf=False)
        self.nodes[self.lateExitNode.index] = self.lateExitNode
        self.leafNodes = sorted([node for node in self.topologicalSortedNodes if node.isLeaf], key=lambda n: n.index)
        self.routingCombinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(self.leafNodes))
        self.routingCombinations = [np.array(route_vec) for route_vec in self.routingCombinations]
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
            if node != self.lateExitNode:
                print("Building Node {0}".format(node.index))
                self.nodeBuildFuncs[node.depth](network=self, node=node)
            else:
                # Build late exit for training and evaluation outputs
                with tf.variable_scope("late_exit"):
                    # Training Exit: Loss and Posteriors
                    self.build_late_exit_inputs()
                    self.lateExitOutput, late_exit_sm_W, late_exit_sm_b = \
                        self.lateExitFunc(network=self, node=self.lateExitNode, x=self.lateExitTrainingInput)
                    self.evalDict["lateExitOutput"] = self.lateExitOutput
                    self.lateExitNode.labelTensor = self.topologicalSortedNodes[0].labelTensor
                    self.lateExitNode.infoGainLoss = tf.constant(0.0)
                    self.apply_late_loss(node=self.lateExitNode, final_feature=self.lateExitOutput,
                                         softmax_weights=late_exit_sm_W, softmax_biases=late_exit_sm_b)
                    # Test Exit: Only Posteriors
                    var_scope = tf.get_variable_scope()
                    var_scope.reuse_variables()
                    test_output, late_exit_sm_W_test, late_exit_sm_b_test = \
                        self.lateExitFunc(network=self, node=self.lateExitNode, x=self.lateExitTestInput)
                    assert late_exit_sm_W == late_exit_sm_W_test and late_exit_sm_b == late_exit_sm_b_test
                    self.lateExitTestLogits = FastTreeNetwork.fc_layer(x=test_output,
                                                                       W=late_exit_sm_W_test, b=late_exit_sm_b_test,
                                                                       node=node, name="late_exit_fc_op")
                    self.lateExitTestPosteriors = tf.nn.softmax(self.lateExitTestLogits)
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

    def calculate_accuracy_late_exit_accuracy(self, sess, dataset, dataset_type):
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        leaf_node_collections, inner_node_collections = \
            self.collect_eval_results_from_network(sess=sess,
                                                   dataset=dataset,
                                                   dataset_type=dataset_type,
                                                   use_masking=True,
                                                   leaf_node_collection_names=[],
                                                   inner_node_collections_names=[
                                                       "posteriors_late_exit", "label_tensor"])
        assert len(inner_node_collections["posteriors_late_exit"]) == 1
        assert len(inner_node_collections["label_tensor"]) == 1
        late_exit_posteriors = inner_node_collections["posteriors_late_exit"][self.lateExitNode.index]
        labels = inner_node_collections["label_tensor"][self.lateExitNode.index]
        predicted_labels = np.argmax(late_exit_posteriors, axis=1)
        assert labels.shape == predicted_labels.shape
        true_count = np.sum(predicted_labels == labels)
        accuracy = true_count / labels.shape[0]
        # Prepare the confusion matrix
        cm = confusion_matrix(y_true=labels, y_pred=predicted_labels)
        return accuracy, cm

    def calculate_model_performance(self, sess, dataset, run_id, epoch_id, iteration):
        # moving_results_1 = sess.run(moving_stat_vars)
        is_evaluation_epoch_at_report_period = \
            epoch_id < GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING \
            and (epoch_id + 1) % GlobalConstants.EPOCH_REPORT_PERIOD == 0
        is_evaluation_epoch_before_ending = \
            epoch_id >= GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING
        if is_evaluation_epoch_at_report_period or is_evaluation_epoch_before_ending:
            training_accuracy, training_confusion = \
                self.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training,
                                        run_id=run_id,
                                        iteration=iteration)
            validation_accuracy, validation_confusion = \
                self.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test,
                                        run_id=run_id,
                                        iteration=iteration)
            validation_accuracy_late = 0.0
            if not self.isBaseline:
                validation_accuracy_late, validation_confusion_late = \
                    self.calculate_accuracy_late_exit_accuracy(sess=sess, dataset=dataset,
                                                               dataset_type=DatasetTypes.test)
                if is_evaluation_epoch_before_ending:
                    # self.save_model(sess=sess, run_id=run_id, iteration=iteration)
                    t0 = time.time()
                    self.save_routing_info(sess=sess, run_id=run_id, iteration=iteration,
                                           dataset=dataset, dataset_type=DatasetTypes.training)
                    t1 = time.time()
                    self.save_routing_info(sess=sess, run_id=run_id, iteration=iteration,
                                           dataset=dataset, dataset_type=DatasetTypes.test)
                    t2 = time.time()
                    print("t1-t0={0}".format(t1 - t0))
                    print("t2-t1={0}".format(t2 - t1))
            DbLogger.write_into_table(
                rows=[(run_id, iteration, epoch_id, training_accuracy,
                       validation_accuracy, validation_accuracy_late,
                       0.0, 0.0, "XXX")], table=DbLogger.logsTable, col_count=9)
