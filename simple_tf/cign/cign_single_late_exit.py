import numpy as np
import tensorflow as tf
import time
import os
import pickle

from algorithms.threshold_optimization_algorithms.threshold_optimization_helpers import RoutingDataset
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
        # Build symbolic networks
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
        # Build late exit for training and evaluation outputs
        with tf.variable_scope("late_exit"):
            # Training Exit: Loss and Posteriors
            with tf.name_scope("merge_leaf_outputs"):
                self.build_late_exit_inputs()
            with tf.name_scope("late_exit_training"):
                self.lateExitOutput, late_exit_sm_W, late_exit_sm_b = \
                    self.lateExitFunc(network=self, node=self.lateExitNode, x=self.lateExitTrainingInput)
                self.evalDict["lateExitOutput"] = self.lateExitOutput
                self.lateExitNode.labelTensor = self.topologicalSortedNodes[0].labelTensor
                self.lateExitNode.infoGainLoss = tf.constant(0.0)
                self.apply_late_loss(node=self.lateExitNode, final_feature=self.lateExitOutput,
                                     softmax_weights=late_exit_sm_W, softmax_biases=late_exit_sm_b)
            # Test Exit: Only Posteriors
            with tf.name_scope("late_exit_test"):
                var_scope = tf.get_variable_scope()
                var_scope.reuse_variables()
                test_output, late_exit_sm_W_test, late_exit_sm_b_test = \
                    self.lateExitFunc(network=self, node=self.lateExitNode, x=self.lateExitTestInput)
                assert late_exit_sm_W == late_exit_sm_W_test and late_exit_sm_b == late_exit_sm_b_test
                self.lateExitTestLogits = FastTreeNetwork.fc_layer(x=test_output,
                                                                   W=late_exit_sm_W_test, b=late_exit_sm_b_test,
                                                                   node=self.lateExitNode, name="late_exit_fc_op")
                self.lateExitTestPosteriors = tf.nn.softmax(self.lateExitTestLogits)
        self.dbName = DbLogger.log_db_path[DbLogger.log_db_path.rindex("/") + 1:]
        print(self.dbName)
        self.nodeCosts = {node.index: node.macCost for node in self.topologicalSortedNodes}
        # Divided by two since there are two duplicate applications; one for the training and one for the test.
        self.nodeCosts[self.lateExitNode.index] = self.lateExitNode.macCost / 2.0
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

    def prepare_evaluation_dictionary(self):
        super().prepare_evaluation_dictionary()
        for k, v in self.lateExitNode.evalDict.items():
            self.evalDict[k] = v
        self.evalDict[UtilityFuncs.get_variable_name(name="label_tensor", node=self.lateExitNode)] \
            = self.lateExitNode.labelTensor

    def calculate_accuracy_late_exit_accuracy(self, sess, dataset, dataset_type):
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        late_exit_collection = {"posteriors_late_exit": {}, "label_tensor": {}}
        while True:
            results, minibatch = self.eval_network(sess=sess, dataset=dataset, use_masking=True)
            late_posteriors_arr = results[self.get_variable_name(name="posteriors_late_exit", node=self.lateExitNode)]
            labels_arr = results[self.get_variable_name(name="label_tensor", node=self.lateExitNode)]
            UtilityFuncs.concat_to_np_array_dict_v2(dct=late_exit_collection["posteriors_late_exit"],
                                                    key=self.lateExitNode.index, array=late_posteriors_arr)
            UtilityFuncs.concat_to_np_array_dict_v2(dct=late_exit_collection["label_tensor"],
                                                    key=self.lateExitNode.index, array=labels_arr)
            if dataset.isNewEpoch:
                break
        for output_name, nodes_arr_dict in late_exit_collection.items():
            for node_idx in nodes_arr_dict.keys():
                if np.isscalar(nodes_arr_dict[node_idx][0]):
                    continue
                nodes_arr_dict[node_idx] = np.concatenate(nodes_arr_dict[node_idx], axis=0)
        assert len(late_exit_collection["posteriors_late_exit"]) == 1
        assert len(late_exit_collection["label_tensor"]) == 1
        late_exit_posteriors = late_exit_collection["posteriors_late_exit"][self.lateExitNode.index]
        labels = late_exit_collection["label_tensor"][self.lateExitNode.index]
        predicted_labels = np.argmax(late_exit_posteriors, axis=1)
        assert labels.shape == predicted_labels.shape
        true_count = np.sum(predicted_labels == labels)
        accuracy = true_count / labels.shape[0]
        # Prepare the confusion matrix
        cm = confusion_matrix(y_true=labels, y_pred=predicted_labels)
        return accuracy, cm

    def save_routing_info(self, sess, run_id, iteration, dataset, dataset_type):
        prev_leaf_outputs = list(GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT)
        GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT.extend(["dense_output", "sparse_output"])
        routing_data = super().save_routing_info(sess=sess, run_id=run_id, iteration=iteration,
                                                 dataset=dataset, dataset_type=dataset_type)
        GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT = prev_leaf_outputs
        # Get the late exit posteriors
        late_posteriors_dict = {}
        for route_vec in self.routingCombinations:
            route_tpl = tuple(route_vec.tolist())
            late_posteriors_dict[route_tpl] = []
        curr_index = 0
        while curr_index < dataset.get_current_sample_count():
            for route_vec in self.routingCombinations:
                route_tpl = tuple(route_vec.tolist())
                leaf_exits = []
                for idx, leaf_node in enumerate(self.leafNodes):
                    dense_output = routing_data.dictionaryOfRoutingData["dense_output"][leaf_node.index][
                                   curr_index: curr_index + GlobalConstants.EVAL_BATCH_SIZE]
                    sparse_output = routing_data.dictionaryOfRoutingData["sparse_output"][leaf_node.index][
                                    curr_index: curr_index + GlobalConstants.EVAL_BATCH_SIZE]
                    assert np.array_equal(dense_output, sparse_output)
                    leaf_exits.append(route_tpl[idx] * dense_output)
                late_exit_input = np.concatenate(leaf_exits, axis=-1)
                feed_dict = {self.classificationDropoutKeepProb: 1.0, self.lateExitTestInput: late_exit_input}
                results = sess.run([self.lateExitTestPosteriors], feed_dict=feed_dict)
                late_posteriors_dict[route_tpl].append(results[0])
            curr_index += GlobalConstants.EVAL_BATCH_SIZE
        for k in late_posteriors_dict.keys():
            late_posteriors_dict[k] = np.concatenate(late_posteriors_dict[k], axis=0)
        data_type = "test" if dataset_type == DatasetTypes.test else "training"
        directory_path = FastTreeNetwork.get_routing_info_path(network_name=self.networkName,
                                                               run_id=run_id, iteration=iteration,
                                                               data_type=data_type)
        routing_data.dictionaryOfRoutingData["lateExitTestPosteriors"] = late_posteriors_dict
        npz_file_name = os.path.abspath(os.path.join(directory_path, "lateExitTestPosteriors"))
        string_arr_dict = {",".join((str(i) for i in k)): v for k, v in late_posteriors_dict.items()}
        UtilityFuncs.save_npz(file_name=npz_file_name, arr_dict=string_arr_dict)
        pickle.dump(self.lateExitNode.opMacCostsDict,
                    open(
                        os.path.abspath(
                            os.path.join(directory_path, "node_{0}_opMacCosts.sav".format(self.lateExitNode.index))),
                        "wb"))
        routing_data.dictionaryOfRoutingData["node_{0}_opMacCosts".format(self.lateExitNode.index)] = \
            self.lateExitNode.opMacCostsDict
        routing_data = RoutingDataset(label_list=routing_data.labelList,
                                      dict_of_data_dicts=routing_data.dictionaryOfRoutingData)
        return routing_data

    def load_routing_info(self, run_id, iteration, data_type):
        prev_leaf_outputs = list(GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT)
        GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT.extend(["dense_output", "sparse_output"])
        routing_data = super().load_routing_info(run_id=run_id,
                                                 iteration=iteration, data_type=data_type)
        GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT = prev_leaf_outputs
        directory_path = FastTreeNetwork.get_routing_info_path(run_id=run_id, iteration=iteration,
                                                               network_name=self.networkName,
                                                               data_type=data_type)
        npz_file_name = os.path.abspath(os.path.join(directory_path, "lateExitTestPosteriors"))
        dict_read = UtilityFuncs.load_npz(file_name=npz_file_name)
        data_dict = {tuple([int(l) for l in k.split(",")]): v for k, v in dict_read.items()}
        routing_data.dictionaryOfRoutingData["lateExitTestPosteriors"] = data_dict
        self.lateExitNode.opMacCostsDict = pickle.load(open(os.path.abspath(os.path.join(directory_path,
                                                                            "node_{0}_opMacCosts.sav"
                                                                            .format(self.lateExitNode.index))), 'rb'))
        routing_data.dictionaryOfRoutingData["node_{0}_opMacCosts".format(self.lateExitNode.index)] = \
            self.lateExitNode.opMacCostsDict
        return routing_data

    def calculate_model_performance(self, sess, dataset, run_id, epoch_id, iteration):
        # moving_results_1 = sess.run(moving_stat_vars)
        # self.test_scatter_nd_behavior(sess=sess, dataset=dataset)
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
                    # self.test_save_load(sess=sess, run_id=run_id, iteration=iteration, dataset=dataset,
                    #                     dataset_type=DatasetTypes.training)
                    # self.test_save_load(sess=sess, run_id=run_id, iteration=iteration, dataset=dataset,
                    #                     dataset_type=DatasetTypes.test)
                    # print("t1-t0={0}".format(t1 - t0))
                    # print("t2-t1={0}".format(t2 - t1))
            DbLogger.write_into_table(
                rows=[(run_id, iteration, epoch_id, training_accuracy,
                       validation_accuracy, validation_accuracy_late,
                       0.0, 0.0, "XXX")], table=DbLogger.logsTable, col_count=9)

    # Unit Tests
    def test_scatter_nd_behavior(self, sess, dataset):
        dataset.set_current_data_set_type(dataset_type=DatasetTypes.test, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        while True:
            results, minibatch = self.eval_network(sess=sess, dataset=dataset, use_masking=True)
            tf_result_arr = results["lateExitTrainingInput"]
            manual_result_arr_shape = list(tf_result_arr.shape)
            manual_result_arr_shape[-1] = int(manual_result_arr_shape[-1] / len(self.leafNodes))
            manual_leaf_exits = []
            for leaf_node in self.leafNodes:
                leaf_exit = np.zeros(shape=tuple(manual_result_arr_shape))
                batch_indices = results[UtilityFuncs.get_variable_name(name="batchIndicesTensor", node=leaf_node)]
                dense_exit = results[UtilityFuncs.get_variable_name(name="dense_output", node=leaf_node)]
                leaf_exit[batch_indices] = dense_exit
                manual_leaf_exits.append(leaf_exit)
            np_result_arr = np.concatenate(manual_leaf_exits, axis=-1)
            assert np.array_equal(tf_result_arr, np_result_arr)
            if dataset.isNewEpoch:
                break

