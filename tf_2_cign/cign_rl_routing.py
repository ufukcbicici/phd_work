from collections import Counter

import numpy as np
import tensorflow as tf
import time
import os
from tf_2_cign.utilities.profiler import Profiler
from auxillary.db_logger import DbLogger
from tf_2_cign.cign_no_mask import CignNoMask
from tf_2_cign.custom_layers.cign_rl_routing_layer import CignRlRoutingLayer
from sklearn.metrics import classification_report
from tf_2_cign.utilities.utilities import Utilities


class CignRlRouting(CignNoMask):
    infeasible_action_penalty = -1000000.0

    def __init__(self,
                 valid_prediction_reward,
                 invalid_prediction_penalty,
                 include_ig_in_reward_calculations,
                 lambda_mac_cost,
                 warm_up_period,
                 cign_rl_train_period,
                 batch_size,
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
                 decision_loss_coeff,
                 q_net_coeff,
                 bn_momentum=0.9):
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
                         decision_loss_coeff,
                         bn_momentum)
        self.validPredictionReward = valid_prediction_reward
        self.invalidPredictionPenalty = invalid_prediction_penalty
        self.lambdaMacCost = lambda_mac_cost
        self.actionSpaces = []
        self.actionSpacesReverse = []
        self.reachabilityMatrices = []
        self.baseEvaluationCost = None
        self.networkActivationCosts = []
        self.networkActivationCostsDict = {}
        self.warmUpPeriod = warm_up_period
        self.cignRlTrainPeriod = cign_rl_train_period
        # self.optimalQtables = []
        # self.regressionTargets = []
        self.includeIgInRewardCalculations = include_ig_in_reward_calculations
        self.qNets = []
        self.qTablesPredicted = []
        self.actionsPredicted = []
        self.qNetOptimizer = None
        self.qNetTrackers = []
        self.mseLoss = tf.keras.losses.MeanSquaredError()
        self.qNetCoeff = q_net_coeff
        self.globalStep = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)

    def save_model(self, run_id, epoch_id=0):
        root_path = os.path.dirname(__file__)
        model_path = os.path.join(root_path, "..", "saved_models", "model_{0}_epoch_{1}".format(run_id, epoch_id))
        os.mkdir(model_path)
        # if os.path.isdir(root_path):
        #     shutil.rmtree(root_path)

        # with open(os.path.join(model_path, "pickle_model.sav"), "wb") as f:
        #     pickle.dump(model, f)
        # Save keras models
        cign_model_path = os.path.join(model_path, "cign_model")
        os.mkdir(cign_model_path)
        cign_model_vars_path = os.path.join(cign_model_path, "variables")
        os.mkdir(cign_model_vars_path)
        self.model.save_weights(cign_model_vars_path)

        for level, q_net in enumerate(self.qNets):
            qnet_model_path = os.path.join(model_path, "q_net_{0}".format(level))
            os.mkdir(qnet_model_path)
            qnet_model_vars_path = os.path.join(qnet_model_path, "variables")
            os.mkdir(qnet_model_vars_path)
            q_net.save_weights(qnet_model_vars_path)

    def load_model(self, run_id, epoch_id=0):
        root_path = os.path.dirname(__file__)
        model_path = os.path.join(root_path, "..", "saved_models", "model_{0}_epoch_{1}".format(run_id, epoch_id))
        # with open(model_path, "rb") as f:
        #     model = pickle.load(f)
        # Load keras models
        cign_model_path = os.path.join(model_path, "cign_model", "variables")
        self.model.load_weights(cign_model_path)
        for level in range(self.networkDepth):
            qnet_model_path = os.path.join(model_path, "q_net_{0}".format(level), "variables")
            self.qNets[level].load_weights(qnet_model_path)

    # OK
    def get_max_trajectory_length(self) -> int:
        return self.networkDepth

    # OK
    def build_action_spaces(self):
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            next_level_node_count = len(self.orderedNodesPerLevel[t + 1])
            action_count = (2 ** next_level_node_count) - 1
            action_space = []
            for action_id in range(action_count):
                action_code = action_id + 1
                l = [int(x) for x in list('{0:0b}'.format(action_code))]
                k = [0] * (next_level_node_count - len(l))
                k.extend(l)
                binary_node_selection = np.array(k)
                action_space.append(binary_node_selection)
            action_space = np.stack(action_space, axis=0)
            self.actionSpaces.append(action_space)
            # Reverse map
            binary_basis = []
            for a_id in range(action_space.shape[1]):
                binary_basis.append(2 ** a_id)
            binary_basis.reverse()
            self.actionSpacesReverse.append(tf.constant(binary_basis))

    # OK
    def build_reachability_matrices(self):
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            actions_t = self.actionSpaces[t]
            if t == 0:
                reachability_matrix_t = np.ones(shape=(1, actions_t.shape[0]), dtype=np.int32)
            else:
                reachability_matrix_t = np.zeros(shape=(self.actionSpaces[t - 1].shape[0], actions_t.shape[0]),
                                                 dtype=np.int32)
                for action_t_minus_one_id in range(self.actionSpaces[t - 1].shape[0]):
                    node_selection_vec_t_minus_one = self.actionSpaces[t - 1][action_t_minus_one_id]
                    selected_nodes_t = [node for i, node in enumerate(self.orderedNodesPerLevel[t])
                                        if node_selection_vec_t_minus_one[i] != 0]
                    next_level_nodes = self.orderedNodesPerLevel[t + 1]
                    reachable_next_level_node_ids = set()
                    next_level_reached_dict = {}
                    for parent_node in selected_nodes_t:
                        child_nodes = {c_node.index for c_node in self.dagObject.children(node=parent_node)}
                        reachable_next_level_node_ids = reachable_next_level_node_ids.union(child_nodes)
                        next_level_reached_dict[parent_node.index] = child_nodes

                    for actions_t_id in range(actions_t.shape[0]):
                        # All selected nodes should have their parents selected in the previous depth
                        node_selection_vec_t = actions_t[actions_t_id]
                        reached_nodes = {node.index for is_reached, node in zip(node_selection_vec_t, next_level_nodes)
                                         if is_reached != 0}
                        is_valid_selection = int(len(reached_nodes.difference(reachable_next_level_node_ids)) == 0)
                        # All selected nodes in the previous depth must have at least one child selected in next depth
                        for parent_node in selected_nodes_t:
                            selection_arr = [_n in reached_nodes for _n in next_level_reached_dict[parent_node.index]]
                            is_valid_selection = is_valid_selection and any(selection_arr)
                        reachability_matrix_t[action_t_minus_one_id, actions_t_id] = is_valid_selection
            self.reachabilityMatrices.append(reachability_matrix_t)

    # OK
    def get_evaluation_costs(self):
        # Q_Net costs
        q_net_costs = {"node_{0}_{1}".format(node.index, k): v for node in self.topologicalSortedNodes
                       for k, v in node.opMacCostsDict.items() if "q_net" in k}
        total_q_net_cost = sum(q_net_costs.values())
        node_costs = {}
        for node in self.topologicalSortedNodes:
            cost = 0.0
            for k in node.opMacCostsDict.keys():
                if "q_net" not in k:
                    cost += node.opMacCostsDict[k]
            node_costs[node.index] = cost
        path_costs = []
        for node in self.leafNodes:
            leaf_ancestors = self.dagObject.ancestors(node=node)
            leaf_ancestors.append(node)
            path_costs.append(sum([node_costs[ancestor.index] for ancestor in leaf_ancestors]))
        self.baseEvaluationCost = np.mean(np.array(path_costs))
        self.networkActivationCosts = []
        self.networkActivationCostsDict = {}
        for action_id in range(self.actionSpaces[-1].shape[0]):
            node_selection = self.actionSpaces[-1][action_id]
            processed_nodes_set = set()
            for node_idx, curr_node in enumerate(self.leafNodes):
                if node_selection[node_idx] == 0:
                    continue
                leaf_ancestors = self.dagObject.ancestors(node=curr_node)
                leaf_ancestors.append(curr_node)
                for ancestor in leaf_ancestors:
                    processed_nodes_set.add(ancestor.index)
            total_cost = sum([node_costs[n_idx] for n_idx in processed_nodes_set])
            self.networkActivationCosts.append(total_cost)
            self.networkActivationCostsDict[tuple(self.actionSpaces[-1][action_id])] = \
                (total_cost / self.baseEvaluationCost) - 1.0
        self.networkActivationCosts = (np.array(self.networkActivationCosts) * (1.0 / self.baseEvaluationCost)) - 1.0

    def get_ig_paths(self, ig_masks_dict):
        ig_paths_matrix = []
        for t in range(self.get_max_trajectory_length() + 1):
            masks = []
            for node in self.orderedNodesPerLevel[t]:
                if not (type(ig_masks_dict[node.index]) is np.ndarray):
                    masks.append(ig_masks_dict[node.index].numpy())
                else:
                    masks.append(ig_masks_dict[node.index])
            # masks = [ig_masks_dict[node.index].numpy()
            #          if type(ig_masks_dict[node.index]) is np.ndarray is False
            #          else ig_masks_dict[node.index]
            #          for node in self.orderedNodesPerLevel[t]]
            mask_matrix = np.stack(masks, axis=-1)
            assert np.array_equal(np.ones_like(mask_matrix[:, 0]), np.sum(mask_matrix, axis=-1))
            ig_indices = np.argmax(mask_matrix, axis=-1)
            min_level_index = min([node.index for node in self.orderedNodesPerLevel[t]])
            ig_indices += min_level_index
            ig_paths_matrix.append(ig_indices)
        ig_paths_matrix = np.stack(ig_paths_matrix, axis=-1)
        return ig_paths_matrix

    def calculate_q_tables_from_network_outputs(self, true_labels, model_outputs):
        posteriors_dict = {k: v.numpy() for k, v in model_outputs["posteriors_dict"].items()}
        ig_masks_dict = {k: v.numpy() for k, v in model_outputs["ig_masks_dict"].items()}
        sample_count = true_labels.shape[0]
        ig_paths_matrix = self.get_ig_paths(ig_masks_dict=ig_masks_dict)
        c = Counter([tuple(ig_paths_matrix[i]) for i in range(ig_paths_matrix.shape[0])])
        # print("Count of ig paths:{0}".format(c))
        regression_targets = []
        optimal_q_tables = []

        for t in range(self.get_max_trajectory_length() - 1, -1, -1):
            action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
            action_count_t = self.actionSpaces[t].shape[0]
            optimal_q_table = np.zeros(shape=(sample_count, action_count_t_minus_one, action_count_t), dtype=np.float32)
            # If in the last layer, take into account the prediction accuracies.
            if t == self.get_max_trajectory_length() - 1:
                posteriors_tensor = []
                for leaf_node in self.leafNodes:
                    if not (type(posteriors_dict[leaf_node.index]) is np.ndarray):
                        posteriors_tensor.append(posteriors_dict[leaf_node.index].numpy())
                    else:
                        posteriors_tensor.append(posteriors_dict[leaf_node.index])
                posteriors_tensor = np.stack(posteriors_tensor, axis=-1)

                # Assert that posteriors are placed correctly.
                min_leaf_index = min([node.index for node in self.leafNodes])
                for leaf_node in self.leafNodes:
                    if not (type(posteriors_dict[leaf_node.index]) is np.ndarray):
                        assert np.array_equal(posteriors_tensor[:, :, leaf_node.index - min_leaf_index],
                                              posteriors_dict[leaf_node.index].numpy())
                    else:
                        assert np.array_equal(posteriors_tensor[:, :, leaf_node.index - min_leaf_index],
                                              posteriors_dict[leaf_node.index])

                # Combine posteriors with respect to the action tuple.
                prediction_correctness_vec_list = []
                calculation_cost_vec_list = []
                min_leaf_id = min([node.index for node in self.leafNodes])
                ig_indices = ig_paths_matrix[:, -1] - min_leaf_id
                for action_id in range(self.actionSpaces[t].shape[0]):
                    routing_decision = self.actionSpaces[t][action_id, :]
                    routing_matrix = np.repeat(routing_decision[np.newaxis, :], axis=0,
                                               repeats=true_labels.shape[0])
                    if self.includeIgInRewardCalculations:
                        # Set Information Gain routed leaf nodes to 1. They are always evaluated.
                        routing_matrix[np.arange(true_labels.shape[0]), ig_indices] = 1
                    weights = np.reciprocal(np.sum(routing_matrix, axis=1).astype(np.float32))
                    routing_matrix_weighted = weights[:, np.newaxis] * routing_matrix
                    weighted_posteriors = posteriors_tensor * routing_matrix_weighted[:, np.newaxis, :]
                    final_posteriors = np.sum(weighted_posteriors, axis=2)
                    predicted_labels = np.argmax(final_posteriors, axis=1)
                    validity_of_predictions_vec = (predicted_labels == true_labels).astype(np.int32)
                    prediction_correctness_vec_list.append(validity_of_predictions_vec)
                    # Get the calculation costs
                    computation_overload_vector = np.apply_along_axis(
                        lambda x: self.networkActivationCostsDict[tuple(x)], axis=1,
                        arr=routing_matrix)
                    calculation_cost_vec_list.append(computation_overload_vector)
                prediction_correctness_matrix = np.stack(prediction_correctness_vec_list, axis=1)
                prediction_correctness_tensor = np.repeat(
                    np.expand_dims(prediction_correctness_matrix, axis=1), axis=1,
                    repeats=action_count_t_minus_one)
                computation_overload_matrix = np.stack(calculation_cost_vec_list, axis=1)
                computation_overload_tensor = np.repeat(
                    np.expand_dims(computation_overload_matrix, axis=1), axis=1,
                    repeats=action_count_t_minus_one)
                # Add to the rewards tensor
                optimal_q_table += (prediction_correctness_tensor == 1
                                    ).astype(np.float32) * self.validPredictionReward
                optimal_q_table += (prediction_correctness_tensor == 0
                                    ).astype(np.float32) * self.invalidPredictionPenalty
                optimal_q_table -= self.lambdaMacCost * computation_overload_tensor
                for idx in range(optimal_q_table.shape[1] - 1):
                    assert np.array_equal(optimal_q_table[:, idx, :], optimal_q_table[:, idx + 1, :])
                regression_targets.append(optimal_q_table[:, 0, :].copy())
            else:
                q_table_next = optimal_q_tables[-1].copy()
                q_table_next = np.max(q_table_next, axis=-1)
                regression_targets.append(q_table_next)
                optimal_q_table = np.expand_dims(q_table_next, axis=1)
                optimal_q_table = np.repeat(optimal_q_table, axis=1, repeats=action_count_t_minus_one)
            reachability_matrix = self.reachabilityMatrices[t].copy().astype(np.float32)
            reachability_matrix = np.repeat(np.expand_dims(reachability_matrix, axis=0), axis=0,
                                            repeats=optimal_q_table.shape[0])
            assert optimal_q_table.shape == reachability_matrix.shape
            optimal_q_table[reachability_matrix == 0] = -np.inf
            optimal_q_tables.append(optimal_q_table)

        regression_targets.reverse()
        optimal_q_tables.reverse()
        return regression_targets

    def convert_model_outputs_to_dict(self, model_output_arr):
        # eval_dict, classification_losses, info_gain_losses, posteriors_dict, \
        # sc_masks_dict, ig_masks_dict, q_tables_predicted, node_outputs_dict, ig_activations_dict \
        model_output_dict = {
            "eval_dict": model_output_arr[0],
            "classification_losses": model_output_arr[1],
            "info_gain_losses": model_output_arr[2],
            "posteriors_dict": model_output_arr[3],
            "sc_masks_dict": model_output_arr[4],
            "ig_masks_dict": model_output_arr[5],
            "q_tables_predicted": model_output_arr[6],
            "node_outputs_dict": model_output_arr[7],
            "ig_activations_dict": model_output_arr[8]
        }
        return model_output_dict

    def calculate_optimal_q_values(self, dataset, batch_size, shuffle_data):
        # Step 1: Evaluate all samples and get the class predictions (posterior probabilities)
        # From there, we are going to calculate what kind of node combinations lead to correct predictions.
        # X_list = []
        # x_features_list = []
        # for t in range(self.get_max_trajectory_length()):
        #     x_features_list.append([])
        #     for idx in range(len(self.orderedNodesPerLevel[t])):
        #         x_features_list[t].append([])
        # y_list = [[] for t in range(self.get_max_trajectory_length() - 1, -1, -1)]
        # q_tables = [[] for t in range(self.get_max_trajectory_length() - 1, -1, -1)]
        #
        # for X, y in dataset:
        #     optimal_q_tables = []
        #     regression_targets = []
        #     model_output = self.run_model(X=X, y=y, iteration=-1, is_training=False, warm_up_period=False)
        #     ig_paths_matrix = self.get_ig_paths(ig_masks_dict=model_output["ig_masks_dict"])
        #     true_labels = y.numpy()
        #     sample_count = true_labels.shape[0]
        #     # Build the optimal Q tables from the final level, recursively up to the top.
        #     for t in range(self.get_max_trajectory_length() - 1, -1, -1):
        #         action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
        #         action_count_t = self.actionSpaces[t].shape[0]
        #         optimal_q_table = np.zeros(shape=(sample_count, action_count_t_minus_one, action_count_t),
        #                                    dtype=np.float32)
        #
        #         # If in the last layer, take into account the prediction accuracies.
        #         if t == self.get_max_trajectory_length() - 1:
        #             posteriors_tensor = []
        #             for leaf_node in self.leafNodes:
        #                 posteriors_tensor.append(model_output["posteriors_dict"][leaf_node.index].numpy())
        #             posteriors_tensor = np.stack(posteriors_tensor, axis=-1)
        #
        #             # Assert that posteriors are placed correctly.
        #             min_leaf_index = min([node.index for node in self.leafNodes])
        #             for leaf_node in self.leafNodes:
        #                 assert np.array_equal(posteriors_tensor[:, :, leaf_node.index - min_leaf_index],
        #                                       model_output["posteriors_dict"][leaf_node.index].numpy())
        #
        #             # Combine posteriors with respect to the action tuple.
        #             prediction_correctness_vec_list = []
        #             calculation_cost_vec_list = []
        #             min_leaf_id = min([node.index for node in self.leafNodes])
        #             ig_indices = ig_paths_matrix[:, -1] - min_leaf_id
        #             for action_id in range(self.actionSpaces[t].shape[0]):
        #                 routing_decision = self.actionSpaces[t][action_id, :]
        #                 routing_matrix = np.repeat(routing_decision[np.newaxis, :], axis=0,
        #                                            repeats=y.shape[0])
        #                 if self.includeIgInRewardCalculations:
        #                     # Set Information Gain routed leaf nodes to 1. They are always evaluated.
        #                     routing_matrix[np.arange(true_labels.shape[0]), ig_indices] = 1
        #                 weights = np.reciprocal(np.sum(routing_matrix, axis=1).astype(np.float32))
        #                 routing_matrix_weighted = weights[:, np.newaxis] * routing_matrix
        #                 weighted_posteriors = posteriors_tensor * routing_matrix_weighted[:, np.newaxis, :]
        #                 final_posteriors = np.sum(weighted_posteriors, axis=2)
        #                 predicted_labels = np.argmax(final_posteriors, axis=1)
        #                 validity_of_predictions_vec = (predicted_labels == true_labels).astype(np.int32)
        #                 prediction_correctness_vec_list.append(validity_of_predictions_vec)
        #                 # Get the calculation costs
        #                 computation_overload_vector = np.apply_along_axis(
        #                     lambda x: self.networkActivationCostsDict[tuple(x)], axis=1,
        #                     arr=routing_matrix)
        #                 calculation_cost_vec_list.append(computation_overload_vector)
        #             prediction_correctness_matrix = np.stack(prediction_correctness_vec_list, axis=1)
        #             prediction_correctness_tensor = np.repeat(
        #                 np.expand_dims(prediction_correctness_matrix, axis=1), axis=1,
        #                 repeats=action_count_t_minus_one)
        #             computation_overload_matrix = np.stack(calculation_cost_vec_list, axis=1)
        #             computation_overload_tensor = np.repeat(
        #                 np.expand_dims(computation_overload_matrix, axis=1), axis=1,
        #                 repeats=action_count_t_minus_one)
        #             # Add to the rewards tensor
        #             optimal_q_table += (prediction_correctness_tensor == 1
        #                                 ).astype(np.float32) * self.validPredictionReward
        #             optimal_q_table += (prediction_correctness_tensor == 0
        #                                 ).astype(np.float32) * self.invalidPredictionPenalty
        #             optimal_q_table -= self.lambdaMacCost * computation_overload_tensor
        #             for idx in range(optimal_q_table.shape[1] - 1):
        #                 assert np.array_equal(optimal_q_table[:, idx, :], optimal_q_table[:, idx + 1, :])
        #             regression_targets.append(optimal_q_table[:, 0, :].copy())
        #         else:
        #             q_table_next = optimal_q_tables[-1].copy()
        #             q_table_next = np.max(q_table_next, axis=-1)
        #             regression_targets.append(q_table_next)
        #             optimal_q_table = np.expand_dims(q_table_next, axis=1)
        #             optimal_q_table = np.repeat(optimal_q_table, axis=1, repeats=action_count_t_minus_one)
        #
        #         reachability_matrix = self.reachabilityMatrices[t].copy().astype(np.float32)
        #         reachability_matrix = np.repeat(np.expand_dims(reachability_matrix, axis=0), axis=0,
        #                                         repeats=optimal_q_table.shape[0])
        #         assert optimal_q_table.shape == reachability_matrix.shape
        #         optimal_q_table[reachability_matrix == 0] = -np.inf
        #         optimal_q_tables.append(optimal_q_table)
        #
        #     X_list.append(X.numpy())
        #     for idx in range(len(regression_targets)):
        #         y_list[idx].append(regression_targets[idx])
        #     for idx in range(len(optimal_q_tables)):
        #         q_tables[idx].append(optimal_q_tables[idx])
        #     # Intermediate features
        #     for t in range(self.get_max_trajectory_length()):
        #         level_nodes = self.orderedNodesPerLevel[t]
        #         for idx, node in enumerate(level_nodes):
        #             x_output = model_output["node_outputs_dict"][node.index]["F"]
        #             x_features_list[t][idx].append(x_output)
        #
        # X_list = np.concatenate(X_list, axis=0)
        # for idx in range(len(y_list)):
        #     y_list[idx] = np.concatenate(y_list[idx], axis=0)
        # for idx in range(len(q_tables)):
        #     q_tables[idx] = np.concatenate(q_tables[idx], axis=0)
        #
        # # Concatenate intermediate features
        # for t in range(self.get_max_trajectory_length()):
        #     level_nodes = self.orderedNodesPerLevel[t]
        #     for idx, node in enumerate(level_nodes):
        #         x_features_list[t][idx] = np.concatenate(x_features_list[t][idx], axis=0)
        #
        # y_list.reverse()
        # q_tables.reverse()
        #
        # # Assertions
        # for t in range(0, self.get_max_trajectory_length()):
        #     action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
        #     y_ = y_list[t]
        #     y_ = np.expand_dims(y_, axis=1)
        #     y_ = np.repeat(y_, axis=1, repeats=action_count_t_minus_one)
        #     reachability_matrix = self.reachabilityMatrices[t].copy().astype(np.float32)
        #     reachability_matrix = np.repeat(np.expand_dims(reachability_matrix, axis=0), axis=0,
        #                                     repeats=y_.shape[0])
        #     assert y_.shape == reachability_matrix.shape
        #     y_[reachability_matrix == 0] = -np.inf
        #     assert np.array_equal(y_, q_tables[t])

        # Collect all data from the network first, the obtain optimal q tables later.
        X_ = []
        x_features_list_all = []
        for t in range(self.get_max_trajectory_length()):
            x_features_list_all.append([])
            for idx in range(len(self.orderedNodesPerLevel[t])):
                x_features_list_all[t].append([])
        y_ = []
        ig_masks = {node.index: [] for node in self.topologicalSortedNodes}
        posteriors = {node.index: [] for node in self.leafNodes}

        for X, y in dataset:
            X_.append(X.numpy())
            y_.append(y.numpy())
            model_output = self.run_model(X=X, y=y, iteration=-1, is_training=False, warm_up_period=False)
            for node in self.topologicalSortedNodes:
                ig_masks[node.index].append(model_output["ig_masks_dict"][node.index].numpy())
                if node.isLeaf:
                    posteriors[node.index].append(model_output["posteriors_dict"][node.index].numpy())
            # Intermediate features
            for t in range(self.get_max_trajectory_length()):
                level_nodes = self.orderedNodesPerLevel[t]
                for idx, node in enumerate(level_nodes):
                    x_output = model_output["node_outputs_dict"][node.index]["F"]
                    x_features_list_all[t][idx].append(x_output)

        X_ = np.concatenate(X_, axis=0)
        y_ = np.concatenate(y_, axis=0)
        for node in self.topologicalSortedNodes:
            ig_masks[node.index] = np.concatenate(ig_masks[node.index], axis=0)
            if node.isLeaf:
                posteriors[node.index] = np.concatenate(posteriors[node.index], axis=0)
        # Concatenate intermediate features
        for t in range(self.get_max_trajectory_length()):
            level_nodes = self.orderedNodesPerLevel[t]
            for idx, node in enumerate(level_nodes):
                x_features_list_all[t][idx] = np.concatenate(x_features_list_all[t][idx], axis=0)

        model_outputs = {"posteriors_dict": posteriors, "ig_masks_dict": ig_masks}
        regs = self.calculate_q_tables_from_network_outputs(true_labels=y_, model_outputs=model_outputs)

        # assert len(regs) == len(y_list)
        # assert len(q_s) == len(q_tables)
        # for idx in range(len(regs)):
        #     assert np.array_equal(regs[idx], y_list[idx])
        #     assert np.array_equal(q_s[idx], q_tables[idx])
        #
        # for t in range(self.get_max_trajectory_length()):
        #     level_nodes = self.orderedNodesPerLevel[t]
        #     for idx, node in enumerate(level_nodes):
        #         assert np.array_equal(x_features_list[t][idx], x_features_list_all[t][idx])

        x_features_linearized = []
        for arr in x_features_list_all:
            for x_feat in arr:
                x_features_linearized.append(x_feat)
        if shuffle_data:
            q_learning_dataset = tf.data.Dataset.from_tensor_slices(
                (X_, *regs, *x_features_linearized)).shuffle(5000).batch(batch_size)
        else:
            q_learning_dataset = tf.data.Dataset.from_tensor_slices(
                (X_, *regs, *x_features_linearized)).batch(batch_size)

        # This is checking for if the Tensorflow data generator works as intended.
        # X_data = []
        # y_data = [[] for _ in range(len(regs))]
        # x_feat_data = []
        # for t in range(self.get_max_trajectory_length()):
        #     x_feat_data.append([])
        #     for idx in range(len(self.orderedNodesPerLevel[t])):
        #         x_feat_data[t].append([])
        #
        # for tpl in q_learning_dataset:
        #     # x_batch, y_1, y_2, x_intermediate_features in q_learning_dataset:
        #     x_batch = tpl[0]
        #     y_1 = tpl[1]
        #     y_2 = tpl[2]
        #     X_data.append(x_batch)
        #     y_data[0].append(y_1)
        #     y_data[1].append(y_2)
        #     x_feat_idx = 0
        #     for t in range(self.get_max_trajectory_length()):
        #         for idx in range(len(self.orderedNodesPerLevel[t])):
        #             x_feat_data[t][idx].append(tpl[3 + x_feat_idx])
        #             x_feat_idx += 1
        #
        # X_data = np.concatenate(X_data, axis=0)
        # assert np.array_equal(X_, X_data)
        # for idx in range(len(y_data)):
        #     y_data[idx] = np.concatenate(y_data[idx], axis=0)
        #     assert np.array_equal(y_data[idx], regs[idx])
        #
        # for t in range(self.get_max_trajectory_length()):
        #     for idx in range(len(self.orderedNodesPerLevel[t])):
        #         x_feat_data[t][idx] = np.concatenate(x_feat_data[t][idx], axis=0)
        #         assert np.array_equal(x_feat_data[t][idx], x_features_list_all[t][idx])

        return q_learning_dataset

    def get_q_net_layer(self, level):
        pass

    def calculate_secondary_routing_matrix(self, level, input_f_tensor, input_ig_routing_matrix):
        assert len(self.scRoutingCalculationLayers) == level

        # This code piece creates the corresponding Q-Net for the current layer, indicated by the variable "level".
        # The Q-Net always takes the aggregated F outputs of the current layer's nodes (sparsified by the sc masks)
        # and outputs raw Q-table predictions. This has been implemented as a separate tf.keras.Model,
        # since they will be used in isolation during the iterative training of the CIGN-RL scheme.
        q_net_input_f_tensor = tf.keras.Input(shape=input_f_tensor.shape[1:],
                                              name="input_f_tensor_q_net_level_{0}".format(level),
                                              dtype=input_f_tensor.dtype)
        q_net_layer = self.get_q_net_layer(level=level)
        q_table_predicted = q_net_layer(q_net_input_f_tensor)
        q_net = tf.keras.Model(inputs=q_net_input_f_tensor, outputs=q_table_predicted)
        q_table_predicted_cign_output = q_net(inputs=input_f_tensor)

        self.qNets.append(q_net)
        self.qTablesPredicted.append(q_table_predicted_cign_output)

        # Now, we are going to calculate the routing matrix (sc matrix) by using the Q-table predictions of the Q-Net,
        # by utilizing the Bellman equation.
        node = self.orderedNodesPerLevel[level][-1]
        routing_calculation_layer = CignRlRoutingLayer(level=level, node=node, network=self, use_ig_in_actions=True)
        past_actions = tf.zeros_like(tf.argmax(q_table_predicted_cign_output, axis=-1)) \
            if level == 0 else self.actionsPredicted[level - 1]
        predicted_actions, secondary_routing_matrix_cign_output = routing_calculation_layer(
            [q_table_predicted_cign_output,
             input_ig_routing_matrix,
             # self.warmUpPeriodInput,
             past_actions])
        self.actionsPredicted.append(predicted_actions)
        self.scRoutingCalculationLayers.append(routing_calculation_layer)
        return secondary_routing_matrix_cign_output

    def build_network(self):
        self.build_action_spaces()
        self.build_reachability_matrices()
        super().build_network()
        self.get_evaluation_costs()

    def get_model_outputs_array(self):
        temp_output_dict = {}
        for node_id, _d in self.nodeOutputsDict.items():
            temp_output_dict[node_id] = {}
            for k, v in _d.items():
                if v is None:
                    continue
                temp_output_dict[node_id][k] = v
        self.nodeOutputsDict = temp_output_dict

        model_output_arr = [self.evalDict,
                            self.classificationLosses,
                            self.informationGainRoutingLosses,
                            self.posteriorsDict,
                            self.scMaskInputsDict,
                            self.igMaskInputsDict,
                            self.qTablesPredicted,
                            self.nodeOutputsDict,
                            self.igActivationsDict]
        return model_output_arr

    def build_trackers(self):
        super().build_trackers()
        self.qNetTrackers = [tf.keras.metrics.Mean(name="q_net_loss_tracker_{0}".format(idx))
                             for idx, _ in enumerate(self.qTablesPredicted)]

    def reset_trackers(self):
        super().reset_trackers()
        for layer_id in range(len(self.qNetTrackers)):
            self.qNetTrackers[layer_id].reset_states()

    def get_feed_dict(self, x, y, iteration, is_training, **kwargs):
        feed_dict = super().get_feed_dict(x=x, y=y, iteration=iteration, is_training=is_training)
        assert "warm_up_period" in kwargs
        feed_dict["warm_up_period"] = kwargs["warm_up_period"]
        return feed_dict

    def calculate_q_net_loss(self, q_net, mse_loss, q_truth, q_pred):
        # Mse Loss
        mse = mse_loss(q_truth, q_pred)
        # Weight decaying
        variables = q_net.trainable_variables
        regularization_losses = []
        for var in variables:
            if var.ref() in self.regularizationCoefficients:
                lambda_coeff = self.regularizationCoefficients[var.ref()]
                regularization_losses.append(lambda_coeff * tf.nn.l2_loss(var))
        total_regularization_loss = tf.add_n(regularization_losses)
        # Total loss
        total_loss = total_regularization_loss + mse
        return total_loss, total_regularization_loss, mse

    def get_regularization_loss(self, is_for_q_nets):
        variables = self.model.trainable_variables
        regularization_losses = []
        for var in variables:
            if var.ref() in self.regularizationCoefficients:
                if is_for_q_nets and "q_net" in var.name:
                    lambda_coeff = self.regularizationCoefficients[var.ref()]
                elif is_for_q_nets and "q_net" not in var.name:
                    lambda_coeff = 0.0
                elif not is_for_q_nets and "q_net" in var.name:
                    lambda_coeff = 0.0
                elif not is_for_q_nets and "q_net" not in var.name:
                    lambda_coeff = self.regularizationCoefficients[var.ref()]
                else:
                    raise NotImplementedError()
                regularization_losses.append(lambda_coeff * tf.nn.l2_loss(var))
        total_regularization_loss = tf.add_n(regularization_losses)
        return total_regularization_loss

    def get_q_learning_batch(self, tpl):
        q_regression_targets = []
        x_feat_data = {}
        # Read original data batch, Q-table regression targets and intermediate features for Q-learning.
        x_batch = tpl[0]
        for t in range(self.get_max_trajectory_length()):
            q_regression_targets.append(tpl[t + 1])
        x_feat_idx = 0
        for t in range(self.get_max_trajectory_length()):
            for node in self.orderedNodesPerLevel[t]:
                x_feat_data[node.index] = tpl[1 + self.get_max_trajectory_length() + x_feat_idx]
                x_feat_idx += 1
        return q_regression_targets, x_feat_data, x_batch.shape[0]

    def get_q_learning_input(self, x_feat_data, route_matrix, batch_size, curr_level):
        level_nodes = self.orderedNodesPerLevel[curr_level]
        routing_prep_layer = self.scRoutingPreparationLayers[curr_level]
        ig_matrices = [
            tf.zeros(shape=(batch_size, self.nodeOutputsDict[node.index]["ig_mask_matrix"].shape[1]),
                     dtype=self.nodeOutputsDict[node.index]["ig_mask_matrix"].dtype)
            for node in level_nodes]
        sc_masks = [tf.identity(route_matrix[:, col_id]) for col_id in range(route_matrix.shape[1])]
        f_outputs = [x_feat_data[node.index] for node in level_nodes]
        input_f_tensor, input_ig_routing_matrix = routing_prep_layer([f_outputs, ig_matrices, sc_masks])
        # Check if input tensor is correctly built.
        last_axis_dim = int(input_f_tensor.shape[-1] / len(level_nodes))
        for idx in range(len(level_nodes)):
            input_x = input_f_tensor.numpy()[..., idx * last_axis_dim:(idx + 1) * last_axis_dim]
            axes = tuple([j + 1 for j in range(len(input_x.shape) - 1)])
            axes_sum = np.sum(input_x, axis=axes)
            sparse_vec = np.logical_not(axes_sum == 0).astype(np.int32)
            assert np.array_equal(sparse_vec, sc_masks[idx].numpy())
        return input_f_tensor, input_ig_routing_matrix

    def eval_q_tables(self, q_tables):
        last_actions = np.zeros(shape=(q_tables[0].shape[0],), dtype=np.int32)
        for curr_level in range(self.get_max_trajectory_length()):
            q_table = q_tables[curr_level].numpy()
            q_table = q_table[np.arange(q_table.shape[0]), last_actions]
            feasibility_matrix = self.reachabilityMatrices[curr_level][last_actions]
            penalty_matrix = np.where(feasibility_matrix.astype(np.bool),
                                      0.0,
                                      CignRlRoutingLayer.infeasible_action_penalty)
            q_table_with_penalties = penalty_matrix + q_table
            last_actions = np.argmax(q_table_with_penalties, axis=-1)
            # best_q_values = np.max(q_table_with_penalties, axis=-1)
        # accuracy = np.sum(best_q_values > 0.0) / best_q_values.shape[0]
        # print("Accuracy={0}".format(accuracy))
        return q_table_with_penalties, q_table

    def eval_q_nets(self, dataset):
        q_truth_tables = []
        q_predicted_tables = []
        for curr_level in range(self.get_max_trajectory_length()):
            q_net = self.qNets[curr_level]
            mse_loss = tf.keras.losses.MeanSquaredError()
            mse_loss_tracker = tf.keras.metrics.Mean(name="mse_loss_tracker")
            level_nodes = self.orderedNodesPerLevel[curr_level]
            previous_action_space = np.ones(shape=(1, 1), dtype=np.int32) \
                if curr_level == 0 else self.actionSpaces[curr_level - 1]

            q_truth = []
            q_predicted = []
            actions = []
            for action_id in range(previous_action_space.shape[0]):
                q_truth_action = []
                q_predicted_action = []
                for tpl in dataset:
                    q_regression_targets, x_feat_data, batch_size = self.get_q_learning_batch(tpl=tpl)
                    a_t_minus_one = action_id * np.ones(shape=(batch_size,), dtype=np.int32)
                    route_matrix = tf.constant(previous_action_space[a_t_minus_one])
                    # Prepare the Q-Net input
                    input_f_tensor, input_ig_routing_matrix = \
                        self.get_q_learning_input(x_feat_data=x_feat_data,
                                                  route_matrix=route_matrix,
                                                  batch_size=batch_size,
                                                  curr_level=curr_level)
                    q_regression_predicted = q_net(inputs=input_f_tensor, training=False)
                    q_truth_action.append(q_regression_targets[curr_level])
                    q_predicted_action.append(q_regression_predicted)
                    actions.append(a_t_minus_one)
                q_truth_action = tf.concat(q_truth_action, axis=0)
                q_predicted_action = tf.concat(q_predicted_action, axis=0)
                q_truth.append(q_truth_action)
                q_predicted.append(q_predicted_action)

            q_truth = tf.stack(q_truth, axis=1)
            q_predicted = tf.stack(q_predicted, axis=1)
            mse = mse_loss(q_truth, q_predicted)
            print("Level:{0} Mse={1}".format(curr_level, mse.numpy()))
            q_truth_tables.append(q_truth)
            q_predicted_tables.append(q_predicted)
        last_q_table_truth, last_q_table_no_penalties_truth = self.eval_q_tables(q_tables=q_truth_tables)
        last_q_table_predicted, last_q_table_no_penalties_predicted = self.eval_q_tables(q_tables=q_predicted_tables)
        # Record ground truth labels
        ground_truth_labels = []
        for idx in range(last_q_table_no_penalties_truth.shape[0]):
            q_row = last_q_table_no_penalties_truth[idx]
            correctness = q_row > 0.0
            correct_labels = set(np.nonzero(correctness)[0])
            ground_truth_labels.append(correct_labels)

        correct_count = 0
        y_pred = []
        for idx in range(last_q_table_predicted.shape[0]):
            selected_index = np.argmax(last_q_table_predicted[idx])
            if selected_index in ground_truth_labels[idx]:
                correct_count += 1
                y_pred.append(True)
            else:
                y_pred.append(False)
        accuracy = correct_count / last_q_table_predicted.shape[0]
        print("Accuracy:{0}".format(accuracy))
        return q_predicted_tables, last_q_table_no_penalties_predicted, y_pred

    def train_q_nets(self, dataset, q_net_epoch_count):
        print("Training Q-Nets")
        # Prepare the Q-Learning dataset
        q_learning_dataset_val = \
            self.calculate_optimal_q_values(dataset=dataset.validationDataTf,
                                            batch_size=self.batchSizeNonTensor,
                                            shuffle_data=True)
        q_learning_dataset_train = \
            self.calculate_optimal_q_values(dataset=dataset.trainDataTf,
                                            batch_size=self.batchSizeNonTensor,
                                            shuffle_data=False)
        q_learning_dataset_test = \
            self.calculate_optimal_q_values(dataset=dataset.testDataTf,
                                            batch_size=self.batchSizeNonTensor,
                                            shuffle_data=False)

        # Training happens here; level by level.
        for curr_level in range(self.get_max_trajectory_length()):
            q_net = self.qNets[curr_level]
            mse_loss = tf.keras.losses.MeanSquaredError()
            mse_loss_tracker = tf.keras.metrics.Mean(name="mse_loss_tracker")
            # Adam Optimizer
            q_net_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            level_nodes = self.orderedNodesPerLevel[curr_level]
            previous_action_space = np.ones(shape=(1, 1), dtype=np.int32) \
                if curr_level == 0 else self.actionSpaces[curr_level - 1]
            for epoch_id in range(q_net_epoch_count):
                mse_loss_tracker.reset_states()
                # Iterate over the dataset.
                for tpl in q_learning_dataset_val:
                    q_regression_targets, x_feat_data, batch_size = self.get_q_learning_batch(tpl=tpl)
                    with tf.GradientTape() as tape:
                        # Sample previous actions
                        a_t_minus_one = np.random.randint(low=0, high=previous_action_space.shape[0],
                                                          size=(batch_size,))
                        route_matrix = tf.constant(previous_action_space[a_t_minus_one])
                        # Prepare the Q-Net input
                        input_f_tensor, input_ig_routing_matrix = \
                            self.get_q_learning_input(x_feat_data=x_feat_data,
                                                      route_matrix=route_matrix,
                                                      batch_size=batch_size,
                                                      curr_level=curr_level)
                        # Calculate the MSE
                        q_regression_predicted = q_net(inputs=input_f_tensor, training=True)
                        total_loss, total_regularization_loss, mse = self.calculate_q_net_loss(
                            q_net=q_net,
                            mse_loss=mse_loss,
                            q_truth=q_regression_targets[curr_level],
                            q_pred=q_regression_predicted)
                    # Apply grads, to the Q-Net grads
                    grads = tape.gradient(total_loss, q_net.trainable_variables)
                    q_net_optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
                    mse_loss_tracker.update_state(total_loss)
                    print("Level:{0} Epoch:{1} Total Loss:{2}".format(curr_level,
                                                                      epoch_id,
                                                                      mse_loss_tracker.result().numpy()))

    def check_q_net_vars(self):
        var_dict = {v.ref(): v.numpy() for v in self.model.variables}
        for level in range(self.networkDepth):
            q_net = self.qNets[level]
            for q_net_var in q_net.variables:
                assert q_net_var.ref() in var_dict
                assert np.array_equal(q_net_var.numpy(), var_dict[q_net_var.ref()])

    def run_q_net_model(self, X, y, iteration):
        with tf.GradientTape() as q_tape:
            model_outputs = self.run_model(
                X=X,
                y=y,
                iteration=iteration,
                is_training=True,
                warm_up_period=False)
            # TODO: Check this
            regression_q_targets, optimal_q_values = self.calculate_q_tables_from_network_outputs(
                true_labels=y.numpy(),
                model_outputs=model_outputs)
            # Q-Net Losses
            q_net_predicted = model_outputs["q_tables_predicted"]
            q_net_predicted_np = []
            q_net_losses = []
            for idx, tpl in enumerate(zip(regression_q_targets, q_net_predicted)):
                q_truth = tpl[0]
                q_predicted = tpl[1]
                q_truth_tensor = tf.convert_to_tensor(q_truth, dtype=q_predicted.dtype)
                mse = self.mseLoss(q_truth_tensor, q_predicted)
                q_net_losses.append(mse)
                q_net_predicted_np.append(q_predicted.numpy())
            full_q_loss = self.qNetCoeff * tf.add_n(q_net_losses)
            q_regularization_loss = self.get_regularization_loss(is_for_q_nets=True)
            total_q_loss = full_q_loss + q_regularization_loss
        q_grads = q_tape.gradient(total_q_loss, self.model.trainable_variables)
        return model_outputs, q_grads, q_net_losses, regression_q_targets, \
               q_net_predicted_np, optimal_q_values, model_outputs

    def train_q_nets_with_full_net(self, dataset, q_net_epoch_count):
        def augment_training_image_fn(image, labels):
            image = tf.image.random_flip_left_right(image)
            return image, labels

        self.build_trackers()
        self.reset_trackers()
        # train_tf = tf.data.Dataset.from_tensor_slices((dataset.valX, dataset.valY)).shuffle(
        #     5000).batch(self.batchSizeNonTensor)
        train_tf = tf.data.Dataset.from_tensor_slices((dataset.valX, dataset.valY)).batch(self.batchSizeNonTensor)
        # q_net_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        # q_net_optimizer = self.get_sgd_optimizer()
        # Kod adı: TONPİLOY
        boundaries = [2000, 5000, 9000]
        values = [0.1, 0.01, 0.001, 0.0001]
        self.globalStep.assign(value=0)
        learning_rate_scheduler_tf = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries, values=values)
        q_net_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_scheduler_tf, momentum=0.9)
        q_net_variables = [v for v in self.model.trainable_variables if "q_net" in v.name]
        q_net_var_set = set([v.name for v in q_net_variables])

        # Just check
        q_net_var_set_from_models = set()
        for level in range(self.get_max_trajectory_length()):
            for q_var in self.qNets[level].variables:
                q_net_var_set_from_models.add(q_var.name)

        assert q_net_var_set_from_models == q_net_var_set
        X_baseline = []
        Q_values_baseline = {}
        posteriors_dict_baseline = {}
        ig_activations_baseline = {}
        actions_predicted_baseline = {}

        iteration = 0
        for X, y in train_tf:
            X_baseline.append(X.numpy())
            model_output_val, q_grads, q_net_losses, regression_q_targets, \
            q_net_predicted, optimal_q_values, model_outputs = self.run_q_net_model(
                X=X,
                y=y,
                iteration=iteration)
            Utilities.append_dict_to_dict(dict_destination=posteriors_dict_baseline,
                                          dict_source=model_outputs["posteriors_dict"])
            Utilities.append_dict_to_dict(dict_destination=ig_activations_baseline,
                                          dict_source=model_outputs["ig_activations_dict"])
            Utilities.append_dict_to_dict(dict_destination=Q_values_baseline,
                                          dict_source=optimal_q_values,
                                          convert_to_numpy=False)
            iteration += 1

        X_baseline = np.concatenate(X_baseline, axis=0)
        Utilities.concatenate_dict_of_arrays(dict_=Q_values_baseline, axis=-1)
        Utilities.concatenate_dict_of_arrays(dict_=posteriors_dict_baseline, axis=0)
        Utilities.concatenate_dict_of_arrays(dict_=ig_activations_baseline, axis=0)

        iteration = 0
        for epoch_id in range(q_net_epoch_count):
            X_epoch = []
            Q_values_epoch = {}
            posteriors_dict_epoch = {}
            ig_activations_epoch = {}

            for layer_id in range(len(self.qNetTrackers)):
                self.qNetTrackers[layer_id].reset_states()

            # Keep track of prediction statistics
            zero_to_one_counters = []
            optimal_decisions_ground_truths = []
            optimal_decisions_predictions = []
            for layer_id in range(self.get_max_trajectory_length()):
                zero_to_one_counters.append(Counter())
                optimal_decisions_ground_truths.append([])
                optimal_decisions_predictions.append([])

            for X, y in train_tf:
                X_epoch.append(X.numpy())
                # DONE
                model_output_val, q_grads, q_net_losses, regression_q_targets, \
                q_net_predicted, optimal_q_values, model_outputs = self.run_q_net_model(
                    X=X,
                    y=y,
                    iteration=iteration)

                Utilities.append_dict_to_dict(dict_destination=posteriors_dict_epoch,
                                              dict_source=model_outputs["posteriors_dict"])
                Utilities.append_dict_to_dict(dict_destination=ig_activations_epoch,
                                              dict_source=model_outputs["ig_activations_dict"])
                Utilities.append_dict_to_dict(dict_destination=Q_values_epoch,
                                              dict_source=optimal_q_values,
                                              convert_to_numpy=False)

                # Keep track of decision statistics
                for layer_id in range(self.get_max_trajectory_length()):
                    c = Counter(np.argmax(regression_q_targets[layer_id], axis=1))
                    zero_to_one_counters[layer_id].update(c)
                    optimal_decisions_ground_truths[layer_id].append(regression_q_targets[layer_id])
                    optimal_decisions_predictions[layer_id].append(q_net_predicted[layer_id])
                # Don't update non-Q-net variables
                q_grads_zeroed = []
                for grad, var in zip(q_grads, self.model.trainable_variables):
                    if var.name in q_net_var_set:
                        q_grads_zeroed.append(grad)
                    else:
                        q_grads_zeroed.append(tf.zeros_like(grad))

                q_net_optimizer.apply_gradients(zip(q_grads_zeroed, self.model.trainable_variables))
                # if np.array_equal(q_grads[10].numpy(), np.zeros_like(q_grads[10].numpy())):
                #     print("Zero grad!")
                assert not np.allclose(q_grads[0].numpy(), np.zeros_like(q_grads[0].numpy()))
                assert not np.allclose(q_grads[10].numpy(), np.zeros_like(q_grads[10].numpy()))
                print("*************************************")
                for layer_id in range(len(self.qNetTrackers)):
                    self.qNetTrackers[layer_id].update_state(q_net_losses[layer_id])
                    print("Epoch:{0} Iteration:{1} Q-Net{2} MSE Loss:{3}".format(
                        epoch_id, iteration,
                        layer_id,
                        self.qNetTrackers[layer_id].result().numpy()))
                print("Epoch:{0} Iteration:{1} Total Loss:{2}".format(
                    epoch_id, iteration,
                    np.sum(np.array(
                        [self.qNetTrackers[layer_id].result().numpy() for layer_id in range(len(self.qNetTrackers))]))
                ))
                print("Lr:{0}".format(q_net_optimizer._decayed_lr(tf.float32).numpy()))
                print("*************************************")
                iteration += 1
                self.globalStep.assign(value=iteration)

            X_epoch = np.concatenate(X_epoch, axis=0)
            assert np.array_equal(X_baseline, X_epoch)
            Utilities.concatenate_dict_of_arrays(dict_=Q_values_epoch, axis=-1)
            Utilities.concatenate_dict_of_arrays(dict_=posteriors_dict_epoch, axis=0)
            Utilities.concatenate_dict_of_arrays(dict_=ig_activations_epoch, axis=0)

            for nid in ig_activations_baseline.keys():
                assert np.allclose(ig_activations_baseline[nid], ig_activations_epoch[nid])

            for nid in posteriors_dict_baseline.keys():
                assert np.allclose(posteriors_dict_baseline[nid], posteriors_dict_epoch[nid])

            for nid in Q_values_baseline.keys():
                assert np.allclose(Q_values_baseline[nid], Q_values_epoch[nid])

            print("X")

            # Keep track of decision statistics
            for layer_id in range(self.get_max_trajectory_length()):
                print("****************Layer {0}****************".format(layer_id))
                print(zero_to_one_counters[layer_id])
                gt_table = np.concatenate(optimal_decisions_ground_truths[layer_id], axis=0)
                pr_table = np.concatenate(optimal_decisions_predictions[layer_id], axis=0)
                gt_optimal_actions = np.argmax(gt_table, axis=1)
                pr_optimal_actions = np.argmax(pr_table, axis=1)
                c_report = classification_report(y_true=gt_optimal_actions, y_pred=pr_optimal_actions)
                print(c_report)
                print("****************Layer {0}****************".format(layer_id))

    def train(self, run_id, dataset, epoch_count, **kwargs):
        q_net_epoch_count = kwargs["q_net_epoch_count"]
        fine_tune_epoch_count = kwargs["fine_tune_epoch_count"]
        is_in_warm_up_period = True
        self.optimizer = self.get_sgd_optimizer()
        self.build_trackers()

        # Group main body CIGN variables and Q Net variables.
        # q_net_variables = [v for v in self.model.trainable_variables if "q_net" in v.name]
        # q_net_var_set = set([v.name for v in q_net_variables])
        # cign_main_body_variables = [v for v in self.model.trainable_variables if v.name not in q_net_var_set]

        epochs_after_warm_up = 0
        iteration = 0
        for epoch_id in range(epoch_count):
            # OK
            iteration, times_list = self.train_cign_body_one_epoch(dataset=dataset.trainDataTf,
                                                                   run_id=run_id,
                                                                   iteration=iteration,
                                                                   is_in_warm_up_period=is_in_warm_up_period)
            self.check_q_net_vars()
            if epoch_id >= self.warmUpPeriod:
                # Run Q-Net learning for the first time.
                if is_in_warm_up_period:
                    self.save_model(run_id=run_id)
                    is_in_warm_up_period = False
                    self.train_q_nets_with_full_net(dataset=dataset, q_net_epoch_count=q_net_epoch_count)
                    self.measure_performance(dataset=dataset,
                                             run_id=run_id,
                                             iteration=iteration,
                                             epoch_id=epoch_id,
                                             times_list=times_list)
                else:
                    is_performance_measured = False
                    if (epochs_after_warm_up + 1) % self.cignRlTrainPeriod == 0:
                        self.train_q_nets_with_full_net(dataset=dataset, q_net_epoch_count=q_net_epoch_count)
                        self.measure_performance(dataset=dataset,
                                                 run_id=run_id,
                                                 iteration=iteration,
                                                 epoch_id=epoch_id,
                                                 times_list=times_list)
                        is_performance_measured = True
                    if (epoch_id >= epoch_count - 10 or epoch_id % self.trainEvalPeriod == 0) \
                            and is_performance_measured is False:
                        self.measure_performance(dataset=dataset,
                                                 run_id=run_id,
                                                 iteration=iteration,
                                                 epoch_id=epoch_id,
                                                 times_list=times_list)
                epochs_after_warm_up += 1

        # Fine tune by merging training and validation sets
        full_train_X = np.concatenate([dataset.trainX, dataset.valX], axis=0)
        full_train_y = np.concatenate([dataset.trainY, dataset.valY], axis=0)
        train_tf = tf.data.Dataset.from_tensor_slices((full_train_X, full_train_y)). \
            shuffle(5000).batch(self.batchSizeNonTensor)

        for epoch_id in range(epoch_count, epoch_count + fine_tune_epoch_count):
            iteration, times_list = self.train_cign_body_one_epoch(dataset=train_tf,
                                                                   run_id=run_id,
                                                                   iteration=iteration,
                                                                   is_in_warm_up_period=is_in_warm_up_period)
            self.measure_performance(dataset=dataset,
                                     run_id=run_id,
                                     iteration=iteration,
                                     epoch_id=epoch_id,
                                     times_list=times_list)

    def eval_verbose(self, run_id, iteration, dataset, dataset_type):
        if dataset is None:
            return 0.0
        y_true = []
        y_pred = []

        q_tables_predicted = []
        for _ in range(self.get_max_trajectory_length()):
            q_tables_predicted.append([])

        leaf_distributions = {node.index: [] for node in self.leafNodes}
        counter = 0
        for X, y in dataset:
            model_output = self.run_model(
                X=X,
                y=y,
                iteration=-1,
                is_training=False)

            for idx, arr in enumerate(model_output["q_tables_predicted"]):
                q_tables_predicted[idx].append(arr)

            leaf_weights = []
            posteriors = []
            for leaf_node in self.leafNodes:
                sc_mask = model_output["sc_masks_dict"][leaf_node.index]
                posterior = model_output["posteriors_dict"][leaf_node.index]
                leaf_weights.append(np.expand_dims(sc_mask, axis=-1))
                posteriors.append(posterior)
                y_leaf = y.numpy()[sc_mask.numpy().astype(np.bool)]
                leaf_distributions[leaf_node.index].extend(y_leaf)
            leaf_weights = np.stack(leaf_weights, axis=-1)
            posteriors = np.stack(posteriors, axis=-1)

            weighted_posteriors = leaf_weights * posteriors
            posteriors_mixture = np.sum(weighted_posteriors, axis=-1)
            y_pred_batch = np.argmax(posteriors_mixture, axis=-1)
            y_pred.append(y_pred_batch)
            y_true.append(y.numpy())
            counter += 1
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        truth_vector = y_true == y_pred
        accuracy = np.mean(truth_vector.astype(np.float))

        # Print sample distribution
        kv_rows = []
        for leaf_node in self.leafNodes:
            c = Counter(leaf_distributions[leaf_node.index])
            str_ = "{0} Node {1} Sample Distribution:{2}".format(dataset_type, leaf_node.index, c)
            print(str_)
            kv_rows.append((run_id, iteration,
                            "{0} Node {1} Sample Distribution".format(dataset_type, leaf_node.index),
                            "{0}".format(c)))
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)

        for idx in range(len(q_tables_predicted)):
            q_tables_predicted[idx] = np.concatenate(q_tables_predicted[idx], axis=0)

        return accuracy, q_tables_predicted, y_true, y_pred

    def evaluate_routing_configuration(self, posteriors_tensor, routing_matrix, true_labels):
        weights = np.reciprocal(np.sum(routing_matrix, axis=1).astype(np.float32))
        routing_matrix_weighted = weights[:, np.newaxis] * routing_matrix
        weighted_posteriors = posteriors_tensor * routing_matrix_weighted[:, np.newaxis, :]
        final_posteriors = np.sum(weighted_posteriors, axis=2)
        predicted_labels = np.argmax(final_posteriors, axis=1)
        validity_of_predictions_vec = (predicted_labels == true_labels).astype(np.int32)
        accuracy = np.mean(validity_of_predictions_vec)
        return accuracy, validity_of_predictions_vec
