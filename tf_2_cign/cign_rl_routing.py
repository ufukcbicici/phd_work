from collections import Counter

import numpy as np
import tensorflow as tf

from tf_2_cign.cign_no_mask import CignNoMask


class CignRlRouting(CignNoMask):
    def __init__(self, valid_prediction_reward, invalid_prediction_penalty, include_ig_in_reward_calculations,
                 lambda_mac_cost,
                 batch_size, input_dims, class_count, node_degrees, decision_drop_probability,
                 classification_drop_probability, decision_wd, classification_wd, information_gain_balance_coeff,
                 softmax_decay_controller, learning_rate_schedule, decision_loss_coeff, bn_momentum=0.9):
        super().__init__(batch_size, input_dims, class_count, node_degrees, decision_drop_probability,
                         classification_drop_probability, decision_wd, classification_wd,
                         information_gain_balance_coeff, softmax_decay_controller, learning_rate_schedule,
                         decision_loss_coeff, bn_momentum)
        self.validPredictionReward = valid_prediction_reward
        self.invalidPredictionPenalty = invalid_prediction_penalty
        self.lambdaMacCost = lambda_mac_cost
        self.actionSpaces = []
        self.reachabilityMatrices = []
        self.baseEvaluationCost = None
        self.networkActivationCosts = []
        self.networkActivationCostsDict = {}
        # self.optimalQtables = []
        # self.regressionTargets = []
        self.includeIgInRewardCalculations = include_ig_in_reward_calculations
        self.qNets = []

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
        path_costs = []
        for node in self.leafNodes:
            leaf_ancestors = self.dagObject.ancestors(node=node)
            leaf_ancestors.append(node)
            path_costs.append(sum([ancestor.macCost for ancestor in leaf_ancestors]))
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
            total_cost = sum([self.nodes[n_idx].macCost for n_idx in processed_nodes_set])
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

    def calculate_q_tables_from_network_outputs(self, true_labels, posteriors_dict, ig_masks_dict):
        sample_count = true_labels.shape[0]
        ig_paths_matrix = self.get_ig_paths(ig_masks_dict=ig_masks_dict)
        c = Counter([tuple(ig_paths_matrix[i]) for i in range(ig_paths_matrix.shape[0])])
        print("Count of ig paths:{0}".format(c))
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
        return regression_targets, optimal_q_tables

    def calculate_optimal_q_values(self, dataset, batch_size):
        # Step 1: Evaluate all samples and get the class predictions (posterior probabilities)
        # From there, we are going to calculate what kind of node combinations lead to correct predictions.
        X_list = []
        y_list = [[] for t in range(self.get_max_trajectory_length() - 1, -1, -1)]
        q_tables = [[] for t in range(self.get_max_trajectory_length() - 1, -1, -1)]

        for X, y in dataset:
            optimal_q_tables = []
            regression_targets = []
            feed_dict = self.get_feed_dict(x=X, y=y, iteration=-1, is_training=False)
            eval_dict, classification_losses, info_gain_losses, posteriors_dict, \
            sc_masks_dict, ig_masks_dict = self.model(inputs=feed_dict, training=False)
            ig_paths_matrix = self.get_ig_paths(ig_masks_dict=ig_masks_dict)
            true_labels = y.numpy()
            sample_count = true_labels.shape[0]
            # Build the optimal Q tables from the final level, recursively up to the top.
            for t in range(self.get_max_trajectory_length() - 1, -1, -1):
                action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
                action_count_t = self.actionSpaces[t].shape[0]
                optimal_q_table = np.zeros(shape=(sample_count, action_count_t_minus_one, action_count_t),
                                           dtype=np.float32)

                # If in the last layer, take into account the prediction accuracies.
                if t == self.get_max_trajectory_length() - 1:
                    posteriors_tensor = []
                    for leaf_node in self.leafNodes:
                        posteriors_tensor.append(posteriors_dict[leaf_node.index].numpy())
                    posteriors_tensor = np.stack(posteriors_tensor, axis=-1)

                    # Assert that posteriors are placed correctly.
                    min_leaf_index = min([node.index for node in self.leafNodes])
                    for leaf_node in self.leafNodes:
                        assert np.array_equal(posteriors_tensor[:, :, leaf_node.index - min_leaf_index],
                                              posteriors_dict[leaf_node.index].numpy())

                    # Combine posteriors with respect to the action tuple.
                    prediction_correctness_vec_list = []
                    calculation_cost_vec_list = []
                    min_leaf_id = min([node.index for node in self.leafNodes])
                    ig_indices = ig_paths_matrix[:, -1] - min_leaf_id
                    for action_id in range(self.actionSpaces[t].shape[0]):
                        routing_decision = self.actionSpaces[t][action_id, :]
                        routing_matrix = np.repeat(routing_decision[np.newaxis, :], axis=0,
                                                   repeats=y.shape[0])
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

            X_list.append(X.numpy())
            for idx in range(len(regression_targets)):
                y_list[idx].append(regression_targets[idx])
            for idx in range(len(optimal_q_tables)):
                q_tables[idx].append(optimal_q_tables[idx])

        X_list = np.concatenate(X_list, axis=0)
        for idx in range(len(y_list)):
            y_list[idx] = np.concatenate(y_list[idx], axis=0)
        for idx in range(len(q_tables)):
            q_tables[idx] = np.concatenate(q_tables[idx], axis=0)

        y_list.reverse()
        q_tables.reverse()

        # Assertions
        for t in range(0, self.get_max_trajectory_length()):
            action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
            y_ = y_list[t]
            y_ = np.expand_dims(y_, axis=1)
            y_ = np.repeat(y_, axis=1, repeats=action_count_t_minus_one)
            reachability_matrix = self.reachabilityMatrices[t].copy().astype(np.float32)
            reachability_matrix = np.repeat(np.expand_dims(reachability_matrix, axis=0), axis=0,
                                            repeats=y_.shape[0])
            assert y_.shape == reachability_matrix.shape
            y_[reachability_matrix == 0] = -np.inf
            assert np.array_equal(y_, q_tables[t])

        # Collect all data from the network first, the obtain optimal q tables later.
        X_ = []
        y_ = []
        ig_masks = {node.index: [] for node in self.topologicalSortedNodes}
        posteriors = {node.index: [] for node in self.leafNodes}

        for X, y in dataset:
            X_.append(X.numpy())
            y_.append(y.numpy())
            feed_dict = self.get_feed_dict(x=X, y=y, iteration=-1, is_training=False)
            eval_dict, classification_losses, info_gain_losses, posteriors_dict, sc_masks_dict, ig_masks_dict = \
                self.model(inputs=feed_dict, training=False)
            for node in self.topologicalSortedNodes:
                ig_masks[node.index].append(ig_masks_dict[node.index].numpy())
                if node.isLeaf:
                    posteriors[node.index].append(posteriors_dict[node.index].numpy())

        X_ = np.concatenate(X_, axis=0)
        y_ = np.concatenate(y_, axis=0)
        for node in self.topologicalSortedNodes:
            ig_masks[node.index] = np.concatenate(ig_masks[node.index], axis=0)
            if node.isLeaf:
                posteriors[node.index] = np.concatenate(posteriors[node.index], axis=0)

        regs, q_s = self.calculate_q_tables_from_network_outputs(true_labels=y_,
                                                                 posteriors_dict=posteriors,
                                                                 ig_masks_dict=ig_masks)

        assert len(regs) == len(y_list)
        assert len(q_s) == len(q_tables)
        for idx in range(len(regs)):
            assert np.array_equal(regs[idx], y_list[idx])
            assert np.array_equal(q_s[idx], q_tables[idx])
        q_learning_dataset = tf.data.Dataset.from_tensor_slices((X_, *regs)).batch(batch_size)

        # This is checking for if the Tensorflow data generator works as intended.
        X_data = []
        y_data = [[] for _ in range(len(regs))]
        for x_batch, y_1, y_2 in q_learning_dataset:
            X_data.append(x_batch)
            y_data[0].append(y_1)
            y_data[1].append(y_2)

        X_data = np.concatenate(X_data, axis=0)
        assert np.array_equal(X_, X_data)
        for idx in range(len(y_data)):
            y_data[idx] = np.concatenate(y_data[idx], axis=0)
            assert np.array_equal(y_data[idx], regs[idx])

        print("X")

        return q_learning_dataset

    def init(self):
        self.build_network()

        # Init RL operations
        self.build_action_spaces()
        self.build_reachability_matrices()
        self.get_evaluation_costs()

        self.build_tf_model()
