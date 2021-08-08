from collections import deque
from collections import Counter
import numpy as np
import tensorflow as tf
from algorithms.info_gain import InfoGainLoss
from auxillary.dag_utilities import Dag
from auxillary.db_logger import DbLogger
from simple_tf.uncategorized.node import Node
from tf_2_cign.cign import Cign
from tf_2_cign.cign_no_mask import CignNoMask
from tf_2_cign.custom_layers.cign_secondary_routing_preparation_layer import CignScRoutingPrepLayer
from tf_2_cign.custom_layers.cign_vanilla_sc_routing_layer import CignVanillaScRoutingLayer
from tf_2_cign.custom_layers.masked_batch_norm import MaskedBatchNormalization
from tf_2_cign.utilities import Utilities
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.cign_masking_layer import CignMaskingLayer
from tf_2_cign.custom_layers.cign_decision_layer import CignDecisionLayer
from tf_2_cign.custom_layers.cign_classification_layer import CignClassificationLayer
from collections import Counter
import time


class CignRlRouting(CignNoMask):
    def __init__(self, valid_action_reward, invalid_action_penalty, include_ig_in_reward_calculations,
                 batch_size, input_dims, class_count, node_degrees, decision_drop_probability,
                 classification_drop_probability, decision_wd, classification_wd, information_gain_balance_coeff,
                 softmax_decay_controller, learning_rate_schedule, decision_loss_coeff, bn_momentum=0.9):
        super().__init__(batch_size, input_dims, class_count, node_degrees, decision_drop_probability,
                         classification_drop_probability, decision_wd, classification_wd,
                         information_gain_balance_coeff, softmax_decay_controller, learning_rate_schedule,
                         decision_loss_coeff, bn_momentum)
        self.validActionReward = valid_action_reward
        self.invalidActionPenalty = invalid_action_penalty
        self.actionSpaces = []
        self.reachabilityMatrices = []
        self.baseEvaluationCost = None
        self.networkActivationCosts = []
        self.networkActivationCostsDict = {}
        self.includeIgInRewardCalculations = include_ig_in_reward_calculations

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
            masks = [ig_masks_dict[node.index].numpy() for node in self.orderedNodesPerLevel[t]]
            mask_matrix = np.stack(masks, axis=-1)
            assert np.array_equal(np.ones_like(mask_matrix[:, 0]), np.sum(mask_matrix, axis=-1))
            ig_indices = np.argmax(mask_matrix, axis=-1)
            min_level_index = min([node.index for node in self.orderedNodesPerLevel[t]])
            ig_indices += min_level_index
            ig_paths_matrix.append(ig_indices)
        ig_paths_matrix = np.stack(ig_paths_matrix, axis=-1)
        return ig_paths_matrix

    def calculate_optimal_q_values(self, dataset):
        # Step 1: Evaluate all samples and get the class predictions (posterior probabilities)
        # From there, we are going to calculate what kind of node combinations lead to correct predictions.
        X_list = []
        y_list = []
        for X, y in dataset:
            feed_dict = self.get_feed_dict(x=X, y=y, iteration=-1, is_training=False)
            eval_dict, classification_losses, info_gain_losses, posteriors_dict, \
            sc_masks_dict, ig_masks_dict = self.model(inputs=feed_dict, training=False)
            self.get_ig_paths(ig_masks_dict=ig_masks_dict)
            # Build the optimal Q tables from the final level, recursively up to the top.
            for t in range(self.get_max_trajectory_length() - 1, 0, -1):
                action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
                action_count_t = self.actionSpaces[t].shape[0]

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
                    for action_id in range(self.actionSpaces[t].shape[0]):
                        routing_decision = self.actionSpaces[t][action_id, :]
                        routing_matrix = np.repeat(routing_decision[np.newaxis, :], axis=0,
                                                   repeats=y.shape[0])
                        weights = np.reciprocal(np.sum(routing_matrix, axis=1).astype(np.float32))
                        routing_matrix_weighted = weights[:, np.newaxis] * routing_matrix
                        weighted_posteriors = posteriors_tensor * routing_matrix_weighted[:, np.newaxis, :]
                        print("X")











    # def calculate_reward_tensors(self):
    #     invalid_action_penalty = self.invalidActionPenalty
    #     valid_prediction_reward = self.validPredictionReward
    #     invalid_prediction_penalty = self.invalidPredictionPenalty
    #     self.rewardTensors = []
    #     label_list = self.routingDataset.labelList
    #     sample_count = label_list.shape[0]
    #     posteriors_tensor = self.posteriorTensors
    #     for t in range(self.get_max_trajectory_length()):
    #         action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
    #         action_count_t = self.actionSpaces[t].shape[0]
    #         reward_shape = (sample_count, action_count_t_minus_one, action_count_t)
    #         rewards_arr = np.zeros(shape=reward_shape, dtype=np.float32)
    #         validity_of_actions_tensor = np.repeat(np.expand_dims(self.reachabilityMatrices[t], axis=0),
    #                                                repeats=sample_count, axis=0)
    #         rewards_arr += (validity_of_actions_tensor == 0.0).astype(np.float32) * invalid_action_penalty
    #         if t == self.get_max_trajectory_length() - 1:
    #             true_labels = label_list
    #             # Prediction Rewards:
    #             # Calculate the prediction results for every state and for every routing decision
    #             prediction_correctness_vec_list = []
    #             calculation_cost_vec_list = []
    #             min_leaf_id = min([node.index for node in self.network.orderedNodesPerLevel[t + 1]])
    #             ig_indices = self.maxLikelihoodPaths[:, -1] - min_leaf_id
    #             for action_id in range(self.actionSpaces[t].shape[0]):
    #                 routing_decision = self.actionSpaces[t][action_id, :]
    #                 routing_matrix = np.repeat(routing_decision[np.newaxis, :], axis=0,
    #                                            repeats=true_labels.shape[0])
    #                 if self.includeIgInRewardCalculations:
    #                     # Set Information Gain routed leaf nodes to 1. They are always evaluated.
    #                     routing_matrix[np.arange(true_labels.shape[0]), ig_indices] = 1
    #                 weights = np.reciprocal(np.sum(routing_matrix, axis=1).astype(np.float32))
    #                 routing_matrix_weighted = weights[:, np.newaxis] * routing_matrix
    #                 assert routing_matrix.shape[1] == posteriors_tensor.shape[2]
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
    #             rewards_arr += (prediction_correctness_tensor == 1).astype(np.float32) * valid_prediction_reward
    #             rewards_arr += (prediction_correctness_tensor == 0).astype(
    #                 np.float32) * invalid_prediction_penalty
    #             rewards_arr -= self.lambdaMacCost * computation_overload_tensor
    #         self.rewardTensors.append(rewards_arr)

    def init(self):
        self.build_network()

        # Init RL operations
        self.build_action_spaces()
        self.build_reachability_matrices()
        self.get_evaluation_costs()

        self.build_tf_model()
