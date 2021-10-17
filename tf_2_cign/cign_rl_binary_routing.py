import numpy as np
import tensorflow as tf
import time

from auxillary.db_logger import DbLogger
from tf_2_cign.cign_rl_routing import CignRlRouting
from tf_2_cign.utilities.profiler import Profiler
from collections import Counter
from tf_2_cign.utilities.utilities import Utilities


class CignRlBinaryRouting(CignRlRouting):
    infeasible_action_penalty = -1000000.0

    def __init__(self, valid_prediction_reward, invalid_prediction_penalty, include_ig_in_reward_calculations,
                 lambda_mac_cost, warm_up_period, cign_rl_train_period, batch_size, input_dims, class_count,
                 node_degrees, decision_drop_probability, classification_drop_probability, decision_wd,
                 classification_wd, information_gain_balance_coeff, softmax_decay_controller, learning_rate_schedule,
                 decision_loss_coeff, q_net_coeff, bn_momentum=0.9):
        super().__init__(valid_prediction_reward, invalid_prediction_penalty, include_ig_in_reward_calculations,
                         lambda_mac_cost, warm_up_period, cign_rl_train_period, batch_size, input_dims, class_count,
                         node_degrees, decision_drop_probability, classification_drop_probability, decision_wd,
                         classification_wd, information_gain_balance_coeff, softmax_decay_controller,
                         learning_rate_schedule, decision_loss_coeff, bn_momentum)

    def calculate_ideal_accuracy(self, dataset):
        posteriors_dict = {}
        ig_masks_dict = {}
        ig_activations_dict = {}
        true_labels = []

        for X, y in dataset:
            model_output = self.run_model(X=X, y=y, iteration=-1, is_training=False, warm_up_period=False)
            posteriors_batch_dict = {k: v.numpy() for k, v in model_output["posteriors_dict"].items()}
            ig_masks_batch_dict = {k: v.numpy() for k, v in model_output["ig_masks_dict"].items()}
            ig_activations_batch_dict = {k: v.numpy() for k, v in model_output["ig_activations_dict"].items()}
            true_labels.append(y)
            Utilities.merge_dict_of_ndarrays(dict_target=posteriors_dict, dict_to_append=posteriors_batch_dict)
            Utilities.merge_dict_of_ndarrays(dict_target=ig_masks_dict, dict_to_append=ig_masks_batch_dict)
            Utilities.merge_dict_of_ndarrays(dict_target=ig_activations_dict, dict_to_append=ig_activations_batch_dict)
        true_labels = np.concatenate(true_labels)
        # Build the posteriors tensor
        posteriors_tensor = np.stack([posteriors_dict[node.index] for node in self.leafNodes], axis=-1)
        ig_routing_matrix = np.zeros(shape=(posteriors_tensor.shape[0], len(self.leafNodes)),
                                     dtype=posteriors_tensor.dtype)
        ig_paths = self.get_ig_paths(ig_masks_dict=ig_masks_dict)
        min_final_level_index = min([node.index for node in self.leafNodes])
        ig_indices = ig_paths[:, -1] - min_final_level_index
        ig_routing_matrix[np.arange(posteriors_tensor.shape[0]), ig_indices] = 1.0

        pure_ig_routing_accuracy, pure_ig_validity_of_predictions_vec = self.evaluate_routing_configuration(
            posteriors_tensor=posteriors_tensor,
            routing_matrix=ig_routing_matrix,
            true_labels=true_labels)
        print("pure_ig_routing_accuracy={0}".format(pure_ig_routing_accuracy))
        routing_configurations = self.calculate_binary_rl_configurations(ig_activations_dict=ig_activations_dict)

        truth_vectors = []
        for action_id in range(routing_configurations.shape[1]):
            routing_matrix = routing_configurations[:, action_id, :]
            routing_accuracy, validity_vec = self.evaluate_routing_configuration(
                posteriors_tensor=posteriors_tensor,
                routing_matrix=routing_matrix,
                true_labels=true_labels)
            print("Action Id:{0} Accuracy:{1}".format(action_id, np.mean(validity_vec)))
            truth_vectors.append(validity_vec)
        truth_vectors = np.stack(truth_vectors, axis=1)
        ideal_sums = np.sum(truth_vectors, axis=1)
        ideal_validity_vector = ideal_sums > 0
        ideal_routing_accuracy = np.mean(ideal_validity_vector)
        print("Ideal Routing Accuracy:{0}".format(ideal_routing_accuracy))

    def calculate_binary_rl_configurations(self, ig_activations_dict):
        # Create action spaces
        # All actions are binary at each tree level
        trajectory_length = self.get_max_trajectory_length()
        all_action_compositions_count = 2 ** trajectory_length
        action_configurations = np.zeros(shape=(all_action_compositions_count, trajectory_length), dtype=np.int32)
        for action_id in range(all_action_compositions_count):
            l = [int(x) for x in list('{0:0b}'.format(action_id))]
            for layer_id in range(trajectory_length):
                if layer_id == len(l):
                    break
                action_configurations[action_id, (trajectory_length - 1) - layer_id] = \
                    l[len(l) - 1 - layer_id]

        # Traverse all samples; calculate the leaf configurations according to the IG results.
        assert len(set([arr.shape[0] for arr in ig_activations_dict.values()]))
        sample_count = ig_activations_dict[self.topologicalSortedNodes[0].index].shape[0]
        min_leaf_index = min([nd.index for nd in self.leafNodes])
        routing_configurations = []
        for sample_id in range(sample_count):
            sample_routing_configurations = []
            for action_id, action_config in enumerate(action_configurations):
                selected_level_nodes = [self.topologicalSortedNodes[0]]
                for level in range(trajectory_length):
                    next_level_nodes = []
                    level_action = action_config[trajectory_length - 1 - level]
                    for selected_node in selected_level_nodes:
                        routing_probs = ig_activations_dict[selected_node.index][sample_id]
                        selected_child_id = np.argmax(routing_probs)
                        child_nodes = self.dagObject.children(node=selected_node)
                        child_nodes_dict = {self.get_node_sibling_index(node=nd): nd for nd in child_nodes}
                        # Only add the node as shown by the information gain
                        if level_action == 0:
                            next_level_nodes.append(child_nodes_dict[selected_child_id])
                        # Add all children
                        else:
                            next_level_nodes.extend(child_nodes)
                    next_level_nodes = sorted(next_level_nodes, key=lambda nd: nd.index)
                    selected_level_nodes = next_level_nodes
                leaf_indices = set([nd.index for nd in selected_level_nodes])
                leaf_configuration = np.array([1 if nd.index in leaf_indices else 0 for nd in self.leafNodes])
                sample_routing_configurations.append(leaf_configuration)
            sample_routing_configurations = np.stack(sample_routing_configurations, axis=0)
            routing_configurations.append(sample_routing_configurations)
        routing_configurations = np.stack(routing_configurations, axis=0)
        return routing_configurations
