import numpy as np
import tensorflow as tf
import time
import os
from scipy.stats import norm
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from auxillary.dag_utilities import Dag
from auxillary.db_logger import DbLogger
from tf_2_cign.cign_rl_routing import CignRlRouting
from tf_2_cign.custom_layers.cign_binary_action_generator_layer import CignBinaryActionGeneratorLayer
from tf_2_cign.custom_layers.cign_binary_action_result_generator_layer import CignBinaryActionResultGeneratorLayer
from tf_2_cign.custom_layers.cign_binary_rl_routing_layer import CignBinaryRlRoutingLayer
from tf_2_cign.custom_layers.cign_rl_routing_layer import CignRlRoutingLayer
from tf_2_cign.custom_layers.cign_test_layer import CignTestLayer
from tf_2_cign.utilities.profiler import Profiler
from collections import Counter
from tf_2_cign.utilities.utilities import Utilities
from collections import deque


class CignRlBinaryRouting(CignRlRouting):
    infeasible_action_penalty = -1000000.0

    def __init__(self, valid_prediction_reward, invalid_prediction_penalty, include_ig_in_reward_calculations,
                 lambda_mac_cost, warm_up_period, cign_rl_train_period, batch_size, input_dims, class_count,
                 node_degrees, decision_drop_probability, classification_drop_probability, decision_wd,
                 classification_wd, information_gain_balance_coeff, softmax_decay_controller, learning_rate_schedule,
                 decision_loss_coeff, q_net_coeff, epsilon_decay_rate, epsilon_step, reward_type, bn_momentum=0.9):
        super().__init__(valid_prediction_reward, invalid_prediction_penalty, include_ig_in_reward_calculations,
                         lambda_mac_cost, warm_up_period, cign_rl_train_period, batch_size, input_dims, class_count,
                         node_degrees, decision_drop_probability, classification_drop_probability, decision_wd,
                         classification_wd, information_gain_balance_coeff, softmax_decay_controller,
                         learning_rate_schedule, decision_loss_coeff, q_net_coeff, bn_momentum)
        # Epsilon hyperparameter for exploration - explotation
        self.afterWarmUpEpochCount = 0
        self.epsilonDecayRate = epsilon_decay_rate
        self.epsilonStep = epsilon_step
        self.rewardType = reward_type
        self.exploreExploitEpsilon = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1.0, decay_steps=self.epsilonStep, decay_rate=self.epsilonDecayRate)
        self.actionResultGenerators = {}
        self.actionCalculatorLayers = []
        self.qNetEndToEndModel = []
        self.enforcedActions = []
        for lid in range(self.get_max_trajectory_length()):
            enforced_actions = tf.keras.Input(shape=(),
                                              name="enforced_actions_{0}".format(lid),
                                              dtype=tf.int64)
            self.enforcedActions.append(enforced_actions)
            self.feedDict["enforced_actions_{0}".format(lid)] = enforced_actions
        self.onlyUseInformationGainForRouting = tf.keras.Input(shape=(),
                                                               name="only_use_information_gain_for_routing",
                                                               dtype=tf.bool)
        self.feedDict["only_use_ig_routing"] = self.onlyUseInformationGainForRouting
        self.qNetInputs = []

    def convert_posteriors_dict_to_tensor(self, posteriors_dict):
        posteriors_tensor = []
        for leaf_node in self.leafNodes:
            if not (type(posteriors_dict[leaf_node.index]) is np.ndarray):
                posteriors_tensor.append(posteriors_dict[leaf_node.index].numpy())
            else:
                posteriors_tensor.append(posteriors_dict[leaf_node.index])
        posteriors_tensor = np.stack(posteriors_tensor, axis=-1)
        return posteriors_tensor

    def calculate_ideal_accuracy(self, dataset):
        posteriors_dict = {}
        ig_masks_dict = {}
        ig_activations_dict = {}
        true_labels = []

        for X, y in dataset:
            model_output = self.run_model(X=X, y=y, iteration=-1,
                                          is_training=False, warm_up_period=False, only_use_ig_routing=False)
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

    def get_action_configurations(self):
        trajectory_length = self.get_max_trajectory_length()
        all_action_compositions_count = 2 ** trajectory_length
        action_configurations = np.zeros(shape=(all_action_compositions_count, trajectory_length), dtype=np.int32)
        actions_dict = {}
        for action_id in range(all_action_compositions_count):
            l = [int(x) for x in list('{0:0b}'.format(action_id))]
            for layer_id in range(trajectory_length):
                if layer_id == len(l):
                    break
                action_configurations[action_id, (trajectory_length - 1) - layer_id] = \
                    l[len(l) - 1 - layer_id]
            action_trajectory_as_tuple = tuple(action_configurations[action_id])
            actions_dict[action_trajectory_as_tuple] = action_id
        # for action_id in range(all_action_compositions_count):
        return action_configurations, actions_dict

    def calculate_binary_rl_configurations(self, ig_activations_dict):
        # Create action spaces
        # All actions are binary at each tree level
        trajectory_length = self.get_max_trajectory_length()
        action_configurations, actions_dict = self.get_action_configurations()
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
                    # level_action = action_config[trajectory_length - 1 - level]
                    level_action = action_config[level]
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

    # We don't have actions spaces as in binary routing as defined in the original RL version.
    # This is more or less for compatibility.
    # def build_action_spaces(self):
    #     max_trajectory_length = self.get_max_trajectory_length()
    #     for t in range(max_trajectory_length):
    #         action_space = np.array([0, 1])
    #         self.actionSpaces.append(action_space)

    # In binary routing, we don't have reachability definition. We always execute one of the two actions.
    def build_reachability_matrices(self):
        pass

    def build_isolated_q_net_model(self, input_f_tensor, level):
        # This code piece creates the corresponding Q-Net for the current layer, indicated by the variable "level".
        # The Q-Net always takes the aggregated F outputs of the current layer's nodes (sparsified by the sc masks)
        # and outputs raw Q-table predictions. This has been implemented as a separate tf.keras.Model,
        # since they will be used in isolation during the iterative training of the CIGN-RL scheme.
        q_net_input_f_tensor = tf.keras.Input(shape=input_f_tensor.shape[1:],
                                              name="input_f_tensor_q_net_level_{0}".format(level),
                                              dtype=input_f_tensor.dtype)
        # global_step_input = tf.keras.Input(shape=(),
        #                                    name="global_step_input_level_{0}".format(level),
        #                                    dtype=tf.int32)
        isolated_q_net_layer = self.get_q_net_layer(level=level)
        q_table_predicted = isolated_q_net_layer(q_net_input_f_tensor)
        # Create the action generator layer. Get predicted actions for the next layer.
        action_generator_layer = CignBinaryActionGeneratorLayer(network=self)
        predicted_actions, explore_exploit_vec, explore_actions, exploit_actions = action_generator_layer(
            q_table_predicted)
        q_net_isolated = tf.keras.Model(inputs=q_net_input_f_tensor, outputs=[q_table_predicted])
        q_net = tf.keras.Model(inputs=q_net_input_f_tensor, outputs=[q_table_predicted,
                                                                     predicted_actions,
                                                                     explore_exploit_vec,
                                                                     explore_actions,
                                                                     exploit_actions])
        return q_net, q_net_isolated

    # Here, some changes are needed.
    def calculate_secondary_routing_matrix(self, level, input_f_tensor, input_ig_routing_matrix):
        assert len(self.scRoutingCalculationLayers) == level
        q_net, isolated_q_net = self.build_isolated_q_net_model(input_f_tensor=input_f_tensor, level=level)
        # Connect the Q-Net to the rest of the network.
        q_table_predicted, predicted_actions, explore_exploit_vec, explore_actions, exploit_actions = \
            q_net(inputs=input_f_tensor)
        q_net_end_to_end_model = tf.keras.Model(inputs=self.feedDict, outputs=[q_table_predicted,
                                                                               predicted_actions,
                                                                               explore_exploit_vec,
                                                                               explore_actions,
                                                                               exploit_actions])
        self.qNetInputs.append(input_f_tensor)
        self.qNetEndToEndModel.append(q_net_end_to_end_model)
        # Save Q-Net and its outputs.
        self.qNets.append(isolated_q_net)
        self.qTablesPredicted.append(q_table_predicted)
        # Pick enforced actions, if they are provided.
        are_enforced_actions_provided = tf.greater(self.enforcedActions[level], -1)
        predicted_actions = tf.where(are_enforced_actions_provided, self.enforcedActions[level], predicted_actions)
        self.actionsPredicted.append(predicted_actions)

        # Get information gain activations for the current level.
        ig_activations = tf.stack(
            [self.igActivationsDict[nd.index] for nd in self.orderedNodesPerLevel[level]], axis=-1)
        # Get secondary routing information for the current level (counter intuitively, it resides at "level-1").
        if level - 1 < 0:
            sc_routing_matrix = tf.expand_dims(tf.ones_like(input_ig_routing_matrix[:, 0]), axis=-1)
        else:
            sc_routing_matrix = self.scRoutingMatricesDict[level - 1]
        # Create the routing layer.
        routing_calculation_layer = CignBinaryRlRoutingLayer(level=level, network=self)
        secondary_routing_matrix_cign_output = routing_calculation_layer(
            [ig_activations,
             sc_routing_matrix,
             predicted_actions,
             explore_exploit_vec,
             explore_actions,
             exploit_actions])
        self.scRoutingCalculationLayers.append(routing_calculation_layer)
        secondary_routing_matrix_cign_output = tf.where(self.onlyUseInformationGainForRouting,
                                                        input_ig_routing_matrix,
                                                        secondary_routing_matrix_cign_output)
        return secondary_routing_matrix_cign_output

    def get_model_outputs_array(self):
        model_output_arr = super().get_model_outputs_array()
        model_output_arr.append(self.actionsPredicted)
        model_output_arr.append(self.scRoutingMatricesDict)
        model_output_arr.append(self.enforcedActions)
        model_output_arr.append(self.igRoutingMatricesDict)
        model_output_arr.append(self.qNetInputs)
        model_output_arr.append(self.qTablesPredicted)
        return model_output_arr

    def convert_model_outputs_to_dict(self, model_output_arr):
        model_output_dict = super(CignRlBinaryRouting, self).convert_model_outputs_to_dict(model_output_arr)
        model_output_dict["actions_predicted"] = model_output_arr[9]
        model_output_dict["sc_routing_matrices_dict"] = model_output_arr[10]
        model_output_dict["enforced_actions"] = model_output_arr[11]
        model_output_dict["ig_routing_matrices_dict"] = model_output_arr[12]
        model_output_dict["q_net_inputs"] = model_output_arr[13]
        model_output_dict["q_tables_predicted"] = model_output_arr[14]
        return model_output_dict

    # OK
    # Next level's possible routing configurations, as the result of the activated nodes in the current layer and
    # their information gain activations.
    def calculate_next_level_action_results_manual(self, level, ig_activations, sc_routing_matrix_curr_level):
        batch_size = sc_routing_matrix_curr_level.shape[0]
        assert len(self.orderedNodesPerLevel[level]) == sc_routing_matrix_curr_level.shape[1]
        next_level_config_action_0 = []
        next_level_config_action_1 = []
        for sample_id in range(batch_size):
            curr_level_config = sc_routing_matrix_curr_level[sample_id]
            next_level_selected_nodes_acton_0 = []
            next_level_selected_nodes_acton_1 = []
            for node_id, node in enumerate(self.orderedNodesPerLevel[level]):
                if curr_level_config[node_id] == 0:
                    continue
                child_nodes = self.dagObject.children(node=node)
                child_nodes_sorted = sorted(child_nodes, key=lambda nd: nd.index)
                curr_sample_ig_activations = ig_activations[sample_id, :, node_id]
                # Action 0: Only max information gain paths.
                arg_max_idx = tf.argmax(curr_sample_ig_activations)
                next_level_node_selected = child_nodes_sorted[arg_max_idx]
                next_level_selected_nodes_acton_0.append(next_level_node_selected)
                # Action 1: Every child in the next layer will be visited.
                next_level_selected_nodes_acton_1.extend(child_nodes_sorted)
            # Convert selected nodes to next level configurations.
            next_level_selected_nodes_acton_0 = set([nd.index for nd in next_level_selected_nodes_acton_0])
            next_level_selected_nodes_acton_1 = set([nd.index for nd in next_level_selected_nodes_acton_1])
            action_0_vec = []
            action_1_vec = []
            for node in self.orderedNodesPerLevel[level + 1]:
                if node.index in next_level_selected_nodes_acton_0:
                    action_0_vec.append(1)
                else:
                    action_0_vec.append(0)
                if node.index in next_level_selected_nodes_acton_1:
                    action_1_vec.append(1)
                else:
                    action_1_vec.append(0)
            action_0_vec = np.array(action_0_vec)
            action_1_vec = np.array(action_1_vec)
            next_level_config_action_0.append(action_0_vec)
            next_level_config_action_1.append(action_1_vec)

        next_level_config_action_0 = np.stack(next_level_config_action_0, axis=0)
        next_level_config_action_1 = np.stack(next_level_config_action_1, axis=0)
        return next_level_config_action_0, next_level_config_action_1

    # OK
    def calculate_sample_action_results(self, ig_activations_dict):
        action_spaces = []
        batch_size = ig_activations_dict[0].shape[0]
        sc_routing_tensor_curr_level = np.expand_dims(
            np.expand_dims(np.ones(shape=[batch_size], dtype=np.int32), axis=-1), axis=0)
        action_spaces.append(sc_routing_tensor_curr_level)
        for level in range(self.get_max_trajectory_length()):
            # IG activations for the nodes in this layer.
            ig_activations = tf.stack([ig_activations_dict[nd.index] for nd in self.orderedNodesPerLevel[level]],
                                      axis=-1)
            # Generate action space generator layer for the first time.
            if level not in self.actionResultGenerators:
                self.actionResultGenerators[level] = \
                    CignBinaryActionResultGeneratorLayer(level=level, network=self)
            # For every past action combination in the current trajectory so far,
            # get the current layer's node configuration. If we are at t. level, there will be actions taken:
            # A_{0:t-1} = a_0,a_1,a_2, ..., a_{t-1}
            # There will be 2^t different paths to the level t (each action is binary).
            actions_each_layer = [[0]]
            actions_each_layer.extend([[0, 1] for _ in range(level)])
            list_of_all_trajectories = Utilities.get_cartesian_product(list_of_lists=actions_each_layer)
            sc_routing_tensor_next_level_shape = [1]
            sc_routing_tensor_next_level_shape.extend([2 for _ in range(level + 1)])
            sc_routing_tensor_next_level_shape.append(batch_size)
            sc_routing_tensor_next_level_shape.append(len(self.orderedNodesPerLevel[level + 1]))
            sc_routing_tensor_next_level = np.zeros(shape=sc_routing_tensor_next_level_shape,
                                                    dtype=sc_routing_tensor_curr_level.dtype)
            for trajectory in list_of_all_trajectories:
                # Given the current trajectory = a_0,a_1,a_2, ..., a_{t-1}, get the configuration of the
                # nodes in the current level, for every sample in the batch
                current_level_routing_matrix = tf.convert_to_tensor(sc_routing_tensor_curr_level[trajectory])
                sc_routing_matrix_action_0, sc_routing_matrix_action_1 = self.actionResultGenerators[level](
                    [ig_activations, current_level_routing_matrix])
                action_0_trajectory = []
                action_0_trajectory.extend(trajectory)
                action_0_trajectory.append(0)
                sc_routing_tensor_next_level[tuple(action_0_trajectory)] = sc_routing_matrix_action_0.numpy()

                action_1_trajectory = []
                action_1_trajectory.extend(trajectory)
                action_1_trajectory.append(1)
                sc_routing_tensor_next_level[tuple(action_1_trajectory)] = sc_routing_matrix_action_1.numpy()

            sc_routing_tensor_curr_level = sc_routing_tensor_next_level
            action_spaces.append(sc_routing_tensor_curr_level)
        return action_spaces

    # OK
    # For every sample in the current minibatch, we calculate ALL possible routing configurations, for every level of
    # the CIGN, as a function of the sample's information gain activations.
    # The array shape for holding configurations at level t is like that:
    # [1,2,...,2 (t times 2),batch_size, 2^t]
    def calculate_sample_action_results_manual(self, ig_activations_dict):
        action_spaces = []
        batch_size = ig_activations_dict[0].shape[0]
        for level in range(self.get_max_trajectory_length() + 1):
            shp = [1]
            shp.extend([2 for _ in range(level)])
            shp.append(batch_size)
            shp.append(len(self.orderedNodesPerLevel[level]))
            sc_routing_tensor_next_level = np.zeros(shape=shp, dtype=np.int32)
            action_spaces.append(sc_routing_tensor_next_level)

        for sample_id in range(batch_size):
            root_action_node = {"level": 0,
                                "action_trajectory": [0],
                                "activated_nodes": [self.topologicalSortedNodes[0]]}
            active_action_nodes = deque()
            active_action_nodes.append(root_action_node)
            processed_action_nodes = []
            # Action tree traversal
            while len(active_action_nodes) > 0:
                curr_action_node = active_action_nodes.popleft()
                curr_level = curr_action_node["level"]
                curr_action_trajectory = curr_action_node["action_trajectory"]
                activated_nodes = curr_action_node["activated_nodes"]
                processed_action_nodes.append(curr_action_node)
                if curr_level == self.get_max_trajectory_length():
                    continue

                # Action 0
                action_0_activated_nodes = []
                for nd in activated_nodes:
                    ig_activations = ig_activations_dict[nd.index][sample_id]
                    arg_max_idx = tf.argmax(ig_activations).numpy()
                    child_cign_nodes = self.dagObject.children(node=nd)
                    child_cign_nodes = sorted(child_cign_nodes, key=lambda x: x.index)
                    action_0_activated_nodes.append(child_cign_nodes[arg_max_idx])
                action_0_activated_nodes = sorted(action_0_activated_nodes, key=lambda x: x.index)
                action_0_trajectory = []
                action_0_trajectory.extend(curr_action_trajectory)
                action_0_trajectory.append(0)
                new_action_0_node = {"level": curr_level + 1,
                                     "action_trajectory": action_0_trajectory,
                                     "activated_nodes": action_0_activated_nodes}
                active_action_nodes.append(new_action_0_node)

                # Action 1
                action_1_activated_nodes = []
                for nd in activated_nodes:
                    child_cign_nodes = self.dagObject.children(node=nd)
                    child_cign_nodes = sorted(child_cign_nodes, key=lambda x: x.index)
                    action_1_activated_nodes.extend(child_cign_nodes)
                action_1_activated_nodes = sorted(action_1_activated_nodes, key=lambda x: x.index)
                action_1_trajectory = []
                action_1_trajectory.extend(curr_action_trajectory)
                action_1_trajectory.append(1)
                new_action_1_node = {"level": curr_level + 1,
                                     "action_trajectory": action_1_trajectory,
                                     "activated_nodes": action_1_activated_nodes}
                active_action_nodes.append(new_action_1_node)

            # Dump the routing configurations into the arrays.
            for action_obj in processed_action_nodes:
                # root_action_node = {"level": 0,
                #                     "action_trajectory": [0],
                #                     "activated_nodes": [self.topologicalSortedNodes[0]]}
                config_arr = []
                node_ids = set([nd.index for nd in action_obj["activated_nodes"]])
                for nd in self.orderedNodesPerLevel[action_obj["level"]]:
                    if nd.index in node_ids:
                        config_arr.append(1)
                    else:
                        config_arr.append(0)
                config_arr = np.array(config_arr)
                coords = []
                coords.extend(action_obj["action_trajectory"])
                coords.append(sample_id)
                action_spaces[action_obj["level"]][tuple(coords)] = config_arr
        return action_spaces

    def calculate_reward_values_from_leaf_level_configurations(self,
                                                               routing_matrix,
                                                               posteriors_tensor,
                                                               true_labels):
        assert routing_matrix.shape[1] == len(self.leafNodes)
        assert posteriors_tensor.shape[2] == len(self.leafNodes)
        # Calculate prediction validity
        weights = np.reciprocal(np.sum(routing_matrix, axis=1).astype(np.float32))
        routing_matrix_weighted = weights[:, np.newaxis] * routing_matrix
        weighted_posteriors = posteriors_tensor * routing_matrix_weighted[:, np.newaxis, :]
        final_posteriors = np.sum(weighted_posteriors, axis=2)
        predicted_labels = np.argmax(final_posteriors, axis=1)
        validity_of_predictions_vec = (predicted_labels == true_labels).astype(np.int32)
        # Calculate the MAC cost
        # Get the calculation costs
        computation_overload_vector = np.apply_along_axis(
            lambda x: self.networkActivationCostsDict[tuple(x)], axis=1,
            arr=routing_matrix)
        rewards_vec = np.zeros_like(computation_overload_vector)
        rewards_vec += (validity_of_predictions_vec == 1).astype(np.float32) * self.validPredictionReward
        rewards_vec += (validity_of_predictions_vec == 0).astype(np.float32) * self.invalidPredictionPenalty
        rewards_vec -= self.lambdaMacCost * computation_overload_vector
        return rewards_vec

    def create_mock_ig_activations(self, batch_size):
        ig_activations_dict = {}
        # Mock IG activations
        for node in self.topologicalSortedNodes:
            if node.isLeaf:
                continue
            ig_arr = tf.random.uniform(
                shape=[batch_size, len(self.dagObject.children(node=node))], dtype=tf.float32)
            ig_activations_dict[node.index] = ig_arr
        return ig_activations_dict

    def create_mock_posteriors(self, true_labels, batch_size, mean_accuracy, std_accuracy):
        class_count = np.max(true_labels) + 1
        posterior_arrays_dict = {nd.index: [] for nd in self.leafNodes}
        leaf_expected_accuracies = {nd.index:
                                        max(min(np.random.normal(loc=mean_accuracy, scale=std_accuracy), 1.0), 0.0)
                                    for nd in self.leafNodes}

        for sample_id in range(batch_size):
            y = true_labels[sample_id]
            for leaf_node in self.leafNodes:
                expected_accuracy = leaf_expected_accuracies[leaf_node.index]
                weights = np.random.uniform(low=-10.0, high=10.0, size=(class_count,))
                max_weight = np.max(weights)
                max_index = np.argmax(weights)
                is_sample_true = np.random.uniform(low=0.0, high=1.0) <= expected_accuracy
                if is_sample_true:
                    # Make the true label weight the maximum
                    weights[y] = max_weight * np.random.uniform(low=1.0, high=3.0)
                posterior = tf.nn.softmax(weights).numpy()
                posterior_arrays_dict[leaf_node.index].append(posterior)

        for leaf_node in self.leafNodes:
            posterior_arrays_dict[leaf_node.index] = np.stack(posterior_arrays_dict[leaf_node.index], axis=0)
            node_predictions = np.argmax(posterior_arrays_dict[leaf_node.index], axis=1)
            accuracy = np.mean(true_labels == node_predictions)
            print("Node {0} Accuracy:{1}".format(leaf_node.index, accuracy))
        posteriors_tensor = np.stack([posterior_arrays_dict[nd.index] for nd in self.leafNodes], axis=-1)
        final_posteriors = np.mean(posteriors_tensor, axis=-1)
        node_predictions = np.argmax(final_posteriors, axis=1)
        accuracy = np.mean(true_labels == node_predictions)
        print("Final Accuracy:{0}".format(accuracy))
        return posterior_arrays_dict

    def create_mock_predicted_actions(self, batch_size):
        actions_predicted = []
        for level in range(self.get_max_trajectory_length()):
            actions_of_this_level = np.random.randint(low=0, high=2, size=(batch_size,))
            actions_predicted.append(actions_of_this_level)
        return actions_predicted

    def create_complete_output_mock_data(self, batch_size, mean_accuracy, std_accuracy):
        # Create mock true labels
        true_labels = np.random.randint(low=0, high=10, size=(batch_size,))
        # Create mock information gain outputs
        ig_activations_dict = self.create_mock_ig_activations(batch_size=batch_size)
        # Create mock posterior
        posterior_arrays_dict = self.create_mock_posteriors(true_labels=true_labels,
                                                            batch_size=batch_size,
                                                            mean_accuracy=mean_accuracy,
                                                            std_accuracy=std_accuracy)
        # Create mock actions predicted.
        actions_predicted = self.create_mock_predicted_actions(batch_size=batch_size)
        return true_labels, ig_activations_dict, posterior_arrays_dict, actions_predicted

    # This method calculates the optimal Q-tables of a given CIGN result, including each sample's true label,
    # predicted posteriors and for each sample the interior information gain activations.
    # The expected result is the optimal Q-values for every possible trajectory in the CIGN.
    # For example, for a binary, depth two CIGN the output will contain:
    #
    # Level 2:
    # Q(s2=x_{00},a_2=0) <---- The utility of taking actions [0,0] for the sample x. (For notational convenience for
    # the root node, the each trajectory starts with a trivial action 0.)
    # Q(s2=x_{00},a_2=1) <---- The utility of taking actions [0,1] for the sample x.
    # Q(s2=x_{01},a_2=0) <---- The utility of taking actions [1,0] for the sample x.
    # Q(s2=x_{01},a_2=1) <---- The utility of taking actions [1,1] for the sample x.
    #
    # Level 1:
    # Q(s1=x_{0},a_1=0) -> Use Bellman Optimality Equation:
    # Without loss of generality:
    # Q(s1=x_{0},a_1=0) = R(s_1=x_{0},a_1=0) + sum_{s_2} p(s_2|s_1=x_{0},a_1=0) * max_{a_2} Q(s2,a_2)
    # Q(s1=x_{0},a_1=0) = R(s_1=x_{0},a_1=0) + max_{a_2} Q(s2=x_{00},a_2) (State transition is deterministic)
    # Q(s1=x_{0},a_1=1) = R(s_1=x_{0},a_1=1) + max_{a_2} Q(s2=x_{01},a_2) (State transition is deterministic)
    def calculate_optimal_q_tables(self, true_labels, posteriors_dict, ig_activations_dict):
        # Calculate the action spaces of each sample.
        action_spaces = self.calculate_sample_action_results(ig_activations_dict=ig_activations_dict)
        # Posteriors tensor for every sample and every leaf node.
        posteriors_tensor = self.convert_posteriors_dict_to_tensor(posteriors_dict=posteriors_dict)
        # Assert that posteriors are placed correctly.
        min_leaf_index = min([node.index for node in self.leafNodes])
        for leaf_node in self.leafNodes:
            if not (type(posteriors_dict[leaf_node.index]) is np.ndarray):
                assert np.array_equal(posteriors_tensor[:, :, leaf_node.index - min_leaf_index],
                                      posteriors_dict[leaf_node.index].numpy())
            else:
                assert np.array_equal(posteriors_tensor[:, :, leaf_node.index - min_leaf_index],
                                      posteriors_dict[leaf_node.index])

        # Calculate the optimal q-tables, starting from the final level.
        optimal_q_tables = {}
        for level in range(self.get_max_trajectory_length(), 0, -1):
            if level < self.get_max_trajectory_length():
                assert level + 1 in optimal_q_tables
            # Get all possible trajectories up to this level.
            actions_each_layer = [[0]]
            actions_each_layer.extend([[0, 1] for _ in range(level)])
            list_of_all_trajectories = Utilities.get_cartesian_product(list_of_lists=actions_each_layer)
            table_shape = action_spaces[level].shape[:-1]
            # The distribution of the dimensions:
            # First 0,1,...,(level - 1) dimensions: The trajectory UP TO THIS level.
            # This excludes the action to be taken in this level.
            # level.th dimension: The action to be taken in this level.
            # The last dimension: The batch size.
            optimal_q_table_for_level = np.zeros(shape=table_shape, dtype=np.float64)

            for trajectory in list_of_all_trajectories:
                if level == self.get_max_trajectory_length():
                    configurations_matrix_for_trajectory = action_spaces[level][tuple(trajectory)]
                    q_vec = self.calculate_reward_values_from_leaf_level_configurations(
                        routing_matrix=configurations_matrix_for_trajectory,
                        posteriors_tensor=posteriors_tensor,
                        true_labels=true_labels
                    )
                    optimal_q_table_for_level[tuple(trajectory)] = q_vec
                else:
                    # Bellman equation:
                    # Results if we take the action 0.
                    trajectory_a0 = []
                    trajectory_a0.extend(trajectory)
                    trajectory_a0.append(0)
                    q_star_a0 = optimal_q_tables[level + 1][tuple(trajectory_a0)]
                    # Results if we take the action 1.
                    trajectory_a1 = []
                    trajectory_a1.extend(trajectory)
                    trajectory_a1.append(1)
                    q_star_a1 = optimal_q_tables[level + 1][tuple(trajectory_a1)]
                    q_table = np.stack([q_star_a0, q_star_a1], axis=1)
                    q_max = np.max(q_table, axis=1)
                    # Rewards
                    if self.rewardType == "Zero Rewards":
                        rewards = np.zeros_like(q_max)
                    else:
                        raise NotImplementedError()
                    q_vec = rewards + q_max
                    optimal_q_table_for_level[tuple(trajectory)] = q_vec
            optimal_q_tables[level] = optimal_q_table_for_level
        return optimal_q_tables

    def calculate_optimal_q_tables_manual(self, true_labels, posteriors_dict, ig_activations_dict):
        action_spaces = self.calculate_sample_action_results_manual(ig_activations_dict=ig_activations_dict)
        optimal_q_tables = {}
        batch_size = true_labels.shape[0]
        for level in range(self.get_max_trajectory_length(), 0, -1):
            optimal_q_table = np.zeros(shape=action_spaces[level].shape[:-1], dtype=np.float64)
            actions_each_layer = [[0]]
            actions_each_layer.extend([[0, 1] for _ in range(level)])
            list_of_all_trajectories = Utilities.get_cartesian_product(list_of_lists=actions_each_layer)
            for trajectory in list_of_all_trajectories:
                if level == self.get_max_trajectory_length():
                    for sample_id in range(batch_size):
                        # Root node
                        active_nodes = [self.topologicalSortedNodes[0]]
                        for t in range(self.get_max_trajectory_length()):
                            next_level_nodes = []
                            action_t_plus_1 = trajectory[t + 1]
                            for active_node in active_nodes:
                                child_nodes = self.dagObject.children(node=active_node)
                                child_nodes_sorted = sorted(child_nodes, key=lambda nd: nd.index)
                                if action_t_plus_1 == 0:
                                    ig_ = ig_activations_dict[active_node.index][sample_id]
                                    selected_idx = np.argmax(ig_)
                                    next_level_nodes.append(child_nodes_sorted[selected_idx])
                                else:
                                    next_level_nodes.extend(child_nodes_sorted)
                            active_nodes = next_level_nodes
                        # Calculate the configuration cost
                        # 1-Classification result
                        final_probabilities = np.zeros_like(posteriors_dict[self.leafNodes[0].index][0])
                        for leaf_node in active_nodes:
                            final_probabilities += posteriors_dict[leaf_node.index][sample_id]
                        final_probabilities = final_probabilities * (1.0 / len(active_nodes))
                        y_predicted = np.argmax(final_probabilities)
                        y_truth = true_labels[sample_id]
                        if y_truth == y_predicted:
                            reward = self.validPredictionReward
                        else:
                            reward = self.invalidPredictionPenalty
                        # 2-MAC cost result
                        leaf_nodes_configuration = [0] * len(self.leafNodes)
                        min_leaf_index = min([nd.index for nd in self.leafNodes])
                        for leaf_node in active_nodes:
                            leaf_nodes_configuration[leaf_node.index - min_leaf_index] = 1
                        computation_overload = self.networkActivationCostsDict[tuple(leaf_nodes_configuration)]
                        reward -= self.lambdaMacCost * computation_overload
                        # Save the result
                        coords = []
                        coords.extend(trajectory)
                        coords.append(sample_id)
                        optimal_q_table[tuple(coords)] = reward
                else:
                    for sample_id in range(batch_size):
                        coord_0 = []
                        coord_0.extend(trajectory)
                        coord_0.append(0)
                        coord_0.append(sample_id)
                        q_0 = optimal_q_tables[level + 1][tuple(coord_0)]
                        coord_1 = []
                        coord_1.extend(trajectory)
                        coord_1.append(1)
                        coord_1.append(sample_id)
                        q_1 = optimal_q_tables[level + 1][tuple(coord_1)]
                        # Rewards
                        if self.rewardType == "Zero Rewards":
                            r_ = 0.0
                        else:
                            raise NotImplementedError()
                        q_final = r_ + max([q_0, q_1])
                        coords = [*trajectory, sample_id]
                        optimal_q_table[tuple(coords)] = q_final
            optimal_q_tables[level] = optimal_q_table
        return optimal_q_tables

    def calculate_q_tables_from_network_outputs(self, true_labels, model_outputs):
        batch_size = true_labels.shape[0]
        posteriors_dict = {k: v.numpy() for k, v in model_outputs["posteriors_dict"].items()}
        actions_predicted = model_outputs["actions_predicted"]
        ig_activations_dict = model_outputs["ig_activations_dict"]

        # Calculate the Q-values for every action
        optimal_q_values = self.calculate_optimal_q_tables(true_labels=true_labels,
                                                           posteriors_dict=posteriors_dict,
                                                           ig_activations_dict=ig_activations_dict)

        # According to the predicted actions, gather the optimal q-values
        regression_q_targets = []
        trajectories = [np.zeros_like(true_labels)]
        for level in range(self.get_max_trajectory_length()):
            coords_action_0 = tuple([*trajectories, np.zeros_like(true_labels), np.arange(batch_size)])
            q_values_action_0 = optimal_q_values[level + 1][coords_action_0]

            coords_action_1 = tuple([*trajectories, np.ones_like(true_labels), np.arange(batch_size)])
            q_values_action_1 = optimal_q_values[level + 1][coords_action_1]

            q_target = np.stack([q_values_action_0, q_values_action_1], axis=-1)
            regression_q_targets.append(q_target)
            # Update the trajectories with the predicted action value
            actions_predicted_this_level = np.array(actions_predicted[level])
            trajectories.append(actions_predicted_this_level)
        # Target q-values
        return regression_q_targets, optimal_q_values

    def calculate_q_tables_from_network_outputs_manual(self,
                                                       true_labels,
                                                       posteriors_dict,
                                                       ig_masks_dict,
                                                       **kwargs):
        batch_size = true_labels.shape[0]
        actions_predicted = kwargs["actions_predicted"]
        ig_activations_dict = kwargs["ig_activations_dict"]

        # Calculate the Q-values for every action
        optimal_q_values = self.calculate_optimal_q_tables_manual(true_labels=true_labels,
                                                                  posteriors_dict=posteriors_dict,
                                                                  ig_activations_dict=ig_activations_dict)
        regression_q_targets = [[] for _ in range(self.get_max_trajectory_length())]
        for sample_id in range(batch_size):
            trajectory = [0]
            for level in range(self.get_max_trajectory_length()):
                coords = []
                coords.extend(trajectory)

                action_0_trajectory = []
                action_0_trajectory.extend(coords)
                action_0_trajectory.append(0)
                action_0_trajectory.append(sample_id)
                q_s_a_0 = optimal_q_values[level + 1][tuple(action_0_trajectory)]

                action_1_trajectory = []
                action_1_trajectory.extend(coords)
                action_1_trajectory.append(1)
                action_1_trajectory.append(sample_id)
                q_s_a_1 = optimal_q_values[level + 1][tuple(action_1_trajectory)]

                regression_q_targets[level].append(np.array([q_s_a_0, q_s_a_1]))
                trajectory.append(actions_predicted[level][sample_id])

        for level in range(self.get_max_trajectory_length()):
            regression_q_targets[level] = np.stack(regression_q_targets[level], axis=0)
        return regression_q_targets

    def get_epoch_states(self, epoch_count, constraints):
        epoch_states = [dict() for _ in range(epoch_count)]
        # Warm up periods
        for epoch_id in range(epoch_count):
            epochs_from_q_training_start = epoch_id - constraints["q_net_train_start_epoch"]
            if epoch_id <= constraints["q_net_train_start_epoch"]:
                epoch_states[epoch_id]["is_in_warm_up_period"] = True
            else:
                epoch_states[epoch_id]["is_in_warm_up_period"] = False

            if epoch_id >= constraints["q_net_train_start_epoch"] and epochs_from_q_training_start % constraints[
                "q_net_train_period"] == 0:
                epoch_states[epoch_id]["do_train_q_nets"] = True
            else:
                epoch_states[epoch_id]["do_train_q_nets"] = False

            if epochs_from_q_training_start >= 0 and ((epoch_id >= epoch_count - 10) or (
                    epochs_from_q_training_start % self.trainEvalPeriod == 0) or
                                                      epoch_states[epoch_id]["do_train_q_nets"]):
                epoch_states[epoch_id]["do_measure_performance"] = True
            else:
                epoch_states[epoch_id]["do_measure_performance"] = False
        return epoch_states

    def run_main_model(self, X, y, iteration, is_in_warm_up_period, only_use_ig_routing):
        with tf.GradientTape() as main_tape:
            model_output = self.run_model(
                X=X,
                y=y,
                iteration=iteration,
                is_training=True,
                warm_up_period=is_in_warm_up_period,
                only_use_ig_routing=only_use_ig_routing)
            classification_losses = model_output["classification_losses"]
            info_gain_losses = model_output["info_gain_losses"]
            # Weight decaying; Q-Net variables excluded
            total_regularization_loss = self.get_regularization_loss(is_for_q_nets=False)
            # Classification losses
            classification_loss = tf.add_n([loss for loss in classification_losses.values()])
            # Information Gain losses
            info_gain_loss = self.decisionLossCoeff * tf.add_n([loss for loss in info_gain_losses.values()])
            # Total loss
            total_loss = total_regularization_loss + info_gain_loss + classification_loss
        # Calculate grads with respect to the main loss
        main_grads = main_tape.gradient(total_loss, self.model.trainable_variables)
        self.softmaxDecayController.update(iteration=iteration + 1)

        # for idx, v in enumerate(self.model.trainable_variables):
        #     if "q_net" not in v.name:
        #         continue
        #     grad_arr = main_grads[idx].numpy()
        #     assert np.array_equal(grad_arr, np.zeros_like(grad_arr))

        return model_output, main_grads, total_loss

    # OK
    def train_cign_body_one_epoch(self,
                                  dataset,
                                  run_id,
                                  iteration,
                                  is_in_warm_up_period,
                                  only_use_ig_routing):
        # self.build_trackers()
        # OK
        self.reset_trackers()
        times_list = []
        # Train for one loop the main CIGN
        for train_X, train_y in dataset:
            profiler = Profiler()
            # OK
            model_output, main_grads, total_loss = \
                self.run_main_model(X=train_X,
                                    y=train_y,
                                    iteration=iteration,
                                    is_in_warm_up_period=is_in_warm_up_period,
                                    only_use_ig_routing=only_use_ig_routing)
            # Check that Q-net variables do not receive gradients
            for grad, var in zip(main_grads, self.model.trainable_variables):
                if "q_net" in var.name:
                    np_grad = grad.numpy()
                    assert np.allclose(np.zeros_like(np_grad), np_grad)
            self.optimizer.apply_gradients(zip(main_grads, self.model.trainable_variables))
            profiler.add_measurement("One Cign Training Iteration")
            # Track losses
            self.track_losses(total_loss=total_loss,
                              classification_losses=model_output["classification_losses"],
                              info_gain_losses=model_output["info_gain_losses"])

            self.print_train_step_info(
                iteration=iteration,
                classification_losses=model_output["classification_losses"],
                info_gain_losses=model_output["info_gain_losses"],
                time_intervals=profiler.get_all_measurements(),
                eval_dict=model_output["eval_dict"])
            times_list.append(sum(profiler.get_all_measurements().values()))
            iteration += 1
            # Print outputs

        self.save_log_data(run_id=run_id,
                           iteration=iteration,
                           eval_dict=model_output["eval_dict"])

        return iteration, times_list

    def run_model(self, **kwargs):
        X = kwargs["X"]
        y = kwargs["y"]
        iteration = kwargs["iteration"]
        is_training = kwargs["is_training"]
        if "warm_up_period" in kwargs:
            warm_up_period = kwargs["warm_up_period"]
        else:
            warm_up_period = False
        feed_dict = self.get_feed_dict(x=X, y=y, iteration=iteration, is_training=is_training,
                                       warm_up_period=warm_up_period)
        for lid in range(self.get_max_trajectory_length()):
            input_name = "enforced_actions_{0}".format(lid)
            if input_name not in kwargs:
                batch_size = X.shape[0]
                feed_dict[input_name] = -1 * tf.ones(shape=(batch_size,))
            else:
                feed_dict[input_name] = kwargs[input_name]

        if "only_use_ig_routing" not in kwargs:
            feed_dict["only_use_ig_routing"] = False
        else:
            feed_dict["only_use_ig_routing"] = kwargs["only_use_ig_routing"]

        # feed_dict["only_use_ig_routing"] = False
        # feed_dict["warm_up_period"] = False

        model_output_arr = self.model(inputs=feed_dict, training=is_training)
        model_output_dict = self.convert_model_outputs_to_dict(model_output_arr=model_output_arr)
        return model_output_dict

    def prepare_anomaly_dataset(self, dataset_tf):
        ig_activations = {}
        inputs_per_tree_level = {}
        targets_per_tree_level = {}
        action_combinations = []

        for _ in range(self.get_max_trajectory_length()):
            action_combinations.append([0, 1])

        action_combinations = Utilities.get_cartesian_product(list_of_lists=action_combinations)

        # Build the training inputs: Intermediate features which are fed to the Q-Nets at every tree layer,
        # for every action trajectory.
        # Build the training targets: Optimal Q-tables for every sample, under every possible trajectory.
        for X, y in dataset_tf:
            # input_source_dict = {}
            # target_source_dict = {}
            for trajectory_nd in action_combinations:
                trajectory_tpl = tuple(trajectory_nd)
                # Model inputs
                kwargs = {"X": X, "y": y, "iteration": 0,
                          "is_training": False, "warm_up_period": False, "only_use_ig_routing": False}

                # Predetermined actions as input
                for lid in range(self.get_max_trajectory_length()):
                    input_name = "enforced_actions_{0}".format(lid)
                    kwargs[input_name] = trajectory_tpl[lid] * np.ones(shape=(self.batchSizeNonTensor,), dtype=np.int64)

                # Run the model
                model_output_dict = self.run_model(**kwargs)

                # Gather Q-Net inputs
                for lid in range(self.get_max_trajectory_length()):
                    idx = (lid, trajectory_tpl)
                    if idx not in inputs_per_tree_level:
                        inputs_per_tree_level[idx] = []
                    inputs_per_tree_level[idx].append(model_output_dict["q_net_inputs"][lid])

                # Gather IG activations
                for node_id in model_output_dict["ig_activations_dict"].keys():
                    idx = (node_id, trajectory_tpl)
                    if idx not in ig_activations:
                        ig_activations[idx] = []
                    ig_activations[idx].append(model_output_dict["ig_activations_dict"][node_id])

                # Gather Q-Net regression targets
                regression_q_targets, optimal_q_values = self.calculate_q_tables_from_network_outputs(
                    true_labels=y.numpy(),
                    model_outputs=model_output_dict)
                for lid in range(self.get_max_trajectory_length()):
                    idx = (lid, trajectory_tpl)
                    if idx not in targets_per_tree_level:
                        targets_per_tree_level[idx] = []
                    targets_per_tree_level[idx].append(regression_q_targets[lid])

        # Merge all collected data
        for k in ig_activations:
            ig_activations[k] = np.concatenate(ig_activations[k], axis=0)

        for k in inputs_per_tree_level:
            inputs_per_tree_level[k] = np.concatenate(inputs_per_tree_level[k], axis=0)

        for k in targets_per_tree_level:
            targets_per_tree_level[k] = np.concatenate(targets_per_tree_level[k], axis=0)

        # Ensure the integrity and correctness of the collected data.
        # Rule - 1: All ig activations should be equal, independent of the followed trajectory.
        for node in self.innerNodes:
            ig_activation_arrs = [arr for idx, arr in ig_activations.items() if idx[0] == node.index]
            res = []
            for idx in range(len(ig_activation_arrs) - 1):
                res.append(np.allclose(ig_activation_arrs[idx], ig_activation_arrs[idx + 1]))
            assert all(res)

        # Rule - 2: When the previous trajectories are equal, all input arrays must be equal.
        inputs = {}
        for lid in range(self.get_max_trajectory_length()):
            for trajectory_nd in action_combinations:
                trajectory_tpl = tuple(trajectory_nd)
                input_arrays_to_be_equal = []
                for idx, arr in inputs_per_tree_level.items():
                    if idx[0] == lid and idx[1][:lid] == trajectory_tpl[:lid]:
                        input_arrays_to_be_equal.append(arr)
                res = []
                for idx in range(len(input_arrays_to_be_equal) - 1):
                    res.append(np.allclose(input_arrays_to_be_equal[idx], input_arrays_to_be_equal[idx + 1]))
                assert all(res)
                final_arr = input_arrays_to_be_equal[0]
                k_ = (lid, trajectory_tpl[:lid])
                if k_ in inputs:
                    assert np.allclose(inputs[k_], final_arr)
                else:
                    inputs[k_] = final_arr

        # Rule - 3: When the previous trajectories are equal, all regression targets (output arrays) must be equal.
        outputs = {}
        for lid in range(self.get_max_trajectory_length()):
            for trajectory_nd in action_combinations:
                trajectory_tpl = tuple(trajectory_nd)
                output_arrays_to_be_equal = []
                for idx, arr in targets_per_tree_level.items():
                    if idx[0] == lid and idx[1][:lid] == trajectory_tpl[:lid]:
                        output_arrays_to_be_equal.append(arr)
                res = []
                for idx in range(len(output_arrays_to_be_equal) - 1):
                    res.append(np.allclose(output_arrays_to_be_equal[idx], output_arrays_to_be_equal[idx + 1]))
                assert all(res)
                final_arr = output_arrays_to_be_equal[0]
                k_ = (lid, trajectory_tpl[:lid])
                if k_ in outputs:
                    assert np.allclose(outputs[k_], final_arr)
                else:
                    outputs[k_] = final_arr

        # Rule - 4: The Q-Net inputs must reflect the information gain activations from the previous layer, and the
        # action taken there.
        ig_activations_temp = {}
        sample_count = set()
        for idx, arr in ig_activations.items():
            if idx[0] not in ig_activations_temp:
                ig_activations_temp[idx[0]] = arr
            sample_count.add(arr.shape[0])
        ig_activations = ig_activations_temp
        assert len(sample_count) == 1
        sample_count = list(sample_count)[0]

        for sample_id in range(sample_count):
            for trajectory_nd in action_combinations:
                trajectory_tpl = tuple(trajectory_nd)
                route_signal = [1]
                for lid in range(self.get_max_trajectory_length()):
                    assert len(route_signal) == len(self.orderedNodesPerLevel[lid])
                    feature_array = inputs_per_tree_level[(lid, trajectory_tpl)][sample_id]
                    feature_width = feature_array.shape[-1]
                    route_width = feature_width // len(route_signal)
                    curr_index = 0
                    for path_id in range(len(route_signal)):
                        start_index = curr_index
                        if path_id < len(route_signal) - 1:
                            end_index = start_index + route_width
                        else:
                            end_index = feature_width
                        route_feature = feature_array[..., start_index:end_index]
                        feat_sum = np.sum(route_feature)
                        is_route_open = route_signal[path_id]
                        if is_route_open:
                            assert feat_sum != 0
                        else:
                            assert feat_sum == 0
                        curr_index = end_index
                    # Prepare routing signal for the next layer
                    action_for_this_level = trajectory_tpl[lid]
                    next_level_route_signal = []
                    for node_layer_id, node in enumerate(self.orderedNodesPerLevel[lid]):
                        if route_signal[node_layer_id] == 0:
                            next_level_route_signal.extend([0] * len(self.dagObject.children(node=node)))
                        else:
                            if action_for_this_level == 0:
                                ig_activation = ig_activations[node.index][sample_id]
                                max_id = np.argmax(ig_activation)
                                for activation_id in range(len(ig_activation)):
                                    next_level_route_signal.append(int(activation_id == max_id))
                            else:
                                next_level_route_signal.extend([1] * len(self.dagObject.children(node=node)))
                    route_signal = next_level_route_signal

        X_per_levels = {}
        y_per_levels = {}
        for lid in range(self.get_max_trajectory_length()):
            correct_array_keys = [tpl for tpl in inputs.keys() if tpl[0] == lid]
            X = [inputs[tpl] for tpl in correct_array_keys]
            X = np.concatenate(X, axis=0)
            y = [outputs[tpl] for tpl in correct_array_keys]
            y = np.concatenate(y, axis=0)
            y = np.argmax(y, axis=-1)
            X_per_levels[lid] = X
            y_per_levels[lid] = y
            histogram = Counter(y_per_levels[lid])
            print("Histogram:{0}".format(histogram))
        return X_per_levels, y_per_levels, ig_activations

    def get_autoencoder(self, q_net_of_level):
        q_net_input_shape = q_net_of_level.layers[0].input_shape[0]
        # input_x = tf.keras.Input(shape=q_net_input_shape[1:], dtype=q_net_of_level.layers[0].dtype)
        # net = input_x
        if q_net_input_shape[1] % 2 == 1:
            input_shape = (q_net_input_shape[1] + 1, q_net_input_shape[2] + 1, q_net_input_shape[3])
        else:
            input_shape = q_net_input_shape[1:]

        # Train an autoencoder
        input_x = tf.keras.Input(shape=input_shape, dtype=q_net_of_level.layers[0].dtype)
        net = input_x
        # if use_bilinear:
        #     net = tf.image.resize(input_x, size=(input_shape[1], input_shape[1]), method="bilinear")
        # else:
        #     net = input_x
        conv_layer = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=3,
                                            strides=2,
                                            padding="same",
                                            use_bias=False,
                                            name="conv1")
        net = conv_layer(net)
        last_conv_layer_shape = net.shape
        batch_norm = tf.keras.layers.BatchNormalization()
        net = batch_norm(net)
        leaky_relu_conv = tf.keras.layers.LeakyReLU(alpha=0.1)
        net = leaky_relu_conv(net)

        flatten = tf.keras.layers.Flatten()
        net = flatten(net)
        flat_dim = net.shape

        dense1 = tf.keras.layers.Dense(units=128)
        net = dense1(net)
        leaky_relu1 = tf.keras.layers.LeakyReLU(alpha=0.1)
        net = leaky_relu1(net)

        dense2 = tf.keras.layers.Dense(units=64)
        net = dense2(net)

        dense3 = tf.keras.layers.Dense(units=128)
        net = dense3(net)
        leaky_relu3 = tf.keras.layers.LeakyReLU(alpha=0.1)
        net = leaky_relu3(net)

        dense4 = tf.keras.layers.Dense(units=flat_dim[1])
        net = dense4(net)

        reshape_to_conv = tf.keras.layers.Reshape(target_shape=last_conv_layer_shape[1:])
        net = reshape_to_conv(net)

        transpose_conv_layer = tf.keras.layers.Conv2DTranspose(input_x.shape[-1],
                                                               kernel_size=3,
                                                               strides=2,
                                                               padding="same",
                                                               name="trans_conv1",
                                                               use_bias=False)
        net = transpose_conv_layer(net)
        batch_norm = tf.keras.layers.BatchNormalization()
        net = batch_norm(net)
        leaky_relu_trans_conv = tf.keras.layers.LeakyReLU(alpha=0.1)
        net = leaky_relu_trans_conv(net)

        # Last linear layer
        linear_1x1_conv = tf.keras.layers.Conv2D(input_x.shape[-1],
                                                 kernel_size=1,
                                                 strides=1,
                                                 padding="same",
                                                 use_bias=False,
                                                 name="linear_1x1_conv")
        net = linear_1x1_conv(net)
        output_x = net
        autoencoder_model = tf.keras.Model(inputs=input_x, outputs=output_x)
        return autoencoder_model

    def get_anomaly_detection_statistics(self, autoencoder, normal_data, anomaly_data):
        # Calculate distributions of the normal and anomaly data
        normal_distances = []
        anomaly_distances = []
        for X_batch in normal_data:
            if X_batch.shape[1] % 2 == 1:
                X_batch = tf.image.resize(X_batch, size=(X_batch.shape[1] + 1,
                                                         X_batch.shape[2] + 1), method="bilinear")
            X_batch_hat = autoencoder(inputs=X_batch, training=False)
            d_ = X_batch_hat - X_batch
            d_ = tf.pow(d_, 2.0)
            error_vector = np.mean(d_, axis=(1, 2, 3))
            normal_distances.append(error_vector)
        normal_distances = np.concatenate(normal_distances)

        for X_batch in anomaly_data:
            if X_batch.shape[1] % 2 == 1:
                X_batch = tf.image.resize(X_batch, size=(X_batch.shape[1] + 1,
                                                         X_batch.shape[2] + 1), method="bilinear")
            X_batch_hat = autoencoder(inputs=X_batch, training=False)
            d_ = X_batch_hat - X_batch
            d_ = tf.pow(d_, 2.0)
            error_vector = np.mean(d_, axis=(1, 2, 3))
            anomaly_distances.append(error_vector)
        anomaly_distances = np.concatenate(anomaly_distances)

        normal_mean = np.mean(normal_distances)
        normal_std = np.std(normal_distances)
        anomaly_mean = np.mean(anomaly_distances)
        anomaly_std = np.std(anomaly_distances)
        return normal_mean, normal_std, anomaly_mean, anomaly_std

    def eval_dataset_with_anomaly_detector(self, autoencoder, normal_data, anomaly_data,
                                           normal_mean, normal_std, anomaly_mean, anomaly_std):
        errors_normal_data = []
        errors_anomaly_data = []
        for X_batch in normal_data:
            if X_batch.shape[1] % 2 == 1:
                X_batch = tf.image.resize(X_batch, size=(X_batch.shape[1] + 1,
                                                         X_batch.shape[2] + 1), method="bilinear")
            x_hat = autoencoder(inputs=X_batch, training=False)
            d_ = x_hat - X_batch
            d_ = tf.pow(d_, 2.0)
            error_vector = np.mean(d_, axis=(1, 2, 3))
            errors_normal_data.append(error_vector)

        for X_batch in anomaly_data:
            if X_batch.shape[1] % 2 == 1:
                X_batch = tf.image.resize(X_batch, size=(X_batch.shape[1] + 1,
                                                         X_batch.shape[2] + 1), method="bilinear")
            x_hat = autoencoder(inputs=X_batch, training=False)
            d_ = x_hat - X_batch
            d_ = tf.pow(d_, 2.0)
            error_vector = np.mean(d_, axis=(1, 2, 3))
            errors_anomaly_data.append(error_vector)

        errors_normal_data = np.concatenate(errors_normal_data)
        errors_anomaly_data = np.concatenate(errors_anomaly_data)
        distances = np.concatenate([errors_normal_data, errors_anomaly_data])
        y_ = np.concatenate(
            [np.zeros(shape=(errors_normal_data.shape[0],), dtype=np.int32),
             np.ones(shape=(errors_anomaly_data.shape[0],), dtype=np.int32)])

        norm_p = norm.pdf(distances, loc=normal_mean, scale=normal_std)
        anorm_p = norm.pdf(distances, loc=anomaly_mean, scale=anomaly_std)
        scores = np.stack([norm_p, anorm_p], axis=-1)
        y_hat = np.argmax(scores, axis=-1)
        report = classification_report(y_, y_hat)
        print(report)
        fpr, tpr, thresholds = metrics.roc_curve(y_, y_hat, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("auc={0}".format(auc))
        return report, auc

    def save_autoencoder(self, autoencoder, run_id, level_id, epoch_id):
        root_path = os.path.dirname(__file__)
        model_path = os.path.join(root_path,
                                  "..", "saved_models", "model_{0}_autoencoder_level_{1}_epoch_{2}".format(run_id,
                                                                                                           level_id,
                                                                                                           epoch_id))
        os.mkdir(model_path)
        autoencoder_model_path = os.path.join(model_path, "model")
        os.mkdir(autoencoder_model_path)
        autoencoder.save_weights(autoencoder_model_path)

    def load_autoencoder(self, autoencoder, run_id, level_id, epoch_id):
        try:
            root_path = os.path.dirname(__file__)
            model_path = os.path.join(root_path,
                                      "..", "saved_models",
                                      "model_{0}_autoencoder_level_{1}_epoch_{2}".format(run_id, level_id, epoch_id))
            autoencoder_model_path = os.path.join(model_path, "model")
            autoencoder.load_weights(autoencoder_model_path)
            return True
        except:
            return False

    def train_q_nets_as_anomaly_detectors(self, run_id, dataset, q_net_epoch_count):
        run_id = DbLogger.get_run_id()
        self.build_trackers()
        self.reset_trackers()
        train_tf = tf.data.Dataset.from_tensor_slices((dataset.valX, dataset.valY)).batch(self.batchSizeNonTensor)
        test_tf = tf.data.Dataset.from_tensor_slices((dataset.testX, dataset.testY)).batch(self.batchSizeNonTensor)

        train_x, train_y, train_ig = self.prepare_anomaly_dataset(dataset_tf=train_tf)
        test_x, test_y, test_ig = self.prepare_anomaly_dataset(dataset_tf=test_tf)

        action_combinations = []
        for _ in range(self.get_max_trajectory_length()):
            action_combinations.append([0, 1])

        for lid in range(self.get_max_trajectory_length()):
            if lid == 0:
                continue
            boundaries = [2000, 5000, 9000]
            values = [0.1, 0.01, 0.001, 0.0001]
            self.globalStep.assign(value=0)
            q_net_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

            # Prepare data
            train_x_normal = train_x[lid][train_y[lid] == 0]
            train_x_anomaly = train_x[lid][train_y[lid] == 1]
            test_x_normal = test_x[lid][test_y[lid] == 0]
            test_x_anomaly = test_x[lid][test_y[lid] == 1]
            normal_train_data = tf.data.Dataset.from_tensor_slices(train_x_normal).shuffle(5000).batch(
                self.batchSizeNonTensor)
            anomaly_train_data = tf.data.Dataset.from_tensor_slices(train_x_anomaly).batch(
                self.batchSizeNonTensor)
            normal_test_data = tf.data.Dataset.from_tensor_slices(test_x_normal).batch(self.batchSizeNonTensor)
            anomaly_test_data = tf.data.Dataset.from_tensor_slices(test_x_anomaly).batch(self.batchSizeNonTensor)

            # Get autoencoder
            autoencoder = self.get_autoencoder(q_net_of_level=self.qNets[lid])
            iteration = 0
            # Train the autoencoder with only normal data
            # ***************** COMMENT THAT PART OUT, LOAD SAVED MODEL *****************
            # Train the q-net model
            iteration = 0
            for epoch_id in range(q_net_epoch_count):
                self.qNetTrackers[lid].reset_states()
                for X_batch in normal_train_data:
                    with tf.GradientTape() as q_tape:
                        if X_batch.shape[1] % 2 == 1:
                            X_batch = tf.image.resize(X_batch, size=(X_batch.shape[1] + 1,
                                                                     X_batch.shape[2] + 1), method="bilinear")
                        X_batch_hat = autoencoder(inputs=X_batch, training=True)
                        mse = self.mseLoss(X_batch, X_batch_hat)
                        # reg_losses = []
                        # for var in q_net.trainable_variables:
                        #     assert var.ref() in self.regularizationCoefficients
                        #     reg_coeff = self.regularizationCoefficients[var.ref()]
                        #     reg_losses.append(reg_coeff * tf.nn.l2_loss(var))
                        # regularization_loss = tf.add_n(reg_losses)
                        # total_loss = mse + regularization_loss
                        total_loss = mse
                    self.qNetTrackers[lid].update_state(mse)
                    q_grads = q_tape.gradient(total_loss, autoencoder.trainable_variables)
                    q_net_optimizer.apply_gradients(zip(q_grads, autoencoder.trainable_variables))
                    iteration += 1
                    print("Level:{0} Iteration:{1} Epoch:{2} MSE:{3}".format(lid,
                                                                             iteration,
                                                                             epoch_id,
                                                                             self.qNetTrackers[
                                                                                 lid].result().numpy()))
                    print("Lr:{0}".format(q_net_optimizer._decayed_lr(tf.float32).numpy()))

                # Get anomaly detector statistics
                normal_mean, normal_std, anomaly_mean, anomaly_std = self.get_anomaly_detection_statistics(
                    autoencoder=autoencoder, normal_data=normal_train_data, anomaly_data=anomaly_train_data)
                print("************Training************")
                train_report, train_auc = self.eval_dataset_with_anomaly_detector(autoencoder=autoencoder,
                                                                                  normal_data=normal_train_data,
                                                                                  anomaly_data=anomaly_train_data,
                                                                                  normal_mean=normal_mean,
                                                                                  normal_std=normal_std,
                                                                                  anomaly_mean=anomaly_mean,
                                                                                  anomaly_std=anomaly_std)
                print("************Test************")
                test_report, test_auc = self.eval_dataset_with_anomaly_detector(autoencoder=autoencoder,
                                                                                normal_data=normal_test_data,
                                                                                anomaly_data=anomaly_test_data,
                                                                                normal_mean=normal_mean,
                                                                                normal_std=normal_std,
                                                                                anomaly_mean=anomaly_mean,
                                                                                anomaly_std=anomaly_std)
                DbLogger.write_into_table(rows=[(run_id, epoch_id, lid, "training", train_report, train_auc),
                                                (run_id, epoch_id, lid, "test", test_report, test_auc)],
                                          table="q_net_anomaly_logs")
                # Save trained autoencoder
                self.save_autoencoder(autoencoder=autoencoder, run_id=run_id, level_id=lid, epoch_id=epoch_id)

            # ***************** COMMENT THAT PART OUT, LOAD SAVE MODEL *****************

            #
            # # Calculate results of anomaly model.
            # for normal_data, anomaly_data, dataset_name in (
            #         [(normal_train_data, anomaly_train_data, "train"),
            #          (normal_test_data, anomaly_test_data, "test")]):
            #     print("*****************{0}*****************".format(dataset_name))
            #     errors_normal_data = []
            #     errors_anomaly_data = []
            #     for X_batch in normal_data:
            #         # x = tf.image.resize(x, size=(32, 32), method="bilinear")
            #         if X_batch.shape[1] % 2 == 1:
            #             X_batch = tf.image.resize(X_batch, size=(X_batch.shape[1] + 1,
            #                                                      X_batch.shape[2] + 1), method="bilinear")
            #         x_hat = autoencoder(inputs=X_batch, training=False)
            #         d_ = x_hat - X_batch
            #         d_ = tf.pow(d_, 2.0)
            #         error_vector = np.mean(d_, axis=(1, 2, 3))
            #         errors_normal_data.append(error_vector)
            #
            #     for X_batch in anomaly_data:
            #         # x = tf.image.resize(x, size=(32, 32), method="bilinear")
            #         if X_batch.shape[1] % 2 == 1:
            #             X_batch = tf.image.resize(X_batch, size=(X_batch.shape[1] + 1,
            #                                                      X_batch.shape[2] + 1), method="bilinear")
            #         x_hat = autoencoder(inputs=X_batch, training=False)
            #         d_ = x_hat - X_batch
            #         d_ = tf.pow(d_, 2.0)
            #         error_vector = np.mean(d_, axis=(1, 2, 3))
            #         errors_anomaly_data.append(error_vector)
            #
            #     errors_normal_data = np.concatenate(errors_normal_data)
            #     errors_anomaly_data = np.concatenate(errors_anomaly_data)
            #
            #     distances = np.concatenate([errors_normal_data, errors_anomaly_data])
            #     y_ = np.concatenate(
            #         [np.zeros(shape=(errors_normal_data.shape[0],), dtype=np.int32),
            #          np.ones(shape=(errors_anomaly_data.shape[0],), dtype=np.int32)])
            #
            #     norm_p = norm.pdf(distances, loc=normal_mean, scale=normal_std)
            #     anorm_p = norm.pdf(distances, loc=anomaly_mean, scale=anomaly_std)
            #     scores = np.stack([norm_p, anorm_p], axis=-1)
            #     y_hat = np.argmax(scores, axis=-1)
            #     report = classification_report(y_, y_hat)
            #     print(report)
            #     fpr, tpr, thresholds = metrics.roc_curve(y_, y_hat, pos_label=1)
            #     auc = metrics.auc(fpr, tpr)
            #     print("auc={0}".format(auc))

    def load_autoencoders(self, dataset, run_id_list, epoch_list, test_models=False):
        assert self.get_max_trajectory_length() == len(run_id_list)
        assert self.get_max_trajectory_length() == len(epoch_list)
        train_tf = tf.data.Dataset.from_tensor_slices((dataset.valX, dataset.valY)).batch(self.batchSizeNonTensor)
        test_tf = tf.data.Dataset.from_tensor_slices((dataset.testX, dataset.testY)).batch(self.batchSizeNonTensor)

        train_x, train_y, train_ig = self.prepare_anomaly_dataset(dataset_tf=train_tf)
        test_x, test_y, test_ig = self.prepare_anomaly_dataset(dataset_tf=test_tf)

        autoencoders = []
        for lid in range(self.get_max_trajectory_length()):
            train_x_normal = train_x[lid][train_y[lid] == 0]
            train_x_anomaly = train_x[lid][train_y[lid] == 1]
            test_x_normal = test_x[lid][test_y[lid] == 0]
            test_x_anomaly = test_x[lid][test_y[lid] == 1]
            normal_train_data = tf.data.Dataset.from_tensor_slices(train_x_normal).shuffle(5000).batch(
                self.batchSizeNonTensor)
            anomaly_train_data = tf.data.Dataset.from_tensor_slices(train_x_anomaly).batch(
                self.batchSizeNonTensor)
            normal_test_data = tf.data.Dataset.from_tensor_slices(test_x_normal).batch(self.batchSizeNonTensor)
            anomaly_test_data = tf.data.Dataset.from_tensor_slices(test_x_anomaly).batch(self.batchSizeNonTensor)

            autoencoder = self.get_autoencoder(q_net_of_level=self.qNets[lid])
            self.load_autoencoder(autoencoder=autoencoder, run_id=run_id_list[lid],
                                  epoch_id=epoch_list[lid], level_id=lid)

            if test_models:
                # Get anomaly detector statistics
                normal_mean, normal_std, anomaly_mean, anomaly_std = self.get_anomaly_detection_statistics(
                    autoencoder=autoencoder, normal_data=normal_train_data, anomaly_data=anomaly_train_data)
                print("normal_mean={0}".format(normal_mean))
                print("normal_std={0}".format(normal_std))
                print("anomaly_mean={0}".format(anomaly_mean))
                print("anomaly_std={0}".format(anomaly_std))
                print("************Training************")
                train_report, train_auc = self.eval_dataset_with_anomaly_detector(autoencoder=autoencoder,
                                                                                  normal_data=normal_train_data,
                                                                                  anomaly_data=anomaly_train_data,
                                                                                  normal_mean=normal_mean,
                                                                                  normal_std=normal_std,
                                                                                  anomaly_mean=anomaly_mean,
                                                                                  anomaly_std=anomaly_std)
                print(train_report)
                print("Auc:{0}".format(train_auc))

                print("************Test************")
                test_report, test_auc = self.eval_dataset_with_anomaly_detector(autoencoder=autoencoder,
                                                                                normal_data=normal_test_data,
                                                                                anomaly_data=anomaly_test_data,
                                                                                normal_mean=normal_mean,
                                                                                normal_std=normal_std,
                                                                                anomaly_mean=anomaly_mean,
                                                                                anomaly_std=anomaly_std)
                print(test_report)
                print("Auc:{0}".format(test_auc))
            autoencoders.append(autoencoder)
        return autoencoders

    def calculate_accuracy_with_anomaly_detectors(self, dataset, anomaly_detectors, selection_thresholds):
        posteriors_dict = {}
        ig_activations_dict = {}
        true_labels = []
        q_net_inputs_per_tree_level_dict = {}
        anomaly_scores = {}
        action_configurations, actions_dict = self.get_action_configurations()

        for X, y in dataset:
            kwargs = {"X": X, "y": y, "iteration": 0,
                      "is_training": False, "warm_up_period": False, "only_use_ig_routing": False}
            # Run the model
            model_output_dict = self.run_model(**kwargs)
            # Obtain Q-Net inputs
            for lid in range(self.get_max_trajectory_length()):
                if lid not in q_net_inputs_per_tree_level_dict:
                    q_net_inputs_per_tree_level_dict[lid] = []
                q_net_inputs_per_tree_level_dict[lid].append(model_output_dict["q_net_inputs"][lid])
            # Obtain IG activations
            for node_id in model_output_dict["ig_activations_dict"].keys():
                if node_id not in ig_activations_dict:
                    ig_activations_dict[node_id] = []
                ig_activations_dict[node_id].append(model_output_dict["ig_activations_dict"][node_id])
            # Obtain true labels
            true_labels.append(y)
            # Obtain posteriors
            for node in self.leafNodes:
                if node.index not in posteriors_dict:
                    posteriors_dict[node.index] = []
                posteriors_dict[node.index].append(model_output_dict["posteriors_dict"][node.index])
            # Run autoencoders
            for lid in range(self.get_max_trajectory_length()):
                X_q = model_output_dict["q_net_inputs"][lid]
                if lid not in anomaly_scores:
                    anomaly_scores[lid] = []
                if X_q.shape[1] % 2 == 1:
                    X_q = tf.image.resize(X_q, size=(X_q.shape[1] + 1, X_q.shape[2] + 1), method="bilinear")
                X_hat = anomaly_detectors[lid](inputs=X_q, training=False)
                d_ = X_hat - X_q
                d_ = tf.pow(d_, 2.0)
                error_vector = np.mean(d_, axis=(1, 2, 3))
                anomaly_scores[lid].append(error_vector)

        # Merge all outputs
        # IG activations
        for k in ig_activations_dict.keys():
            ig_activations_dict[k] = np.concatenate(ig_activations_dict[k], axis=0)
        # Posterior probabilities
        for k in posteriors_dict.keys():
            posteriors_dict[k] = np.concatenate(posteriors_dict[k], axis=0)
        # True labels
        true_labels = np.concatenate(true_labels, axis=0)
        # Anomaly Scores for every CIGN level
        for lid in range(self.get_max_trajectory_length()):
            anomaly_scores[lid] = np.concatenate(anomaly_scores[lid], axis=0)

        posteriors_tensor = np.stack([posteriors_dict[node.index] for node in self.leafNodes], axis=-1)
        routing_configurations = self.calculate_binary_rl_configurations(ig_activations_dict=ig_activations_dict)
        # Determine actions with respect to anomaly thresholds.
        actions_taken = []
        for lid in range(self.get_max_trajectory_length()):
            actions = tf.cast(anomaly_scores[lid] >= selection_thresholds[lid], dtype=tf.int64)
            actions_taken.append(actions)
        action_trajectories = np.stack(actions_taken, axis=-1)
        action_id_list = np.apply_along_axis(lambda row: actions_dict[tuple(row)], axis=1, arr=action_trajectories)
        routing_matrix = routing_configurations[np.arange(start=0, stop=action_id_list.shape[0]), action_id_list, :]
        routing_accuracy, validity_vec = self.evaluate_routing_configuration(
            posteriors_tensor=posteriors_tensor,
            routing_matrix=routing_matrix,
            true_labels=true_labels)
        print("routing_accuracy={0}".format(routing_accuracy))

    def train_q_nets_separately(self, run_id, dataset, q_net_epoch_count):
        self.build_trackers()
        self.reset_trackers()
        train_tf = tf.data.Dataset.from_tensor_slices((dataset.valX, dataset.valY)).batch(self.batchSizeNonTensor)
        test_tf = tf.data.Dataset.from_tensor_slices((dataset.testX, dataset.testY)).batch(self.batchSizeNonTensor)

        train_x, train_y, train_ig = self.prepare_anomaly_dataset(dataset_tf=train_tf)
        test_x, test_y, test_ig = self.prepare_anomaly_dataset(dataset_tf=test_tf)

        sample_count = dataset.valX.shape[0]
        action_combinations = []

        ig_activations = {}
        inputs_per_tree_level = {}
        targets_per_tree_level = {}
        for _ in range(self.get_max_trajectory_length()):
            action_combinations.append([0, 1])

        action_combinations = Utilities.get_cartesian_product(list_of_lists=action_combinations)

        def augment_training_image_fn(image, labels):
            image = tf.image.random_flip_left_right(image)
            return image, labels

            # for lid in range(self.get_max_trajectory_length()):
            #     if lid == 0:
            #         continue
            #     route_signal = []
            #     for node in self.orderedNodesPerLevel[lid - 1]:
            #         ig_activation = ig_activations[node.inde][sample_id]
            #         max_id = np.argmax(ig_activation)
            #         for activation_id in range(len(ig_activation)):
            #             route_signal.append(int(activation_id == max_id))

        # Training procedure
        # q_net_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        # q_net_optimizer = self.get_sgd_optimizer()
        # Kod adı: TONPİLOY

        X_per_levels = {}
        y_per_levels = {}
        for lid in range(self.get_max_trajectory_length()):
            boundaries = [2000, 5000, 9000]
            values = [0.1, 0.01, 0.001, 0.0001]
            self.globalStep.assign(value=0)

            q_net_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            # learning_rate_scheduler_tf = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            #     boundaries=boundaries, values=values)
            # q_net_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_scheduler_tf, momentum=0.9)

            correct_array_keys = [tpl for tpl in inputs.keys() if tpl[0] == lid]
            X = [inputs[tpl] for tpl in correct_array_keys]
            X = np.concatenate(X, axis=0)
            y = [outputs[tpl] for tpl in correct_array_keys]
            y = np.concatenate(y, axis=0)

            X_per_levels[lid] = X
            y_per_levels[lid] = y
            test_ratio = 0.1
            assert X.shape[0] == y.shape[0]
            assert X.shape[0] % sample_count == 0

            # train_indices, test_indices = train_test_split(np.arange(sample_count), test_size=test_ratio)
            # train_indices_multiplied = []
            # multiplier = X.shape[0] // sample_count
            # for idx in train_indices:
            #     for coefficient in range(multiplier):
            #         train_indices_multiplied.append(idx + coefficient * sample_count)
            #
            # test_indices_multiplied = []
            # for idx in test_indices:
            #     for coefficient in range(multiplier):
            #         test_indices_multiplied.append(idx + coefficient * sample_count)

            X_train = X[train_indices_multiplied]
            X_test = X[test_indices_multiplied]
            y_train = y[train_indices_multiplied]
            y_test = y[test_indices_multiplied]

            y_train_labels = np.argmax(y_train, axis=-1)
            y_test_labels = np.argmax(y_test, axis=-1)

            X_train_normal = X_train[y_train_labels == 0]
            X_train_anomaly = X_train[y_train_labels == 1]
            X_test_normal = X_test[y_test_labels == 0]
            X_test_anomaly = X_test[y_test_labels == 1]

            normal_train_data = tf.data.Dataset.from_tensor_slices(X_train_normal).shuffle(5000).batch(
                self.batchSizeNonTensor)
            anomaly_train_data = tf.data.Dataset.from_tensor_slices(X_train_anomaly).shuffle(5000).batch(
                self.batchSizeNonTensor)
            normal_test_data = tf.data.Dataset.from_tensor_slices(X_test_normal).batch(self.batchSizeNonTensor)
            anomaly_test_data = tf.data.Dataset.from_tensor_slices(X_test_anomaly).batch(self.batchSizeNonTensor)

            q_net_train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(5000).batch(
                self.batchSizeNonTensor)
            q_net_test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
                self.batchSizeNonTensor)

            q_net = self.qNets[lid]

            # Analysis of optimal decisions
            optimal_decisions_ground_truth = []
            for X_batch, y_batch in q_net_train_dataset:
                optimal_decisions_ground_truth.append(y_batch)
            optimal_decisions_ground_truth = np.concatenate(optimal_decisions_ground_truth, axis=0)
            histogram = Counter(np.argmax(optimal_decisions_ground_truth, axis=-1))
            print("Histogram:{0}".format(histogram))

            # Load trained autoencoder
            root_path = os.path.dirname(__file__)
            model_path = os.path.join(root_path, "..", "saved_models", "model_{0}_autoencoder".format(run_id))
            autoencoder_model_path = os.path.join(model_path, "model")
            autoencoder_model.load_weights(autoencoder_model_path)

            # ***************** COMMENT THAT PART OUT, LOAD SAVE MODEL *****************
            # # Train the q-net model
            # iteration = 0
            # for epoch_id in range(q_net_epoch_count):
            #     self.qNetTrackers[lid].reset_states()
            #     # Keep track of prediction statistics
            #     action_counters = []
            #     optimal_decisions_ground_truth = []
            #     optimal_decisions_prediction = []
            #
            #     for X_batch in normal_train_data:
            #         with tf.GradientTape() as q_tape:
            #             X_batch_hat = autoencoder_model(inputs=X_batch, training=True)
            #             mse = self.mseLoss(X_batch, X_batch_hat)
            #             # reg_losses = []
            #             # for var in q_net.trainable_variables:
            #             #     assert var.ref() in self.regularizationCoefficients
            #             #     reg_coeff = self.regularizationCoefficients[var.ref()]
            #             #     reg_losses.append(reg_coeff * tf.nn.l2_loss(var))
            #             # regularization_loss = tf.add_n(reg_losses)
            #             # total_loss = mse + regularization_loss
            #             total_loss = mse
            #         self.qNetTrackers[lid].update_state(mse)
            #         q_grads = q_tape.gradient(total_loss, autoencoder_model.trainable_variables)
            #         q_net_optimizer.apply_gradients(zip(q_grads, autoencoder_model.trainable_variables))
            #         iteration += 1
            #         print("Level:{0} Iteration:{1} Epoch:{2} MSE:{3}".format(lid,
            #                                                                  iteration,
            #                                                                  epoch_id,
            #                                                                  self.qNetTrackers[lid].result().numpy()))
            #         print("Lr:{0}".format(q_net_optimizer._decayed_lr(tf.float32).numpy()))
            #
            # # Save trained autoencoder
            # root_path = os.path.dirname(__file__)
            # model_path = os.path.join(root_path, "..", "saved_models", "model_{0}_autoencoder".format(run_id))
            # os.mkdir(model_path)
            # autoencoder_model_path = os.path.join(model_path, "model")
            # os.mkdir(autoencoder_model_path)
            # autoencoder_model.save_weights(autoencoder_model_path)
            # ***************** COMMENT THAT PART OUT, LOAD SAVE MODEL *****************

            normal_distances = []
            anomaly_distances = []
            for X_batch in normal_train_data:
                X_batch_hat = autoencoder_model(inputs=X_batch, training=False)
                d_ = X_batch_hat - X_batch
                d_ = tf.pow(d_, 2.0)
                error_vector = np.mean(d_, axis=(1, 2, 3))
                normal_distances.append(error_vector)

            for X_batch in anomaly_train_data:
                X_batch_hat = autoencoder_model(inputs=X_batch, training=False)
                d_ = X_batch_hat - X_batch
                d_ = tf.pow(d_, 2.0)
                error_vector = np.mean(d_, axis=(1, 2, 3))
                anomaly_distances.append(error_vector)

            normal_distances = np.concatenate(normal_distances)
            anomaly_distances = np.concatenate(anomaly_distances)

            normal_mean = np.mean(normal_distances)
            normal_std = np.std(normal_distances)
            anomaly_mean = np.mean(anomaly_distances)
            anomaly_std = np.std(anomaly_distances)

            print("X")

            for dataset, dataset_type in [(q_net_train_dataset, "training"), (q_net_test_dataset, "test")]:
                ground_truth = []
                predictions = []
                for X_batch, y_batch in dataset:
                    y_hat = q_net(inputs=X_batch)
                    ground_truth.append(y_batch)
                    predictions.append(y_hat)
                ground_truth = np.concatenate(ground_truth, axis=0)
                predictions = np.concatenate(predictions, axis=0)
                ground_truth = np.argmax(ground_truth, axis=-1)
                predictions = np.argmax(predictions, axis=-1)
                report = classification_report(ground_truth, predictions)
                print("Dataset:{0} Report:{1}".format(dataset_type, report))

            # conv_transpose = tf.keras.layers.Conv2DTranspose(64,
            #                                                  kernel_size=self.kernelSize,
            #                                                  strides=3,
            #                                                  padding="same",
            #                                                  name="trans_conv_{0}".format(block_id),
            #                                                  use_bias=False)

        #     # Train the q-net model
        #     iteration = 0
        #     for epoch_id in range(q_net_epoch_count):
        #         self.qNetTrackers[lid].reset_states()
        #         # Keep track of prediction statistics
        #         action_counters = []
        #         optimal_decisions_ground_truth = []
        #         optimal_decisions_prediction = []
        #
        #         for X_batch, y_batch in q_net_train_dataset:
        #             with tf.GradientTape() as q_tape:
        #                 y_hat = q_net(inputs=X_batch)
        #                 mse = self.mseLoss(y_batch, y_hat)
        #                 reg_losses = []
        #                 for var in q_net.trainable_variables:
        #                     assert var.ref() in self.regularizationCoefficients
        #                     reg_coeff = self.regularizationCoefficients[var.ref()]
        #                     reg_losses.append(reg_coeff * tf.nn.l2_loss(var))
        #                 regularization_loss = tf.add_n(reg_losses)
        #                 total_loss = mse + regularization_loss
        #             self.qNetTrackers[lid].update_state(mse)
        #             q_grads = q_tape.gradient(total_loss, q_net.trainable_variables)
        #             q_net_optimizer.apply_gradients(zip(q_grads, q_net.trainable_variables))
        #             iteration += 1
        #             print("Level:{0} Iteration:{1} Epoch:{2} MSE:{3}".format(lid,
        #                                                                      iteration,
        #                                                                      epoch_id,
        #                                                                      self.qNetTrackers[lid].result().numpy()))
        #             print("Lr:{0}".format(q_net_optimizer._decayed_lr(tf.float32).numpy()))
        #
        #         if (epoch_id + 1) % 10 == 0:
        #             for dataset, dataset_type in [(q_net_train_dataset, "training"), (q_net_test_dataset, "test")]:
        #                 ground_truth = []
        #                 predictions = []
        #                 for X_batch, y_batch in dataset:
        #                     y_hat = q_net(inputs=X_batch)
        #                     ground_truth.append(y_batch)
        #                     predictions.append(y_hat)
        #                 ground_truth = np.concatenate(ground_truth, axis=0)
        #                 predictions = np.concatenate(predictions, axis=0)
        #                 ground_truth = np.argmax(ground_truth, axis=-1)
        #                 predictions = np.argmax(predictions, axis=-1)
        #                 report = classification_report(ground_truth, predictions)
        #                 print("Dataset:{0} Report:{1}".format(dataset_type, report))
        #
        #     print("X")
        # print("X")

    def train(self, run_id, dataset, epoch_count, **kwargs):
        # q_net_epoch_count = kwargs["q_net_epoch_count"]
        # fine_tune_epoch_count = kwargs["fine_tune_epoch_count"]
        # warm_up_epoch_count = kwargs["warm_up_epoch_count"]
        # is_in_warm_up_period = True

        constraints = {"q_net_epoch_count": kwargs["q_net_epoch_count"],
                       "fine_tune_epoch_count": kwargs["fine_tune_epoch_count"],
                       "warm_up_epoch_count": kwargs["warm_up_epoch_count"],
                       "q_net_train_start_epoch": kwargs["q_net_train_start_epoch"],
                       "q_net_train_period": kwargs["q_net_train_period"]}
        # OK
        self.optimizer = self.get_sgd_optimizer()
        # OK
        self.build_trackers()
        states_dict = self.get_epoch_states(constraints=constraints, epoch_count=epoch_count)

        # Train loop
        iteration = 0
        for epoch_id in range(epoch_count):
            # One epoch training of the main CIGN body, without Q-Nets - OK
            #
            # for train_X, train_y in dataset:
            #
            #
            #
            #

            iteration, times_list = self.train_cign_body_one_epoch(
                dataset=dataset.trainDataTf,
                run_id=run_id,
                iteration=iteration,
                is_in_warm_up_period=states_dict[epoch_id]["is_in_warm_up_period"])
            self.check_q_net_vars()
            # If it is the valid epoch, train the Q-Nets
            if states_dict[epoch_id]["do_train_q_nets"]:
                self.save_model(run_id=run_id, epoch_id=epoch_id)
                # Freeze the main CIGN and train the Q-Nets.
                self.train_q_nets_with_full_net(dataset=dataset, q_net_epoch_count=kwargs["q_net_epoch_count"])
            # Measure the model performance, if it is a valid epoch.
            if states_dict[epoch_id]["do_measure_performance"]:
                self.measure_performance(dataset=dataset,
                                         run_id=run_id,
                                         iteration=iteration,
                                         epoch_id=epoch_id,
                                         times_list=times_list)

        # Train the routing for the last time.
        self.train_q_nets_with_full_net(dataset=dataset, q_net_epoch_count=kwargs["q_net_epoch_count"])

        # Fine tune the model.
        # Fine tune by merging training and validation sets
        full_train_X = np.concatenate([dataset.trainX, dataset.valX], axis=0)
        full_train_y = np.concatenate([dataset.trainY, dataset.valY], axis=0)
        train_tf = tf.data.Dataset.from_tensor_slices((full_train_X, full_train_y)). \
            shuffle(5000).batch(self.batchSizeNonTensor)

        for epoch_id in range(epoch_count, epoch_count + kwargs["fine_tune_epoch_count"]):
            iteration, times_list = self.train_cign_body_one_epoch(dataset=train_tf,
                                                                   run_id=run_id,
                                                                   iteration=iteration,
                                                                   is_in_warm_up_period=False)
            self.measure_performance(dataset=dataset,
                                     run_id=run_id,
                                     iteration=iteration,
                                     epoch_id=epoch_id,
                                     times_list=times_list)

    def train_using_q_nets_as_post_processing(self, run_id, dataset, epoch_count, **kwargs):
        constraints = {"q_net_epoch_count": kwargs["q_net_epoch_count"],
                       "fine_tune_epoch_count": kwargs["fine_tune_epoch_count"],
                       "warm_up_epoch_count": kwargs["warm_up_epoch_count"],
                       "q_net_train_start_epoch": kwargs["q_net_train_start_epoch"],
                       "q_net_train_period": kwargs["q_net_train_period"]}
        # OK
        self.optimizer = self.get_sgd_optimizer()
        self.build_trackers()
        states_dict = self.get_epoch_states(constraints=constraints, epoch_count=epoch_count)

        # Train loop
        iteration = 0
        for epoch_id in range(epoch_count):
            # TODO: Go on from here.
            iteration, times_list = self.train_cign_body_one_epoch(
                dataset=dataset.trainDataTf,
                run_id=run_id,
                iteration=iteration,
                is_in_warm_up_period=states_dict[epoch_id]["is_in_warm_up_period"],
                only_use_ig_routing=not states_dict[epoch_id]["is_in_warm_up_period"])
            if states_dict[epoch_id]["do_measure_performance"]:
                self.measure_performance(dataset=dataset,
                                         run_id=run_id,
                                         iteration=iteration,
                                         epoch_id=epoch_id,
                                         times_list=times_list,
                                         only_use_ig_routing=True)
        # assert epoch_id == epoch_count
        self.save_model(run_id=run_id, epoch_id=epoch_count - 1)

    def measure_performance(self, dataset, run_id, iteration=-1, epoch_id=0, times_list=tuple([0]), **kwargs):
        only_use_ig_routing = kwargs["only_use_ig_routing"]

        accuracy_dict = {}
        for dataset_type, ds in zip(
                ["train", "validation", "test"],
                [dataset.trainDataTf, dataset.validationDataTf, dataset.testDataTf]):
            if ds is None:
                accuracy_dict[dataset_type] = 0.0
                continue
            accuracy, q_losses = self.eval(run_id=run_id, iteration=iteration, dataset=ds,
                                           dataset_type=dataset_type, only_use_ig_routing=only_use_ig_routing)
            print("{0} accuracy:{1}".format(dataset_type, accuracy))
            print("{0} Q Losses:{1}".format(dataset_type, q_losses))
            accuracy_dict[dataset_type] = accuracy
        mean_time_passed = np.mean(np.array(times_list))
        DbLogger.write_into_table(
            rows=[(run_id, iteration, epoch_id, accuracy_dict["train"],
                   accuracy_dict["validation"], accuracy_dict["test"],
                   mean_time_passed, 0.0, "XXX")], table=DbLogger.logsTable)

    def calculate_total_q_net_loss(self, model_output, y):
        regression_q_targets, optimal_q_values = self.calculate_q_tables_from_network_outputs(
            true_labels=y.numpy(), model_outputs=model_output)
        # Q-Net Losses
        q_net_predicted = model_output["q_tables_predicted"]
        q_net_losses = []
        for idx, tpl in enumerate(zip(regression_q_targets, q_net_predicted)):
            q_truth = tpl[0]
            q_predicted = tpl[1]
            q_truth_tensor = tf.convert_to_tensor(q_truth, dtype=q_predicted.dtype)
            mse = self.mseLoss(q_truth_tensor, q_predicted)
            q_net_losses.append(mse)
        # full_q_loss = self.qNetCoeff * tf.add_n(q_net_losses)
        # if add_regularization:
        #     q_regularization_loss = self.get_regularization_loss(is_for_q_nets=True)
        #     total_q_loss = full_q_loss + q_regularization_loss
        # else:
        #     total_q_loss = full_q_loss
        return q_net_losses

    # SEEMS OK
    def eval(self, run_id, iteration, dataset, dataset_type, **kwargs):
        if dataset is None:
            return 0.0
        y_true = []
        y_pred = []
        q_loss_list = [[] for _ in range(self.get_max_trajectory_length())]
        leaf_distributions = {node.index: [] for node in self.leafNodes}
        only_use_ig_routing = kwargs["only_use_ig_routing"]

        for X, y in dataset:
            model_output = self.run_model(
                X=X,
                y=y,
                iteration=-1,
                is_training=False,
                only_use_ig_routing=only_use_ig_routing)
            y_pred_batch, leaf_distributions_batch = self.calculate_predictions_of_batch(
                model_output=model_output, y=y)
            for leaf_node in self.leafNodes:
                leaf_distributions[leaf_node.index].extend(leaf_distributions_batch[leaf_node.index])
            y_pred.append(y_pred_batch)
            y_true.append(y.numpy())
            q_net_losses = self.calculate_total_q_net_loss(model_output=model_output, y=y)
            for idx, q_loss in enumerate(q_net_losses):
                q_loss_list[idx].append(q_loss.numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        truth_vector = y_true == y_pred
        accuracy = np.mean(truth_vector.astype(np.float))
        for idx, q_losses in enumerate(q_loss_list):
            mean_q_loss = np.mean(np.array(q_losses))
            q_loss_list[idx] = mean_q_loss

        # Print sample distribution
        kv_rows = []
        for leaf_node in self.leafNodes:
            c = Counter(leaf_distributions[leaf_node.index])
            str_ = "{0} Node {1} Sample Distribution:{2}".format(dataset_type, leaf_node.index, c)
            print(str_)
            kv_rows.append((run_id, iteration,
                            "{0} Node {1} Sample Distribution".format(dataset_type, leaf_node.index),
                            "{0}".format(c)))
        for idx, q_loss in enumerate(q_loss_list):
            kv_rows.append((run_id, iteration,
                            "{0} Q-Net{1} MSE Loss".format(dataset_type, idx),
                            np.asscalar(q_loss)))
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)
        return accuracy, q_loss_list

    def print_train_step_info(self, **kwargs):
        iteration = kwargs["iteration"]
        time_intervals = kwargs["time_intervals"]
        eval_dict = kwargs["eval_dict"]
        # Print outputs
        print("************************************")
        print("Iteration {0}".format(iteration))
        for k, v in time_intervals.items():
            print("{0}={1}".format(k, v))
        self.print_losses(eval_dict=eval_dict)

        q_str = "Q-Net Losses: "
        for idx, q_net_loss in enumerate(self.qNetTrackers):
            q_str += "Q-Net_{0}:{1} ".format(idx, self.qNetTrackers[idx].result().numpy())

        print(q_str)
        print("Temperature:{0}".format(self.softmaxDecayController.get_value()))
        print("Lr:{0}".format(self.optimizer._decayed_lr(tf.float32).numpy()))
        print("************************************")

    def track_losses(self, **kwargs):
        super().track_losses(**kwargs)
        q_net_losses = kwargs["q_net_losses"]
        if q_net_losses is not None:
            for idx, q_net_loss in enumerate(q_net_losses):
                self.qNetTrackers[idx].update_state(q_net_loss)

    def train_end_to_end(self, run_id, dataset, epoch_count, **kwargs):
        constraints = {"q_net_epoch_count": kwargs["q_net_epoch_count"],
                       "fine_tune_epoch_count": kwargs["fine_tune_epoch_count"],
                       "warm_up_epoch_count": kwargs["warm_up_epoch_count"],
                       "q_net_train_start_epoch": kwargs["q_net_train_start_epoch"],
                       "q_net_train_period": kwargs["q_net_train_period"]}
        # OK
        self.optimizer = self.get_sgd_optimizer()
        # OK
        self.build_trackers()
        states_dict = self.get_epoch_states(constraints=constraints, epoch_count=epoch_count)
        iteration = 0
        iterations_in_warm_up = 0
        for epoch_id in range(epoch_count):
            is_in_warm_up_period = states_dict[epoch_id]["is_in_warm_up_period"]
            self.reset_trackers()
            for train_X, train_y in dataset.trainDataTf:
                profiler = Profiler()
                with tf.GradientTape() as main_tape:
                    model_output = self.run_model(
                        X=train_X,
                        y=train_y,
                        iteration=iteration,
                        is_training=True,
                        warm_up_period=is_in_warm_up_period,
                        only_use_ig_routing=False)
                    regression_q_targets, optimal_q_values = self.calculate_q_tables_from_network_outputs(
                        true_labels=train_y.numpy(), model_outputs=model_output)
                    classification_losses = model_output["classification_losses"]
                    info_gain_losses = model_output["info_gain_losses"]
                    q_net_outputs = model_output["q_tables_predicted"]

                    # L2 Loss for regularization
                    regularization_losses = []
                    for var in self.model.trainable_variables:
                        assert var.ref() in self.regularizationCoefficients
                        lambda_coeff = self.regularizationCoefficients[var.ref()]
                        regularization_losses.append(lambda_coeff * tf.nn.l2_loss(var))
                    regularization_loss_total = tf.add_n(regularization_losses)

                    # Classification losses
                    classification_loss = tf.add_n([loss for loss in classification_losses.values()])

                    # Information Gain losses
                    info_gain_loss = self.decisionLossCoeff * tf.add_n([loss for loss in info_gain_losses.values()])

                    # Q-Net Loss
                    q_net_losses = []
                    for lid in range(self.get_max_trajectory_length()):
                        q_truth = regression_q_targets[lid]
                        q_hat = q_net_outputs[lid]
                        mse = self.mseLoss(q_truth, q_hat)
                        q_net_losses.append(mse)
                    total_qnet_loss = tf.add_n(q_net_losses)
                    # Update Q Nets only outside warm-up period
                    if is_in_warm_up_period:
                        total_loss = classification_loss + regularization_loss_total + \
                                     info_gain_loss + 0.0 * total_qnet_loss
                    else:
                        total_loss = classification_loss + regularization_loss_total + info_gain_loss + total_qnet_loss
                grads = main_tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                iteration += 1
                self.softmaxDecayController.update(iteration=iteration)
                if is_in_warm_up_period:
                    iterations_in_warm_up += 1
                else:
                    self.globalStep.assign(value=iteration - iterations_in_warm_up)
                profiler.add_measurement(label="One Iteration")
                self.track_losses(total_loss=total_loss,
                                  classification_losses=model_output["classification_losses"],
                                  info_gain_losses=model_output["info_gain_losses"],
                                  q_net_losses=q_net_losses)
                self.print_train_step_info(
                    iteration=iteration,
                    classification_losses=model_output["classification_losses"],
                    info_gain_losses=model_output["info_gain_losses"],
                    time_intervals=profiler.get_all_measurements(),
                    eval_dict=model_output["eval_dict"])
            # times_list.append(sum(profiler.get_all_measurements().values()))
            if states_dict[epoch_id]["do_measure_performance"]:
                self.measure_performance(dataset=dataset,
                                         run_id=run_id,
                                         iteration=iteration,
                                         epoch_id=epoch_id,
                                         times_list=[0],
                                         only_use_ig_routing=False)