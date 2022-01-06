import numpy as np
import tensorflow as tf
import time

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
        q_net_layer = self.get_q_net_layer(level=level)
        q_table_predicted = q_net_layer(q_net_input_f_tensor)
        # Create the action generator layer. Get predicted actions for the next layer.
        action_generator_layer = CignBinaryActionGeneratorLayer(network=self)
        predicted_actions, explore_exploit_vec, explore_actions, exploit_actions = action_generator_layer(
            q_table_predicted)
        q_net = tf.keras.Model(inputs=q_net_input_f_tensor, outputs=[q_table_predicted,
                                                                     predicted_actions,
                                                                     explore_exploit_vec,
                                                                     explore_actions,
                                                                     exploit_actions])
        return q_net

    # Here, some changes are needed.
    def calculate_secondary_routing_matrix(self, level, input_f_tensor, input_ig_routing_matrix):
        assert len(self.scRoutingCalculationLayers) == level
        q_net = self.build_isolated_q_net_model(input_f_tensor=input_f_tensor, level=level)
        # Connect the Q-Net to the rest of the network.
        q_table_predicted, predicted_actions, explore_exploit_vec, explore_actions, exploit_actions = \
            q_net(inputs=input_f_tensor)

        q_net_end_to_end_model = tf.keras.Model(inputs=self.feedDict, outputs=[q_table_predicted,
                                                                               predicted_actions,
                                                                               explore_exploit_vec,
                                                                               explore_actions,
                                                                               exploit_actions])
        self.qNetEndToEndModel.append(q_net_end_to_end_model)
        # Save Q-Net and its outputs.
        self.qNets.append(q_net)
        self.qTablesPredicted.append(q_table_predicted)
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
        return secondary_routing_matrix_cign_output

    def get_model_outputs_array(self):
        model_output_arr = super().get_model_outputs_array()
        model_output_arr.append(self.actionsPredicted)
        model_output_arr.append(self.scRoutingMatricesDict)
        return model_output_arr

    def convert_model_outputs_to_dict(self, model_output_arr):
        model_output_dict = super(CignRlBinaryRouting, self).convert_model_outputs_to_dict(model_output_arr)
        model_output_dict["actions_predicted"] = model_output_arr[9]
        model_output_dict["sc_routing_matrices_dict"] = model_output_arr[10]
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
        # ig_masks_dict = {k: v.numpy() for k, v in model_outputs["ig_masks_dict"].items()}
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
            if epoch_id < constraints["q_net_train_start_epoch"]:
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
        model_output_arr = self.model(inputs=feed_dict, training=is_training)
        model_output_dict = self.convert_model_outputs_to_dict(model_output_arr=model_output_arr)
        return model_output_dict

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
