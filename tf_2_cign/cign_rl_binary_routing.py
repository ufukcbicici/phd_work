import numpy as np
import tensorflow as tf
import time

from auxillary.dag_utilities import Dag
from auxillary.db_logger import DbLogger
from tf_2_cign.cign_rl_routing import CignRlRouting
from tf_2_cign.custom_layers.cign_binary_action_generator_layer import CignBinaryActionGeneratorLayer
from tf_2_cign.custom_layers.cign_binary_action_space_generator_layer import CignBinaryActionSpaceGeneratorLayer
from tf_2_cign.custom_layers.cign_binary_rl_routing_layer import CignBinaryRlRoutingLayer
from tf_2_cign.custom_layers.cign_rl_routing_layer import CignRlRoutingLayer
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
                 decision_loss_coeff, q_net_coeff, epsilon_decay_rate, epsilon_step, bn_momentum=0.9):
        super().__init__(valid_prediction_reward, invalid_prediction_penalty, include_ig_in_reward_calculations,
                         lambda_mac_cost, warm_up_period, cign_rl_train_period, batch_size, input_dims, class_count,
                         node_degrees, decision_drop_probability, classification_drop_probability, decision_wd,
                         classification_wd, information_gain_balance_coeff, softmax_decay_controller,
                         learning_rate_schedule, decision_loss_coeff, bn_momentum)
        # Epsilon hyperparameter for exploration - explotation
        self.globalStep = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
        self.epsilonDecayRate = epsilon_decay_rate
        self.epsilonStep = epsilon_step
        self.exploreExploitEpsilon = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1.0, decay_steps=self.epsilonStep, decay_rate=self.epsilonDecayRate)
        self.actionSpaceGenerators = {}
        self.actionCalculatorLayers = {}

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

    # TODO: Test this
    # Here, some changes are needed.
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

        # Action generator
        # action_generator_layer = CignBinaryActionGeneratorLayer(network=self)
        # actions, explore_exploit_vec = action_generator_layer([q_table_predicted_cign_output, self.globalStep])

        # Get information gain activations from the current level.
        node = self.orderedNodesPerLevel[level][-1]
        ig_activations = tf.stack(
            [self.igActivationsDict[nd.index] for nd in self.orderedNodesPerLevel[level]], axis=-1)
        routing_calculation_layer = CignBinaryRlRoutingLayer(level=level, network=self)
        # Get secondary routing information from the previous layer also.
        if level - 1 < 0:
            sc_routing_matrix = tf.expand_dims(tf.ones_like(input_ig_routing_matrix[:, 0]), axis=-1)
        else:
            sc_routing_matrix = self.scRoutingMatricesDict[level - 1]
        # explore_actions = tf.random.uniform(shape=[q_table_predicted.shape[0]], dtype=tf.int32,
        #                                     minval=0, maxval=2)
        predicted_actions, secondary_routing_matrix_cign_output = routing_calculation_layer(
            [q_table_predicted_cign_output,
             # self.warmUpPeriodInput,
             ig_activations,
             sc_routing_matrix])
        self.actionsPredicted.append(predicted_actions)
        self.scRoutingCalculationLayers.append(routing_calculation_layer)
        return secondary_routing_matrix_cign_output

    def get_model_outputs_array(self):
        model_output_arr = super().get_model_outputs_array()
        model_output_arr.append(self.actionsPredicted)
        model_output_arr.append(self.scRoutingMatricesDict)
        return model_output_arr

    def get_model_outputs_dict(self, model_output_arr):
        model_output_dict = super(CignRlBinaryRouting, self).get_model_outputs_dict(model_output_arr)
        model_output_dict["actions_predicted"] = model_output_arr[9]
        model_output_dict["sc_routing_matrices_dict"] = model_output_arr[10]
        return model_output_dict

    def calculate_next_level_configurations_manuel(self, level, ig_activations, sc_routing_matrix_curr_level):
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

    def calculate_sample_action_space(self, ig_activations_dict):
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
            if level not in self.actionSpaceGenerators:
                self.actionSpaceGenerators[level] = \
                    CignBinaryActionSpaceGeneratorLayer(level=level, network=self)
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
                sc_routing_matrix_action_0, sc_routing_matrix_action_1 = self.actionSpaceGenerators[level](
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

    def calculate_sample_action_space_manuel(self, ig_activations_dict):
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













    def calculate_q_tables_from_network_outputs(self,
                                                true_labels,
                                                posteriors_dict,
                                                ig_masks_dict,
                                                **kwargs):
        actions_predicted = kwargs["actions_predicted"]
        sc_routing_matrices_dict = kwargs["sc_routing_matrices_dict"]

        # 1) Calculate the action spaces of each sample.
        # action_spaces = []
        # sc_routing_tensor_curr_level = tf.expand_dims(
        #     tf.expand_dims(tf.ones_like(true_labels), axis=-1), axis=0)
        # action_spaces.append(sc_routing_tensor_curr_level)
        # for level in range(self.get_max_trajectory_length()):
        #     # IG activations for the nodes in this layer.
        #     ig_activations = tf.stack([ig_masks_dict[nd.index] for nd in self.orderedNodesPerLevel[level]],
        #                               axis=-1)
        #     # Generate action space generator layer for the first time.
        #     if level not in self.actionSpaceGenerators:
        #         self.actionSpaceGenerators[level] = \
        #             CignBinaryActionSpaceGeneratorLayer(level=level, network=self)
        #     # For every past action combination in the current trajectory so far,
        #     # get the current layer's node configuration. If we are at t. level, there will be actions taken:
        #     # A_{0:t-1} = a_0,a_1,a_2, ..., a_{t-1}
        #     # There will be 2^t different paths to the level t (each action is binary).
        #     actions_each_layer = [[0]]
        #     actions_each_layer.extend([[0, 1] for _ in range(level)])
        #     list_of_all_trajectories = Utilities.get_cartesian_product(list_of_lists=actions_each_layer)
        #     sc_routing_tensor_next_level_shape = [1]
        #     sc_routing_tensor_next_level_shape.extend([2 for _ in range(level + 1)])
        #     sc_routing_tensor_next_level_shape.append(true_labels.shape[0])
        #     sc_routing_tensor_next_level_shape.append(len(self.orderedNodesPerLevel[level + 1]))
        #     sc_routing_tensor_next_level = tf.zeros(shape=sc_routing_tensor_next_level_shape,
        #                                             dtype=sc_routing_tensor_curr_level.dtype)
        #     for trajectory in list_of_all_trajectories:
        #         # Given the current trajectory = a_0,a_1,a_2, ..., a_{t-1}, get the configuration of the
        #         # nodes in the current level, for every sample in the batch
        #         current_level_routing_matrix = sc_routing_tensor_curr_level[trajectory]
        #         sc_routing_matrix_action_0, sc_routing_matrix_action_1 = self.actionSpaceGenerators[level](
        #             [ig_activations, current_level_routing_matrix])
        #         action_0_trajectory = []
        #         action_0_trajectory.extend(trajectory)
        #         action_0_trajectory.append(0)
        #         sc_routing_tensor_next_level[action_0_trajectory] = sc_routing_matrix_action_0
        #
        #         action_1_trajectory = []
        #         action_1_trajectory.extend(trajectory)
        #         action_1_trajectory.append(1)
        #         sc_routing_tensor_next_level[action_1_trajectory] = sc_routing_matrix_action_1
        #
        #     sc_routing_tensor_curr_level = sc_routing_tensor_next_level
        #     action_spaces.append(sc_routing_tensor_curr_level)

        # 2) Using the calculated action spaces, now calculate the optimal Q tables for each sample.

        # ig_activations = inputs[0]
        # sc_routing_matrix_prev_level = inputs[1]
        # sc_routing_matrices = \
        #     self.actionSpaceGenerators[level]([ig_activations, sc_routing_matrix_prev_level])

        # for level in range(self.get_max_trajectory_length()):
        #     ig_activations = tf.stack(
        #         [self.igActivationsDict[nd.index] for nd in self.orderedNodesPerLevel[level]], axis=-1)
        #     if level == 0:
        #         sc_routing_matrix_prev_level = tf.expand_dims(tf.ones_like(true_labels), axis=-1)
        #     else:
        #         sc_routing_matrix_prev_level = self.scRoutingMatricesDict[level - 1]
        #     if level not in self.actionSpaceGenerators:
        #         self.actionSpaceGenerators[level] = CignBinaryActionSpaceGeneratorLayer(level=level, network=self)
        #     sc_routing_matrices = \
        #         self.actionSpaceGenerators[level]([ig_activations, sc_routing_matrix_prev_level])
        #     sc_routing_matrices = tf.stack(sc_routing_matrices, axis=-1)
        #     action_spaces.append(sc_routing_matrices)

        # sc_routing_matrices = self.actionSpaceGeneratorLayer([ig_activations, sc_routing_matrix_prev_level])
        # sc_routing_matrix = tf.where(final_actions, sc_routing_matrices[1], sc_routing_matrices[0])
        #
        #
        # for nd_idx, node in enumerate(self.orderedNodesPerLevel[level]):
        # # ************ final_actions[i] == 0 ************
        # sc_routing_matrix_action_0 = []
        # for nd_idx in range(self.nodeCountInThisLevel):
        #     activations_nd = ig_activations[:, :, nd_idx]
        #     sc_routing_vector = sc_routing_matrix_prev_level[:, nd_idx]
        #     ig_indices = tf.argmax(activations_nd, axis=-1)
        #     ig_routing_matrix = tf.one_hot(ig_indices, activations_nd.shape[1])
        #     rl_routing_matrix = sc_routing_vector * ig_routing_matrix
        #     sc_routing_matrix_action_0.append(rl_routing_matrix)
        # # ************ final_actions[i] == 0 ************
        #
        # # ************ final_actions[i] == 1 ************
        # sc_routing_matrix_action_1 = []
        # for nd_idx in range(self.nodeCountInThisLevel):
        #     sc_routing_vector = sc_routing_matrix_prev_level[:, nd_idx]
        #     rl_routing_matrix = tf.ones_like(sc_routing_matrix_action_0[nd_idx])
        #     rl_routing_matrix = sc_routing_vector * rl_routing_matrix
        #     sc_routing_matrix_action_1.append(rl_routing_matrix)
        # # ************ final_actions[i] == 1 ************

        # # Get the sc routing outputs of this level.
        # sc_outputs = sc_routing_matrices_dict[level]

        # for t in range(self.get_max_trajectory_length() - 1, -1, -1):

        # sample_count = true_labels.shape[0]
        # ig_paths_matrix = self.get_ig_paths(ig_masks_dict=ig_masks_dict)
        # c = Counter([tuple(ig_paths_matrix[i]) for i in range(ig_paths_matrix.shape[0])])
        # # print("Count of ig paths:{0}".format(c))
        # regression_targets = []
        # optimal_q_tables = []
        #
        # for t in range(self.get_max_trajectory_length() - 1, -1, -1):
        #     action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
        #     action_count_t = self.actionSpaces[t].shape[0]
        #     optimal_q_table = np.zeros(shape=(sample_count, action_count_t_minus_one, action_count_t), dtype=np.float32)
        #     # If in the last layer, take into account the prediction accuracies.
        #     if t == self.get_max_trajectory_length() - 1:
        #         posteriors_tensor = []
        #         for leaf_node in self.leafNodes:
        #             if not (type(posteriors_dict[leaf_node.index]) is np.ndarray):
        #                 posteriors_tensor.append(posteriors_dict[leaf_node.index].numpy())
        #             else:
        #                 posteriors_tensor.append(posteriors_dict[leaf_node.index])
        #         posteriors_tensor = np.stack(posteriors_tensor, axis=-1)
        #
        #         # Assert that posteriors are placed correctly.
        #         min_leaf_index = min([node.index for node in self.leafNodes])
        #         for leaf_node in self.leafNodes:
        #             if not (type(posteriors_dict[leaf_node.index]) is np.ndarray):
        #                 assert np.array_equal(posteriors_tensor[:, :, leaf_node.index - min_leaf_index],
        #                                       posteriors_dict[leaf_node.index].numpy())
        #             else:
        #                 assert np.array_equal(posteriors_tensor[:, :, leaf_node.index - min_leaf_index],
        #                                       posteriors_dict[leaf_node.index])
        #
        #         # Combine posteriors with respect to the action tuple.
        #         prediction_correctness_vec_list = []
        #         calculation_cost_vec_list = []
        #         min_leaf_id = min([node.index for node in self.leafNodes])
        #         ig_indices = ig_paths_matrix[:, -1] - min_leaf_id
        #         for action_id in range(self.actionSpaces[t].shape[0]):
        #             routing_decision = self.actionSpaces[t][action_id, :]
        #             routing_matrix = np.repeat(routing_decision[np.newaxis, :], axis=0,
        #                                        repeats=true_labels.shape[0])
        #             if self.includeIgInRewardCalculations:
        #                 # Set Information Gain routed leaf nodes to 1. They are always evaluated.
        #                 routing_matrix[np.arange(true_labels.shape[0]), ig_indices] = 1
        #             weights = np.reciprocal(np.sum(routing_matrix, axis=1).astype(np.float32))
        #             routing_matrix_weighted = weights[:, np.newaxis] * routing_matrix
        #             weighted_posteriors = posteriors_tensor * routing_matrix_weighted[:, np.newaxis, :]
        #             final_posteriors = np.sum(weighted_posteriors, axis=2)
        #             predicted_labels = np.argmax(final_posteriors, axis=1)
        #             validity_of_predictions_vec = (predicted_labels == true_labels).astype(np.int32)
        #             prediction_correctness_vec_list.append(validity_of_predictions_vec)
        #             # Get the calculation costs
        #             computation_overload_vector = np.apply_along_axis(
        #                 lambda x: self.networkActivationCostsDict[tuple(x)], axis=1,
        #                 arr=routing_matrix)
        #             calculation_cost_vec_list.append(computation_overload_vector)
        #         prediction_correctness_matrix = np.stack(prediction_correctness_vec_list, axis=1)
        #         prediction_correctness_tensor = np.repeat(
        #             np.expand_dims(prediction_correctness_matrix, axis=1), axis=1,
        #             repeats=action_count_t_minus_one)
        #         computation_overload_matrix = np.stack(calculation_cost_vec_list, axis=1)
        #         computation_overload_tensor = np.repeat(
        #             np.expand_dims(computation_overload_matrix, axis=1), axis=1,
        #             repeats=action_count_t_minus_one)
        #         # Add to the rewards tensor
        #         optimal_q_table += (prediction_correctness_tensor == 1
        #                             ).astype(np.float32) * self.validPredictionReward
        #         optimal_q_table += (prediction_correctness_tensor == 0
        #                             ).astype(np.float32) * self.invalidPredictionPenalty
        #         optimal_q_table -= self.lambdaMacCost * computation_overload_tensor
        #         for idx in range(optimal_q_table.shape[1] - 1):
        #             assert np.array_equal(optimal_q_table[:, idx, :], optimal_q_table[:, idx + 1, :])
        #         regression_targets.append(optimal_q_table[:, 0, :].copy())
        #     else:
        #         q_table_next = optimal_q_tables[-1].copy()
        #         q_table_next = np.max(q_table_next, axis=-1)
        #         regression_targets.append(q_table_next)
        #         optimal_q_table = np.expand_dims(q_table_next, axis=1)
        #         optimal_q_table = np.repeat(optimal_q_table, axis=1, repeats=action_count_t_minus_one)
        #     reachability_matrix = self.reachabilityMatrices[t].copy().astype(np.float32)
        #     reachability_matrix = np.repeat(np.expand_dims(reachability_matrix, axis=0), axis=0,
        #                                     repeats=optimal_q_table.shape[0])
        #     assert optimal_q_table.shape == reachability_matrix.shape
        #     optimal_q_table[reachability_matrix == 0] = -np.inf
        #     optimal_q_tables.append(optimal_q_table)
        #
        # regression_targets.reverse()
        # optimal_q_tables.reverse()
        # return regression_targets, optimal_q_tables

    # TODO: Rewrite this according to the binary routing logic.
    # def run_q_net_model(self, X, y, iteration, is_in_warm_up_period):
    #     with tf.GradientTape() as q_tape:
    #         model_output_val = self.run_model(
    #             X=X,
    #             y=y,
    #             iteration=iteration,
    #             is_training=True,
    #             warm_up_period=is_in_warm_up_period)
    #         # Calculate target values for the Q-Nets
    #         posteriors_val = {k: v.numpy() for k, v in model_output_val["posteriors_dict"].items()}
    #         ig_masks_val = {k: v.numpy() for k, v in model_output_val["ig_masks_dict"].items()}
    #         regs, q_s = self.calculate_q_tables_from_network_outputs(true_labels=y.numpy(),
    #                                                                  posteriors_dict=posteriors_val,
    #                                                                  ig_masks_dict=ig_masks_val)
    #         # Q-Net Losses
    #         q_net_predicted = model_output_val["q_tables_predicted"]
    #         q_net_losses = []
    #         for idx, tpl in enumerate(zip(regs, q_net_predicted)):
    #             q_truth = tpl[0]
    #             q_predicted = tpl[1]
    #             q_truth_tensor = tf.convert_to_tensor(q_truth, dtype=q_predicted.dtype)
    #             mse = self.mseLoss(q_truth_tensor, q_predicted)
    #             q_net_losses.append(mse)
    #         full_q_loss = self.qNetCoeff * tf.add_n(q_net_losses)
    #         q_regularization_loss = self.get_regularization_loss(is_for_q_nets=True)
    #         total_q_loss = full_q_loss + q_regularization_loss
    #     q_grads = q_tape.gradient(total_q_loss, self.model.trainable_variables)
    #     return model_output_val, q_grads, q_net_losses
