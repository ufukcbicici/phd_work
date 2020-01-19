import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.policy_gradients_network import \
    PolicyGradientsNetwork, TrajectoryHistory


class TreeDepthState:
    def __init__(self, state_ids, state_vecs, max_likelihood_selections):
        self.stateIds = state_ids
        self.stateVectors = state_vecs
        self.maxLikelihoodSelections = max_likelihood_selections


class TreeDepthPolicyNetwork(PolicyGradientsNetwork):
    def __init__(self, l2_lambda,
                 network_name, run_id, iteration, degree_list, data_type, output_names, used_feature_names,
                 hidden_layers, test_ratio=0.2):
        self.hiddenLayers = hidden_layers
        super().__init__(l2_lambda, network_name, run_id, iteration, degree_list, data_type, output_names,
                         used_feature_names, test_ratio=test_ratio)
        assert len(self.hiddenLayers) == self.get_max_trajectory_length()

    def prepare_state_features(self, data):
        # Prepare Policy Gradients State Data
        root_node = [node for node in self.network.topologicalSortedNodes if node.isRoot]
        assert len(root_node) == 1
        features_dict = {}
        for node in self.innerNodes:
            # array_list = [data.get_dict(feature_name)[node.index] for feature_name in self.networkFeatureNames]
            array_list = []
            for feature_name in self.usedFeatureNames:
                feature_arr = data.get_dict(feature_name)[node.index]
                if len(feature_arr.shape) > 2:
                    shape_as_list = list(feature_arr.shape)
                    mean_axes = tuple([i for i in range(1, len(shape_as_list) - 1, 1)])
                    feature_arr = np.mean(feature_arr, axis=mean_axes)
                array_list.append(feature_arr)
            feature_vectors = np.concatenate(array_list, axis=-1)
            features_dict[node.index] = feature_vectors
        return features_dict

    def build_action_spaces(self):
        max_trajectory_length = self.get_max_trajectory_length()
        self.actionSpaces = []
        for t in range(max_trajectory_length):
            next_level_node_count = len(self.network.orderedNodesPerLevel[t + 1])
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

    def build_policy_networks(self, time_step):
        # Create the policy network
        hidden_layers = list(self.hiddenLayers[time_step])
        hidden_layers.append(self.actionSpaces[time_step].shape[0])
        net = self.inputs[time_step]
        for layer_id, layer_dim in enumerate(hidden_layers):
            if layer_id < len(hidden_layers) - 1:
                net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
            else:
                net = tf.layers.dense(inputs=net, units=layer_dim, activation=None)
        _logits = net
        self.logits.append(_logits)
        self.policies.append(tf.nn.softmax(_logits))

    def sample_initial_states(self, data, features_dict, ml_selections_arr, state_sample_count, samples_per_state):
        total_sample_count = data.labelList.shape[0]
        sample_indices = np.random.choice(total_sample_count, state_sample_count, replace=False)
        sample_indices = np.repeat(sample_indices, repeats=samples_per_state)
        feature_arr = features_dict[self.network.topologicalSortedNodes[0].index]
        initial_state_vectors = feature_arr[sample_indices, :]
        state_ml_selections = ml_selections_arr[sample_indices, :]
        history = TrajectoryHistory(state_ids=sample_indices, max_likelihood_routes=state_ml_selections)
        history.states.append(initial_state_vectors)
        return history

    def get_max_trajectory_length(self):
        return int(self.network.depth - 1)

    def get_reachability_matrices(self):
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            actions_t = self.actionSpaces[t]
            if t == 0:
                reachability_matrix_t = np.ones(shape=(1, actions_t.shape[0]))
            else:
                reachability_matrix_t = np.zeros(shape=(self.actionSpaces[t - 1].shape[0], actions_t.shape[0]))
                for action_t_minus_one_id in range(self.actionSpaces[t - 1].shape[0]):
                    node_selection_vec_t_minus_one = self.actionSpaces[t - 1][action_t_minus_one_id]
                    selected_nodes_t = [node for i, node in enumerate(self.network.orderedNodesPerLevel[t])
                                        if node_selection_vec_t_minus_one[i] != 0]
                    next_level_nodes = self.network.orderedNodesPerLevel[t+1]
                    reachable_next_level_node_ids = set()
                    for parent_node in selected_nodes_t:
                        child_nodes = {c_node.index for c_node in self.network.dagObject.children(node=parent_node)}
                        reachable_next_level_node_ids = reachable_next_level_node_ids.union(child_nodes)

                    for actions_t_id in range(actions_t.shape[0]):
                        node_selection_vec_t = actions_t[actions_t_id]
                        reached_nodes = {node.index for is_reached, node in zip(node_selection_vec_t, next_level_nodes)
                                         if is_reached != 0}
                        is_valid_selection = int(len(reached_nodes.difference(reachable_next_level_node_ids)) == 0)
                        reachability_matrix_t[action_t_minus_one_id, actions_t_id] = is_valid_selection
            self.reachabilityMatrices.append(reachability_matrix_t)

    def sample_from_policy(self, history, time_step):
        assert len(history.states) == time_step + 1
        assert len(history.actions) == time_step
        feed_dict = {self.inputs[t]: history.states[t] for t in range(time_step + 1)}
        results = self.tfSession.run([self.policies[time_step], self.policySamples[time_step]], feed_dict=feed_dict)
        policy_samples = results[-1]
        history.actions.append(policy_samples)
        routing_decisions_t = self.actionSpaces[time_step][history.actions[time_step], :]
        history.routingDecisions.append(routing_decisions_t)

    def state_transition(self, history, features_dict, time_step):
        nodes_in_level = self.network.orderedNodesPerLevel[time_step + 1]
        feature_arrays = [features_dict[node.index][history.stateIds] for node in nodes_in_level]
        # routing_decisions_t = self.actionSpaces[time_step][history.actions[time_step], :]
        routing_decisions_t = history.routingDecisions[time_step]
        weighted_feature_arrays = [np.expand_dims(routing_decisions_t[:, idx], axis=1) * feature_arrays[idx]
                                   for idx in range(len(nodes_in_level))]
        indicator_arr = np.stack([np.any(arr, axis=1) for arr in weighted_feature_arrays], axis=1).astype(np.int32)
        assert np.array_equal(indicator_arr, routing_decisions_t)

        # routing_decisions_t_v2 = \
        #     np.apply_along_axis(lambda a: self.actionSpaces[time_step][np.asscalar(a), :], axis=1,
        #                         arr=np.expand_dims(history.actions[time_step], axis=1))
        # assert np.array_equal(routing_decisions_t, routing_decisions_t_v2)

        states_t_plus_1 = np.concatenate(weighted_feature_arrays, axis=1)
        history.states.append(states_t_plus_1)

    def reward_calculation(self, data, history, posteriors_tensor, time_step):
        rewards_arr = np.zeros(shape=history.stateIds.shape, dtype=np.float32)
        if time_step - 1 < 0:
            actions_t_minus_one = np.zeros(shape=history.stateIds.shape, dtype=np.int32)
        else:
            actions_t_minus_one = history.actions[time_step - 1]
        # Asses availability of the decisions
        action_t = history.actions[time_step]
        validity_of_actions_arr = self.reachabilityMatrices[time_step][actions_t_minus_one, action_t]


        # if time_step < self.get_max_trajectory_length() - 1:
        #     rewards_arr = np.zeros(shape=history.stateIds.shape, dtype=np.float32)
        # else:
        #     # Get the true labels for all samples
        #     true_labels = data.labelList[history.stateIds]
        #     routing_costs = self.networkActivationCosts[history.actions[self.get_max_trajectory_length() - 1]]




    def train(self, state_sample_count, samples_per_state):
        self.tfSession = tf.Session()
        init = tf.global_variables_initializer()
        self.tfSession.run(init)
        self.sample_trajectories(data=self.validationData, features_dict=self.validationFeaturesDict,
                                 ml_selections_arr=self.validationMLPaths, state_sample_count=state_sample_count,
                                 samples_per_state=samples_per_state)
        print("X")


def main():
    # run_id = 715
    # network_name = "Cifar100_CIGN_MultiGpuSingleLateExit"
    # iteration = 119100

    run_id = 453
    network_name = "FashionNet_Lite"
    iteration = 43680

    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature"]
    used_output_names = ["pre_branch_feature"]
    policy_gradients_routing_optimizer = TreeDepthPolicyNetwork(l2_lambda=0.0,
                                                                network_name=network_name,
                                                                run_id=run_id,
                                                                iteration=iteration,
                                                                degree_list=[2, 2],
                                                                data_type="test",
                                                                output_names=output_names,
                                                                used_feature_names=used_output_names,
                                                                test_ratio=0.2,
                                                                hidden_layers=[[128], [256]])
    state_sample_count = policy_gradients_routing_optimizer.validationData.labelList.shape[0]
    samples_per_state = 100
    policy_gradients_routing_optimizer.train(state_sample_count=state_sample_count, samples_per_state=samples_per_state)


if __name__ == "__main__":
    main()
    # def prepare_sampling_feed_dict(self, curr_time_step):
    #     feed_dict = {}
    #     for tau in range(curr_time_step):
    #         total_sample_count = data.labelList.shape[0]
    #         sample_indices = np.random.choice(total_sample_count, state_sample_count, replace=False)
    #         sample_indices = np.repeat(sample_indices, repeats=samples_per_state)

    # def prepare_state_features(self, data):
    #     # Prepare Policy Gradients State Data
    #     root_node = [node for node in self.network.topologicalSortedNodes if node.isRoot]
    #     assert len(root_node) == 1
    #     for idx in range(data.labelList.shape[0]):
    #
    #     for tree_level in range(self.network.depth - 1):
    #         state_vectors_for_each_tree_level.append([])
    #         routes_per_sample.append([])
    #         route_combination_count.append([])
    #     for idx in range(routing_dataset.labelList.shape[0]):
    #         route_arr = greedy_routes[idx]
    #         for tree_level in range(self.network.depth - 1):
    #             # Gather all feature dicts
    #             level_nodes = self.network.orderedNodesPerLevel[tree_level]
    #             route_combinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(level_nodes))
    #             route_combinations = [route for route in route_combinations if sum(route) > 0]
    #             min_level_id = min([node.index for node in level_nodes])
    #             selected_node_id = route_arr[tree_level]
    #             valid_node_selections = set()
    #             for route in route_combinations:
    #                 r = np.array(route)
    #                 r[selected_node_id - min_level_id] = 1
    #                 valid_node_selections.add(tuple(r))
    #             route_combination_count[tree_level].append(len(valid_node_selections))
    #             for route_combination in valid_node_selections:
    #                 level_features_list = []
    #                 for feature_name in self.featuresUsed:
    #                     feature_vectors_per_node = [routing_dataset.get_dict(feature_name)[node.index][idx]
    #                                                 for node in level_nodes]
    #                     weighted_vectors = [route_weight * f_vec for route_weight, f_vec in
    #                                         zip(route_combination, feature_vectors_per_node)]
    #                     feature_vector = np.concatenate(weighted_vectors, axis=-1)
    #                     level_features_list.append(feature_vector)
    #                 state_vector_for_curr_level = np.concatenate(level_features_list, axis=-1)
    #                 state_vectors_for_each_tree_level[tree_level].append(state_vector_for_curr_level)
    #                 routes_per_sample[tree_level].append(route_combination)
    #     for arr in route_combination_count:
    #         assert len(set(arr)) == 1
    #     for tree_level in range(len(state_vectors_for_each_tree_level)):
    #         state_vectors_for_each_tree_level[tree_level] = np.stack(state_vectors_for_each_tree_level[tree_level],
    #                                                                  axis=0)
    #         routes_per_sample[tree_level] = np.stack(routes_per_sample[tree_level], axis=0)
    #     return state_vectors_for_each_tree_level, routes_per_sample
