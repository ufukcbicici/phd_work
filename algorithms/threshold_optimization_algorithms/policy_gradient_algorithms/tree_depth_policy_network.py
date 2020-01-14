import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.policy_gradients_network import \
    PolicyGradientsNetwork


class TreeDepthState:
    def __init__(self, state_id, state_vec, max_likelihood_selection):
        self.id = state_id
        self.stateVector = state_vec
        self.maxLikelihoodSelection = max_likelihood_selection


class TreeDepthPolicyNetwork(PolicyGradientsNetwork):
    def __init__(self, l2_lambda,
                 network_name, run_id, iteration, degree_list, data_type, output_names,
                 test_ratio=0.2):
        super().__init__(l2_lambda, network_name, run_id, iteration, degree_list, data_type, output_names,
                         test_ratio=test_ratio)

    def prepare_state_features(self, data):
        # Prepare Policy Gradients State Data
        root_node = [node for node in self.network.topologicalSortedNodes if node.isRoot]
        assert len(root_node) == 1
        features_dict = {}
        for node in self.innerNodes:
            array_list = [data.get_dict(feature_name)[node.index] for feature_name in self.networkFeatureNames]
            feature_vectors = np.concatenate(array_list, axis=-1)
            features_dict[node.index] = feature_vectors
        return features_dict

    def sample_initial_states(self, data, features_dict, state_sample_count, samples_per_state):
        total_sample_count = data.labelList.shape[0]
        sample_indices = np.random.choice(total_sample_count, state_sample_count, replace=False)
        sample_indices = np.repeat(sample_indices, repeats=samples_per_state)
        feature_arr = features_dict[self.network.topologicalSortedNodes[0].index]
        initial_states = features_dict[sample_indices, :]

    def state_transition(self, history):
        pass


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