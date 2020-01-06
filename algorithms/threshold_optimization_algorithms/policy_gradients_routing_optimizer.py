import numpy as np
import tensorflow as tf
from algorithms.threshold_optimization_algorithms.combinatorial_routing_optimizer import CombinatorialRoutingOptimizer
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork


class TreeLevelRoutingOptimizer:
    def __init__(self, branching_state_vectors, hidden_layers, action_space_size):
        self.branchingStateVectors = branching_state_vectors
        # self.hiddenLayers = [self.branchingStateVectors.shape[-1]]
        # self.hiddenLayers.extend(hidden_layers)
        self.hiddenLayers = hidden_layers
        self.inputs = tf.placeholder(dtype=tf.int64, shape=[None, self.branchingStateVectors.shape[-1]],
                                     name="inputs")
        self.actionCount = int(action_space_size)
        self.hiddenLayers.append(self.actionCount)
        # Policy MLP
        self.net = self.inputs
        for layer_dim in self.hiddenLayers:
            self.net = tf.layers.dense(inputs=self.net, units=layer_dim, activation=tf.nn.relu)
        self.pi = tf.nn.softmax(self.net)


class PolicyGradientsRoutingOptimizer(CombinatorialRoutingOptimizer):
    def __init__(self, network_name, run_id, iteration, degree_list, data_type, output_names, test_ratio, features_used,
                 hidden_layers):
        super().__init__(network_name, run_id, iteration, degree_list, data_type, output_names)
        self.testRatio = test_ratio
        self.featuresUsed = features_used
        self.hiddenLayers = hidden_layers
        # Apply Validation - Test split to the routing data
        self.validationData, self.testData = self.routingData.apply_validation_test_split(test_ratio=self.testRatio)
        # Greedy Information Gain Paths
        self.validationDataPaths = self.get_max_likelihood_paths(
            branch_probs=self.validationData.get_dict("branch_probs"))
        self.testDataPaths = self.get_max_likelihood_paths(branch_probs=self.testData.get_dict("branch_probs"))
        self.validationStateFeatures = self.prepare_features_for_dataset(routing_dataset=self.validationData,
                                                                         greedy_routes=self.validationDataPaths)
        self.testStateFeatures = self.prepare_features_for_dataset(routing_dataset=self.testData,
                                                                   greedy_routes=self.testDataPaths)

        for level in range(self.network.depth - 1):
            if level != self.network.depth - 2:
                continue
            # self.reward_function(states=self.validationStateFeatures[level], labels=self.validationData.labelList,
            #                      routes=self.validationDataPaths, level=level)
            # self.reward_function(states=self.testStateFeatures[level], labels=self.testData.labelList,
            #                      routes=self.testDataPaths, level=level)

        self.policyGradientOptimizers = []
        for tree_level in range(self.network.depth, 1, -1):
            action_space_size = self.get_action_space_size(tree_level=tree_level)
            policy_gradient_optimizer = TreeLevelRoutingOptimizer(
                branching_state_vectors=self.validationStateFeatures[tree_level],
                hidden_layers=hidden_layers[tree_level], action_space_size=action_space_size)
            self.policyGradientOptimizers.append(policy_gradient_optimizer)
            print(tree_level)

    def get_action_space_size(self, tree_level):
        level_node_count = 2 ** (tree_level - 1)
        action_space_size = 2 ** (level_node_count - 1)
        return action_space_size

    # def reward_function(self, states, labels, routes, level):
    #     if level != self.network.depth - 2:
    #         raise NotImplementedError()
    #     assert states.shape[0] == labels.shape[0] == routes.shape[0]
    #     rewards = []
    #     for idx in range(states.shape[0]):
    #         route

    def prepare_features_for_dataset(self, routing_dataset, greedy_routes):
        # Prepare Policy Gradients State Data
        root_node = [node for node in self.network.topologicalSortedNodes if node.isRoot]
        assert len(root_node) == 1
        state_vectors_for_each_tree_level = []
        route_combination_count = None
        for tree_level in range(self.network.depth - 1):
            state_vectors_for_each_tree_level.append([])
        for idx in range(routing_dataset.labelList.shape[0]):
            route_arr = greedy_routes[idx]
            for tree_level in range(self.network.depth - 1):
                # Gather all feature dicts
                level_nodes = [node for node in self.network.topologicalSortedNodes if node.depth == tree_level]
                level_nodes = sorted(level_nodes, key=lambda node: node.index)
                route_combinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(level_nodes))
                route_combinations = [route for route in route_combinations if sum(route) > 0]
                min_level_id = min([node.index for node in level_nodes])
                selected_node_id = route_arr[tree_level]
                valid_node_selections = set()
                for route in route_combinations:
                    r = np.array(route)
                    r[selected_node_id - min_level_id] = 1
                    valid_node_selections.add(tuple(r))
                if route_combination_count is None:
                    route_combination_count = len(valid_node_selections)
                else:
                    assert route_combination_count == len(valid_node_selections)
                for route_combination in valid_node_selections:
                    level_features_list = []
                    for feature_name in self.featuresUsed:
                        feature_vectors_per_node = [routing_dataset.get_dict(feature_name)[node.index][idx]
                                                    for node in level_nodes]
                        weighted_vectors = [route_weight * f_vec for route_weight, f_vec in
                                            zip(route_combination, feature_vectors_per_node)]
                        feature_vector = np.concatenate(weighted_vectors, axis=-1)
                        level_features_list.append(feature_vector)
                    state_vector_for_curr_level = np.concatenate(level_features_list, axis=-1)
                    state_vectors_for_each_tree_level[tree_level].append(state_vector_for_curr_level)
                print("X")
        return None

        # while True:
        #     if any([node.isLeaf for node in curr_level_nodes]):
        #         break
        #     # Gather all feature dicts
        #     route_combinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(curr_level_nodes))
        #     route_combinations = [route for route in route_combinations if sum(route) > 0]
        #     sample_features_tensor = []
        #     for route_combination in route_combinations:
        #         level_features_list = []
        #         for feature_name in self.featuresUsed:
        #             feature_matrices_per_node = [routing_dataset.get_dict(feature_name)[node.index]
        #                                          for node in curr_level_nodes]
        #             weighted_matrices = [route_weight * f_matrix for route_weight, f_matrix in
        #                                  zip(route_combination, feature_matrices_per_node)]
        #             feature_matrix = np.concatenate(weighted_matrices, axis=-1)
        #             level_features_list.append(feature_matrix)
        #         level_features = np.concatenate(level_features_list, axis=-1)
        #         sample_features_tensor.append(level_features)
        #     sample_features_tensor = np.stack(sample_features_tensor, axis=-1)
        #     state_vectors_for_each_tree_level.append(sample_features_tensor)
        #     # Forward to the next level.
        #     curr_level += 1
        #     next_level_node_ids = set()
        #     for curr_level_node in curr_level_nodes:
        #         child_nodes = self.network.dagObject.children(node=curr_level_node)
        #         for child_node in child_nodes:
        #             next_level_node_ids.add(child_node.index)
        #     curr_level_nodes = [self.network.nodes[node_id] for node_id in sorted(list(next_level_node_ids))]
        # return state_vectors_for_each_tree_level

    # def prepare_features_for_dataset(self, routing_dataset, greedy_routes):
    #     # Prepare Policy Gradients State Data
    #     root_node = [node for node in self.network.topologicalSortedNodes if node.isRoot]
    #     assert len(root_node) == 1
    #     curr_level_nodes = root_node
    #     curr_level = 0
    #     state_vectors_for_each_tree_level = []
    #     while True:
    #         if any([node.isLeaf for node in curr_level_nodes]):
    #             break
    #         # Gather all feature dicts
    #         route_combinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(curr_level_nodes))
    #         route_combinations = [route for route in route_combinations if sum(route) > 0]
    #         sample_features_tensor = []
    #         for route_combination in route_combinations:
    #             level_features_list = []
    #             for feature_name in self.featuresUsed:
    #                 feature_matrices_per_node = [routing_dataset.get_dict(feature_name)[node.index]
    #                                              for node in curr_level_nodes]
    #                 weighted_matrices = [route_weight * f_matrix for route_weight, f_matrix in
    #                                      zip(route_combination, feature_matrices_per_node)]
    #                 feature_matrix = np.concatenate(weighted_matrices, axis=-1)
    #                 level_features_list.append(feature_matrix)
    #             level_features = np.concatenate(level_features_list, axis=-1)
    #             sample_features_tensor.append(level_features)
    #         sample_features_tensor = np.stack(sample_features_tensor, axis=-1)
    #         state_vectors_for_each_tree_level.append(sample_features_tensor)
    #         # Forward to the next level.
    #         curr_level += 1
    #         next_level_node_ids = set()
    #         for curr_level_node in curr_level_nodes:
    #             child_nodes = self.network.dagObject.children(node=curr_level_node)
    #             for child_node in child_nodes:
    #                 next_level_node_ids.add(child_node.index)
    #         curr_level_nodes = [self.network.nodes[node_id] for node_id in sorted(list(next_level_node_ids))]
    #     return state_vectors_for_each_tree_level


def main():
    run_id = 715
    network_name = "Cifar100_CIGN_MultiGpuSingleLateExit"
    iteration = 119100
    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature"]
    policy_gradients_routing_optimizer = PolicyGradientsRoutingOptimizer(network_name=network_name, run_id=run_id,
                                                                         iteration=iteration,
                                                                         degree_list=[2, 2], data_type="test",
                                                                         output_names=output_names,
                                                                         test_ratio=0.2,
                                                                         features_used=["branching_feature"],
                                                                         hidden_layers=[[128], [256]])


main()
