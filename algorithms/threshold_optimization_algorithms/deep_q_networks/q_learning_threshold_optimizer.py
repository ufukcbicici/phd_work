import numpy as np
import tensorflow as tf

from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.policy_gradients_network import \
    RoutingDataForMDP
from auxillary.general_utility_funcs import UtilityFuncs


class QLearningThresholdOptimizer:
    invalid_action_penalty = -100.0
    valid_prediction_reward = 100.0
    invalid_prediction_penalty = 0.0
    INCLUDE_IG_IN_REWARD_CALCULATIONS = True

    def __init__(self, validation_data, test_data, network, network_name, run_id,
                 used_feature_names, q_learning_func, lambda_mac_cost):
        self.validationData = validation_data
        self.testData = test_data
        self.network = network
        self.networkName = network_name
        self.runId = run_id
        self.usedFeatureNames = used_feature_names
        self.qLearningFunc = q_learning_func
        self.actionSpaces = None
        self.reachabilityMatrices = []
        self.baseEvaluationCost = None
        self.networkActivationCosts = None
        self.networkActivationCostsDict = None
        self.lambdaMacCost = lambda_mac_cost
        self.validationRewards = []
        self.testRewards = []
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        self.leafIndices = {node.index: idx for idx, node in enumerate(self.leafNodes)}
        self.validationData = validation_data
        self.testData = test_data
        self.validationMLPaths = self.get_max_likelihood_paths(
            branch_probs=self.validationData.get_dict("branch_probs"))
        self.testMLPaths = self.get_max_likelihood_paths(branch_probs=self.testData.get_dict("branch_probs"))
        self.validationFeaturesDict = self.prepare_state_features(data=self.validationData)
        self.testFeaturesDict = self.prepare_state_features(data=self.testData)
        self.validationPosteriorsTensor = \
            np.stack([self.validationData.get_dict("posterior_probs")[node.index] for node in self.leafNodes], axis=2)
        self.testPosteriorsTensor = \
            np.stack([self.testData.get_dict("posterior_probs")[node.index] for node in self.leafNodes], axis=2)
        self.validationDataForMDP = RoutingDataForMDP(
            routing_dataset=self.validationData,
            features_dict=self.validationFeaturesDict,
            ml_paths=self.validationMLPaths,
            posteriors_tensor=self.validationPosteriorsTensor)
        self.testDataForMDP = RoutingDataForMDP(
            routing_dataset=self.testData,
            features_dict=self.testFeaturesDict,
            ml_paths=self.testMLPaths,
            posteriors_tensor=self.testPosteriorsTensor)
        self.build_action_spaces()
        self.get_evaluation_costs()
        self.get_reachability_matrices()
        self.calculate_reward_tensors()

    def get_max_likelihood_paths(self, branch_probs):
        sample_sizes = list(set([arr.shape[0] for arr in branch_probs.values()]))
        assert len(sample_sizes) == 1
        sample_size = sample_sizes[0]
        max_likelihood_paths = []
        for idx in range(sample_size):
            curr_node = self.network.topologicalSortedNodes[0]
            route = []
            while True:
                route.append(curr_node.index)
                if curr_node.isLeaf:
                    break
                routing_distribution = branch_probs[curr_node.index][idx]
                arg_max_child_index = np.argmax(routing_distribution)
                child_nodes = self.network.dagObject.children(node=curr_node)
                child_nodes = sorted(child_nodes, key=lambda c_node: c_node.index)
                curr_node = child_nodes[arg_max_child_index]
            max_likelihood_paths.append(np.array(route))
        max_likelihood_paths = np.stack(max_likelihood_paths, axis=0)
        return max_likelihood_paths

    def prepare_state_features(self, data):
        # if self.policyNetworkFunc == "mlp":
        #     super().prepare_state_features(data=data)
        # elif self.policyNetworkFunc == "cnn":
        root_node = [node for node in self.network.topologicalSortedNodes if node.isRoot]
        assert len(root_node) == 1
        features_dict = {}
        for node in self.innerNodes:
            # array_list = [data.get_dict(feature_name)[node.index] for feature_name in self.networkFeatureNames]
            array_list = []
            for feature_name in self.usedFeatureNames:
                feature_arr = data.get_dict(feature_name)[node.index]
                if self.qLearningFunc == "mlp":
                    if len(feature_arr.shape) > 2:
                        shape_as_list = list(feature_arr.shape)
                        mean_axes = tuple([i for i in range(1, len(shape_as_list) - 1, 1)])
                        feature_arr = np.mean(feature_arr, axis=mean_axes)
                elif self.qLearningFunc == "cnn":
                    assert len(feature_arr.shape) == 4
                array_list.append(feature_arr)
            feature_vectors = np.concatenate(array_list, axis=-1)
            features_dict[node.index] = feature_vectors
        return features_dict

    def get_max_trajectory_length(self) -> int:
        return int(self.network.depth - 1)

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

    def get_evaluation_costs(self):
        list_of_lists = []
        path_costs = []
        for node in self.leafNodes:
            list_of_lists.append([0, 1])
            leaf_ancestors = self.network.dagObject.ancestors(node=node)
            leaf_ancestors.append(node)
            path_costs.append(sum([self.network.nodeCosts[ancestor.index] for ancestor in leaf_ancestors]))
        self.baseEvaluationCost = np.mean(np.array(path_costs))
        self.networkActivationCosts = []
        self.networkActivationCostsDict = {}
        for action_id in range(self.actionSpaces[-1].shape[0]):
            node_selection = self.actionSpaces[-1][action_id]
            processed_nodes_set = set()
            for node_idx, curr_node in enumerate(self.leafNodes):
                if node_selection[node_idx] == 0:
                    continue
                leaf_ancestors = self.network.dagObject.ancestors(node=curr_node)
                leaf_ancestors.append(curr_node)
                for ancestor in leaf_ancestors:
                    processed_nodes_set.add(ancestor.index)
            total_cost = sum([self.network.nodeCosts[n_idx] for n_idx in processed_nodes_set])
            self.networkActivationCosts.append(total_cost)
            self.networkActivationCostsDict[tuple(self.actionSpaces[-1][action_id])] = \
                (total_cost / self.baseEvaluationCost) - 1.0
        self.networkActivationCosts = (np.array(self.networkActivationCosts) * (1.0 / self.baseEvaluationCost)) - 1.0

    def get_reachability_matrices(self):
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
                    selected_nodes_t = [node for i, node in enumerate(self.network.orderedNodesPerLevel[t])
                                        if node_selection_vec_t_minus_one[i] != 0]
                    next_level_nodes = self.network.orderedNodesPerLevel[t + 1]
                    reachable_next_level_node_ids = set()
                    next_level_reached_dict = {}
                    for parent_node in selected_nodes_t:
                        child_nodes = {c_node.index for c_node in self.network.dagObject.children(node=parent_node)}
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

    def calculate_reward_tensors(self):
        invalid_action_penalty = QLearningThresholdOptimizer.invalid_action_penalty
        valid_prediction_reward = QLearningThresholdOptimizer.valid_prediction_reward
        invalid_prediction_penalty = QLearningThresholdOptimizer.invalid_prediction_penalty

        for t in range(self.get_max_trajectory_length()):
            action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
            action_count_t = self.actionSpaces[t].shape[0]
            for dataset in [self.validationDataForMDP, self.testDataForMDP]:
                reward_shape = (dataset.routingDataset.labelList.shape[0], action_count_t_minus_one, action_count_t)
                rewards_arr = np.zeros(shape=reward_shape, dtype=np.float32)
                validity_of_actions_tensor = np.repeat(
                    np.expand_dims(self.reachabilityMatrices[t], axis=0),
                    repeats=dataset.routingDataset.labelList.shape[0], axis=0)
                rewards_arr += (validity_of_actions_tensor == 0.0).astype(np.float32) * invalid_action_penalty
                if t == self.get_max_trajectory_length() - 1:
                    true_labels = dataset.routingDataset.labelList
                    # Prediction Rewards:
                    # Calculate the prediction results for every state and for every routing decision
                    prediction_correctness_vec_list = []
                    calculation_cost_vec_list = []
                    min_leaf_id = min([node.index for node in self.network.orderedNodesPerLevel[t + 1]])
                    ig_indices = dataset.mlPaths[:, -1] - min_leaf_id
                    for action_id in range(self.actionSpaces[t].shape[0]):
                        routing_decision = self.actionSpaces[t][action_id, :]
                        routing_matrix = np.repeat(routing_decision[np.newaxis, :], axis=0,
                                                   repeats=true_labels.shape[0])
                        if QLearningThresholdOptimizer.INCLUDE_IG_IN_REWARD_CALCULATIONS:
                            # Set Information Gain routed leaf nodes to 1. They are always evaluated.
                            routing_matrix[np.arange(true_labels.shape[0]), ig_indices] = 1
                        weights = np.reciprocal(np.sum(routing_matrix, axis=1).astype(np.float32))
                        routing_matrix_weighted = weights[:, np.newaxis] * routing_matrix
                        assert routing_matrix.shape[1] == dataset.posteriorsTensor.shape[2]
                        weighted_posteriors = dataset.posteriorsTensor * routing_matrix_weighted[:, np.newaxis, :]
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
                        np.expand_dims(prediction_correctness_matrix, axis=1), axis=1, repeats=action_count_t_minus_one)
                    computation_overload_matrix = np.stack(calculation_cost_vec_list, axis=1)
                    computation_overload_tensor = np.repeat(
                        np.expand_dims(computation_overload_matrix, axis=1), axis=1, repeats=action_count_t_minus_one)
                    # Add to the rewards tensor
                    rewards_arr += (prediction_correctness_tensor == 1).astype(np.float32) * valid_prediction_reward
                    rewards_arr += (prediction_correctness_tensor == 0).astype(np.float32) * invalid_prediction_penalty
                    rewards_arr -= self.lambdaMacCost * computation_overload_tensor
                if dataset == self.validationDataForMDP:
                    self.validationRewards.append(rewards_arr)
                else:
                    self.testRewards.append(rewards_arr)
        self.validationDataForMDP.rewardTensors = self.validationRewards
        self.testDataForMDP.rewardTensors = self.testRewards

    # Classic Off Policy Q Learning Implementation
    def train(self, level, **kwargs):
        if level != self.get_max_trajectory_length() - 1:
            raise NotImplementedError()

        episode_count = kwargs["episode_count"]
        discount_factor = kwargs["discount_factor"]
        epsilon_discount_factor = kwargs["epsilon_discount_factor"]
        epsilon = 0.6
        sample_count = self.validationDataForMDP.routingDataset.labelList.shape[0]
        action_count_t_minus_one = 1 if level == 0 else self.actionSpaces[level - 1].shape[0]
        action_count_t = self.actionSpaces[level].shape[0]
        Q_table = np.zeros(shape=(sample_count * action_count_t_minus_one, action_count_t), dtype=np.float32)
        rewards_tensor = self.validationRewards[level]

        # Enumerate all state combinations
        state_list = UtilityFuncs.get_cartesian_product(
            [[sample_id for sample_id in range(sample_count)],
             [a_t_minus_one for a_t_minus_one in range(action_count_t_minus_one)]])
        state_matrix = np.array(state_list)
        rewards_matrix = rewards_tensor[state_matrix[:, 0], state_matrix[:, 1], :]
        # Check if we have correctly built the rewards matrix
        assert all([
            np.array_equal(rewards_tensor[s_id],
                           rewards_matrix[action_count_t_minus_one * s_id:action_count_t_minus_one * (s_id + 1)])
            for s_id in range(sample_count)])
        for episode_id in range(episode_count):
            # Sample epsilon greedy for every state.
            # If 1, choose uniformly over all actions. If 0, choose the best action.
            epsilon_greedy_sampling_choices = np.random.choice(a=[0, 1], size=len(state_list),
                                                               p=[1.0 - epsilon, epsilon])
            random_selection = np.random.choice(action_count_t, size=len(state_list))
            greedy_selection = np.argmax(Q_table, axis=1)
            selected_actions = np.where(epsilon_greedy_sampling_choices, random_selection, greedy_selection)
            # This was to control that np.where() works as intended
            # assert all([selected_actions[idx] == random_selection[idx] if epsilon_greedy_sampling_choices[idx] == 1 else
            #             selected_actions[idx] == greedy_selection[idx] for idx in
            #             range(epsilon_greedy_sampling_choices.shape[0])])
            print("X")

            # for state in state_list:
            #     print("X")
            epsilon *= epsilon_discount_factor

        print("X")
