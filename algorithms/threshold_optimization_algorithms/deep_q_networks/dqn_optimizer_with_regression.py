import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs


class DqnWithRegression:
    invalid_action_penalty = -1.0
    valid_prediction_reward = 1.0
    invalid_prediction_penalty = 0.0
    INCLUDE_IG_IN_REWARD_CALCULATIONS = True

    CONV_FEATURES = [64]
    FILTER_SIZES = [1]
    STRIDES = [1]
    HIDDEN_LAYERS = [128, 64]
    MAX_POOL = [None]

    def __init__(self, routing_dataset, network, network_name, run_id, used_feature_names, q_learning_func,
                 lambda_mac_cost, max_experience_count=100000):
        self.routingDataset = routing_dataset
        self.network = network
        self.networkName = network_name
        self.runId = run_id
        self.usedFeatureNames = used_feature_names
        self.qLearningFunc = q_learning_func
        self.lambdaMacCost = lambda_mac_cost
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        self.leafIndices = {node.index: idx for idx, node in enumerate(self.leafNodes)}
        # Data containers
        self.maxLikelihoodPaths = None
        self.stateFeatures = {}
        self.posteriorTensors = None
        self.actionSpaces = []
        self.networkActivationCosts = None
        self.networkActivationCostsDict = None
        self.baseEvaluationCost = 0.0
        self.reachabilityMatrices = None
        self.rewardTensors = None
        # Init data structures
        self.get_max_likelihood_paths()
        self.prepare_state_features()
        self.prepare_posterior_tensors()
        self.build_action_spaces()
        self.get_evaluation_costs()
        self.get_reachability_matrices()
        self.calculate_reward_tensors()
        # # Neural network components
        # self.experienceReplayTable = None
        # self.maxExpCount = max_experience_count
        # self.stateInput = None
        # self.qFunction = None
        # self.selectedQs = None
        # self.stateCount = tf.placeholder(dtype=tf.int32, name="stateCount")
        # self.stateRange = tf.range(0, self.stateCount, 1)
        # self.actionSelection = None
        # self.selectionMatrix = None
        # self.rewardVector = None
        # self.lossVector = None
        # self.lossValue = None
        # self.optimizer = None
        # self.totalLoss = None
        # self.globalStep = tf.Variable(0, name='global_step', trainable=False)
        # self.l2LambdaTf = tf.placeholder(dtype=tf.float32, name="l2LambdaTf")
        # self.l2Loss = None
        # self.build_q_function()
        # self.session = tf.Session()
        # # The following is for testing, can comment out later.
        # # self.test_likelihood_consistency()
        # print("X")

    # OK
    def get_max_trajectory_length(self) -> int:
        return int(self.network.depth - 1)

    # OK
    def get_max_likelihood_paths(self):
        branch_probs = self.routingDataset.get_dict("branch_probs")
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
        self.maxLikelihoodPaths = np.stack(max_likelihood_paths, axis=0)

    # OK
    def prepare_state_features(self):
        # if self.policyNetworkFunc == "mlp":
        #     super().prepare_state_features(data=data)
        # elif self.policyNetworkFunc == "cnn":
        root_node = [node for node in self.network.topologicalSortedNodes if node.isRoot]
        assert len(root_node) == 1
        for node in self.innerNodes:
            # array_list = [data.get_dict(feature_name)[node.index] for feature_name in self.networkFeatureNames]
            array_list = []
            for feature_name in self.usedFeatureNames:
                feature_arr = self.routingDataset.get_dict(feature_name)[node.index]
                if self.qLearningFunc == "mlp":
                    if len(feature_arr.shape) > 2:
                        shape_as_list = list(feature_arr.shape)
                        mean_axes = tuple([i for i in range(1, len(shape_as_list) - 1, 1)])
                        feature_arr = np.mean(feature_arr, axis=mean_axes)
                        assert len(feature_arr.shape) == 2
                elif self.qLearningFunc == "cnn":
                    assert len(feature_arr.shape) == 4
                array_list.append(feature_arr)
            feature_vectors = np.concatenate(array_list, axis=-1)
            self.stateFeatures[node.index] = feature_vectors

    # OK
    def prepare_posterior_tensors(self):
        self.posteriorTensors = \
            np.stack([self.routingDataset.get_dict("posterior_probs")[node.index] for node in self.leafNodes], axis=2)

    # OK
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

    # OK
    def get_evaluation_costs(self):
        path_costs = []
        for node in self.leafNodes:
            leaf_ancestors = self.network.dagObject.ancestors(node=node)
            leaf_ancestors.append(node)
            path_costs.append(sum([self.routingDataset.nodeCosts[ancestor.index] for ancestor in leaf_ancestors]))
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
            total_cost = sum([self.routingDataset.nodeCosts[n_idx] for n_idx in processed_nodes_set])
            self.networkActivationCosts.append(total_cost)
            self.networkActivationCostsDict[tuple(self.actionSpaces[-1][action_id])] = \
                (total_cost / self.baseEvaluationCost) - 1.0
        self.networkActivationCosts = (np.array(self.networkActivationCosts) * (1.0 / self.baseEvaluationCost)) - 1.0

    # OK
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

    # OK
    def calculate_reward_tensors(self):
        invalid_action_penalty = DqnWithRegression.invalid_action_penalty
        valid_prediction_reward = DqnWithRegression.valid_prediction_reward
        invalid_prediction_penalty = DqnWithRegression.invalid_prediction_penalty
        self.rewardTensors = []
        label_list = self.routingDataset.labelList
        sample_count = label_list.shape[0]
        posteriors_tensor = self.posteriorTensors
        for t in range(self.get_max_trajectory_length()):
            action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
            action_count_t = self.actionSpaces[t].shape[0]
            reward_shape = (sample_count, action_count_t_minus_one, action_count_t)
            rewards_arr = np.zeros(shape=reward_shape, dtype=np.float32)
            validity_of_actions_tensor = np.repeat(np.expand_dims(self.reachabilityMatrices[t], axis=0),
                                                   repeats=sample_count, axis=0)
            rewards_arr += (validity_of_actions_tensor == 0.0).astype(np.float32) * invalid_action_penalty
            if t == self.get_max_trajectory_length() - 1:
                true_labels = label_list
                # Prediction Rewards:
                # Calculate the prediction results for every state and for every routing decision
                prediction_correctness_vec_list = []
                calculation_cost_vec_list = []
                min_leaf_id = min([node.index for node in self.network.orderedNodesPerLevel[t + 1]])
                ig_indices = self.maxLikelihoodPaths[:, -1] - min_leaf_id
                for action_id in range(self.actionSpaces[t].shape[0]):
                    routing_decision = self.actionSpaces[t][action_id, :]
                    routing_matrix = np.repeat(routing_decision[np.newaxis, :], axis=0,
                                               repeats=true_labels.shape[0])
                    if DqnWithRegression.INCLUDE_IG_IN_REWARD_CALCULATIONS:
                        # Set Information Gain routed leaf nodes to 1. They are always evaluated.
                        routing_matrix[np.arange(true_labels.shape[0]), ig_indices] = 1
                    weights = np.reciprocal(np.sum(routing_matrix, axis=1).astype(np.float32))
                    routing_matrix_weighted = weights[:, np.newaxis] * routing_matrix
                    assert routing_matrix.shape[1] == posteriors_tensor.shape[2]
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
                rewards_arr += (prediction_correctness_tensor == 1).astype(np.float32) * valid_prediction_reward
                rewards_arr += (prediction_correctness_tensor == 0).astype(
                    np.float32) * invalid_prediction_penalty
                rewards_arr -= self.lambdaMacCost * computation_overload_tensor
            self.rewardTensors.append(rewards_arr)
