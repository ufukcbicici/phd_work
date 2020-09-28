import numpy as np
import tensorflow as tf

from auxillary.general_utility_funcs import UtilityFuncs


class MultiIterationDQN:
    invalid_action_penalty = -1.0
    valid_prediction_reward = 1.0
    invalid_prediction_penalty = 0.0
    INCLUDE_IG_IN_REWARD_CALCULATIONS = True

    CONV_FEATURES = [[32], [64]]
    HIDDEN_LAYERS = [[128, 64], [128, 64]]
    FILTER_SIZES = [[1], [1]]
    STRIDES = [[1], [1]]
    MAX_POOL = [[None], [None]]

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
        self.maxLikelihoodPaths = {}
        self.stateFeatures = {}
        self.posteriorTensors = {}
        self.actionSpaces = []
        self.networkActivationCosts = []
        self.networkActivationCostsDict = {}
        self.baseEvaluationCost = 0.0
        self.reachabilityMatrices = []
        self.rewardTensors = {}
        # Init data structures
        self.get_max_likelihood_paths()
        self.prepare_state_features()
        self.prepare_posterior_tensors()
        self.build_action_spaces()
        self.get_evaluation_costs()
        self.get_reachability_matrices()
        self.calculate_reward_tensors()
        # Neural network components
        self.experienceReplayTable = None
        self.maxExpCount = max_experience_count
        self.stateInputs = []
        self.qFuncs = []
        self.selectedQValues = []
        self.stateCount = tf.placeholder(dtype=tf.int32, name="stateCount")
        self.stateRange = tf.range(0, self.stateCount, 1)
        self.actionSelections = []
        self.selectionIndices = []
        self.rewardVectors = []
        self.lossVectors = []
        self.lossValues = []
        self.optimizers = []
        self.totalLosses = []
        self.globalStep = tf.Variable(0, name='global_step', trainable=False)
        self.l2LambdaTf = tf.placeholder(dtype=tf.float32, name="l2LambdaTf")
        self.l2Loss = None
        for level in range(self.get_max_trajectory_length()):
            self.build_q_function(level=level)
        self.session = tf.Session()
        # The following is for testing, can comment out later.
        # self.test_likelihood_consistency()
        print("X")

    def get_max_trajectory_length(self) -> int:
        return int(self.network.depth - 1)

    def get_max_likelihood_paths(self):
        for iteration in self.routingDataset.iterations:
            branch_probs = self.routingDataset.dictOfDatasets[iteration].get_dict("branch_probs")
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
            self.maxLikelihoodPaths[iteration] = max_likelihood_paths

    def prepare_state_features(self):
        # if self.policyNetworkFunc == "mlp":
        #     super().prepare_state_features(data=data)
        # elif self.policyNetworkFunc == "cnn":
        root_node = [node for node in self.network.topologicalSortedNodes if node.isRoot]
        assert len(root_node) == 1
        for iteration in self.routingDataset.iterations:
            features_dict = {}
            for node in self.innerNodes:
                # array_list = [data.get_dict(feature_name)[node.index] for feature_name in self.networkFeatureNames]
                array_list = []
                for feature_name in self.usedFeatureNames:
                    feature_arr = self.routingDataset.dictOfDatasets[iteration].get_dict(feature_name)[node.index]
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
                features_dict[node.index] = feature_vectors
            self.stateFeatures[iteration] = features_dict

    def prepare_posterior_tensors(self):
        for iteration in self.routingDataset.iterations:
            self.posteriorTensors[iteration] = \
                np.stack([self.routingDataset.dictOfDatasets[iteration].
                         get_dict("posterior_probs")[node.index] for node in self.leafNodes], axis=2)

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
        invalid_action_penalty = MultiIterationDQN.invalid_action_penalty
        valid_prediction_reward = MultiIterationDQN.valid_prediction_reward
        invalid_prediction_penalty = MultiIterationDQN.invalid_prediction_penalty

        for iteration in self.routingDataset.iterations:
            self.rewardTensors[iteration] = []
            label_list = self.routingDataset.dictOfDatasets[iteration].labelList
            sample_count = label_list.shape[0]
            posteriors_tensor = self.posteriorTensors[iteration]
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
                    ig_indices = self.maxLikelihoodPaths[iteration][:, -1] - min_leaf_id
                    for action_id in range(self.actionSpaces[t].shape[0]):
                        routing_decision = self.actionSpaces[t][action_id, :]
                        routing_matrix = np.repeat(routing_decision[np.newaxis, :], axis=0,
                                                   repeats=true_labels.shape[0])
                        if MultiIterationDQN.INCLUDE_IG_IN_REWARD_CALCULATIONS:
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
                        np.expand_dims(prediction_correctness_matrix, axis=1), axis=1, repeats=action_count_t_minus_one)
                    computation_overload_matrix = np.stack(calculation_cost_vec_list, axis=1)
                    computation_overload_tensor = np.repeat(
                        np.expand_dims(computation_overload_matrix, axis=1), axis=1, repeats=action_count_t_minus_one)
                    # Add to the rewards tensor
                    rewards_arr += (prediction_correctness_tensor == 1).astype(np.float32) * valid_prediction_reward
                    rewards_arr += (prediction_correctness_tensor == 0).astype(np.float32) * invalid_prediction_penalty
                    rewards_arr -= self.lambdaMacCost * computation_overload_tensor
                self.rewardTensors[iteration].append(rewards_arr)

    def build_q_function(self, level):
        if level != self.get_max_trajectory_length() - 1:
            self.stateInputs.append(None)
            self.qFuncs.append(None)
            self.actionSelections.append(None)
            self.selectedQValues.append(None)
            self.selectionIndices.append(None)
            self.rewardVectors.append(None)
            self.lossVectors.append(None)
            self.lossValues.append(None)
            self.totalLosses.append(None)
            self.optimizers.append(None)
        else:
            if self.qLearningFunc == "cnn":
                nodes_at_level = self.network.orderedNodesPerLevel[level]
                shapes_list = [self.stateFeatures[iteration][node.index].shape
                               for iteration in self.routingDataset.iterations for node in nodes_at_level]
                assert len(set(shapes_list)) == 1
                entry_shape = list(shapes_list[0])
                entry_shape[0] = None
                entry_shape[-1] = len(nodes_at_level) * entry_shape[-1]
                tf_state_input = tf.placeholder(dtype=tf.float32, shape=entry_shape,
                                                name="state_inputs_{0}".format(level))
                self.stateInputs.append(tf_state_input)
                self.build_cnn_q_network(level=level)
            # Get selected q values; build the regression loss
            tf_selected_action_input = tf.placeholder(dtype=tf.int32, shape=[None],
                                                      name="action_inputs_{0}".format(level))
            self.actionSelections.append(tf_selected_action_input)
            selection_matrix = tf.stack([self.stateRange, tf_selected_action_input], axis=1)
            self.selectionIndices.append(selection_matrix)
            selected_q_values = tf.gather_nd(self.qFuncs[level], selection_matrix)
            self.selectedQValues.append(selected_q_values)
            reward_vector = tf.placeholder(dtype=tf.float32, shape=[None],
                                           name="reward_vector_{0}".format(level))
            self.rewardVectors.append(reward_vector)
            # Loss functions
            mse_vector = tf.square(selected_q_values - reward_vector)
            self.lossVectors.append(mse_vector)
            mse_loss = tf.reduce_mean(mse_vector)
            self.lossValues.append(mse_loss)
            self.get_l2_loss()
            total_loss = mse_loss + self.l2Loss
            self.totalLosses.append(total_loss)
            optimizer = tf.train.AdamOptimizer().minimize(total_loss, global_step=self.globalStep)
            self.optimizers.append(optimizer)

    def build_cnn_q_network(self, level):
        hidden_layers = MultiIterationDQN.HIDDEN_LAYERS[level]
        hidden_layers.append(self.actionSpaces[level].shape[0])
        conv_features = MultiIterationDQN.CONV_FEATURES[level]
        filter_sizes = MultiIterationDQN.FILTER_SIZES[level]
        strides = MultiIterationDQN.STRIDES[level]
        pools = MultiIterationDQN.MAX_POOL[level]

        with tf.variable_scope("dqn_level_{0}".format(level)):
            net = self.stateInputs[level]
            conv_layer_id = 0
            for conv_feature, filter_size, stride, max_pool in zip(conv_features, filter_sizes, strides, pools):
                in_filters = net.get_shape().as_list()[-1]
                out_filters = conv_feature
                kernel = [filter_size, filter_size, in_filters, out_filters]
                strides = [1, stride, stride, 1]
                W = tf.get_variable("conv_layer_kernel_{0}_t{1}".format(conv_layer_id, level), kernel,
                                    trainable=True)
                b = tf.get_variable("conv_layer_bias_{0}_t{1}".format(conv_layer_id, level), [kernel[-1]],
                                    trainable=True)
                net = tf.nn.conv2d(net, W, strides, padding='SAME')
                net = tf.nn.bias_add(net, b)
                net = tf.nn.relu(net)
                if max_pool is not None:
                    net = tf.nn.max_pool(net, ksize=[1, max_pool, max_pool, 1], strides=[1, max_pool, max_pool, 1],
                                         padding='SAME')
                conv_layer_id += 1
            # net = tf.contrib.layers.flatten(net)
            net_shape = net.get_shape().as_list()
            net = tf.nn.avg_pool(net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
            net_shape = net.get_shape().as_list()
            net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
            for layer_id, layer_dim in enumerate(hidden_layers):
                if layer_id < len(hidden_layers) - 1:
                    net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
                else:
                    net = tf.layers.dense(inputs=net, units=layer_dim, activation=None)
            q_values = net
            self.qFuncs.append(q_values)

    def get_l2_loss(self):
        # L2 Loss
        tvars = tf.trainable_variables()
        self.l2Loss = tf.constant(0.0)
        for tv in tvars:
            if 'kernel' in tv.name:
                self.l2Loss += self.l2LambdaTf * tf.nn.l2_loss(tv)
            # self.paramL2Norms[tv.name] = tf.nn.l2_loss(tv)

    def get_max_likelihood_accuracy(self, iterations, sample_indices):
        min_leaf_id = min([node.index for node in self.network.orderedNodesPerLevel[self.network.depth - 1]])
        predicted_labels_list = []
        true_labels_list = []
        for iteration in iterations:
            indices_for_iteration = np.array(list(map(lambda idx: self.routingDataset.linkageInfo[(idx, iteration)],
                                                      sample_indices)))
            posteriors_for_iteration = self.posteriorTensors[iteration][indices_for_iteration]
            ml_indices_for_iteration = self.maxLikelihoodPaths[iteration][indices_for_iteration]
            true_labels_for_iteration = self.routingDataset.dictOfDatasets[iteration].labelList[indices_for_iteration]
            ml_leaf_indices_for_iteration = ml_indices_for_iteration[:, -1] - min_leaf_id
            # Check indexing validity
            # selection_matrix = np.zeros(shape=(posteriors_for_iteration.shape[0], posteriors_for_iteration.shape[2]))
            # selection_matrix[np.arange(selection_matrix.shape[0]), ml_leaf_indices_for_iteration] = 1.0
            # selected_posteriors_v1 = np.sum(posteriors_for_iteration * np.expand_dims(selection_matrix, axis=1), axis=2)
            # selected_posteriors_v2 = posteriors_for_iteration[np.arange(posteriors_for_iteration.shape[0]), :,
            #                          ml_leaf_indices_for_iteration]
            # assert np.array_equal(selected_posteriors_v1, selected_posteriors_v2)
            selected_posteriors = posteriors_for_iteration[np.arange(posteriors_for_iteration.shape[0]), :,
                                  ml_leaf_indices_for_iteration]
            predicted_labels = np.argmax(selected_posteriors, axis=1)
            predicted_labels_list.append(predicted_labels)
            true_labels_list.append(true_labels_for_iteration)
        all_predicted_labels = np.concatenate(predicted_labels_list)
        all_true_labels = np.concatenate(true_labels_list)
        assert all_predicted_labels.shape == all_true_labels.shape
        correct_count = np.sum((all_predicted_labels == all_true_labels).astype(np.float32))
        accuracy = correct_count / all_true_labels.shape[0]
        return accuracy

    def get_state_features(self, samples, iterations, action_ids_t_minus_1, level):
        nodes_in_level = self.network.orderedNodesPerLevel[level]
        features = []
        for s_id, it, a_t_1 in zip(samples, iterations, action_ids_t_minus_1):
            sample_id_in_iteration = self.routingDataset.linkageInfo[(s_id, it)]
            route_decision = np.array([1]) if level == 0 else self.actionSpaces[level - 1][a_t_1]
            feature = [route_decision[idx] * self.stateFeatures[it][node.index][sample_id_in_iteration]
                       for idx, node in enumerate(nodes_in_level)]
            feature = np.concatenate(feature, axis=-1)
            features.append(feature)
        features = np.stack(features, axis=0)
        return features

    def get_rewards(self, samples, iterations, action_ids_t_minus_1, action_ids_t, level):
        rewards = []
        if action_ids_t is not None:
            for s_id, it, a_t_1, a_t in zip(samples, iterations, action_ids_t_minus_1, action_ids_t):
                sample_id_in_iteration = self.routingDataset.linkageInfo[(s_id, it)]
                reward_t = self.rewardTensors[it][level][sample_id_in_iteration, a_t_1, a_t]
                rewards.append(reward_t)
            rewards = np.array(rewards)
        else:
            for s_id, it, a_t_1 in zip(samples, iterations, action_ids_t_minus_1):
                sample_id_in_iteration = self.routingDataset.linkageInfo[(s_id, it)]
                reward_array = self.rewardTensors[it][level][sample_id_in_iteration, a_t_1]
                rewards.append(reward_array)
            rewards = np.array(rewards)
        return rewards

    def add_to_the_experience_table(self, experience_matrix):
        if self.experienceReplayTable is None:
            self.experienceReplayTable = np.copy(experience_matrix)
        else:
            self.experienceReplayTable = np.concatenate([self.experienceReplayTable, experience_matrix], axis=0)
            if self.experienceReplayTable.shape[0] > self.maxExpCount:
                num_rows_to_delete = self.experienceReplayTable.shape[0] - self.maxExpCount
                self.experienceReplayTable = self.experienceReplayTable[num_rows_to_delete:]
                assert self.experienceReplayTable.shape[0] == self.maxExpCount

    def fill_experience_replay_table(self, level, sample_count, epsilon):
        state_count = self.routingDataset.trainingIndices.shape[0]
        action_count_t_minus_one = 1 if level == 0 else self.actionSpaces[level - 1].shape[0]
        action_count_t = self.actionSpaces[level].shape[0]
        sample_ids = np.random.choice(self.routingDataset.trainingIndices, sample_count, replace=True)
        iterations = np.random.choice(self.routingDataset.iterations, sample_count, replace=True)
        actions_t_minus_1 = np.random.choice(action_count_t_minus_one, sample_count)
        # Build the current state: s_t
        state_matrix_t = np.stack([sample_ids, iterations, actions_t_minus_1], axis=1)
        # Get the state features for state_matrix_t
        state_features_t = self.get_state_features(samples=sample_ids, iterations=iterations,
                                                   action_ids_t_minus_1=actions_t_minus_1, level=level)
        # Check if the state features are correctly built
        self.test_state_features(state_features_t=state_features_t, actions_t_minus_1=actions_t_minus_1, level=level)
        # Get the Q Table for states: s_t
        Q_table = self.session.run([self.qFuncs[level]],
                                   feed_dict={
                                       self.stateInputs[level]: state_features_t})[0]
        # Sample a_t epsilon greedy for every state.
        # If 1, choose uniformly over all actions. If 0, choose the best action.
        epsilon_greedy_sampling_choices = np.random.choice(a=[0, 1], size=state_matrix_t.shape[0],
                                                           p=[1.0 - epsilon, epsilon])
        random_selection = np.random.choice(action_count_t, size=state_matrix_t.shape[0])
        greedy_selection = np.argmax(Q_table, axis=1)
        actions_t = np.where(epsilon_greedy_sampling_choices, random_selection, greedy_selection)
        rewards_t = self.get_rewards(samples=sample_ids, iterations=iterations, action_ids_t_minus_1=actions_t_minus_1,
                                     action_ids_t=actions_t, level=level)
        # Store into the experience replay table. Note that normally, we store (s_t,a_t,r_t,s_{t+1}) into the
        # experience replay table. But in our case, the state transition distribution p(s_{t+1}|s_t,a_t) is
        # deterministic. We can only store s_t,a_t and we can calculate s_{t+1} from this pair later.
        experience_matrix = np.concatenate([state_matrix_t,
                                            np.expand_dims(actions_t, axis=1),
                                            np.expand_dims(rewards_t, axis=1)], axis=1)
        self.add_to_the_experience_table(experience_matrix=experience_matrix)

    def train(self, level, **kwargs):
        sample_count = kwargs["sample_count"]
        episode_count = kwargs["episode_count"]
        discount_factor = kwargs["discount_factor"]
        epsilon_discount_factor = kwargs["epsilon_discount_factor"]
        learning_rate = kwargs["learning_rate"]
        epsilon = 1.0
        if level != self.get_max_trajectory_length() - 1:
            raise NotImplementedError()
        self.session.run(tf.global_variables_initializer())
        # If we use only information gain for routing (ML: Maximum likelihood routing)
        print("Whole Data ML Accuracy{0}".format(
            self.get_max_likelihood_accuracy(iterations=self.routingDataset.iterations,
                                             sample_indices=np.arange(self.routingDataset.labelList.shape[0]))))
        print("Training Set ML Accuracy:{0}".format(
            self.get_max_likelihood_accuracy(iterations=self.routingDataset.iterations,
                                             sample_indices=self.routingDataset.trainingIndices)))
        print("Test Set ML Accuracy:{0}".format(
            self.get_max_likelihood_accuracy(iterations=self.routingDataset.testIterations,
                                             sample_indices=self.routingDataset.testIndices)))
        # Fill the experience replay table: Solve the cold start problem.
        self.fill_experience_replay_table(level=level, sample_count=10 * sample_count, epsilon=epsilon)
        losses = []
        for episode_id in range(episode_count):
            self.fill_experience_replay_table(level=level, sample_count=sample_count, epsilon=epsilon)
            # Sample batch of experiences from the table
            experience_ids = np.random.choice(self.experienceReplayTable.shape[0], sample_count, replace=False)
            experiences_sampled = self.experienceReplayTable[experience_ids]
            sampled_state_tuples = experiences_sampled[:, 0:3].astype(np.int32)
            sampled_actions = experiences_sampled[:, 3].astype(np.int32)
            sampled_rewards = experiences_sampled[:, 4]
            # Add Gradient Descent Step
            # Get the state features for state_matrix_t
            state_features_t = self.get_state_features(samples=sampled_state_tuples[:, 0],
                                                       iterations=sampled_state_tuples[:, 1],
                                                       action_ids_t_minus_1=sampled_state_tuples[:, 2], level=level)
            # results = self.session.run(
            #     [self.qFuncs[level],
            #      self.selectionIndices[level],
            #      self.selectedQValues[level],
            #      self.lossVectors[level],
            #      self.lossValues[level],
            #      self.totalLosses[level],
            #      self.l2Loss], feed_dict={self.stateCount: experiences_sampled.shape[0],
            #                               self.stateInputs[level]: state_features_t,
            #                               self.actionSelections[level]: sampled_actions,
            #                               self.rewardVectors[level]: sampled_rewards,
            #                               self.l2LambdaTf: 0.0})
            results = self.session.run([self.totalLosses[level], self.optimizers[level]],
                                       feed_dict={self.stateCount: experiences_sampled.shape[0],
                                                  self.stateInputs[level]: state_features_t,
                                                  self.actionSelections[level]: sampled_actions,
                                                  self.rewardVectors[level]: sampled_rewards,
                                                  self.l2LambdaTf: 0.0})
            # Adjust epsilon
            epsilon *= epsilon_discount_factor
            total_loss = results[0]
            losses.append(total_loss)
            if len(losses) % 100 == 0:
                self.measure_performance(level=level, losses=losses, data_type="validation")
                self.measure_performance(level=level, losses=losses, data_type="test")
                losses = []

    def get_all_possible_state_features(self, sample_indices, iterations, level):
        action_count_t_minus_one = 1 if level == 0 else self.actionSpaces[level - 1].shape[0]
        # Enumerate all possible states
        all_state_tuples = UtilityFuncs.get_cartesian_product(
            [sample_indices, iterations, [a_t_minus_one for a_t_minus_one in range(action_count_t_minus_one)]])
        complete_state_matrix = np.array(all_state_tuples)
        complete_state_features = self.get_state_features(
            samples=complete_state_matrix[:, 0], iterations=complete_state_matrix[:, 1],
            action_ids_t_minus_1=complete_state_matrix[:, 2], level=level)
        return complete_state_features, complete_state_matrix

    def create_q_table(self, sample_indices, iterations, level):
        # Enumerate all possible states
        complete_state_features, complete_state_matrix = self.get_all_possible_state_features(
            sample_indices=sample_indices,
            iterations=iterations, level=level)
        assert complete_state_features.shape[0] == complete_state_matrix.shape[0]
        # Get q-values for every state
        q_vals = self.session.run([self.qFuncs[level]], feed_dict={self.stateInputs[level]: complete_state_features})[0]
        assert q_vals.shape[0] == complete_state_matrix.shape[0]
        q_table = {}
        for state_tuple, q_val in zip(complete_state_matrix, q_vals):
            q_table[tuple(state_tuple)] = q_val
        return q_table

    def execute_bellman_equation(self, sample_indices, iterations, discount_rate):
        last_level = self.get_max_trajectory_length() - 1
        # sample_ids_for_iterations = np.array([self.routingDataset.linkageInfo[(s_id, it)]
        #                                       for s_id, it in zip(sample_indices, iterations)])
        # Execute the Bellman Equation
        # Step 1: Get the Q*(s,a) for the last level.
        # TODO: Later, implement this in a vectorized way.
        Q_table_T = self.create_q_table(sample_indices=sample_indices, iterations=iterations, level=last_level)
        Q_tables = {last_level: Q_table_T}
        for t in range(last_level - 1, -1, -1):
            action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
            action_count_t = self.actionSpaces[t].shape[0]
            q_table_t = {t: {}}

            # all_state_tuples = UtilityFuncs.get_cartesian_product(
            #     [sample_indices, iterations, [a_t_minus_one for a_t_minus_one in range(action_count_t_minus_one)]])
            # complete_state_matrix = np.array(all_state_tuples)

            for s_id, it, a_t_minus_1 in zip(sample_indices, iterations, range(action_count_t_minus_one)):
                sample_id_for_iteration = self.routingDataset.linkageInfo[(s_id, it)]
                q_table_t[t][(s_id, it, a_t_minus_1)] = np.array([np.nan] * action_count_t)
                for a_t in range(action_count_t):
                    # E[r_{t+1}] = \sum_{r_{t+1}}r_{t+1}p(r_{t+1}|s_{t},a_{t})
                    # Since in our case p(r_{t+1}|s_{t},a_{t}) is deterministic, it is a lookup into the rewards table.
                    # s_{t}: (sample_id, iteration, a_{t-1})
                    r_t_plus_1 = self.rewardTensors[it][t][sample_id_for_iteration, a_t_minus_1, a_t]
                    # \sum_{s_{t+1}} p(s_{t+1}|s_{t},a_{t}) max_{a_{t+1}} Q*(s_{t+1},a_{t+1})
                    # Since in our case p(s_{t+1}|s_{t},a_{t}) is deterministic,
                    # it is a lookup into the Q*(s_{t+1},a_{t+1}) table.
                    # Again:
                    # s_{t}: (sample_id, iteration, a_{t-1}). Then we have:
                    # s_{t+1}: (sample_id, iteration, a_t)
                    # Get the Q* values, belonging to s_{t+1}.
                    q_values = Q_tables[t+1][(s_id, it, a_t)]
                    q_t = r_t_plus_1 + discount_rate*np.max(q_values)
                    # Save the result into Q* table for the current time step; for the state tuple:
                    # s_{t}: (sample_id, iteration, a_{t-1})
                    q_table_t[t][(s_id, it, a_t_minus_1)][a_t] = q_t








        # # Enumerate all possible states
        # complete_state_features, complete_state_matrix = self.get_all_possible_state_features(
        #     sample_indices=sample_indices,
        #     iterations=iterations, level=last_level)
        # Q_table_T = self.session.run([self.qFuncs[last_level]], feed_dict={self.stateInputs[last_level]:
        #                                                                        complete_state_features})[0]
        # Q_tables = []
        # Q_t_plus_one = np.copy(Q_table_T)
        # assert np.prod(Q_t_plus_one.shape) == len(sample_indices) * len(iterations) * action_count_t_minus_one * \
        #        action_count_t
        # Q_tables.append(Q_t_plus_one)

        # for t in range(last_level - 1, -1, -1):
        #     action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
        #     action_count_t = self.actionSpaces[t].shape[0]
        #     all_state_tuples = UtilityFuncs.get_cartesian_product(
        #         [sample_indices, iterations, [a_t_minus_one for a_t_minus_one in range(action_count_t_minus_one)]])
        #     complete_state_features, complete_state_matrix = self.get_all_possible_state_features(
        #         sample_indices=sample_indices,
        #         iterations=iterations, level=t)
        #     expected_rewards = self.get_rewards(samples=complete_state_matrix[:, 0],
        #                                         iterations=complete_state_matrix[:, 1],
        #                                         action_ids_t_minus_1=complete_state_matrix[:, 2],
        #                                         action_ids_t=None, level=t)
        #     Q_t = []
        #     for a_t in range(action_count_t):
        #         # E[r_{t+1}] = \sum_{r_{t+1}}r_{t+1}p(r_{t+1}|s_{t},a_{t})
        #         # Since in our case p(r_{t+1}|s_{t},a_{t}) is deterministic, it is a lookup into the rewards table.
        #         r_t_plus_1 = expected_rewards[:, a_t]
        #         # \sum_{s_{t+1}} p(s_{t+1}|s_{t},a_{t}) max_{a_{t+1}} Q*(s_{t+1},a_{t+1})
        #         # Since in our case p(s_{t+1}|s_{t},a_{t}) is deterministic,
        #         # it is a lookup into the Q*(s_{t+1},a_{t+1}) table.
        #         Q_tables[-1]






        print("X")

        # dataset = self.validationDataForMDP if data_type == "validation" else self.testDataForMDP
        # last_level = self.get_max_trajectory_length() - 1
        # state_count = dataset.routingDataset.labelList.shape[0]
        # action_count_t_minus_one = 1 if last_level == 0 else self.actionSpaces[last_level - 1].shape[0]
        # action_count_t = self.actionSpaces[last_level].shape[0]
        # rewards = self.validationRewards if data_type == "validation" else self.testRewards
        # # Execute the Bellman Equation
        # # Step 1: Get the Q*(s,a) for the last level.
        # state_ids = np.array([sample_id for sample_id in range(state_count)])
        # complete_state_matrix = UtilityFuncs.get_cartesian_product(
        #     [state_ids, [a_t_minus_one for a_t_minus_one in range(action_count_t_minus_one)]])
        # complete_state_matrix = np.array(complete_state_matrix)
        # complete_state_features = self.get_state_features(state_matrix=complete_state_matrix, level=last_level,
        #                                                   data_type=data_type)
        # Q_table_T = self.session.run([self.qFuncs[last_level]],
        #                              feed_dict={self.stateInputs[last_level]: complete_state_features})[0]
        # Q_tables = []
        # Q_t_plus_one = np.copy(Q_table_T)
        # assert np.prod(Q_t_plus_one.shape) == state_count * action_count_t_minus_one * action_count_t
        # Q_t_plus_one = np.reshape(Q_t_plus_one, newshape=(state_count, action_count_t_minus_one, action_count_t))
        # Q_tables.append(Q_t_plus_one)
        # for t in range(last_level - 1, -1, -1):
        #     action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
        #     action_count_t = self.actionSpaces[t].shape[0]
        #     states_matrix = UtilityFuncs.get_cartesian_product(
        #         [[sample_id for sample_id in range(state_count)],
        #          [a_t_minus_one for a_t_minus_one in range(action_count_t_minus_one)]])
        #     states_matrix = np.array(states_matrix)
        #     expected_rewards = rewards[t][states_matrix[:, 0], states_matrix[:, 1], :]
        #     Q_t = []
        #     for a in range(action_count_t):
        #         # E[r_{t+1}] = \sum_{r_{t+1}}r_{t+1}p(r_{t+1}|s_{t},a_{t})
        #         # Since in our case p(r_{t+1}|s_{t},a_{t}) is deterministic, it is a lookup into the rewards table.
        #         reward_vector = expected_rewards[:, a]
        #         # \sum_{s_{t+1}} p(s_{t+1}|s_{t},a_{t}) max_{a_{t+1}} Q*(s_{t+1},a_{t+1})
        #         # Since in our case p(s_{t+1}|s_{t},a_{t}) is deterministic,
        #         # it is a lookup into the Q*(s_{t+1},a_{t+1}) table.
        #         Q_t_plus_one_given_a = Q_t_plus_one[state_ids, a * np.ones_like(state_ids), :]
        #         assert reward_vector.shape[0] == Q_t_plus_one_given_a.shape[0] * action_count_t_minus_one
        #         Q_t_plus_one_given_a = np.repeat(Q_t_plus_one_given_a, axis=0, repeats=action_count_t_minus_one)
        #         q_values = np.max(Q_t_plus_one_given_a, axis=1)
        #         assert reward_vector.shape[0] == q_values.shape[0]
        #         Q_t.append(reward_vector + q_values)
        #     Q_t = np.stack(Q_t, axis=1)
        #     Q_t_plus_one = np.reshape(Q_t, newshape=(state_count, action_count_t_minus_one, action_count_t))
        #     Q_tables.append(Q_t_plus_one)
        # Q_tables.reverse()
        # # Now we are ready to calculate the optimal trajectories and calculate the optimal accuracy and computation load
        # # for every state.
        # states_matrix = UtilityFuncs.get_cartesian_product(
        #     [[sample_id for sample_id in range(state_count)], [0]])
        # states_matrix = np.array(states_matrix)
        # for t in range(self.get_max_trajectory_length()):
        #     Q_table = Q_tables[t][states_matrix[:, 0], states_matrix[:, 1], :]
        #     actions_t = np.argmax(Q_table, axis=1)
        #     states_matrix[:, 1] = actions_t
        #     print("Bellman Decision Distribution Level:{0}:{1}".format(t, Counter(states_matrix[:, 1])))
        # # The last layer actions determine the routing decisions.
        # routing_decisions = self.actionSpaces[-1][states_matrix[:, 1], :]
        # truth_vector = self.calculate_results_from_routing_decisions(states_matrix[:, 0], routing_decisions, data_type)
        # self.calculate_results_by_sampling(q_tables=Q_tables, state_ids=states_matrix[:, 0], data_type=data_type,
        #                                    sample_count=1000)
        # return routing_decisions, truth_vector

    # Test methods
    def test_likelihood_consistency(self):
        for idx in range(self.routingDataset.labelList.shape[0]):
            path_array = []
            for iteration in self.routingDataset.iterations:
                iteration_id = self.routingDataset.linkageInfo[(idx, iteration)]
                path_array.append(self.maxLikelihoodPaths[iteration][iteration_id])
            path_array = np.stack(path_array, axis=0)
            print("X")

    def test_state_features(self, state_features_t, actions_t_minus_1, level):
        s_t = np.mean(state_features_t, (1, 2))
        s_t = np.stack([np.mean(s_t[:, 0:s_t.shape[1] // 2], axis=1),
                        np.mean(s_t[:, s_t.shape[1] // 2:s_t.shape[1]], axis=1)], axis=1)
        r_t = np.stack([self.actionSpaces[level - 1][idx] for idx in actions_t_minus_1], axis=0)
        assert np.array_equal(s_t != 0, r_t)
