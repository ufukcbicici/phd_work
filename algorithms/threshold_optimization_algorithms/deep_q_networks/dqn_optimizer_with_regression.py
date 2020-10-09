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
        self.maxLikelihoodPaths = None
        self.stateFeatures = {}
        self.posteriorTensors = None
        self.actionSpaces = []
        self.networkActivationCosts = None
        self.networkActivationCostsDict = None
        self.baseEvaluationCost = 0.0
        self.reachabilityMatrices = []
        self.rewardTensors = None
        self.optimalQTables = []
        # Init data structures
        self.get_max_likelihood_paths()
        self.prepare_state_features()
        self.prepare_posterior_tensors()
        self.build_action_spaces()
        self.get_evaluation_costs()
        self.get_reachability_matrices()
        self.calculate_reward_tensors()
        # # Neural network components
        self.stateInputs = [None] * self.get_max_trajectory_length()
        self.qFuncs = [None] * self.get_max_trajectory_length()
        self.selectedQValues = [None] * self.get_max_trajectory_length()
        self.stateCount = tf.placeholder(dtype=tf.int32, name="stateCount")
        self.stateRange = tf.range(0, self.stateCount, 1)
        self.actionSelections = [None] * self.get_max_trajectory_length()
        self.selectionIndices = [None] * self.get_max_trajectory_length()
        self.rewardMatrices = [None] * self.get_max_trajectory_length()
        self.lossMatrices = [None] * self.get_max_trajectory_length()
        self.regressionLossValues = [None] * self.get_max_trajectory_length()
        self.optimizers = [None] * self.get_max_trajectory_length()
        self.totalLosses = [None] * self.get_max_trajectory_length()
        self.globalSteps = [tf.Variable(0, name='global_step_{0}'.format(idx), trainable=False)
                            for idx in range(self.get_max_trajectory_length())]
        self.l2LambdaTf = tf.placeholder(dtype=tf.float32, name="l2LambdaTf")
        self.l2Losses = [None] * self.get_max_trajectory_length()
        for level in range(self.get_max_trajectory_length()):
            self.build_q_function(level=level)
        self.lrBoundaries = [5000, 10000, 20000]
        self.lrValues = [0.1, 0.01, 0.001, 0.0001]
        # self.learningRate = tf.train.piecewise_constant(self.globalStep, self.lrBoundaries, self.lrValues)
        self.session = tf.Session()
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

    def build_cnn_q_network(self, level):
        hidden_layers = DqnWithRegression.HIDDEN_LAYERS[level]
        hidden_layers.append(self.actionSpaces[level].shape[0])
        conv_features = DqnWithRegression.CONV_FEATURES[level]
        filter_sizes = DqnWithRegression.FILTER_SIZES[level]
        strides = DqnWithRegression.STRIDES[level]
        pools = DqnWithRegression.MAX_POOL[level]
        net = self.stateInputs[level]
        conv_layer_id = 0
        for conv_feature, filter_size, stride, max_pool in zip(conv_features, filter_sizes, strides, pools):
            in_filters = net.get_shape().as_list()[-1]
            out_filters = conv_feature
            kernel = [filter_size, filter_size, in_filters, out_filters]
            strides = [1, stride, stride, 1]
            W = tf.get_variable("conv_layer_kernel_{0}".format(conv_layer_id), kernel, trainable=True)
            b = tf.get_variable("conv_layer_bias_{0}".format(conv_layer_id), [kernel[-1]], trainable=True)
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
        self.qFuncs[level] = net

    def get_l2_loss(self, level):
        # L2 Loss
        tvars = tf.trainable_variables(scope="dqn_{0}".format(level))
        self.l2Losses[level] = tf.constant(0.0)
        for tv in tvars:
            if 'kernel' in tv.name:
                self.l2Losses[level] += self.l2LambdaTf * tf.nn.l2_loss(tv)
            # self.paramL2Norms[tv.name] = tf.nn.l2_loss(tv)

    def build_loss(self, level):
        # Get selected q values; build the regression loss: MSE or Huber between Last layer Q outputs and the reward
        self.rewardMatrices[level] = tf.placeholder(dtype=tf.float32, shape=[None, self.actionSpaces[level].shape[0]],
                                                    name="reward_matrix_{0}".format(level))
        self.lossMatrices[level] = tf.square(self.qFuncs[level] - self.rewardMatrices[level])
        self.regressionLossValues[level] = tf.reduce_mean(self.lossMatrices[level])
        self.get_l2_loss(level=level)
        self.totalLosses[level] = self.regressionLossValues[level] + self.l2Losses[level]
        self.optimizers[level] = tf.train.AdamOptimizer().minimize(self.totalLosses[level],
                                                                   global_step=self.globalSteps[level])
        # self.optimizer = tf.train.MomentumOptimizer(self.learningRate, 0.9).\
        #     minimize(self.totalLoss, global_step=self.globalStep)

    def get_max_likelihood_accuracy(self, sample_indices):
        min_leaf_id = min([node.index for node in self.network.orderedNodesPerLevel[self.network.depth - 1]])
        posteriors = self.posteriorTensors[sample_indices]
        ml_indices = self.maxLikelihoodPaths[sample_indices][:, -1] - min_leaf_id
        true_labels = self.routingDataset.labelList[sample_indices]
        selected_posteriors = posteriors[np.arange(posteriors.shape[0]), :, ml_indices]
        predicted_labels = np.argmax(selected_posteriors, axis=1)
        correct_count = np.sum((predicted_labels == true_labels).astype(np.float32))
        accuracy = correct_count / true_labels.shape[0]
        return accuracy

    # OK
    def build_q_function(self, level):
        with tf.variable_scope("dqn_{0}".format(level)):
            if self.qLearningFunc == "cnn":
                nodes_at_level = self.network.orderedNodesPerLevel[level]
                shapes_list = [self.stateFeatures[node.index].shape for node in nodes_at_level]
                assert len(set(shapes_list)) == 1
                entry_shape = list(shapes_list[0])
                entry_shape[0] = None
                entry_shape[-1] = len(nodes_at_level) * entry_shape[-1]
                self.stateInputs[level] = tf.placeholder(dtype=tf.float32, shape=entry_shape,
                                                         name="state_inputs_{0}".format(level))
                self.build_cnn_q_network(level=level)
                self.build_loss(level=level)

    def log_meta_data(self, kwargs):
        # If we use only information gain for routing (ML: Maximum likelihood routing)
        whole_data_ml_accuracy = self.get_max_likelihood_accuracy(sample_indices=np.arange(
            self.routingDataset.labelList.shape[0]))
        training_ml_accuracy = self.get_max_likelihood_accuracy(sample_indices=self.routingDataset.trainingIndices)
        test_ml_accuracy = self.get_max_likelihood_accuracy(sample_indices=self.routingDataset.testIndices)
        # Fill the explanation string for the experiment
        kwargs["whole_data_ml_accuracy"] = whole_data_ml_accuracy
        kwargs["training_ml_accuracy"] = training_ml_accuracy
        kwargs["test_ml_accuracy"] = test_ml_accuracy
        kwargs["invalid_action_penalty"] = DqnWithRegression.invalid_action_penalty
        kwargs["valid_prediction_reward"] = DqnWithRegression.valid_prediction_reward
        kwargs["invalid_prediction_penalty"] = DqnWithRegression.invalid_prediction_penalty
        kwargs["INCLUDE_IG_IN_REWARD_CALCULATIONS"] = DqnWithRegression.INCLUDE_IG_IN_REWARD_CALCULATIONS
        kwargs["CONV_FEATURES"] = DqnWithRegression.CONV_FEATURES
        kwargs["HIDDEN_LAYERS"] = DqnWithRegression.HIDDEN_LAYERS
        kwargs["FILTER_SIZES"] = DqnWithRegression.FILTER_SIZES
        kwargs["STRIDES"] = DqnWithRegression.STRIDES
        kwargs["MAX_POOL"] = DqnWithRegression.MAX_POOL
        kwargs["lambdaMacCost"] = self.lambdaMacCost
        run_id = DbLogger.get_run_id()
        explanation_string = "DQN Experiment. RunID:{0}\n".format(run_id)
        for k, v in kwargs.items():
            explanation_string += "{0}:{1}\n".format(k, v)
        print("Whole Data ML Accuracy{0}".format(whole_data_ml_accuracy))
        print("Training Set ML Accuracy:{0}".format(training_ml_accuracy))
        print("Test Set ML Accuracy:{0}".format(test_ml_accuracy))
        DbLogger.write_into_table(rows=[(run_id, explanation_string)], table=DbLogger.runMetaData, col_count=2)
        return run_id

    def get_state_features(self, sample_indices, action_ids_t_minus_1, level):
        nodes_in_level = self.network.orderedNodesPerLevel[level]
        assert len({len(sample_indices), len(action_ids_t_minus_1)}) == 1
        routing_decisions = np.array([1] * len(sample_indices)) if level == 0 else \
            self.actionSpaces[level - 1][action_ids_t_minus_1]
        weighted_feature_arrays = [routing_decisions[:, idx] * self.stateFeatures[node.index][sample_indices]
                                   for idx, node in enumerate(nodes_in_level)]
        features = np.concatenate(weighted_feature_arrays, axis=-1)
        return features

    def create_q_table(self, level, sample_indices, action_ids_t_minus_1, batch_size=5000):
        q_values = []
        assert len({len(sample_indices), len(action_ids_t_minus_1)}) == 1
        for offset in range(0, sample_indices.shape[0], batch_size):
            start_idx = offset
            end_idx = min(start_idx + offset, sample_indices.shape[0])
            sample_indices_batch = sample_indices[start_idx: end_idx]
            action_ids_t_minus_1_batch = action_ids_t_minus_1[start_idx: end_idx]
            state_features = self.get_state_features(sample_indices=sample_indices_batch,
                                                     action_ids_t_minus_1=action_ids_t_minus_1_batch,
                                                     level=level)
            q_vals = self.session.run([self.qFuncs[level]], feed_dict={self.stateInputs[level]: state_features})[0]
            q_values.append(q_vals)
        q_values = np.concatenate(q_values, axis=0)
        return q_values

    def calculate_q_tables_with_dqn(self, discount_rate, dqn_lowest_level=np.inf):
        q_tables = [None] * self.get_max_trajectory_length()
        last_level = self.get_max_trajectory_length()
        total_sample_count = self.rewardTensors[0].shape[0]
        for t in range(last_level - 1, -1, -1):
            action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
            if t >= dqn_lowest_level:
                state_id_tuples = UtilityFuncs.get_cartesian_product(
                    list_of_lists=[list(range(total_sample_count)),
                                   list(range(action_count_t_minus_one))])
                q_table_predicted = self.create_q_table(level=t, sample_indices=state_id_tuples[:, 0],
                                                        action_ids_t_minus_1=state_id_tuples[:, 1])
                # Reshape for further processing
                assert q_table_predicted.shape[0] == total_sample_count * action_count_t_minus_one \
                       and len(q_table_predicted.shape) == 2
                q_table_predicted = np.reshape(q_table_predicted,
                                               newshape=(total_sample_count, action_count_t_minus_one,
                                                         q_table_predicted.shape[1]))
                q_tables[t] = q_table_predicted
            else:
                # Get the rewards for that time step
                if t == last_level - 1:
                    q_tables[t] = self.rewardTensors[t]
                else:
                    rewards_t = self.rewardTensors[t]
                    q_next = self.optimalQTables[t + 1]
                    q_star = np.max(q_next, axis=-1)
                    q_t = rewards_t + discount_rate * q_star[:, np.newaxis, :]
                    q_tables[t] = q_t
        return q_tables

    def calculate_q_tables_for_test(self, discount_rate, dqn_lowest_level=np.inf):
        last_level = self.get_max_trajectory_length()
        total_sample_count = self.rewardTensors[0].shape[0]
        q_tables = [None] * self.get_max_trajectory_length()
        for t in range(last_level - 1, -1, -1):
            action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
            action_count_t = self.actionSpaces[t].shape[0]
            if t >= dqn_lowest_level:
                state_id_tuples = UtilityFuncs.get_cartesian_product(
                    list_of_lists=[list(range(total_sample_count)),
                                   list(range(action_count_t_minus_one))])
                q_table_predicted = self.create_q_table(level=t, sample_indices=state_id_tuples[:, 0],
                                                        action_ids_t_minus_1=state_id_tuples[:, 1])
                # Reshape for further processing
                assert q_table_predicted.shape[0] == total_sample_count * action_count_t_minus_one \
                       and len(q_table_predicted.shape) == 2
                q_table_predicted = np.reshape(q_table_predicted,
                                               newshape=(total_sample_count, action_count_t_minus_one,
                                                         q_table_predicted.shape[1]))
                q_tables[t] = q_table_predicted
            else:
                q_tables[t] = np.zeros(shape=(total_sample_count, action_count_t_minus_one, action_count_t))
                all_state_tuples = UtilityFuncs.get_cartesian_product(
                    [range(total_sample_count), [a_t_minus_one for a_t_minus_one in range(action_count_t_minus_one)]])
                for s_id, a_t_minus_1 in all_state_tuples:
                    for a_t in range(action_count_t):
                        # E[r_{t+1}] = \sum_{r_{t+1}}r_{t+1}p(r_{t+1}|s_{t},a_{t})
                        # Since in our case p(r_{t+1}|s_{t},a_{t}) is deterministic, it is a lookup into the rewards table.
                        # s_{t}: (sample_id, iteration, a_{t-1})
                        r_t = self.rewardTensors[t][s_id, a_t_minus_1, a_t]
                        q_t = r_t
                        if t < last_level - 1:
                            q_t_plus_1 = q_tables[t + 1][s_id, a_t]
                            q_t += np.max(q_t_plus_1)
                        q_tables[t][s_id, a_t_minus_1, a_t] = q_t
        return q_tables

    # Calculate the estimated Q-Table vs actual Q-table divergence and related scores for the given layer.
    def measure_performance(self, level, Q_table, sample_indices, action_ids_t_minus_1):
        assert len({len(sample_indices), len(action_ids_t_minus_1)}) == 1
        # Calculate the mean policy value
        mean_policy_value = np.mean(np.max(Q_table, axis=1))
        # Calculate the MSE between the Q_{t}^{predicted}(s,a) and Q_{t}^{actual}(s,a).
        y = []
        y_hat = []

    def execute_bellman_equation(self, level, sample_indices, action_ids_t_minus_1, discount_rate):
        last_level = self.get_max_trajectory_length() - 1
        # sample_ids_for_iterations = np.array([self.routingDataset.linkageInfo[(s_id, it)]
        #                                       for s_id, it in zip(sample_indices, iterations)])
        # Execute the Bellman Equation
        # Step 1: Get the Q*(s,a) for the last level.
        Q_table_T = self.create_q_table(level=level, sample_indices=sample_indices,
                                        action_ids_t_minus_1=action_ids_t_minus_1)
        # Calculate the mean policy value and the MSE for the provided samples
        # mean_policy_value, mse_score = self.measure_performance(sample_indices=sample_indices, iterations=iterations,
        #                                                         Q_table_T=Q_table_T)

        # Q_tables = {last_level: Q_table_T}
        # R_table_T = self.create_r_table(sample_indices=sample_indices, iterations=iterations)
        # R_tables = {last_level: R_table_T}
        # # Calculate the mean policy value and the MSE for the provided samples
        # mean_policy_value, mse_score = self.measure_performance(sample_indices=sample_indices, iterations=iterations,
        #                                                         Q_table_T=Q_table_T)
        # for t in range(last_level - 1, -1, -1):
        #     action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
        #     action_count_t = self.actionSpaces[t].shape[0]
        #     Q_tables[t] = {}
        #     R_tables[t] = {}
        #     all_state_tuples = UtilityFuncs.get_cartesian_product(
        #         [sample_indices, iterations, [a_t_minus_one for a_t_minus_one in range(action_count_t_minus_one)]])
        #     for s_id, it, a_t_minus_1 in all_state_tuples:
        #         sample_id_for_iteration = self.routingDataset.linkageInfo[(s_id, it)]
        #         Q_tables[t][(s_id, it, a_t_minus_1)] = np.array([np.nan] * action_count_t)
        #         R_tables[t][(s_id, it, a_t_minus_1)] = np.array([np.nan] * action_count_t)
        #         for a_t in range(action_count_t):
        #             # E[r_{t+1}] = \sum_{r_{t+1}}r_{t+1}p(r_{t+1}|s_{t},a_{t})
        #             # Since in our case p(r_{t+1}|s_{t},a_{t}) is deterministic, it is a lookup into the rewards table.
        #             # s_{t}: (sample_id, iteration, a_{t-1})
        #             r_t_plus_1 = self.rewardTensors[it][t][sample_id_for_iteration, a_t_minus_1, a_t]
        #             # \sum_{s_{t+1}} p(s_{t+1}|s_{t},a_{t}) max_{a_{t+1}} Q*(s_{t+1},a_{t+1})
        #             # Since in our case p(s_{t+1}|s_{t},a_{t}) is deterministic,
        #             # it is a lookup into the Q*(s_{t+1},a_{t+1}) table.
        #             # Again:
        #             # s_{t}: (sample_id, iteration, a_{t-1}). Then we have:
        #             # s_{t+1}: (sample_id, iteration, a_t)
        #             # Get the Q* values, belonging to s_{t+1}.
        #             q_values = Q_tables[t + 1][(s_id, it, a_t)]
        #             q_t = r_t_plus_1 + discount_rate * np.max(q_values)
        #             r_values = R_tables[t + 1][(s_id, it, a_t)]
        #             r_t = r_t_plus_1 + discount_rate * np.max(r_values)
        #             # Save the result into Q* table for the current time step; for the state tuple:
        #             # s_{t}: (sample_id, iteration, a_{t-1})
        #             Q_tables[t][(s_id, it, a_t_minus_1)][a_t] = q_t
        #             R_tables[t][(s_id, it, a_t_minus_1)][a_t] = r_t
        #     # Confirm that no nan entries left
        #     for v in Q_tables[t].values():
        #         assert np.sum(np.isnan(v)) == 0
        #     for v in R_tables[t].values():
        #         assert np.sum(np.isnan(v)) == 0
        # accuracy, computation_cost, optimal_accuracy, optimal_calculation_cost = \
        #     self.calculate_results_from_routing_decisions(
        #         sample_ids=sample_indices,
        #         iterations=iterations,
        #         Q_tables=Q_tables,
        #         R_tables=R_tables)
        # print("Mean Policy Value:{0}".format(accuracy))
        # print("MSE:{0}".format(mse_score))
        # print("Accuracy:{0}".format(accuracy))
        # print("Computation Cost:{0}".format(computation_cost))
        # print("Optimal Accuracy:{0}".format(optimal_accuracy))
        # print("Optimal Computation Cost:{0}".format(optimal_calculation_cost))
        # return mean_policy_value, mse_score, accuracy, computation_cost, optimal_accuracy, optimal_calculation_cost

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
        kwargs["lrValues"] = self.lrValues
        kwargs["lrBoundaries"] = self.lrBoundaries
        run_id = self.log_meta_data(kwargs=kwargs)
        losses = []
        # Calculate the ultimate, optimal Q Tables.
        self.optimalQTables = self.calculate_q_tables_with_dqn(discount_rate=discount_factor)
        optimal_q_tables_test = self.calculate_q_tables_for_test(discount_rate=discount_factor)
        assert len(self.optimalQTables) == len(optimal_q_tables_test)
        for t in range(len(self.optimalQTables)):
            assert np.allclose(self.optimalQTables[t], optimal_q_tables_test[t])
        print("X")

        # self.calculate_optimal_q_tables(discount_rate=discount_factor)
        # Test the accuracy evaluations
        # self.evaluate(run_id=run_id, episode_id=-1, discount_factor=discount_factor)
        # for episode_id in range(episode_count):
        #     print("Episode:{0}".format(episode_id))
        #     sample_ids = np.random.choice(self.routingDataset.trainingIndices, sample_count, replace=True)
        #     iterations = np.random.choice(self.routingDataset.iterations, sample_count, replace=True)
        #     actions_t_minus_1 = np.random.choice(self.actionSpaces[level - 1].shape[0], sample_count, replace=True)
        #     rewards_matrix = self.get_rewards(samples=sample_ids, iterations=iterations,
        #                                       action_ids_t_minus_1=actions_t_minus_1, action_ids_t=None, level=level)
        #     state_features = self.get_state_features(samples=sample_ids,
        #                                              iterations=iterations,
        #                                              action_ids_t_minus_1=actions_t_minus_1, level=level)
        #     results = self.session.run([self.totalLoss, self.lossMatrix, self.lossValue, self.optimizer],
        #                                feed_dict={self.stateCount: sample_count,
        #                                           self.stateInput: state_features,
        #                                           self.rewardMatrix: rewards_matrix,
        #                                           self.l2LambdaTf: 0.0})
        #     total_loss = results[0]
        #     losses.append(total_loss)
        #     if len(losses) % 10 == 0:
        #         print("Episode:{0} MSE:{1}".format(episode_id, np.mean(np.array(losses))))
        #         losses = []
        #     if (episode_id + 1) % 200 == 0:
        #         if (episode_id + 1) == 10000:
        #             print("X")
        #         self.evaluate(run_id=run_id, episode_id=episode_id, discount_factor=discount_factor)