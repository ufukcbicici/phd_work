import numpy as np
import tensorflow as tf
from collections import deque, Counter

from algorithms.threshold_optimization_algorithms.deep_q_networks.q_learning_threshold_optimizer import \
    QLearningThresholdOptimizer
from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.policy_gradients_network import \
    RoutingDataForMDP
from auxillary.general_utility_funcs import UtilityFuncs


class DeepQThresholdOptimizer(QLearningThresholdOptimizer):
    CONV_FEATURES = [[32], [64]]
    HIDDEN_LAYERS = [[128, 64], [128, 64]]
    FILTER_SIZES = [[1], [1]]
    STRIDES = [[1], [1]]
    MAX_POOL = [[None], [None]]

    def __init__(self, validation_data, test_data, network, network_name, run_id, used_feature_names, q_learning_func,
                 lambda_mac_cost, max_experience_count=100000):
        super().__init__(validation_data, test_data, network, network_name, run_id, used_feature_names, q_learning_func,
                         lambda_mac_cost)
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

    def add_to_the_experience_table(self, experience_matrix):
        if self.experienceReplayTable is None:
            self.experienceReplayTable = np.copy(experience_matrix)
        else:
            self.experienceReplayTable = np.concatenate([self.experienceReplayTable, experience_matrix], axis=0)
            if self.experienceReplayTable.shape[0] > self.maxExpCount:
                num_rows_to_delete = self.experienceReplayTable.shape[0] - self.maxExpCount
                self.experienceReplayTable = self.experienceReplayTable[num_rows_to_delete:]

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
                entry_shape = list(self.validationFeaturesDict[nodes_at_level[0].index].shape)
                entry_shape[0] = None
                entry_shape[-1] = len(nodes_at_level) * entry_shape[-1]
                tf_state_input = tf.placeholder(dtype=tf.float32, shape=entry_shape, name="inputs_{0}".format(level))
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
        hidden_layers = DeepQThresholdOptimizer.HIDDEN_LAYERS[level]
        hidden_layers.append(self.actionSpaces[level].shape[0])
        conv_features = DeepQThresholdOptimizer.CONV_FEATURES[level]
        filter_sizes = DeepQThresholdOptimizer.FILTER_SIZES[level]
        strides = DeepQThresholdOptimizer.STRIDES[level]
        pools = DeepQThresholdOptimizer.MAX_POOL[level]

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

    def get_state_features(self, state_matrix, level, type="validation"):
        assert type in {"validation", "test"}
        route_decisions = np.zeros(shape=(state_matrix.shape[0],)) if level == 0 else \
            self.actionSpaces[level - 1][state_matrix[:, 1]]
        list_of_feature_tensors = [self.validationFeaturesDict[node.index] if type == "validation" else
                                   self.testFeaturesDict[node.index] for node in
                                   self.network.orderedNodesPerLevel[level]]
        list_of_sampled_tensors = [feature_tensor[state_matrix[:, 0], :] for feature_tensor in list_of_feature_tensors]
        list_of_coeffs = []
        for idx in range(len(list_of_sampled_tensors)):
            route_coeffs = route_decisions[:, idx]
            for _ in range(len(list_of_sampled_tensors[idx].shape) - 1):
                route_coeffs = np.expand_dims(route_coeffs, axis=-1)
            list_of_coeffs.append(route_coeffs)
        list_of_sparse_tensors = [feature_tensor * coeff_arr
                                  for feature_tensor, coeff_arr in zip(list_of_sampled_tensors, list_of_coeffs)]
        # This is for debugging
        # manuel_route_matrix = np.stack([np.array([int(np.sum(tensor) != 0) for tensor in _l])
        #                                 for _l in list_of_sparse_tensors], axis=1)
        # assert np.array_equal(route_decisions, manuel_route_matrix)
        state_features = np.concatenate(list_of_sparse_tensors, axis=-1)
        return state_features

    # def sample_trajectory(self, level, sample_count, epsilon, type="validation"):
    #     state_count = self.validationDataForMDP.routingDataset.labelList.shape[0]
    #     action_count_t_minus_one = 1 if level == 0 else self.actionSpaces[level - 1].shape[0]
    #     state_ids = np.random.choice(state_count, sample_count, replace=False)
    #     action_ids = np.random.choice(action_count_t_minus_one, sample_count)
    #     state_matrix = np.stack([state_ids, action_ids], axis=1)
    #     state_features = self.get_state_features(state_matrix=state_matrix, level=level, type=type)
    #     state_q_values = self.session.run([self.qFuncs[level]],
    #                                       feed_dict={
    #                                           self.stateInputs[level]: state_features})[0]
    #     rewards_tensor = self.validationRewards[level]
    #     rewards_matrix = rewards_tensor[state_matrix[:, 0], state_matrix[:, 1], :]
    #
    #     # route_decisions = np.zeros(shape=(state_matrix.shape[0],)) if level == 0 else \
    #     #     self.actionSpaces[level - 1][state_matrix[:, 1]]
    #     print("X")

    def train(self, level, **kwargs):
        if level != self.get_max_trajectory_length() - 1:
            raise NotImplementedError()
        self.session.run(tf.global_variables_initializer())
        self.evaluate_ml_routing_accuracies()
        sample_count = kwargs["sample_count"]
        episode_count = kwargs["episode_count"]
        discount_factor = kwargs["discount_factor"]
        epsilon_discount_factor = kwargs["epsilon_discount_factor"]
        learning_rate = kwargs["learning_rate"]
        # Fill the experience replay table: Solve the cold start problem.
        # self.sample_trajectory(level=level, sample_count=sample_count)
        state_count = self.validationDataForMDP.routingDataset.labelList.shape[0]
        action_count_t_minus_one = 1 if level == 0 else self.actionSpaces[level - 1].shape[0]
        action_count_t = self.actionSpaces[level].shape[0]
        rewards_tensor = self.validationRewards[level]
        epsilon = 1.0
        losses = []
        for episode_id in range(episode_count):
            print("episode_id:{0}".format(episode_id))
            # Sample (state,action,reward) tuples and store in the experience replay table
            state_ids = np.random.choice(state_count, sample_count, replace=False)
            action_ids = np.random.choice(action_count_t_minus_one, sample_count)
            state_matrix = np.stack([state_ids, action_ids], axis=1)
            rewards_matrix = rewards_tensor[state_matrix[:, 0], state_matrix[:, 1], :]
            state_features = self.get_state_features(state_matrix=state_matrix, level=level, type="validation")
            Q_table = self.session.run([self.qFuncs[level]],
                                       feed_dict={
                                           self.stateInputs[level]: state_features})[0]
            # Sample epsilon greedy for every state.
            # If 1, choose uniformly over all actions. If 0, choose the best action.
            epsilon_greedy_sampling_choices = np.random.choice(a=[0, 1], size=state_matrix.shape[0],
                                                               p=[1.0 - epsilon, epsilon])
            random_selection = np.random.choice(action_count_t, size=state_matrix.shape[0])
            greedy_selection = np.argmax(Q_table, axis=1)
            selected_actions = np.where(epsilon_greedy_sampling_choices, random_selection, greedy_selection)
            rewards = rewards_matrix[np.arange(state_matrix.shape[0]), selected_actions]
            # Store into the experience replay table
            experience_matrix = np.concatenate([state_matrix,
                                                np.expand_dims(selected_actions, axis=1),
                                                np.expand_dims(rewards, axis=1)], axis=1)
            self.add_to_the_experience_table(experience_matrix=experience_matrix)
            # Sample batch of experiences from the table
            experience_ids = np.random.choice(self.experienceReplayTable.shape[0], sample_count, replace=False)
            experiences_sampled = self.experienceReplayTable[experience_ids]
            sampled_state_ids = experiences_sampled[:, 0:2].astype(np.int32)
            sampled_actions = experiences_sampled[:, 2].astype(np.int32)
            sampled_rewards = experiences_sampled[:, 3]
            # Add Gradient Descent Step
            sampled_state_features = self.get_state_features(state_matrix=sampled_state_ids, level=level,
                                                             type="validation")
            # results = self.session.run(
            #     [self.qFuncs[level],
            #      self.selectionIndices[level],
            #      self.selectedQValues[level],
            #      self.lossVectors[level],
            #      self.lossValues[level],
            #      self.totalLosses[level],
            #      self.l2Loss], feed_dict={self.stateCount: experiences_sampled.shape[0],
            #                               self.stateInputs[level]: sampled_state_features,
            #                               self.actionSelections[level]: sampled_actions,
            #                               self.rewardVectors[level]: sampled_rewards,
            #                               self.l2LambdaTf: 0.0})
            results = self.session.run([self.totalLosses[level], self.optimizers[level]],
                                       feed_dict={self.stateCount: experiences_sampled.shape[0],
                                                  self.stateInputs[level]: sampled_state_features,
                                                  self.actionSelections[level]: sampled_actions,
                                                  self.rewardVectors[level]: sampled_rewards,
                                                  self.l2LambdaTf: 0.0})
            epsilon *= epsilon_discount_factor
            total_loss = results[0]
            losses.append(total_loss)
            if len(losses) % 10 == 0:
                print("MSE:{0}".format(np.mean(np.array(losses))))
                # Enumerate all state combinations
                complete_state_matrix = UtilityFuncs.get_cartesian_product(
                    [[sample_id for sample_id in range(sample_count)],
                     [a_t_minus_one for a_t_minus_one in range(action_count_t_minus_one)]])
                complete_state_matrix = np.array(complete_state_matrix)
                complete_state_features = self.get_state_features(state_matrix=complete_state_matrix,
                                                                  level=level, type="validation")
                complete_Q_table = self.session.run([self.qFuncs[level]],
                                                    feed_dict={
                                                        self.stateInputs[level]: complete_state_features})[0]
                mean_policy_value = np.mean(np.max(complete_Q_table, axis=1))
                print("mean_policy_value:{0}".format(mean_policy_value))
                mean_sample_policy_value = np.mean(
                    np.array([np.max(complete_Q_table[3 * idx:3 * (idx + 1)]) for idx in range(sample_count)])
                )
                print("mean_sample_policy_value:{0}".format(mean_sample_policy_value))

