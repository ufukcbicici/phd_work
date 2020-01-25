import tensorflow as tf
import numpy as np

from collections import Counter
from auxillary.db_logger import DbLogger
from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.policy_gradients_network import \
    PolicyGradientsNetwork, TrajectoryHistory


class TreeDepthState:
    def __init__(self, state_ids, state_vecs, max_likelihood_selections):
        self.stateIds = state_ids
        self.stateVectors = state_vecs
        self.maxLikelihoodSelections = max_likelihood_selections


class TreeDepthPolicyNetwork(PolicyGradientsNetwork):
    INVALID_ACTION_PENALTY = -10.0
    VALID_PREDICTION_REWARD = 1.0
    INVALID_PREDICTION_PENALTY = 0.0
    LAMBDA_MAC_COST = 0.0
    BASELINE_UPDATE_GAMMA = 0.99

    def __init__(self, l2_lambda,
                 network_name, run_id, iteration, degree_list, data_type, output_names, used_feature_names,
                 hidden_layers, use_baselines, state_sample_count, trajectory_per_state_sample_count, test_ratio=0.2):
        self.hiddenLayers = hidden_layers
        super().__init__(l2_lambda, network_name, run_id, iteration, degree_list, data_type, output_names,
                         used_feature_names, use_baselines,
                         state_sample_count, trajectory_per_state_sample_count, test_ratio=test_ratio)
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

    def get_previous_actions(self, history, time_step):
        if time_step - 1 < 0:
            actions_t_minus_one = np.zeros(shape=history.stateIds.shape, dtype=np.int32)
        else:
            actions_t_minus_one = history.actions[time_step - 1]
        return actions_t_minus_one

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
        net = self.stateInputs[time_step]
        for layer_id, layer_dim in enumerate(hidden_layers):
            if layer_id < len(hidden_layers) - 1:
                net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
            else:
                net = tf.layers.dense(inputs=net, units=layer_dim, activation=None)
        _logits = net
        self.logits.append(_logits)
        self.policies.append(tf.nn.softmax(_logits))
        self.logPolicies.append(tf.log(self.policies[-1]))

    def build_policy_gradient_loss(self):
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            selected_policy_indices = tf.stack([tf.range(0, self.trajectoryCount, 1), self.selectedPolicyInputs[t]],
                                               axis=1)
            log_selected_policies = tf.gather_nd(self.logPolicies[t], selected_policy_indices)
            self.selectedLogPolicySamples.append(log_selected_policies)
            # TODO: ADD BASELINE HERE WHEN THE TIME COMES
            proxy_loss_step_t = self.selectedLogPolicySamples[t] * (self.cumulativeRewards[t] - self.baselinesTf[t])
            self.proxyLossTrajectories.append(proxy_loss_step_t)
        self.proxyLossVector = tf.add_n(self.proxyLossTrajectories)
        self.proxyLoss = tf.reduce_mean(self.proxyLossVector)

    def sample_initial_states(self, routing_data, state_sample_count, samples_per_state, state_ids=None) \
            -> TrajectoryHistory:
        if state_ids is None:
            total_sample_count = routing_data.routingDataset.labelList.shape[0]
            sample_indices = np.random.choice(total_sample_count, state_sample_count, replace=False)
        else:
            sample_indices = state_ids
        sample_indices = np.repeat(sample_indices, repeats=samples_per_state)
        feature_arr = routing_data.featuresDict[self.network.topologicalSortedNodes[0].index]
        initial_state_vectors = feature_arr[sample_indices, :]
        state_ml_selections = routing_data.mlPaths[sample_indices, :]
        history = TrajectoryHistory(state_ids=sample_indices, max_likelihood_routes=state_ml_selections)
        history.states.append(initial_state_vectors)
        return history

    def get_max_trajectory_length(self) -> int:
        return int(self.network.depth - 1)

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

    def sample_from_policy(self, routing_data, history, time_step, select_argmax, ignore_invalid_actions):
        assert len(history.states) == time_step + 1
        assert len(history.actions) == time_step
        feed_dict = {self.stateInputs[t]: history.states[t] for t in range(time_step + 1)}
        if not select_argmax:
            # sampling_op = self.policyArgMaxSamples[time_step] if select_argmax else self.policySamples[time_step]
            results = self.tfSession.run([self.policies[time_step], self.policySamples[time_step]], feed_dict=feed_dict)
            policies = results[0]
            policy_samples = results[-1]
        else:
            results = self.tfSession.run([self.policies[time_step]], feed_dict=feed_dict)
            policies = results[0]
            # Determine valid actions
            actions_t_minus_one = self.get_previous_actions(history=history, time_step=time_step)
            reachability_matrix = self.reachabilityMatrices[time_step][actions_t_minus_one, :]
            assert policies.shape == reachability_matrix.shape
            policies = policies * reachability_matrix if ignore_invalid_actions else policies
            policy_samples = np.argmax(policies, axis=1)
        history.policies.append(policies)
        history.actions.append(policy_samples)
        routing_decisions_t = self.actionSpaces[time_step][history.actions[time_step], :]
        history.routingDecisions.append(routing_decisions_t)

    def state_transition(self, history, routing_data, time_step):
        nodes_in_level = self.network.orderedNodesPerLevel[time_step + 1]
        feature_arrays = [routing_data.featuresDict[node.index][history.stateIds] for node in nodes_in_level]
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

    def reward_calculation(self, routing_data, history, time_step):
        rewards_arr = np.zeros(shape=history.stateIds.shape, dtype=np.float32)
        # Check if valid actions
        actions_t_minus_one = self.get_previous_actions(history=history, time_step=time_step)
        # Asses availability of the decisions
        action_t = history.actions[time_step]
        validity_of_actions_vec = self.reachabilityMatrices[time_step][actions_t_minus_one, action_t]
        validity_rewards = np.array([TreeDepthPolicyNetwork.INVALID_ACTION_PENALTY, 0.0])
        rewards_arr += validity_rewards[validity_of_actions_vec]
        # If in the last step, calculate reward according to the accuracy + computation cost
        if time_step == self.get_max_trajectory_length() - 1:
            # Prediction Rewards
            validity_of_predictions_vec = self.calculate_accuracy_of_trajectories(routing_data=routing_data,
                                                                                  history=history)
            validity_of_prediction_rewards = np.array([TreeDepthPolicyNetwork.INVALID_PREDICTION_PENALTY,
                                                       TreeDepthPolicyNetwork.VALID_PREDICTION_REWARD])
            prediction_rewards = validity_of_prediction_rewards[validity_of_predictions_vec]
            rewards_arr += prediction_rewards
            # Computation Cost Penalties
            activation_cost_arr = self.networkActivationCosts[history.actions[time_step]]
            activation_cost_arr = -TreeDepthPolicyNetwork.LAMBDA_MAC_COST * activation_cost_arr
            rewards_arr += activation_cost_arr
        history.rewards.append(rewards_arr)

    def calculate_routing_accuracy(self, routing_data, state_batch_size):
        data_count = routing_data.routingDataset.labelList.shape[0]
        id_list = list(range(data_count))
        ignore_invalid_action_flags = [False, True]
        accuracy_types_list = ["All Actions", "Only Valid Actions"]
        accuracy_dict = {}
        for ignore_invalid_actions, accuracy_type in zip(ignore_invalid_action_flags, accuracy_types_list):
            validity_vectors = []
            for idx in range(0, data_count, state_batch_size):
                curr_sample_ids = id_list[idx:idx + state_batch_size]
                history = self.sample_trajectories(routing_data=routing_data,
                                                   state_sample_count=None,
                                                   samples_per_state=1,
                                                   state_ids=curr_sample_ids,
                                                   select_argmax=True,
                                                   ignore_invalid_actions=ignore_invalid_actions)
                validity_of_predictions_vec = \
                    self.calculate_accuracy_of_trajectories(routing_data=routing_data, history=history)
                validity_vectors.append(validity_of_predictions_vec)
            validity_vector = np.concatenate(validity_vectors)
            total_correct_count = np.sum(validity_vector)
            accuracy = total_correct_count / validity_vector.shape[0]
            accuracy_dict[accuracy_type] = accuracy
        return accuracy_dict

    def update_baselines(self, history):
        max_trajectory_length = self.get_max_trajectory_length()
        gamma = TreeDepthPolicyNetwork.BASELINE_UPDATE_GAMMA
        for t in range(max_trajectory_length):
            forward_rewards = np.stack([history.rewards[_t] for _t in range(t, max_trajectory_length)], axis=1)
            cumulative_rewards = np.sum(forward_rewards, axis=1)
            actions_t_minus_one = self.get_previous_actions(history=history, time_step=t)
            delta_arr = np.zeros_like(self.baselinesNp[t])
            count_arr = np.zeros_like(self.baselinesNp[t])
            for state_id, action_t_minus_1, reward in zip(history.stateIds, actions_t_minus_one, cumulative_rewards):
                delta_arr[state_id, action_t_minus_1] += reward
                count_arr[state_id, action_t_minus_1] += 1.0
            # update_indicator = count_arr > 0.0
            mean_delta_arr = delta_arr * np.reciprocal(count_arr)
            new_baseline_arr = gamma * self.baselinesNp[t] + (1.0 - gamma) * mean_delta_arr
            nan_mask = np.logical_not(np.isnan(new_baseline_arr))
            self.baselinesNp[t] = np.where(nan_mask, new_baseline_arr, self.baselinesNp[t])
            # self.baselinesNp[t] += gamma * self.baselinesNp[t] + (1.0 - gamma) * mean_delta_arr

    def train(self, max_num_of_iterations=7500):
        self.evaluate_ml_routing_accuracies()
        self.evaluate_policy_values()
        self.evaluate_routing_accuracies()

        exp_str = self.get_explanation()
        run_id = DbLogger.get_run_id()
        DbLogger.write_into_table(rows=[(run_id, exp_str)], table=DbLogger.runMetaData, col_count=2)

        for iteration_id in range(max_num_of_iterations):
            # Sample a set of trajectories
            history = self.sample_trajectories(routing_data=self.validationDataForMDP,
                                               state_sample_count=self.stateSampleCount,
                                               samples_per_state=self.trajectoryPerStateSampleCount,
                                               state_ids=None,
                                               select_argmax=False,
                                               ignore_invalid_actions=False)
            # Calculate the policy gradient, update the network.
            # Fill the feed dict
            feed_dict = {}
            for t in range(self.get_max_trajectory_length()):
                feed_dict[self.stateInputs[t]] = history.states[t]
                feed_dict[self.selectedPolicyInputs[t]] = history.actions[t]
                feed_dict[self.rewards[t]] = history.rewards[t]
                actions_t_minus_one = self.get_previous_actions(history=history, time_step=t)
                feed_dict[self.baselinesTf[t]] = self.baselinesNp[t][history.stateIds, actions_t_minus_one]
            feed_dict[self.l2LambdaTf] = self.l2Lambda

            results = self.tfSession.run([self.logPolicies,
                                          self.selectedLogPolicySamples,
                                          self.cumulativeRewards,
                                          self.proxyLossTrajectories,
                                          self.proxyLossVector,
                                          self.proxyLoss,
                                          self.optimizer], feed_dict=feed_dict)
            # Update baselines
            if self.useBaselines:
                self.update_baselines(history=history)
            if any([np.any(np.isinf(log_policy_arr)) for log_policy_arr in results[0]]):
                print("Contains inf!!!")
            # self.evaluate_ml_routing_accuracies()
            if iteration_id % 10 == 0:
                print("***********Iteration {0}***********".format(iteration_id))
                validation_policy_value, test_policy_value = self.evaluate_policy_values()
                validation_accuracy, test_accuracy = self.evaluate_routing_accuracies()
                DbLogger.write_into_table(rows=[(run_id,
                                                 iteration_id,
                                                 validation_policy_value,
                                                 test_policy_value,
                                                 validation_accuracy["All Actions"],
                                                 test_accuracy["All Actions"],
                                                 validation_accuracy["Only Valid Actions"],
                                                 test_accuracy["Only Valid Actions"])],
                                          table="policy_gradients_results", col_count=8)
                print("***********Iteration {0}***********".format(iteration_id))
            # print("X")
        print("X")

    def grid_search(self):
        for l2_lambda in [0.0, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005]:
            self.l2Lambda = l2_lambda
            self.train()

    def get_explanation(self):
        explanation = ""
        explanation += "INVALID_ACTION_PENALTY={0}\n".format(TreeDepthPolicyNetwork.INVALID_ACTION_PENALTY)
        explanation += "VALID_PREDICTION_REWARD={0}\n".format(TreeDepthPolicyNetwork.VALID_PREDICTION_REWARD)
        explanation += "INVALID_PREDICTION_PENALTY={0}\n".format(TreeDepthPolicyNetwork.INVALID_PREDICTION_PENALTY)
        explanation += "LAMBDA_MAC_COST={0}\n".format(TreeDepthPolicyNetwork.LAMBDA_MAC_COST)
        explanation += "BASELINE_UPDATE_GAMMA={0}\n".format(TreeDepthPolicyNetwork.BASELINE_UPDATE_GAMMA)
        explanation += "Hidden Layers={0}\n".format(self.hiddenLayers)
        explanation += "Network Name:{0}\n".format(self.networkName)
        explanation += "Network Run Id:{0}\n".format(self.networkRunId)
        explanation += "Use Baselines:{0}\n".format(self.useBaselines)
        explanation += "stateSampleCount:{0}\n".format(self.stateSampleCount)
        explanation += "trajectoryPerStateSampleCount:{0}\n".format(self.trajectoryPerStateSampleCount)
        explanation += "l2Lambda:{0}\n".format(self.l2Lambda)
        val_ml_accuracy, test_ml_accuracy = self.evaluate_ml_routing_accuracies()
        explanation += "val_ml_accuracy:{0}\n".format(val_ml_accuracy)
        explanation += "test_ml_accuracy:{0}\n".format(test_ml_accuracy)
        return explanation


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
    state_sample_count = 8000
    samples_per_state = 100

    policy_gradients_routing_optimizer = TreeDepthPolicyNetwork(l2_lambda=0.0,
                                                                network_name=network_name,
                                                                run_id=run_id,
                                                                iteration=iteration,
                                                                degree_list=[2, 2],
                                                                data_type="test",
                                                                output_names=output_names,
                                                                used_feature_names=used_output_names,
                                                                test_ratio=0.2,
                                                                use_baselines=True,
                                                                state_sample_count=state_sample_count,
                                                                trajectory_per_state_sample_count=samples_per_state,
                                                                hidden_layers=[[128], [256]])
    # policy_gradients_routing_optimizer.train()
    policy_gradients_routing_optimizer.grid_search()


if __name__ == "__main__":
    main()
