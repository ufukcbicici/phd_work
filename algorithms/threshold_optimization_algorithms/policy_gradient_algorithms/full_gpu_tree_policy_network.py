import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.policy_gradients_network import \
    TrajectoryHistory
from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.tree_depth_policy_network import \
    TreeDepthPolicyNetwork
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork


class FullGpuTreePolicyGradientsNetwork(TreeDepthPolicyNetwork):
    INVALID_ACTION_PENALTY = -10.0
    VALID_PREDICTION_REWARD = 1.0
    INVALID_PREDICTION_PENALTY = 0.0
    LAMBDA_MAC_COST = 0.0
    BASELINE_UPDATE_GAMMA = 0.99

    def __init__(self, validation_data, test_data, l2_lambda, network, network_name, run_id, iteration, degree_list,
                 output_names, used_feature_names, hidden_layers, use_baselines, state_sample_count,
                 trajectory_per_state_sample_count):
        self.stateInputTransformed = []
        self.softmaxDecay = tf.placeholder(dtype=tf.float32, name="softmaxDecay", shape=[])
        self.isSamplingTrajectory = tf.placeholder(dtype=tf.bool, name="isSamplingTrajectory")
        self.ignoreInvalidActions = tf.placeholder(dtype=tf.bool, name="ignoreInvalidActions")
        self.rewardTensorsTf = []
        self.selectedRewards = []
        self.cumRewards = []
        self.trajectoryValues = None
        self.trajectoryValuesSum = None
        self.policyValue = None
        self.actionSpacesTf = []
        self.reachabilityMatricesTf = []
        self.validationRewards = []
        self.testRewards = []
        self.routingDecisions = []
        self.rangeIndex = None
        self.argMaxActions = []
        self.finalActions = []
        self.resultsDict = {}
        self.stateIds = tf.placeholder(dtype=tf.int32, name="stateIds", shape=[None])
        super().__init__(validation_data, test_data, l2_lambda, network, network_name, run_id, iteration, degree_list,
                         output_names, used_feature_names, hidden_layers, use_baselines, state_sample_count,
                         trajectory_per_state_sample_count)

    def build_action_spaces(self):
        super().build_action_spaces()
        for t in range(self.get_max_trajectory_length()):
            self.actionSpacesTf.append(tf.constant(self.actionSpaces[t]))
        self.resultsDict["actionSpacesTf"] = self.actionSpacesTf

    def calculate_reward_tensors(self):
        invalid_action_penalty = FullGpuTreePolicyGradientsNetwork.INVALID_ACTION_PENALTY
        valid_prediction_reward = FullGpuTreePolicyGradientsNetwork.VALID_PREDICTION_REWARD
        invalid_prediction_penalty = FullGpuTreePolicyGradientsNetwork.INVALID_PREDICTION_PENALTY
        calculation_cost_modifier = FullGpuTreePolicyGradientsNetwork.LAMBDA_MAC_COST
        for t in range(self.get_max_trajectory_length()):
            action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
            action_count_t = self.actionSpaces[t].shape[0]
            self.rewardTensorsTf.append(
                tf.placeholder(dtype=tf.float32, name="rewardTensors_{0}".format(t),
                               shape=(None, action_count_t_minus_one, action_count_t)))
            self.reachabilityMatricesTf.append(tf.constant(self.reachabilityMatrices[t]))
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
                    for action_id in range(self.actionSpaces[t].shape[0]):
                        routing_decision = self.actionSpaces[t][action_id, :]
                        weight = 1.0 / np.sum(routing_decision)
                        routing_decision_weighted = weight * routing_decision
                        assert routing_decision.shape[0] == dataset.posteriorsTensor.shape[2]
                        weighted_posteriors = dataset.posteriorsTensor * routing_decision_weighted
                        final_posteriors = np.sum(weighted_posteriors, axis=2)
                        predicted_labels = np.argmax(final_posteriors, axis=1)
                        validity_of_predictions_vec = (predicted_labels == true_labels).astype(np.int32)
                        prediction_correctness_vec_list.append(validity_of_predictions_vec)
                    prediction_correctness_matrix = np.stack(prediction_correctness_vec_list, axis=1)
                    prediction_correctness_tensor = np.repeat(
                        np.expand_dims(prediction_correctness_matrix, axis=1), axis=1, repeats=action_count_t_minus_one)
                    rewards_arr += (prediction_correctness_tensor == 1).astype(np.float32) * valid_prediction_reward
                    rewards_arr += (prediction_correctness_tensor == 0).astype(np.float32) * invalid_prediction_penalty
                    # Calculation Cost Rewards (Penalties):
                    # Calculate the Cost for every last layer routing combination
                    cost_arr = np.expand_dims(self.networkActivationCosts, axis=0)
                    cost_arr = np.repeat(cost_arr, axis=0, repeats=action_count_t_minus_one)
                    cost_arr = np.expand_dims(cost_arr, axis=0)
                    cost_arr = np.repeat(cost_arr, axis=0, repeats=true_labels.shape[0])
                    cost_arr = calculation_cost_modifier * cost_arr
                    rewards_arr -= cost_arr
                if dataset == self.validationDataForMDP:
                    self.validationRewards.append(rewards_arr)
                else:
                    self.testRewards.append(rewards_arr)
        self.validationDataForMDP.rewardTensors = self.validationRewards
        self.testDataForMDP.rewardTensors = self.testRewards

    def build_state_inputs(self, time_step):
        ordered_nodes_at_level = self.network.orderedNodesPerLevel[time_step]
        inputs_list = [self.validationFeaturesDict[node.index] for node in ordered_nodes_at_level]
        shape_set = {input_arr.shape for input_arr in inputs_list}
        assert len(shape_set) == 1
        feat_shape = list(shape_set)[0]
        input_shape = [None]
        input_shape.extend(feat_shape[1:])
        tf_list = [
            tf.placeholder(dtype=tf.float32, shape=input_shape,
                           name="inputs_t{0}_node{1}".format(time_step, node.index))
            for node in ordered_nodes_at_level]
        self.stateInputs.append(tf_list)
        if time_step == 0:
            assert len(self.stateInputTransformed) == 0
            self.stateInputTransformed.append(self.stateInputs[0][0])
            state_input_shape = tf.shape(self.stateInputs[0][0])
            self.trajectoryCount = tf.gather_nd(state_input_shape, [0])
            self.rangeIndex = tf.range(0, self.trajectoryCount, 1)
            self.resultsDict["stateInputTransformed_{0}".format(time_step)] = self.stateInputTransformed[time_step]
            self.resultsDict["trajectoryCount"] = self.trajectoryCount
            self.resultsDict["rangeIndex"] = self.rangeIndex
        else:
            routing_decisions = self.routingDecisions[time_step - 1]
            list_of_indices = []
            list_of_coefficients = []
            weighted_state_inputs = []
            for action_id, state_input in enumerate(self.stateInputs[time_step]):
                routing_indices = tf.stack([self.rangeIndex, action_id * tf.ones_like(self.rangeIndex)], axis=1)
                list_of_indices.append(routing_indices)
                route_coefficients = tf.gather_nd(routing_decisions, routing_indices)
                list_of_coefficients.append(route_coefficients)
                weighted_state_inputs.append(tf.cast(tf.expand_dims(route_coefficients, axis=1), dtype=tf.float32)
                                             * state_input)
            assert len(self.stateInputTransformed) == time_step
            self.stateInputTransformed.append(tf.concat(values=weighted_state_inputs, axis=1))
            self.resultsDict["routing_indices_{0}".format(time_step)] = list_of_indices
            self.resultsDict["routing_weights_{0}".format(time_step)] = list_of_coefficients
            self.resultsDict["stateInputTransformed_{0}".format(time_step)] = self.stateInputTransformed[time_step]

    #     routing_decisions = self.routingDecisions[time_step]
    #     list_of_indices = []
    #     list_of_coefficients = []
    #     weighted_state_inputs = []
    #     for action_id, state_input in enumerate(self.stateInputs[time_step]):
    #         routing_indices = tf.stack([self.rangeIndex, action_id * tf.ones_like(self.rangeIndex)], axis=1)
    #         list_of_indices.append(routing_indices)
    #         route_coefficients = tf.gather_nd(routing_decisions, routing_indices)
    #         list_of_coefficients.append(route_coefficients)
    #         weighted_state_inputs.append(route_coefficients * state_input)
    #     self.stateInputTransformed.append(tf.concat(values=weighted_state_inputs, axis=1))
    #     assert len(self.stateInputTransformed) == time_step + 1
    #     self.resultsDict["routing_indices_{0}".format(time_step)] = list_of_indices
    #     self.resultsDict["routing_weights_{0}".format(time_step)] = list_of_coefficients
    #     self.resultsDict["stateInputTransformed_{0}".format(time_step + 1)] = self.stateInputTransformed[time_step + 1]

    # Override this for different type of policy network implementations
    def build_policy_networks(self, time_step):
        pass

    def sample_from_policy_tf(self, time_step):
        sampled_actions = FastTreeNetwork.sample_from_categorical_v2(probs=self.policies[time_step])
        actions_t_minus_one = tf.zeros_like(self.stateIds) if time_step == 0 else self.finalActions[time_step - 1]
        reachability_matrix = tf.gather_nd(self.reachabilityMatricesTf[time_step],
                                           tf.expand_dims(actions_t_minus_one, axis=1))
        argmax_actions_all = tf.cast(tf.argmax(self.policies[time_step], axis=1), dtype=tf.int32)
        valid_policies = self.policies[time_step] * tf.cast(reachability_matrix, dtype=tf.float32)
        argmax_actions_only_valid = tf.cast(tf.argmax(valid_policies, axis=1), dtype=tf.int32)
        argmax_actions = tf.where(self.ignoreInvalidActions, argmax_actions_only_valid, argmax_actions_all)
        samples = tf.where(self.isSamplingTrajectory, sampled_actions, argmax_actions)
        routing_decisions = tf.gather_nd(self.actionSpacesTf[time_step], tf.expand_dims(samples, axis=1))
        self.policySamples.append(sampled_actions)
        self.argMaxActions.append(argmax_actions)
        self.finalActions.append(samples)
        self.routingDecisions.append(routing_decisions)
        self.resultsDict["reachability_matrix_{0}".format(time_step)] = reachability_matrix
        self.resultsDict["valid_policies_{0}".format(time_step)] = valid_policies
        self.resultsDict["policySamples_{0}".format(time_step)] = sampled_actions
        self.resultsDict["argmax_actions_all_{0}".format(time_step)] = argmax_actions_all
        self.resultsDict["argmax_actions_only_valid_{0}".format(time_step)] = argmax_actions_only_valid
        self.resultsDict["argMaxActions_{0}".format(time_step)] = argmax_actions
        self.resultsDict["routingDecisions_{0}".format(time_step)] = routing_decisions
        self.resultsDict["finalActions_{0}".format(time_step)] = samples

    def calculate_reward(self, time_step):
        # Get state indices
        actions_t_minus_one = tf.zeros_like(self.stateIds) if time_step == 0 else self.finalActions[time_step - 1]
        actions_t = self.finalActions[time_step]
        reward_indices = tf.stack([self.stateIds, actions_t_minus_one, actions_t], axis=1)
        reward_tensor = self.rewardTensorsTf[time_step]
        selected_rewards = tf.gather_nd(reward_tensor, reward_indices)
        self.selectedRewards.append(selected_rewards)
        self.resultsDict["selected_rewards_{0}".format(time_step)] = selected_rewards
        self.resultsDict["reward_indices_{0}".format(time_step)] = reward_indices

    def calculate_cumulative_rewards_tf(self):
        # Cumulative Rewards
        max_trajectory_length = self.get_max_trajectory_length()
        for t1 in range(max_trajectory_length):
            rew_list = [self.selectedRewards[t2] for t2 in range(t1, max_trajectory_length, 1)]
            cum_sum = tf.add_n(rew_list)
            self.cumulativeRewards.append(cum_sum)
            self.resultsDict["cumulativeRewards_{0}".format(t1)] = cum_sum

    def calculate_policy_value_tf(self):
        self.trajectoryValues = tf.add_n(self.selectedRewards)
        self.trajectoryValuesSum = tf.reduce_sum(self.trajectoryValues)
        self.policyValue = tf.reduce_mean(self.trajectoryValues)
        self.resultsDict["trajectoryValues"] = self.trajectoryValues
        self.resultsDict["trajectoryValuesSum"] = self.trajectoryValuesSum
        self.resultsDict["policyValue"] = self.policyValue

    def build_baselines_tf(self, time_step):
        baseline_input = tf.placeholder(dtype=tf.float32, shape=[None], name="baselines_{0}".format(time_step))
        self.baselinesTf.append(baseline_input)
        if time_step - 1 < 0:
            baseline_np = np.zeros(shape=(self.validationDataForMDP.routingDataset.labelList.shape[0], 1))
        else:
            baseline_np = np.zeros(shape=(self.validationDataForMDP.routingDataset.labelList.shape[0],
                                          self.actionSpaces[time_step - 1].shape[0]))
        self.baselinesNp.append(baseline_np)

    def build_policy_gradient_loss(self):
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            selected_policy_indices = tf.stack([tf.range(0, self.trajectoryCount, 1), self.finalActions[t]],
                                               axis=1)
            log_selected_policies = tf.gather_nd(self.logPolicies[t], selected_policy_indices)
            self.selectedLogPolicySamples.append(log_selected_policies)
            proxy_loss_step_t = self.selectedLogPolicySamples[t] * (self.cumulativeRewards[t] - self.baselinesTf[t])
            self.proxyLossTrajectories.append(proxy_loss_step_t)
        self.proxyLossVector = tf.add_n(self.proxyLossTrajectories)
        self.proxyLoss = tf.reduce_mean(self.proxyLossVector)
        self.resultsDict["loss_selectedLogPolicySamples"] = self.selectedLogPolicySamples
        self.resultsDict["loss_proxyLossTrajectories"] = self.proxyLossTrajectories
        self.resultsDict["loss_proxyLossVector"] = self.proxyLossVector
        self.resultsDict["loss_proxyLoss"] = self.proxyLoss

    def build_networks(self):
        self.calculate_reward_tensors()
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            # Handle the first step: s_0 and state transitions s_{t} = f(s_{t-1},a_{t-1})
            self.build_state_inputs(time_step=t)
            # Build the network generating the policy at time step t: \pi_t(a_t|s_t)
            self.build_policy_networks(time_step=t)
            # Sample actions from the policy network: a_t ~ \pi_t(a_t|s_t)
            # Or select the maximum likely action: a_t = argmax_{a_t} \pi_t(a_t|s_t)
            self.sample_from_policy_tf(time_step=t)
            # Select a reward r_t = r(s_t, a_t)
            self.calculate_reward(time_step=t)
            # Baselines for variance reduction
            self.build_baselines_tf(time_step=t)
        self.calculate_policy_value_tf()
        self.calculate_cumulative_rewards_tf()
        self.build_policy_gradient_loss()
        self.get_l2_loss()
        self.build_optimizer()

    def sample_trajectories(self, sess, routing_data, state_sample_count, samples_per_state,
                            select_argmax, ignore_invalid_actions, state_ids) \
            -> TrajectoryHistory:
        # if state_ids is None, sample from state distribution
        # Sample from s1 ~ p(s1)
        history = self.sample_initial_states(routing_data=routing_data,
                                             state_sample_count=state_sample_count,
                                             samples_per_state=samples_per_state,
                                             state_ids=state_ids)
        # Prepare all state inputs for all time steps
        feed_dict = {self.isSamplingTrajectory: not select_argmax,
                     self.ignoreInvalidActions: ignore_invalid_actions,
                     self.softmaxDecay: 1.0,
                     self.stateIds: history.stateIds}
        for t in range(self.get_max_trajectory_length()):
            # State inputs
            for idx, state_input in enumerate(self.stateInputs[t]):
                features = \
                    routing_data.featuresDict[self.network.orderedNodesPerLevel[t][idx].index][history.stateIds, :]
                feed_dict[state_input] = features
            # Reward inputs
            feed_dict[self.rewardTensorsTf[t]] = routing_data.rewardTensors[t]

        results_dict = {k: v for k, v in self.resultsDict.items() if "loss_" not in k}
        results = sess.run(results_dict, feed_dict)
        # Build the history object
        history.states = []
        for t in range(self.get_max_trajectory_length()):
            # State inputs
            history.states.append(results["stateInputTransformed_{0}".format(t)])
            history.policies.append(results["policies_{0}".format(t)])
            history.actions.append(results["finalActions_{0}".format(t)])
            history.rewards.append(results["selected_rewards_{0}".format(t)])
            history.routingDecisions.append(results["routingDecisions_{0}".format(t)])
            history.validPolicies.append(results["valid_policies_{0}".format(t)])
        return history
