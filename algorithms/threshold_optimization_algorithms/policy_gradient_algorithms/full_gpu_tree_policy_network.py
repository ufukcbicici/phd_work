import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.policy_gradients_network import \
    TrajectoryHistory
from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.tree_depth_policy_network import \
    TreeDepthPolicyNetwork
from auxillary.db_logger import DbLogger
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
        if time_step - 1 < 0:
            baseline_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="baselines_{0}".format(time_step))
            baseline_np = np.zeros(shape=(self.validationDataForMDP.routingDataset.labelList.shape[0], 1))
        else:
            baseline_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.actionSpaces[time_step - 1].shape[0]],
                                         name="baselines_{0}".format(time_step))
            baseline_np = np.zeros(shape=(self.validationDataForMDP.routingDataset.labelList.shape[0],
                                          self.actionSpaces[time_step - 1].shape[0]))
        self.baselinesTf.append(baseline_tf)
        self.baselinesNp.append(baseline_np)

    def build_policy_gradient_loss(self):
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            actions_t_minus_one = tf.zeros_like(self.stateIds) if t == 0 else self.finalActions[t - 1]
            actions_t = self.finalActions[t]
            selected_policy_indices = tf.stack([tf.range(0, self.trajectoryCount, 1), actions_t], axis=1)
            log_selected_policies = tf.gather_nd(self.logPolicies[t], selected_policy_indices)
            self.selectedLogPolicySamples.append(log_selected_policies)
            baseline_indices = tf.stack([tf.range(0, self.trajectoryCount, 1), actions_t_minus_one], axis=1)
            baseline_t = tf.gather_nd(self.baselinesTf[t], baseline_indices)
            proxy_loss_step_t = self.selectedLogPolicySamples[t] * (self.cumulativeRewards[t] - baseline_t)
            self.proxyLossTrajectories.append(proxy_loss_step_t)
            self.resultsDict["loss_baseline_indices_{0}".format(t)] = baseline_indices
            self.resultsDict["loss_baseline_{0}".format(t)] = baseline_t
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

    def populate_history(self, history, results):
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
        history = self.populate_history(history=history, results=results)
        return history

    def calculate_routing_accuracy(self, sess, routing_data, state_batch_size) -> (dict, dict):
        data_count = routing_data.routingDataset.labelList.shape[0]
        id_list = list(range(data_count))
        ignore_invalid_action_flags = [False, True]
        combine_with_ig_list = [False, True]
        accuracy_dict = {}
        computation_overload_dict = {}
        cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[[False, True], [False, True]])
        for tpl in cartesian_product:
            ignore_invalid_actions = tpl[0]
            combine_with_ig = tpl[1]
            validity_vectors = []
            overload_vectors = []
            for idx in range(0, data_count, state_batch_size):
                curr_sample_ids = id_list[idx:idx + state_batch_size]
                history = self.sample_trajectories(routing_data=routing_data,
                                                   state_sample_count=None,
                                                   samples_per_state=1,
                                                   state_ids=curr_sample_ids,
                                                   select_argmax=True,
                                                   ignore_invalid_actions=ignore_invalid_actions,
                                                   sess=sess)
                validity_of_predictions_vec, computation_overload_vec = \
                    self.calculate_accuracy_of_trajectories(routing_data=routing_data, history=history,
                                                            combine_with_ig=combine_with_ig)
                validity_vectors.append(validity_of_predictions_vec)
                overload_vectors.append(computation_overload_vec)
            validity_vector = np.concatenate(validity_vectors)
            total_correct_count = np.sum(validity_vector)
            accuracy = total_correct_count / validity_vector.shape[0]
            accuracy_dict[(ignore_invalid_actions, combine_with_ig)] = accuracy
            overload_vector = np.concatenate(overload_vectors)
            assert validity_vector.shape == overload_vector.shape
            mean_overload = np.sum(overload_vectors) / overload_vector.shape[0]
            computation_overload_dict[(ignore_invalid_actions, combine_with_ig)] = mean_overload
        return accuracy_dict, computation_overload_dict

    def evaluate_routing_accuracies(self, sess):
        validation_accuracy_dict, val_computation_overload_dict = \
            self.calculate_routing_accuracy(sess=sess,
                                            routing_data=self.validationDataForMDP,
                                            state_batch_size=1000)
        test_accuracy_dict, test_computation_overload_dict = \
            self.calculate_routing_accuracy(sess=sess, routing_data=self.testDataForMDP,
                                            state_batch_size=100)
        print("validation_accuracy={0}".format(validation_accuracy_dict))
        print("test_accuracy={0}".format(test_accuracy_dict))
        print("val_computation_overload_dict={0}".format(val_computation_overload_dict))
        print("test_computation_overload_dict={0}".format(test_computation_overload_dict))
        return validation_accuracy_dict, test_accuracy_dict, \
               val_computation_overload_dict, test_computation_overload_dict

    def save_results_to_db(self, run_id, iteration_id, validation_policy_value, test_policy_value,
                           is_test, accuracy_dict, computation_overload_dict):
        cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[[False, True], [False, True]])
        for tpl in cartesian_product:
            ignore_invalid_actions = tpl[0]
            combine_with_ig = tpl[1]
            DbLogger.write_into_table(rows=[(run_id,
                                             iteration_id,
                                             validation_policy_value,
                                             test_policy_value,
                                             is_test,
                                             ignore_invalid_actions,
                                             combine_with_ig,
                                             accuracy_dict[(ignore_invalid_actions, combine_with_ig)],
                                             computation_overload_dict[(ignore_invalid_actions, combine_with_ig)])],
                                      table="policy_gradients_results", col_count=9)

    def train(self, sess, max_num_of_iterations=15000):
        sess.run(tf.initialize_all_variables())
        self.evaluate_ml_routing_accuracies()
        self.evaluate_policy_values(sess=sess)
        self.evaluate_routing_accuracies(sess=sess)
        routing_data = self.validationDataForMDP
        exp_str = self.get_explanation()
        run_id = DbLogger.get_run_id()
        DbLogger.write_into_table(rows=[(run_id, exp_str)], table=DbLogger.runMetaData, col_count=2)
        for iteration_id in range(max_num_of_iterations):
            # Sample a set of trajectories
            # Sample from s1 ~ p(s1)
            history = self.sample_initial_states(routing_data=routing_data,
                                                 state_sample_count=self.stateSampleCount,
                                                 samples_per_state=self.trajectoryPerStateSampleCount,
                                                 state_ids=None)
            # Prepare all state inputs for all time steps, the reward matrices and the baselines
            feed_dict = {self.isSamplingTrajectory: True,
                         self.ignoreInvalidActions: False,
                         self.softmaxDecay: 1.0,
                         self.stateIds: history.stateIds,
                         self.l2LambdaTf: self.l2Lambda}
            for t in range(self.get_max_trajectory_length()):
                # State inputs
                for idx, state_input in enumerate(self.stateInputs[t]):
                    features = \
                        routing_data.featuresDict[self.network.orderedNodesPerLevel[t][idx].index][history.stateIds, :]
                    feed_dict[state_input] = features
                # Reward inputs
                feed_dict[self.rewardTensorsTf[t]] = routing_data.rewardTensors[t]
                # Baseline inputs
                feed_dict[self.baselinesTf[t]] = self.baselinesNp[t]
            # Execute the optimizer
            run_dict = {k: v for k, v in self.resultsDict.items() if "loss" in k}
            run_dict.update({k: v for k, v in self.resultsDict.items() if "stateInputTransformed" in k})
            run_dict.update({k: v for k, v in self.resultsDict.items() if "policies" in k})
            run_dict.update({k: v for k, v in self.resultsDict.items() if "finalActions" in k})
            run_dict.update({k: v for k, v in self.resultsDict.items() if "selected_rewards" in k})
            run_dict.update({k: v for k, v in self.resultsDict.items() if "routingDecisions" in k})
            run_dict.update({k: v for k, v in self.resultsDict.items() if "valid_policies" in k})
            run_dict.update({k: v for k, v in self.resultsDict.items() if "log_policies" in k})
            run_dict["optimizer"] = self.optimizer
            results = sess.run(run_dict, feed_dict=feed_dict)
            # Populate the history object
            # Build the history object
            history = self.populate_history(history=history, results=results)
            # Update baselines
            if self.useBaselines:
                self.update_baselines(history=history)
            log_policy_arrays = [v for k, v in results.items() if "log_policies" in k]
            if any([np.any(np.isinf(log_policy_arr)) for log_policy_arr in log_policy_arrays]):
                print("Contains inf!!!")
            print("X")
            if iteration_id % 100 == 0 or iteration_id == max_num_of_iterations - 1:
                print("***********Iteration {0}***********".format(iteration_id))
                validation_policy_value, test_policy_value = self.evaluate_policy_values(sess=sess)
                validation_accuracy_dict, test_accuracy_dict, val_computation_overload_dict, \
                    test_computation_overload_dict = self.evaluate_routing_accuracies(sess=sess)
                self.save_results_to_db(run_id=run_id,
                                        iteration_id=iteration_id,
                                        validation_policy_value=validation_policy_value,
                                        test_policy_value=test_policy_value,
                                        is_test=False,
                                        accuracy_dict=validation_accuracy_dict,
                                        computation_overload_dict=val_computation_overload_dict)
                self.save_results_to_db(run_id=run_id,
                                        iteration_id=iteration_id,
                                        validation_policy_value=validation_policy_value,
                                        test_policy_value=test_policy_value,
                                        is_test=True,
                                        accuracy_dict=test_accuracy_dict,
                                        computation_overload_dict=test_computation_overload_dict)
                print("***********Iteration {0}***********".format(iteration_id))
