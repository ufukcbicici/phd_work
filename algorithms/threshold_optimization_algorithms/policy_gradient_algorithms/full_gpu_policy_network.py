import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.policy_gradients_network import \
    TrajectoryHistory
from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.tree_depth_policy_network import \
    TreeDepthPolicyNetwork
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork


class FullGpuPolicyGradientsNetwork(TreeDepthPolicyNetwork):
    INVALID_ACTION_PENALTY = -10.0
    VALID_PREDICTION_REWARD = 1.0
    INVALID_PREDICTION_PENALTY = 0.0
    LAMBDA_MAC_COST = 0.0
    BASELINE_UPDATE_GAMMA = 0.99

    def __init__(self, validation_data, test_data, l2_lambda, network, network_name, run_id, iteration, degree_list,
                 output_names, used_feature_names, hidden_layers, use_baselines, state_sample_count,
                 trajectory_per_state_sample_count):
        self.stateInputTransformed = []
        self.softmaxDecay = tf.placeholder(dtype=tf.float32, name="softmaxDecay")
        self.actionSpacesTf = []
        self.resultsDict = {}
        super().__init__(validation_data, test_data, l2_lambda, network, network_name, run_id, iteration, degree_list,
                         output_names, used_feature_names, hidden_layers, use_baselines, state_sample_count,
                         trajectory_per_state_sample_count)

    def build_action_spaces(self):
        super().build_action_spaces()
        for t in range(self.get_max_trajectory_length()):
            self.actionSpacesTf.append(tf.constant(self.actionSpaces[t]))
        self.resultsDict["actionSpacesTf"] = self.actionSpacesTf

    def reward_calculation_tf(self):
        for t in range(self.get_max_trajectory_length()):
            action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
            action_count_t = self.actionSpaces[t].shape[0]
            for dataset in [self.validationDataForMDP, self.testDataForMDP]:
                reward_shape = (dataset.routingDataset.labelList.shape[0], action_count_t_minus_one, action_count_t)
                rewards_arr = np.zeros(shape=reward_shape, dtype=np.float32)
                validity_of_actions_tensor = np.repeat(
                    np.expand_dims(self.reachabilityMatrices[t], axis=0),
                    repeats=dataset.routingDataset.labelList.shape[0], axis=0)
        print("X")

        # rewards_arr = np.zeros(shape=history.stateIds.shape, dtype=np.float32)
        # # Check if valid actions
        # actions_t_minus_one = self.get_previous_actions(history=history, time_step=time_step)
        # # Asses availability of the decisions
        # action_t = history.actions[time_step]
        # validity_of_actions_vec = self.reachabilityMatrices[time_step][actions_t_minus_one, action_t]
        # validity_rewards = np.array([TreeDepthPolicyNetwork.INVALID_ACTION_PENALTY, 0.0])
        # rewards_arr += validity_rewards[validity_of_actions_vec]
        # # If in the last step, calculate reward according to the accuracy + computation cost
        # if time_step == self.get_max_trajectory_length() - 1:
        #     # Prediction Rewards
        #     validity_of_predictions_vec = self.calculate_accuracy_of_trajectories(routing_data=routing_data,
        #                                                                           history=history)
        #     validity_of_prediction_rewards = np.array([TreeDepthPolicyNetwork.INVALID_PREDICTION_PENALTY,
        #                                                TreeDepthPolicyNetwork.VALID_PREDICTION_REWARD])
        #     prediction_rewards = validity_of_prediction_rewards[validity_of_predictions_vec]
        #     rewards_arr += prediction_rewards
        #     # Computation Cost Penalties
        #     activation_cost_arr = self.networkActivationCosts[history.actions[time_step]]
        #     activation_cost_arr = -TreeDepthPolicyNetwork.LAMBDA_MAC_COST * activation_cost_arr
        #     rewards_arr += activation_cost_arr
        # history.rewards.append(rewards_arr)

    def build_state_inputs(self, time_step):
        ordered_nodes_at_level = self.network.orderedNodesPerLevel[time_step]
        inputs_list = [self.validationFeaturesDict[node.index] for node in ordered_nodes_at_level]
        shape_set = {input_arr.shape for input_arr in inputs_list}
        assert len(shape_set) == 1
        feat_shape = list(shape_set)[0]
        input_shape = [None]
        input_shape.extend(feat_shape.shape[1:])
        tf_list = [
            tf.placeholder(dtype=tf.float32, shape=input_shape,
                           name="inputs_t{0}_node{1}".format(time_step, node.index))
            for node in ordered_nodes_at_level]
        self.stateInputs.append([tf_list])

    # Override this for different type of policy network implementations
    def build_policy_networks(self, time_step):
        pass

    def sample_from_policy_tf(self, time_step):
        samples = FastTreeNetwork.sample_from_categorical_v2(probs=self.policies[time_step])
        self.policySamples.append(samples)
        self.resultsDict["policy_samples_{0}".format(time_step)] = samples

    def build_state_transition(self, time_step):
        if time_step == 0:
            assert len(self.stateInputTransformed) == 0
            self.stateInputTransformed.append(self.stateInputs[0])

    def build_networks(self):
        self.reward_calculation_tf()
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            self.build_state_inputs(time_step=t)
            self.build_policy_networks(time_step=t)
            self.sample_from_policy_tf(time_step=t)

        #     reward_input = tf.placeholder(dtype=tf.float32, shape=[None], name="rewards_{0}".format(t))
        #     self.rewards.append(reward_input)
        #     baseline_input = tf.placeholder(dtype=tf.float32, shape=[None], name="baselines_{0}".format(t))
        #     self.baselinesTf.append(baseline_input)
        #     if t - 1 < 0:
        #         baseline_np = np.zeros(shape=(self.validationDataForMDP.routingDataset.labelList.shape[0], 1))
        #     else:
        #         baseline_np = np.zeros(shape=(self.validationDataForMDP.routingDataset.labelList.shape[0],
        #                                       self.actionSpaces[t - 1].shape[0]))
        #     self.baselinesNp.append(baseline_np)
        #     # Policy sampling
        #     sampler = FastTreeNetwork.sample_from_categorical_v2(probs=self.policies[t])
        #     self.policySamples.append(sampler)
        #     selected_policy_input = tf.placeholder(dtype=tf.int32, shape=[None],
        #                                            name="selected_policy_input_{0}".format(t))
        #     self.selectedPolicyInputs.append(selected_policy_input)
        #     # Argmax actions
        #     argmax_actions = tf.argmax(self.policies[t], axis=1)
        #     self.policyArgMaxSamples.append(argmax_actions)
        # # Get the total number of trajectories
        # state_input_shape = tf.shape(self.stateInputs[0])
        # self.trajectoryCount = tf.gather_nd(state_input_shape, [0])
        # # Cumulative Rewards
        # for t1 in range(max_trajectory_length):
        #     rew_list = [self.rewards[t2] for t2 in range(t1, max_trajectory_length, 1)]
        #     cum_sum = tf.add_n(rew_list)
        #     self.cumulativeRewards.append(cum_sum)
        # # Building the proxy loss and the policy gradient
        # self.build_policy_gradient_loss()
        # self.get_l2_loss()
        # self.build_optimizer()

    # Prevent policies from reaching exactly to 1.0
    # def clamp_policies(self, time_step):
    #     epsilon = 1e-30
    #     policy = self.policies[time_step]
    #     policy_count = self.actionSpaces[time_step].shape[1]
    #     inf_mask = tf.greater_equal(policy, 1.0 - epsilon)
    #     inf_detection_vector = tf.reduce_any(inf_mask, axis=1)
    #     # inf_detection_vector =
    #
    #
    #
    #     self.resultsDict["inf_mask_{0}".format(time_step)] = inf_mask
    #     self.resultsDict["inf_detection_vector_{0}".format(time_step)] = inf_detection_vector

    # state_input_t = self.stateInputTransformed[time_step]
    # hidden_layers = list(self.hiddenLayers[time_step])
    # hidden_layers.append(self.actionSpaces[time_step].shape[0])
    # net = state_input_t
    # for layer_id, layer_dim in enumerate(hidden_layers):
    #     if layer_id < len(hidden_layers) - 1:
    #         net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
    #     else:
    #         net = tf.layers.dense(inputs=net, units=layer_dim, activation=None)
    # _logits = net
    # _logits = _logits / self.softmaxDecay
    # self.logits.append(_logits)
    # self.policies.append(tf.nn.softmax(_logits))
    # self.logPolicies.append(tf.log(self.policies[-1]))

    # def sample_initial_states(self, routing_data, state_sample_count, samples_per_state, state_ids=None) \
    #         -> TrajectoryHistory:
    #     history = self.sample_initial_states(routing_data=routing_data,
    #                                          state_sample_count=state_sample_count,
    #                                          samples_per_state=samples_per_state,
    #                                          state_ids=state_ids)
    #     for t in range(1, self.get_max_trajectory_length()):
    #         nodes_in_level = self.network.orderedNodesPerLevel[t]
    #         feature_arrays = [routing_data.featuresDict[node.index][history.stateIds] for node in nodes_in_level]
    #
    #
    #     return history
    #
    # def sample_trajectories(self, routing_data, state_sample_count, samples_per_state,
    #                         select_argmax, ignore_invalid_actions, state_ids) \
    #         -> TrajectoryHistory:
    #     # if state_ids is None, sample from state distribution
    #     # Sample from s1 ~ p(s1)
    #     history = self.sample_initial_states(routing_data=routing_data,
    #                                          state_sample_count=state_sample_count,
    #                                          samples_per_state=samples_per_state,
    #                                          state_ids=state_ids)
    #     max_trajectory_length = self.get_max_trajectory_length()
    #     for t in range(max_trajectory_length):
    #         # Sample from a_t ~ p(a_t|history(t))
    #         self.sample_from_policy(routing_data=routing_data, history=history, time_step=t,
    #                                 select_argmax=select_argmax, ignore_invalid_actions=ignore_invalid_actions)
    #         # Get the reward: r_t ~ p(r_t|history(t))
    #         self.reward_calculation(routing_data=routing_data, history=history, time_step=t)
    #         # State transition s_{t+1} ~ p(s_{t+1}|history(t))
    #         if t < max_trajectory_length - 1:
    #             self.state_transition(routing_data=routing_data, history=history, time_step=t)
    #     return history


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
    network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
    routing_data = network.load_routing_info(run_id=run_id, iteration=iteration, data_type="test",
                                             output_names=output_names)
    validation_data, test_data = routing_data.apply_validation_test_split(test_ratio=0.1)

    wd_list = [0.0] * 10
    # [0.0, 0.00005, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005]
    state_sample_count_list = [1000]
    samples_per_state_list = [1]
    cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[wd_list,
                                                                          state_sample_count_list,
                                                                          samples_per_state_list])

    for tpl in cartesian_product:
        l2_wd = tpl[0]
        state_sample_count = tpl[1]
        samples_per_state = tpl[2]
        policy_gradients_routing_optimizer = \
            FullGpuPolicyGradientsNetwork(l2_lambda=l2_wd,
                                          network=network,
                                          network_name=network_name,
                                          run_id=run_id,
                                          iteration=iteration,
                                          degree_list=[2, 2],
                                          output_names=output_names,
                                          used_feature_names=used_output_names,
                                          use_baselines=True,
                                          state_sample_count=state_sample_count,
                                          trajectory_per_state_sample_count=samples_per_state,
                                          hidden_layers=[[128], [256]],
                                          validation_data=validation_data,
                                          test_data=test_data)
        policy_gradients_routing_optimizer.train(max_num_of_iterations=15000)


if __name__ == "__main__":
    main()
