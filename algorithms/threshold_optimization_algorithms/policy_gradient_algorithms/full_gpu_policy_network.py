import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.policy_gradients_network import \
    TrajectoryHistory
from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.tree_depth_policy_network import \
    TreeDepthPolicyNetwork
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
        super().__init__(validation_data, test_data, l2_lambda, network, network_name, run_id, iteration, degree_list,
                         output_names, used_feature_names, hidden_layers, use_baselines, state_sample_count,
                         trajectory_per_state_sample_count)
        self.stateInputTransformed = []
        self.softmaxDecay = tf.placeholder(dtype=tf.float32, name="softmaxDecay")
        self.actionSpacesTf = []
        self.resultsDict = {}

    def build_action_spaces(self):
        super().build_action_spaces()
        for t in range(self.get_max_trajectory_length()):
            self.actionSpacesTf.append(tf.constant(self.actionSpaces[t]))
        self.resultsDict["actionSpacesTf"] = self.actionSpacesTf

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
