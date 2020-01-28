import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.policy_gradients_network import \
    TrajectoryHistory
from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.tree_depth_policy_network import \
    TreeDepthPolicyNetwork


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

    def sample_trajectories(self, routing_data, state_sample_count, samples_per_state,
                            select_argmax, ignore_invalid_actions, state_ids) \
            -> TrajectoryHistory:
        # if state_ids is None, sample from state distribution
        # Sample from s1 ~ p(s1)
        history = self.sample_initial_states(routing_data=routing_data,
                                             state_sample_count=state_sample_count,
                                             samples_per_state=samples_per_state,
                                             state_ids=state_ids)
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            # Sample from a_t ~ p(a_t|history(t))
            self.sample_from_policy(routing_data=routing_data, history=history, time_step=t,
                                    select_argmax=select_argmax, ignore_invalid_actions=ignore_invalid_actions)
            # Get the reward: r_t ~ p(r_t|history(t))
            self.reward_calculation(routing_data=routing_data, history=history, time_step=t)
            # State transition s_{t+1} ~ p(s_{t+1}|history(t))
            if t < max_trajectory_length - 1:
                self.state_transition(routing_data=routing_data, history=history, time_step=t)
        return history