import tensorflow as tf
import numpy as np
from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.full_gpu_tree_policy_network import \
    FullGpuTreePolicyGradientsNetwork


class FullGpuTreePolicyTreeNetworkValidActions(FullGpuTreePolicyGradientsNetwork):
    def __init__(self, validation_data, test_data, l2_lambda, network, network_name, run_id, iteration, degree_list,
                 output_names, used_feature_names, policy_network_func, hidden_layers, use_baselines,
                 state_sample_count, trajectory_per_state_sample_count):
        self.passiveWeight = tf.constant(-1e+10)
        self.epsilonProb = tf.constant(1e-10)
        super().__init__(validation_data, test_data, l2_lambda, network, network_name, run_id, iteration, degree_list,
                         output_names, used_feature_names, policy_network_func, hidden_layers, use_baselines,
                         state_sample_count, trajectory_per_state_sample_count)

    def build_policy_generators(self, time_step):
        actions_t_minus_one = tf.zeros_like(self.stateIds) if time_step == 0 else self.finalActions[time_step - 1]
        reachability_matrix = tf.gather_nd(self.reachabilityMatricesTf[time_step],
                                           tf.expand_dims(actions_t_minus_one, axis=1))
        _logits = self.logits[time_step] / self.softmaxDecay
        passive_weights_matrix = tf.ones_like(_logits, dtype=tf.float32) * self.passiveWeight
        sparse_logits = tf.where(tf.cast(reachability_matrix, tf.bool), _logits, passive_weights_matrix)
        policies_non_modified = tf.nn.softmax(sparse_logits)
        policies = policies_non_modified + tf.where(tf.cast(reachability_matrix, tf.bool),
                                                    tf.zeros_like(policies_non_modified, dtype=tf.float32),
                                                    tf.ones_like(policies_non_modified,
                                                                 dtype=tf.float32) * self.epsilonProb)
        self.policies.append(policies)
        self.logPolicies.append(tf.log(self.policies[-1]))
        self.resultsDict["logits_{0}".format(time_step)] = self.logits[time_step]
        self.resultsDict["reachability_matrix_{0}".format(time_step)] = reachability_matrix
        self.resultsDict["policies_{0}".format(time_step)] = self.policies[time_step]
        self.resultsDict["logPolicies_{0}".format(time_step)] = self.logPolicies[time_step]

    # def sample_from_policy_tf(self, time_step):
    #     sampled_actions = FastTreeNetwork.sample_from_categorical_v2(probs=self.policies[time_step])
    #     actions_t_minus_one = tf.zeros_like(self.stateIds) if time_step == 0 else self.finalActions[time_step - 1]
    #     reachability_matrix = tf.gather_nd(self.reachabilityMatricesTf[time_step],
    #                                        tf.expand_dims(actions_t_minus_one, axis=1))
    #     argmax_actions_all = tf.cast(tf.argmax(self.policies[time_step], axis=1), dtype=tf.int32)
    #     valid_policies = self.policies[time_step] * tf.cast(reachability_matrix, dtype=tf.float32)
    #     argmax_actions_only_valid = tf.cast(tf.argmax(valid_policies, axis=1), dtype=tf.int32)
    #     argmax_actions = tf.where(self.ignoreInvalidActions, argmax_actions_only_valid, argmax_actions_all)
    #     samples = tf.where(self.isSamplingTrajectory, sampled_actions, argmax_actions)
    #     routing_decisions = tf.gather_nd(self.actionSpacesTf[time_step], tf.expand_dims(samples, axis=1))
    #     self.policySamples.append(sampled_actions)
    #     self.argMaxActions.append(argmax_actions)
    #     self.finalActions.append(samples)
    #     self.routingDecisions.append(routing_decisions)
    #     self.resultsDict["reachability_matrix_{0}".format(time_step)] = reachability_matrix
    #     self.resultsDict["valid_policies_{0}".format(time_step)] = valid_policies
    #     self.resultsDict["policySamples_{0}".format(time_step)] = sampled_actions
    #     self.resultsDict["argmax_actions_all_{0}".format(time_step)] = argmax_actions_all
    #     self.resultsDict["argmax_actions_only_valid_{0}".format(time_step)] = argmax_actions_only_valid
    #     self.resultsDict["argMaxActions_{0}".format(time_step)] = argmax_actions
    #     self.resultsDict["routingDecisions_{0}".format(time_step)] = routing_decisions
    #     self.resultsDict["finalActions_{0}".format(time_step)] = samples