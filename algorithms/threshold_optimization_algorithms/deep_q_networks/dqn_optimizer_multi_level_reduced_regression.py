import tensorflow as tf

from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_multi_level_regression import \
    DqnMultiLevelRegression
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_with_regression import DqnWithRegression


class DqnMultiLevelReducedRegression(DqnMultiLevelRegression):
    def __init__(self, routing_dataset, network, network_name, run_id, used_feature_names, dqn_func,
                 lambda_mac_cost, valid_prediction_reward, invalid_prediction_penalty,
                 include_ig_in_reward_calculations,
                 feature_type,
                 dqn_parameters):
        self.selectionIndices = [tf.placeholder(dtype=tf.int32, name="selectionIndices_{0}".format(idx),
                                                shape=[None, 2]) for idx in range(network.depth - 1)]
        self.selectedRewards = [None] * int(network.depth - 1)
        self.lossVectors = [None] * int(network.depth - 1)
        super().__init__(routing_dataset, network, network_name, run_id, used_feature_names, dqn_func,
                         lambda_mac_cost,
                         -100,
                         valid_prediction_reward,
                         invalid_prediction_penalty,
                         include_ig_in_reward_calculations,
                         feature_type,
                         dqn_parameters)
        self.useReachability = True
        self.build_total_loss()

    def build_loss(self, level):
        # Get selected q values; build the regression loss: MSE or Huber between Last layer Q outputs and the reward
        self.rewardMatrices[level] = tf.placeholder(dtype=tf.float32, shape=[None, self.actionSpaces[level].shape[0]],
                                                    name="reward_matrix_{0}".format(level))
        update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS, scope="dqn_{0}".format(level))
        with tf.control_dependencies(update_ops):
            self.selectedRewards[level] = tf.gather_nd(self.rewardMatrices[level], self.selectionIndices[level])
            self.selectedQValues[level] = tf.gather_nd(self.qFuncs[level], self.selectionIndices[level])
            self.lossVectors[level] = tf.square(self.selectedQValues[level] - self.selectedRewards[level])
            self.regressionLossValues[level] = tf.reduce_mean(self.lossVectors[level])
            self.get_l2_loss(level=level)
            self.totalLosses[level] = self.regressionLossValues[level] + self.l2Losses[level]

    def prepare_explanation_string(self, kwargs):
        kwargs["SE_REDUCTION_RATIO"] = self.dqnParameters["Squeeze_And_Excitation"]["SE_REDUCTION_RATIO"]
        explanation_string = super().prepare_explanation_string(kwargs=kwargs)
        return explanation_string

    def fill_eval_list_feed_dict(self, level, eval_list, feed_dict, **kwargs):
        state_features = kwargs["state_features"]
        optimal_q_values = kwargs["optimal_q_values"]
        idx_array = kwargs["idx_array"]
        eval_list.extend(
            [self.totalLosses[level], self.selectedRewards[level], self.selectedQValues[level],
             self.lossVectors[level], self.regressionLossValues[level]])
        feed_dict[self.stateInputs[level]] = state_features
        feed_dict[self.rewardMatrices[level]] = optimal_q_values
        feed_dict[self.selectionIndices[level]] = idx_array
