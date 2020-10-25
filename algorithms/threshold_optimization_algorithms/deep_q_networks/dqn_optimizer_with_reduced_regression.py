import tensorflow as tf

from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_with_regression import DqnWithRegression


class DqnWithReducedRegression(DqnWithRegression):
    def __init__(self, routing_dataset, network, network_name, run_id, used_feature_names, dqn_func,
                 lambda_mac_cost, valid_prediction_reward, invalid_prediction_penalty,
                 include_ig_in_reward_calculations,
                 feature_type,
                 dqn_parameters):
        self.selectionIndices = tf.placeholder(dtype=tf.int32, name="selectionIndices", shape=[None, 2])
        self.selectedRewards = [None] * int(network.depth - 1)
        self.lossVectors = [None] * int(network.depth - 1)
        super().__init__(routing_dataset, network, network_name, run_id, used_feature_names, dqn_func,
                         lambda_mac_cost,
                         -1.0e10,
                         valid_prediction_reward,
                         invalid_prediction_penalty,
                         include_ig_in_reward_calculations,
                         feature_type,
                         dqn_parameters)
        self.useReachability = True

    def build_loss(self, level):
        # Get selected q values; build the regression loss: MSE or Huber between Last layer Q outputs and the reward
        self.rewardMatrices[level] = tf.placeholder(dtype=tf.float32, shape=[None, self.actionSpaces[level].shape[0]],
                                                    name="reward_matrix_{0}".format(level))
        update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS, scope="dqn_{0}".format(level))
        with tf.control_dependencies(update_ops):
            self.selectedRewards[level] = tf.gather_nd(self.rewardMatrices[level], self.selectionIndices)
            self.selectedQValues[level] = tf.gather_nd(self.qFuncs[level], self.selectionIndices)
            self.lossVectors[level] = tf.square(self.selectedQValues[level] - self.selectedRewards[level])
            self.regressionLossValues[level] = tf.reduce_mean(self.lossVectors[level])
            self.get_l2_loss(level=level)
            self.totalLosses[level] = self.regressionLossValues[level] + self.l2Losses[level]
            self.optimizers[level] = tf.train.AdamOptimizer().minimize(self.totalLosses[level],
                                                                       global_step=self.globalSteps[level])
        # self.optimizer = tf.train.MomentumOptimizer(self.learningRate, 0.9).\
        #     minimize(self.totalLoss, global_step=self.globalStep)

    def prepare_explanation_string(self, kwargs):
        explanation_string = super().prepare_explanation_string(kwargs=kwargs)
        kwargs["SE_REDUCTION_RATIO"] = self.dqnParameters["Squeeze_And_Excitation"]["SE_REDUCTION_RATIO"]
        return explanation_string

    def run_training_step(self, level, **kwargs):
        sample_count = kwargs["sample_count"]
        state_features = kwargs["state_features"]
        optimal_q_values = kwargs["optimal_q_values"]
        idx_array = kwargs["idx_array"]
        l2_lambda = kwargs["l2_lambda"]
        results = self.session.run([self.totalLosses[level],
                                    self.selectedRewards[level],
                                    self.selectedQValues[level],
                                    self.lossVectors[level],
                                    self.regressionLossValues[level],
                                    self.optimizers[level]],
                                   feed_dict={self.stateCount: sample_count,
                                              self.stateInputs[level]: state_features,
                                              self.rewardMatrices[level]: optimal_q_values,
                                              self.selectionIndices: idx_array,
                                              self.isTrain: True,
                                              self.l2LambdaTf: l2_lambda})
        return results
