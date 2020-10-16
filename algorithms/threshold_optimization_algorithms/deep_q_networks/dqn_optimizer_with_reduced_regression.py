import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_with_regression import DqnWithRegression
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs


class DqnWithReducedRegression(DqnWithRegression):
    invalid_action_penalty = -1.0e10
    valid_prediction_reward = 1.0
    invalid_prediction_penalty = 0.0
    INCLUDE_IG_IN_REWARD_CALCULATIONS = False

    CONV_FEATURES = [[32], [64]]
    HIDDEN_LAYERS = [[128, 64], [128, 64]]
    FILTER_SIZES = [[1], [1]]
    STRIDES = [[1], [1]]
    MAX_POOL = [[None], [None]]

    def __init__(self, routing_dataset, network, network_name, run_id, used_feature_names, q_learning_func,
                 lambda_mac_cost, valid_prediction_reward, invalid_prediction_penalty, feature_type,
                 max_experience_count=100000):
        self.selectionIndices = tf.placeholder(dtype=tf.int32, name="selectionIndices", shape=[None, 2])
        self.selectedRewards = [None] * int(network.depth - 1)
        self.lossVectors = [None] * int(network.depth - 1)
        super().__init__(routing_dataset, network, network_name, run_id, used_feature_names, q_learning_func,
                         lambda_mac_cost,
                         DqnWithReducedRegression.invalid_action_penalty,
                         valid_prediction_reward,
                         invalid_prediction_penalty,
                         feature_type,
                         max_experience_count)

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

    def get_selection_indices(self, level, actions_t_minus_1, non_zeros=True):
        reachability_matrix = self.reachabilityMatrices[level][actions_t_minus_1]
        if non_zeros:
            indices = np.nonzero(reachability_matrix == 1)
        else:
            indices = np.nonzero(reachability_matrix == 0)
        idx_array = np.stack([indices[0], indices[1]], axis=1)
        return idx_array

    def calculate_q_tables_with_dqn(self, discount_rate, dqn_lowest_level=np.inf):
        q_tables = [None] * self.get_max_trajectory_length()
        last_level = self.get_max_trajectory_length()
        total_sample_count = self.rewardTensors[0].shape[0]
        for t in range(last_level - 1, -1, -1):
            action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
            if t >= dqn_lowest_level:
                state_id_tuples = self.get_state_tuples(sample_indices=list(range(total_sample_count)), level=t)
                q_table_predicted = self.create_q_table(level=t, sample_indices=state_id_tuples[:, 0],
                                                        action_ids_t_minus_1=state_id_tuples[:, 1])
                # Set non accesible indices to -np.inf
                idx_array = self.get_selection_indices(level=t, actions_t_minus_1=state_id_tuples[:, 1],
                                                       non_zeros=False)
                q_table_predicted[idx_array[:, 0], idx_array[:, 1]] = DqnWithReducedRegression.invalid_action_penalty
                # Reshape for further processing
                assert q_table_predicted.shape[0] == total_sample_count * action_count_t_minus_one \
                       and len(q_table_predicted.shape) == 2
                q_table_predicted = np.reshape(q_table_predicted,
                                               newshape=(total_sample_count, action_count_t_minus_one,
                                                         q_table_predicted.shape[1]))
                q_tables[t] = q_table_predicted
            else:
                # Get the rewards for that time step
                if t == last_level - 1:
                    q_tables[t] = self.rewardTensors[t]
                else:
                    rewards_t = self.rewardTensors[t]
                    q_next = q_tables[t + 1]
                    q_star = np.max(q_next, axis=-1)
                    q_t = rewards_t + discount_rate * q_star[:, np.newaxis, :]
                    q_tables[t] = q_t
        return q_tables

    # Calculate the estimated Q-Table vs actual Q-table divergence and related scores for the given layer.
    def measure_performance(self, level, Q_tables_whole, sample_indices):
        state_id_tuples = self.get_state_tuples(sample_indices=sample_indices, level=level)
        Q_table_predicted = Q_tables_whole[level][state_id_tuples[:, 0], state_id_tuples[:, 1]]
        Q_table_truth = self.optimalQTables[level][state_id_tuples[:, 0], state_id_tuples[:, 1]]
        # Calculate the mean policy value
        mean_policy_value = np.mean(np.max(Q_table_predicted, axis=1))
        # Calculate the MSE between the Q_{t}^{predicted}(s,a) and Q_{t}^{actual}(s,a).
        assert Q_table_predicted.shape == Q_table_truth.shape
        # Account for -np.inf entries
        reachability_matrix = self.reachabilityMatrices[level][state_id_tuples[:, 1]]
        assert Q_table_predicted.shape == reachability_matrix.shape
        y = Q_table_truth[reachability_matrix == 1]
        y_hat = Q_table_predicted[reachability_matrix == 1]
        mse_score = mean_squared_error(y_true=y, y_pred=y_hat)
        return mean_policy_value, mse_score

    def train(self, level, **kwargs):
        sample_count = kwargs["sample_count"]
        episode_count = kwargs["episode_count"]
        discount_factor = kwargs["discount_factor"]
        epsilon_discount_factor = kwargs["epsilon_discount_factor"]
        learning_rate = kwargs["learning_rate"]
        epsilon = 1.0
        if level != self.get_max_trajectory_length() - 1:
            raise NotImplementedError()
        self.session.run(tf.global_variables_initializer())
        # If we use only information gain for routing (ML: Maximum likelihood routing)
        kwargs["lrValues"] = self.lrValues
        kwargs["lrBoundaries"] = self.lrBoundaries
        run_id = self.log_meta_data(kwargs=kwargs)
        losses = []
        # Calculate the ultimate, optimal Q Tables.
        self.optimalQTables = self.calculate_q_tables_with_dqn(discount_rate=discount_factor)
        # These are for testing purposes
        # optimal_q_tables_test = self.calculate_q_tables_for_test(discount_rate=discount_factor)
        # assert len(self.optimalQTables) == len(optimal_q_tables_test)
        # for t in range(len(self.optimalQTables)):
        #     assert np.allclose(self.optimalQTables[t], optimal_q_tables_test[t])
        self.evaluate(run_id=run_id, episode_id=-1, level=level, discount_factor=discount_factor)
        for episode_id in range(episode_count):
            print("Episode:{0}".format(episode_id))
            sample_ids = np.random.choice(self.routingDataset.trainingIndices, sample_count, replace=True)
            actions_t_minus_1 = np.random.choice(self.actionSpaces[level - 1].shape[0], sample_count, replace=True)
            optimal_q_values = self.optimalQTables[level][sample_ids, actions_t_minus_1]
            idx_array = self.get_selection_indices(level=level, actions_t_minus_1=actions_t_minus_1)
            for s_id, a_t_minus_1 in zip(sample_ids, actions_t_minus_1):
                if (s_id, a_t_minus_1) not in self.processedPairs:
                    self.processedPairs[(s_id, a_t_minus_1)] = 0
                self.processedPairs[(s_id, a_t_minus_1)] += 1
            state_features = self.get_state_features(sample_indices=sample_ids,
                                                     action_ids_t_minus_1=actions_t_minus_1,
                                                     level=level)
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
                                                  self.l2LambdaTf: 0.0})
            total_loss = results[0]
            losses.append(total_loss)
            if len(losses) % 10 == 0:
                print("Episode:{0} MSE:{1}".format(episode_id, np.mean(np.array(losses))))
                losses = []
            if (episode_id + 1) % 200 == 0:
                if (episode_id + 1) == 10000:
                    print("X")
                self.evaluate(run_id=run_id, episode_id=episode_id, level=level, discount_factor=discount_factor)
