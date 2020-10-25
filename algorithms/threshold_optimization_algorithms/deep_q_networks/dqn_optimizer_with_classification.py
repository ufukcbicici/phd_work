import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import os
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_with_regression import DqnWithRegression
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from sklearn.metrics import log_loss


class DqnWithClassification(DqnWithRegression):
    def __init__(self, routing_dataset, network, network_name, run_id, used_feature_names, dqn_func,
                 lambda_mac_cost, valid_prediction_reward, invalid_prediction_penalty,
                 include_ig_in_reward_calculations,
                 feature_type,
                 dqn_parameters):
        self.selectionLabels = tf.placeholder(dtype=tf.int32, name="selectionLabels", shape=(None,))
        self.crossEntropyLossValues = [None] * int(network.depth - 1)
        self.softmaxOutputs = [None] * int(network.depth - 1)
        self.lossVectors = [None] * int(network.depth - 1)
        super().__init__(routing_dataset, network, network_name, run_id, used_feature_names, dqn_func,
                         lambda_mac_cost,
                         -1.0e10,
                         valid_prediction_reward,
                         invalid_prediction_penalty,
                         include_ig_in_reward_calculations,
                         dqn_parameters,
                         feature_type)
        self.useReachability = True

    def build_loss(self, level):
        # Get selected q values; build the regression loss: MSE or Huber between Last layer Q outputs and the reward
        update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS, scope="dqn_{0}".format(level))
        with tf.control_dependencies(update_ops):
            self.lossVectors[level] = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.selectionLabels,
                                                                                     logits=self.qFuncs[level])
            self.softmaxOutputs[level] = tf.nn.softmax(self.qFuncs[level])
            self.crossEntropyLossValues[level] = tf.reduce_mean(self.lossVectors[level])
            self.get_l2_loss(level=level)
            self.totalLosses[level] = self.crossEntropyLossValues[level] + self.l2Losses[level]
            self.optimizers[level] = tf.train.AdamOptimizer().minimize(self.totalLosses[level],
                                                                       global_step=self.globalSteps[level])

    def calculate_q_table_with_dqn(self, level):
        action_count_t_minus_one = 1 if level == 0 else self.actionSpaces[level - 1].shape[0]
        total_sample_count = self.rewardTensors[0].shape[0]
        state_id_tuples = self.get_state_tuples(sample_indices=list(range(total_sample_count)), level=level)
        q_table_predicted = self.calculate_q_values(level=level, sample_indices=state_id_tuples[:, 0],
                                                    action_ids_t_minus_1=state_id_tuples[:, 1])
        # Set non accesible indices to -np.inf
        idx_array = self.get_selection_indices(level=level, actions_t_minus_1=state_id_tuples[:, 1],
                                               non_zeros=False)
        q_table_predicted[idx_array[:, 0], idx_array[:, 1]] = DqnWithClassification.invalid_action_penalty
        # Reshape for further processing
        assert q_table_predicted.shape[0] == total_sample_count * action_count_t_minus_one and \
               len(q_table_predicted.shape) == 2
        q_table_predicted = np.reshape(q_table_predicted,
                                       newshape=(total_sample_count, action_count_t_minus_one,
                                                 q_table_predicted.shape[1]))
        return q_table_predicted

    # Calculate the estimated Q-Table vs actual Q-table divergence and related scores for the given layer.
    def measure_performance(self, level, Q_tables_whole, sample_indices):
        state_id_tuples = self.get_state_tuples(sample_indices=sample_indices, level=level)
        Q_table_predicted = Q_tables_whole[level][state_id_tuples[:, 0], state_id_tuples[:, 1]]
        Q_table_truth = self.optimalQTables[level][state_id_tuples[:, 0], state_id_tuples[:, 1]]
        # Truth labels
        truth_labels = np.argmax(Q_table_truth, axis=1)
        # Prediction labels
        prediction_labels = np.argmax(Q_table_predicted, axis=1)
        # Cross entropy loss
        exp_q_table = np.exp(Q_table_predicted)
        predicted_probs = exp_q_table * np.reciprocal(np.sum(exp_q_table, axis=1, keepdims=True))
        # selected_probs = predicted_probs[np.arange(predicted_probs.shape[0]), truth_labels]
        # cross_entropy_loss = np.mean(-np.log(selected_probs))
        cross_entropy_loss = log_loss(truth_labels, predicted_probs,
                                      labels=np.arange(self.actionSpaces[level].shape[0]))
        accuracy = np.mean(truth_labels == prediction_labels)
        return accuracy, cross_entropy_loss

    def evaluate(self, run_id, episode_id, level, discount_factor):
        # Get the q-tables for all samples
        q_tables = self.calculate_estimated_q_tables(dqn_level=level)
        print("***********Training Set***********")
        training_mean_accuracy, training_cross_entropy_loss = \
            self.measure_performance(level=level, Q_tables_whole=q_tables,
                                     sample_indices=self.routingDataset.trainingIndices)
        # training_mean_policy_value_optimal, training_mse_score_optimal = \
        #     self.measure_performance(level=level,
        #                              Q_tables_whole=self.optimalQTables,
        #                              sample_indices=self.routingDataset.trainingIndices)
        _, _, training_accuracy, training_computation_cost = \
            self.execute_bellman_equation(Q_tables=q_tables,
                                          sample_indices=self.routingDataset.trainingIndices)
        _, _, training_accuracy_optimal, training_computation_cost_optimal = \
            self.execute_bellman_equation(Q_tables=self.optimalQTables,
                                          sample_indices=self.routingDataset.trainingIndices)
        # self.result_analysis(level=level, q_tables=self.optimalQTables, q_hat_tables=q_tables,
        #                      indices=self.routingDataset.trainingIndices)
        print("***********Test Set***********")
        test_mean_accuracy, test_cross_entropy_loss = \
            self.measure_performance(level=level,
                                     Q_tables_whole=q_tables,
                                     sample_indices=self.routingDataset.testIndices)
        # test_mean_policy_value_optimal, test_mse_score_optimal = \
        #     self.measure_performance(level=level,
        #                              Q_tables_whole=self.optimalQTables,
        #                              sample_indices=self.routingDataset.testIndices)
        _, _, test_accuracy, test_computation_cost = \
            self.execute_bellman_equation(Q_tables=q_tables,
                                          sample_indices=self.routingDataset.testIndices)
        _, _, test_accuracy_optimal, test_computation_cost_optimal = \
            self.execute_bellman_equation(Q_tables=self.optimalQTables,
                                          sample_indices=self.routingDataset.testIndices)
        print("training_mean_accuracy:{0} training_cross_entropy_loss:{1}"
              .format(training_mean_accuracy, training_cross_entropy_loss))
        print("test_mean_accuracy:{0} test_cross_entropy_loss:{1}"
              .format(test_mean_accuracy, test_cross_entropy_loss))
        print("test_accuracy_optimal:{0} test_computation_cost_optimal:{1}"
              .format(test_accuracy_optimal, test_computation_cost_optimal))
        DbLogger.write_into_table(
            rows=[(run_id, episode_id,
                   np.asscalar(training_mean_accuracy), training_cross_entropy_loss,
                   np.asscalar(training_accuracy), np.asscalar(training_computation_cost),
                   np.asscalar(test_mean_accuracy), test_cross_entropy_loss,
                   np.asscalar(test_accuracy), np.asscalar(test_computation_cost))],
            table="deep_q_learning_logs", col_count=10)

    def train(self, level, **kwargs):
        self.setup_before_training(level=level, **kwargs)
        sample_count = kwargs["sample_count"]
        l2_lambda = kwargs["l2_lambda"]
        run_id = kwargs["run_id"]
        discount_factor = kwargs["discount_factor"]
        episode_count = kwargs["episode_count"]
        self.saver = tf.train.Saver()
        losses = []
        # These are for testing purposes
        optimal_q_tables_test = self.calculate_q_tables_for_test(discount_rate=discount_factor)
        assert len(self.optimalQTables) == len(optimal_q_tables_test)
        for t in range(len(self.optimalQTables)):
            assert np.allclose(self.optimalQTables[t], optimal_q_tables_test[t])
        self.evaluate(run_id=run_id, episode_id=-1, level=level, discount_factor=discount_factor)
        for episode_id in range(episode_count):
            print("Episode:{0}".format(episode_id))
            sample_ids = np.random.choice(self.routingDataset.trainingIndices, sample_count, replace=True)
            actions_t_minus_1 = np.random.choice(self.actionSpaces[level - 1].shape[0], sample_count, replace=True)
            optimal_q_values = self.optimalQTables[level][sample_ids, actions_t_minus_1]
            selection_labels = np.argmax(optimal_q_values, axis=1)
            for s_id, a_t_minus_1 in zip(sample_ids, actions_t_minus_1):
                if (s_id, a_t_minus_1) not in self.processedPairs:
                    self.processedPairs[(s_id, a_t_minus_1)] = 0
                self.processedPairs[(s_id, a_t_minus_1)] += 1
            state_features = self.get_state_features(sample_indices=sample_ids,
                                                     action_ids_t_minus_1=actions_t_minus_1,
                                                     level=level)
            results = self.session.run([self.totalLosses[level],
                                        self.selectionLabels[level],
                                        self.lossVectors[level],
                                        self.crossEntropyLossValues[level],
                                        self.l2Losses[level],
                                        self.optimizers[level]],
                                       feed_dict={self.stateCount: sample_count,
                                                  self.stateInputs[level]: state_features,
                                                  self.selectionLabels: selection_labels,
                                                  self.isTrain: True,
                                                  self.l2LambdaTf: l2_lambda})
            total_loss = results[0]
            losses.append(total_loss)
            if len(losses) % 10 == 0:
                print("Episode:{0} MSE:{1}".format(episode_id, np.mean(np.array(losses))))
                losses = []
            if (episode_id + 1) % 200 == 0:
                self.evaluate(run_id=run_id, episode_id=episode_id, level=level, discount_factor=discount_factor)
        model_path = os.path.join("..", "dqn_models", "dqn_run_id_{0}".format(run_id))
        os.mkdir(model_path)
        self.saver.save(self.session, os.path.join(model_path, "dqn_run_id_{0}.ckpt".format(run_id)))
