import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import os

from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_multi_level_regression import \
    DqnMultiLevelRegression
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_with_regression import DqnWithRegression
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from sklearn.metrics import log_loss


class DqnWithClassification(DqnMultiLevelRegression):
    def __init__(self, routing_dataset, network, network_name, run_id, used_feature_names, dqn_func,
                 lambda_mac_cost, valid_prediction_reward, invalid_prediction_penalty,
                 include_ig_in_reward_calculations,
                 feature_type,
                 dqn_parameters):
        self.selectionLabels = [tf.placeholder(dtype=tf.int32, name="selectionLabels_{0}".format(idx),
                                               shape=(None,)) for idx in range(network.depth - 1)]

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

    def prepare_explanation_string(self, kwargs):
        kwargs["SE_REDUCTION_RATIO"] = self.dqnParameters["Squeeze_And_Excitation"]["SE_REDUCTION_RATIO"]
        explanation_string = super().prepare_explanation_string(kwargs=kwargs)
        return explanation_string

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

    def evaluate(self, run_id, episode_id, discount_factor, level=None):
        # Get the q-tables for all samples
        last_level = self.get_max_trajectory_length()
        # Get the q-tables for all samples
        q_tables = self.calculate_estimated_q_tables()
        results_dict = {}
        for data_type, indices in zip(["training", "test"],
                                      [self.routingDataset.trainingIndices, self.routingDataset.testIndices]):
            results_dict[data_type] = {}
            _, _, accuracy, computation_cost = self.execute_bellman_equation(Q_tables=q_tables, sample_indices=indices)
            _, _, accuracy_optimal, computation_cost_optimal = self.execute_bellman_equation(
                Q_tables=self.optimalQTables,
                sample_indices=indices)
            results_dict[data_type]["accuracy"] = accuracy
            results_dict[data_type]["computation_cost"] = computation_cost
            results_dict[data_type]["accuracy_optimal"] = accuracy_optimal
            results_dict[data_type]["computation_cost_optimal"] = computation_cost_optimal
            for level in range(last_level):
                print("***********{0}***********".format(data_type))
                mean_accuracy, cross_entropy_loss = self.measure_performance(level=level,
                                                                             Q_tables_whole=q_tables,
                                                                             sample_indices=indices)
                mean_accuracy_optimal, cross_entropy_loss_optimal = self.measure_performance(level=level,
                                                                                             Q_tables_whole=
                                                                                             self.optimalQTables,
                                                                                             sample_indices=indices)
                results_dict[data_type]["mean_accuracy_{0}".format(level)] = mean_accuracy
                results_dict[data_type]["cross_entropy_loss_{0}".format(level)] = cross_entropy_loss
                results_dict[data_type]["mean_accuracy_optimal_{0}".format(level)] = mean_accuracy_optimal
                results_dict[data_type][
                    "cross_entropy_loss_optimal_{0}".format(level)] = cross_entropy_loss_optimal
        training_total_ce_list = [results_dict["training"][k] for k in results_dict["training"].keys()
                                  if "cross_entropy_loss" in k and "optimal" not in k]
        training_total_ce = sum(training_total_ce_list)
        training_accuracy = results_dict["training"]["accuracy"]
        training_computation_cost = results_dict["training"]["computation_cost"]
        test_total_ce_list = [results_dict["test"][k] for k in results_dict["test"].keys()
                              if "cross_entropy_loss" in k and "optimal" not in k]
        test_total_ce = sum(test_total_ce_list)
        test_accuracy = results_dict["test"]["accuracy"]
        test_computation_cost = results_dict["test"]["computation_cost"]
        log_string = str(results_dict)

        DbLogger.write_into_table(
            rows=[(run_id,
                   episode_id,
                   0.0,
                   np.asscalar(training_total_ce),
                   np.asscalar(training_accuracy),
                   np.asscalar(training_computation_cost),
                   0.0,
                   np.asscalar(test_total_ce),
                   np.asscalar(test_accuracy),
                   np.asscalar(test_computation_cost),
                   log_string)],
            table="deep_q_learning_logs", col_count=11)

    def fill_eval_list_feed_dict(self, level, eval_list, feed_dict, **kwargs):
        state_features = kwargs["state_features"]
        selection_labels = kwargs["selection_labels"]
        eval_list.extend(
            [self.totalLosses[level], self.selectionLabels[level],
             self.lossVectors[level], self.crossEntropyLossValues[level], self.l2Losses[level]])
        feed_dict[self.stateInputs[level]] = state_features
        feed_dict[self.selectionLabels[level]] = selection_labels

    def train(self, level=None, **kwargs):
        self.saver = tf.train.Saver()
        losses = []
        self.setup_before_training(kwargs=kwargs)
        # These are for testing purposes
        optimal_q_tables_test = self.calculate_q_tables_for_test(discount_rate=kwargs["discount_factor"])
        assert len(self.optimalQTables) == len(optimal_q_tables_test)
        for t in range(len(self.optimalQTables)):
            assert np.allclose(self.optimalQTables[t], optimal_q_tables_test[t])
        self.evaluate(run_id=kwargs["run_id"], episode_id=-1, discount_factor=kwargs["discount_factor"])
        sample_count = kwargs["sample_count"]
        l2_lambda = kwargs["l2_lambda"]
        measurement_period = kwargs["measurement_period"]
        eval_list = [self.totalSystemLoss, self.totalOptimizer]
        feed_dict = {self.stateCount: sample_count,
                     self.isTrain: True,
                     self.l2LambdaTf: l2_lambda}
        for episode_id in range(kwargs["episode_count"]):
            print("Episode:{0}".format(episode_id))
            last_level = self.get_max_trajectory_length()
            for t in range(last_level):
                sample_ids = np.random.choice(self.routingDataset.trainingIndices, sample_count, replace=True)
                action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
                actions_t_minus_1 = np.random.choice(action_count_t_minus_one, sample_count, replace=True)
                optimal_q_values = self.optimalQTables[t][sample_ids, actions_t_minus_1]
                selection_labels = np.argmax(optimal_q_values, axis=1)
                state_features = self.get_state_features(sample_indices=sample_ids,
                                                         action_ids_t_minus_1=actions_t_minus_1,
                                                         level=t)
                self.fill_eval_list_feed_dict(level=t, eval_list=eval_list,
                                              feed_dict=feed_dict, state_features=state_features,
                                              selection_labels=selection_labels)
            results = self.session.run(eval_list, feed_dict=feed_dict)
            total_loss = results[0]
            losses.append(total_loss)
            if len(losses) % 10 == 0:
                print("Episode:{0} MSE:{1}".format(episode_id, np.mean(np.array(losses))))
                losses = []
            if (episode_id + 1) % measurement_period == 0:
                self.evaluate(run_id=kwargs["run_id"], episode_id=episode_id, discount_factor=kwargs["discount_factor"])
        model_path = os.path.join("..", "dqn_models", "dqn_run_id_{0}".format(kwargs["run_id"]))
        os.mkdir(model_path)
        self.saver.save(self.session, os.path.join(model_path, "dqn_run_id_{0}.ckpt".format(kwargs["run_id"])))