import tensorflow as tf
import numpy as np
import os
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_with_regression import DqnWithRegression
from auxillary.db_logger import DbLogger


class DqnMultiLevelRegression(DqnWithRegression):
    def __init__(self, routing_dataset, network, network_name, run_id, used_feature_names, dqn_func, lambda_mac_cost,
                 invalid_action_penalty, valid_prediction_reward, invalid_prediction_penalty,
                 include_ig_in_reward_calculations, feature_type, dqn_parameters):
        super().__init__(routing_dataset, network, network_name, run_id, used_feature_names, dqn_func, lambda_mac_cost,
                         invalid_action_penalty, valid_prediction_reward, invalid_prediction_penalty,
                         include_ig_in_reward_calculations, feature_type, dqn_parameters)
        self.totalSystemLoss = None
        self.totalOptimizer = None
        self.totalGlobalStep = tf.Variable(0, name="total_global_step", trainable=False)

    def build_total_loss(self):
        with tf.control_dependencies(self.totalLosses):
            self.totalSystemLoss = tf.add_n(self.totalLosses)
            self.totalOptimizer = tf.train.AdamOptimizer().minimize(self.totalSystemLoss,
                                                                    global_step=self.totalGlobalStep)

    def setup_before_training(self, kwargs, level=None):
        discount_factor = kwargs["discount_factor"]
        self.session.run(tf.global_variables_initializer())
        # If we use only information gain for routing (ML: Maximum likelihood routing)
        kwargs["lrValues"] = self.lrValues
        kwargs["lrBoundaries"] = self.lrBoundaries
        self.calculate_ml_performance(kwargs=kwargs)
        explanation_string = self.prepare_explanation_string(kwargs=kwargs)
        DbLogger.write_into_table(rows=[(kwargs["run_id"], explanation_string)], table=DbLogger.runMetaData,
                                  col_count=2)
        # Calculate the ultimate, optimal Q Tables.
        self.calculate_optimal_q_tables(discount_rate=discount_factor)

    def calculate_estimated_q_tables(self, dqn_level=None):
        arrays = []
        last_level = self.get_max_trajectory_length()
        for t in range(last_level):
            arrays.append(self.calculate_q_table_with_dqn(level=t))
        return arrays

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
                mean_policy_value, mse_score = self.measure_performance(level=level,
                                                                        Q_tables_whole=q_tables,
                                                                        sample_indices=indices)
                mean_policy_value_optimal, mse_score_optimal = self.measure_performance(level=level,
                                                                                        Q_tables_whole=
                                                                                        self.optimalQTables,
                                                                                        sample_indices=indices)
                results_dict[data_type]["mean_policy_value_{0}".format(level)] = mean_policy_value
                results_dict[data_type]["mse_score_{0}".format(level)] = mse_score
                results_dict[data_type]["mean_policy_value_optimal_{0}".format(level)] = mean_policy_value_optimal
                results_dict[data_type]["mse_score_optimal_{0}".format(level)] = mse_score_optimal
        training_total_mse = sum(
            [results_dict["training"][k] for k in results_dict["training"].keys() if "mse_score" in k])
        training_accuracy = results_dict["training"]["accuracy"]
        training_computation_cost = results_dict["training"]["computation_cost"]
        test_total_mse = sum(
            [results_dict["test"][k] for k in results_dict["test"].keys() if "mse_score" in k])
        test_accuracy = results_dict["test"]["accuracy"]
        test_computation_cost = results_dict["test"]["computation_cost"]
        log_string = str(results_dict)

        DbLogger.write_into_table(
            rows=[(run_id,
                   episode_id,
                   0.0,
                   np.asscalar(training_total_mse),
                   np.asscalar(training_accuracy),
                   np.asscalar(training_computation_cost),
                   0.0,
                   np.asscalar(test_total_mse),
                   np.asscalar(test_accuracy),
                   np.asscalar(test_computation_cost),
                   log_string)],
            table="deep_q_learning_logs", col_count=11)

    def run_training_step(self, level=None, **kwargs):
        pass

    def fill_eval_list_feed_dict(self, level, eval_list, feed_dict, **kwargs):
        state_features = kwargs["state_features"]
        optimal_q_values = kwargs["optimal_q_values"]
        eval_list.extend([self.totalLosses[level], self.lossMatrices[level], self.regressionLossValues[level]])
        feed_dict[self.stateInputs[level]] = state_features
        feed_dict[self.rewardMatrices[level]] = optimal_q_values

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
                idx_array = self.get_selection_indices(level=t, actions_t_minus_1=actions_t_minus_1)
                optimal_q_values = self.optimalQTables[t][sample_ids, actions_t_minus_1]
                state_features = self.get_state_features(sample_indices=sample_ids,
                                                         action_ids_t_minus_1=actions_t_minus_1,
                                                         level=t)
                self.fill_eval_list_feed_dict(level=t, eval_list=eval_list,
                                              feed_dict=feed_dict, state_features=state_features,
                                              optimal_q_values=optimal_q_values, idx_array=idx_array)
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
