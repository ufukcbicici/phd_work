import tensorflow as tf
import numpy as np
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_with_regression import DqnWithRegression
from auxillary.db_logger import DbLogger


class DqnMultiLevelRegression(DqnWithRegression):
    def __init__(self, routing_dataset, network, network_name, run_id, used_feature_names, dqn_func, lambda_mac_cost,
                 invalid_action_penalty, valid_prediction_reward, invalid_prediction_penalty,
                 include_ig_in_reward_calculations, feature_type, dqn_parameters):
        super().__init__(routing_dataset, network, network_name, run_id, used_feature_names, dqn_func, lambda_mac_cost,
                         invalid_action_penalty, valid_prediction_reward, invalid_prediction_penalty,
                         include_ig_in_reward_calculations, feature_type, dqn_parameters)

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
                   np.asscalar(test_computation_cost)),
                  log_string],
            table="deep_q_learning_logs", col_count=11)
