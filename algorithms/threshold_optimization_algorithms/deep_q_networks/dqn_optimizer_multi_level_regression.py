import tensorflow as tf
import numpy as np
import os
import pickle

from sklearn.neural_network import MLPClassifier, MLPRegressor

from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_networks import DeepQNetworks
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_with_regression import DqnWithRegression
from auxillary.db_logger import DbLogger
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import imblearn
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.linear_model import LogisticRegression


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

    @staticmethod
    def get_optimal_actions_list_from_q_table(q_table):
        a_truth_actions = []
        for idx in range(q_table.shape[0]):
            max_score = np.max(q_table[idx])
            max_indices = tuple(sorted(np.nonzero(q_table[idx] == max_score)[0]))
            a_truth_actions.append(max_indices)
        return a_truth_actions

    def get_formatted_input(self, action_id, level):
        batch_size = 10000
        all_indices = np.arange(self.routingDataset.labelList.shape[0])
        actions_arr = np.zeros_like(all_indices)
        actions_arr[:] = action_id
        X = self.get_state_features(sample_indices=np.arange(self.routingDataset.labelList.shape[0]),
                                    action_ids_t_minus_1=actions_arr,
                                    level=level)
        sess = tf.Session()
        X_shape = list(X.shape)
        X_shape[0] = None
        x_input = tf.placeholder(dtype=tf.float32, shape=X_shape, name="x_input")
        x_output = DeepQNetworks.global_average_pooling(net_input=x_input)
        X_arr = []
        for batch_idx in range(0, all_indices.shape[0], batch_size):
            X_batch = X[batch_idx: batch_idx + batch_size]
            X_formatted_batch = sess.run([x_output], feed_dict={x_input: X_batch})[0]
            X_arr.append(X_formatted_batch)
        X_formatted = np.concatenate(X_arr, axis=0)
        return X_formatted

    def convert_q_table_to_regression_target(self, level, action_id, Q_table):
        reachability_vector = self.reachabilityMatrices[level][action_id]
        valid_column_ids = sorted(np.nonzero(reachability_vector)[0])
        valid_columns = [Q_table[:, idx] for idx in valid_column_ids]
        regression_Q = np.stack(valid_columns, axis=-1)
        return regression_Q

    def convert_regression_target_to_q_table(self, level, action_id, R_table):
        reachability_vector = self.reachabilityMatrices[level][action_id]
        valid_column_ids = sorted(np.nonzero(reachability_vector)[0])
        Q_table = np.zeros_like(self.optimalQTables[level][:, action_id, :])
        Q_table[:] = -np.inf
        for R_table_id, Q_table_id in enumerate(valid_column_ids):
            Q_table[:, Q_table_id] = R_table[:, R_table_id]
        return Q_table

    def process_estimated_q_table_with_nn(self, estimated_q, q_train):
        q_centroids = set([tuple(arr) for arr in q_train])
        q_centroids = np.stack([arr for arr in q_centroids], axis=0)
        distance_matrix = []
        for arr in estimated_q:
            differences = q_centroids - np.expand_dims(arr, axis=0)
            squared_differences = np.square(differences)
            squared_distances = np.sum(squared_differences, axis=1)
            euclidean_distances = np.sqrt(squared_distances)
            distance_matrix.append(euclidean_distances)
        distance_matrix = np.stack(distance_matrix, axis=0)
        selected_ids = np.argmin(distance_matrix, axis=1)
        processed_q = q_centroids[selected_ids]
        return processed_q

    def measure_action_accuracy(self, Q_train, Q_train_pred, Q_test, Q_test_pred):
        a_train_optimal = self.get_optimal_actions_list_from_q_table(Q_train)
        a_train_predicted = self.get_optimal_actions_list_from_q_table(Q_train_pred)
        a_test_optimal = self.get_optimal_actions_list_from_q_table(Q_test)
        a_test_predicted = self.get_optimal_actions_list_from_q_table(Q_test_pred)
        comparison_vector_train = np.array([len(set(s1).intersection(set(s2))) > 0
                                            for s1, s2 in zip(a_train_optimal, a_train_predicted)])
        comparison_vector_test = np.array([len(set(s1).intersection(set(s2))) > 0
                                           for s1, s2 in zip(a_test_optimal, a_test_predicted)])
        train_accuracy = np.mean(comparison_vector_train)
        test_accuracy = np.mean(comparison_vector_test)
        print("train_accuracy={0}".format(train_accuracy))
        print("test_accuracy={0}".format(test_accuracy))

    def train_non_deep_learning_regression(self, **kwargs):
        tf.reset_default_graph()
        self.calculate_optimal_q_tables(discount_rate=kwargs["discount_factor"])
        last_level = self.get_max_trajectory_length()
        under_sample_ratio = 2.0
        over_sample_ratio = 2.5
        estimated_q_tables = []
        estimated_and_processed_q_tables = []
        for t in range(last_level):
            action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
            q_estimated = np.zeros_like(self.optimalQTables[t])
            q_estimated_processed = np.zeros_like(self.optimalQTables[t])
            for action_id in range(action_count_t_minus_one):
                file_name = "regression_model_level_{0}_action_{1}.sav".format(t, action_id)
                X = self.get_formatted_input(action_id=action_id, level=t)
                all_indices = np.arange(self.routingDataset.labelList.shape[0])
                actions_arr = np.zeros_like(all_indices)
                actions_arr[:] = action_id
                Q = self.convert_q_table_to_regression_target(level=t,
                                                              action_id=action_id,
                                                              Q_table=self.optimalQTables[t][all_indices, actions_arr])
                X_train = X[self.routingDataset.trainingIndices]
                Q_train = Q[self.routingDataset.trainingIndices]
                X_test = X[self.routingDataset.testIndices]
                Q_test = Q[self.routingDataset.testIndices]
                if os.path.exists(file_name):
                    f = open(file_name, "rb")
                    model = pickle.load(f)
                    f.close()
                else:
                    # Regression pipeline
                    standard_scaler = StandardScaler()
                    pca = PCA()
                    mlp = MLPRegressor()
                    pipe = Pipeline(steps=[("scaler", standard_scaler),
                                           ('pca', pca),
                                           ('mlp', mlp)])
                    param_grid = \
                        [{
                            "pca__n_components": [None],
                            "mlp__hidden_layer_sizes": [(64, 32)],
                            "mlp__activation": ["relu"],
                            "mlp__solver": ["adam"],
                            # "mlp__learning_rate": ["adaptive"],
                            "mlp__alpha": [25.0, 50.0, 100.0, 150.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0],
                            "mlp__max_iter": [10000],
                            "mlp__early_stopping": [True],
                            "mlp__n_iter_no_change": [25]
                        }]
                    search = GridSearchCV(pipe, param_grid, n_jobs=8, cv=10, verbose=10,
                                          scoring=["neg_mean_squared_error", "r2"], refit="neg_mean_squared_error")
                    search.fit(X_train, Q_train)
                    print("Best parameter (CV score=%0.3f):" % search.best_score_)
                    print(search.best_params_)
                    model = search.best_estimator_
                    f = open(file_name, "wb")
                    pickle.dump(model, f)
                    f.close()
                Q_train_pred = model.predict(X_train)
                Q_test_pred = model.predict(X_test)
                Q_pred = model.predict(X)
                Q_pred_converted = self.convert_regression_target_to_q_table(level=t,
                                                                             action_id=action_id,
                                                                             R_table=Q_pred)
                self.measure_action_accuracy(Q_train=Q_train, Q_train_pred=Q_train_pred,
                                             Q_test=Q_test, Q_test_pred=Q_test_pred)
                q_estimated[:, action_id, :] = Q_pred_converted
                Q_train_pred_processed = self.process_estimated_q_table_with_nn(estimated_q=Q_train_pred,
                                                                                q_train=Q_train)
                Q_test_pred_processed = self.process_estimated_q_table_with_nn(estimated_q=Q_test_pred, q_train=Q_train)
                self.measure_action_accuracy(Q_train=Q_train, Q_train_pred=Q_train_pred_processed,
                                             Q_test=Q_test, Q_test_pred=Q_test_pred_processed)
                Q_pred_processed = self.process_estimated_q_table_with_nn(estimated_q=Q_pred, q_train=Q_train)
                Q_pred_processed_converted = self.convert_regression_target_to_q_table(level=t,
                                                                             action_id=action_id,
                                                                             R_table=Q_pred_processed)
                q_estimated_processed[:, action_id, :] = Q_pred_processed_converted
            estimated_q_tables.append(q_estimated)
            estimated_and_processed_q_tables.append(q_estimated_processed)
        _, _, training_accuracy, training_computation_cost = self.execute_bellman_equation(
            Q_tables=estimated_q_tables, sample_indices=self.routingDataset.trainingIndices)
        _, _, test_accuracy, test_computation_cost = self.execute_bellman_equation(
            Q_tables=estimated_q_tables, sample_indices=self.routingDataset.testIndices)
        print("training_accuracy:{0} training_computation_cost:{1}".format(
            training_accuracy, training_computation_cost))
        print("test_accuracy:{0} test_computation_cost:{1}".format(
            test_accuracy, test_computation_cost))

    def train_non_deep_learning_classification(self, **kwargs):
        tf.reset_default_graph()
        self.calculate_optimal_q_tables(discount_rate=kwargs["discount_factor"])
        last_level = self.get_max_trajectory_length()
        under_sample_ratio = 2.0
        over_sample_ratio = 2.5
        estimated_q_tables = []
        for t in range(last_level):
            action_count_t_minus_one = 1 if t == 0 else self.actionSpaces[t - 1].shape[0]
            q_estimated = np.zeros_like(self.optimalQTables[t])
            for action_id in range(action_count_t_minus_one):
                file_name = "classifier_model_level_{0}_action_{1}.sav".format(t, action_id)
                X_ = {}
                Q_ = {}
                for data_type, indices in zip(["training", "test"],
                                              [self.routingDataset.trainingIndices, self.routingDataset.testIndices]):
                    idx_arr = np.array(indices)
                    actions_arr = np.zeros_like(idx_arr)
                    actions_arr[:] = action_id
                    X = self.get_state_features(sample_indices=idx_arr, action_ids_t_minus_1=actions_arr, level=t)
                    X_[data_type] = np.reshape(X, newshape=(X.shape[0], np.prod(X.shape[1:])))
                    Q_[data_type] = self.optimalQTables[t][idx_arr, actions_arr]
                a_truth_actions = self.get_optimal_actions_list_from_q_table(q_table=Q_["training"])
                le = LabelEncoder()
                y_truth = le.fit_transform(a_truth_actions)
                if os.path.exists(file_name):
                    f = open(file_name, "rb")
                    model = pickle.load(f)
                    f.close()
                else:
                    label_counter = Counter(y_truth)
                    most_common_two_labels = label_counter.most_common(n=2)
                    curr_ratio = most_common_two_labels[0][1] / most_common_two_labels[1][1]
                    # Undersample
                    X_hat, y_hat = X_["training"], y_truth
                    if curr_ratio > under_sample_ratio:
                        new_sample_counts = {
                            most_common_two_labels[0][0]: int(under_sample_ratio * most_common_two_labels[1][1]),
                            most_common_two_labels[1][0]: most_common_two_labels[1][1]
                        }
                        sample_counts = {}
                        for k, v in label_counter.items():
                            if k not in new_sample_counts:
                                sample_counts[k] = v
                            else:
                                sample_counts[k] = new_sample_counts[k]
                        under_sampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=sample_counts)
                        X_hat, y_hat = under_sampler.fit_resample(X=X_hat, y=y_hat)
                    print("X")
                    # Oversample
                    label_counter = Counter(y_hat)
                    max_sample_count = label_counter.most_common(n=1)[0][1]
                    over_sample_counts = {}
                    for k, v in label_counter.items():
                        ratio_to_max = max_sample_count / v
                        over_sample_counts[k] = int(min(over_sample_ratio, ratio_to_max) * v)
                    over_sampler = imblearn.over_sampling.SMOTE(sampling_strategy=over_sample_counts,
                                                                n_jobs=8, k_neighbors=3)
                    X_hat, y_hat = over_sampler.fit_resample(X=X_hat, y=y_hat)
                    # Classification pipeline
                    standard_scaler = StandardScaler()
                    pca = PCA()
                    mlp = MLPClassifier()
                    pipe = Pipeline(steps=[("scaler", standard_scaler),
                                           ('pca', pca),
                                           ('mlp', mlp)])
                    param_grid = \
                        [{
                            "pca__n_components": [100],
                            "mlp__hidden_layer_sizes": [(64, 32)],
                            "mlp__activation": ["relu"],
                            "mlp__solver": ["adam"],
                            # "mlp__learning_rate": ["adaptive"],
                            "mlp__alpha": [0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
                            "mlp__max_iter": [10000],
                            "mlp__early_stopping": [True],
                            "mlp__n_iter_no_change": [100]
                        }]
                    search = GridSearchCV(pipe, param_grid, n_jobs=4, cv=5, verbose=10,
                                          scoring=["accuracy", "f1_weighted", "f1_micro", "f1_macro",
                                                   "balanced_accuracy"],
                                          refit="accuracy")
                    search.fit(X_hat, y_hat)
                    print("Best parameter (CV score=%0.3f):" % search.best_score_)
                    print(search.best_params_)
                    model = search.best_estimator_
                    f = open(file_name, "wb")
                    pickle.dump(model, f)
                    f.close()
                # Classify training and test sets
                y_pred = {"training": model.predict(X_["training"]),
                          "test": model.predict(X_["test"])}
                print("*************Training*************")
                print(classification_report(y_pred=y_pred["training"], y_true=y_truth))
                print("*************Test*************")
                a_truth_actions_test = self.get_optimal_actions_list_from_q_table(q_table=Q_["test"])
                y_truth_test = le.transform(a_truth_actions_test)
                print(classification_report(y_pred=y_pred["test"], y_true=y_truth_test))
                # Process the whole data
                idx_arr = np.arange(self.routingDataset.labelList.shape[0])
                actions_arr = np.zeros_like(idx_arr)
                actions_arr[:] = action_id
                X_whole = self.get_state_features(sample_indices=idx_arr, action_ids_t_minus_1=actions_arr, level=t)
                X_whole = np.reshape(X_whole, newshape=(X_whole.shape[0], np.prod(X_whole.shape[1:])))
                estimated_actions = model.predict(X_whole)
                inverse_actions = le.inverse_transform(estimated_actions)
                valid_entries = np.array([[idx, jdx] for idx in range(inverse_actions.shape[0])
                                          for jdx in inverse_actions[idx]])
                actions_arr = np.zeros_like(valid_entries[:, 0])
                actions_arr[:] = action_id
                q_estimated[valid_entries[:, 0], actions_arr, valid_entries[:, 1]] = 1
            estimated_q_tables.append(q_estimated)
        _, _, training_accuracy, training_computation_cost = self.execute_bellman_equation(
            Q_tables=estimated_q_tables, sample_indices=self.routingDataset.trainingIndices)
        _, _, test_accuracy, test_computation_cost = self.execute_bellman_equation(
            Q_tables=estimated_q_tables, sample_indices=self.routingDataset.testIndices)
        print("training_accuracy:{0} training_computation_cost:{1}".format(
            training_accuracy, training_computation_cost))
        print("test_accuracy:{0} test_computation_cost:{1}".format(
            test_accuracy, test_computation_cost))

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
