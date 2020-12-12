import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from bayes_opt import BayesianOptimization
from datetime import datetime
import pickle

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor

from algorithms.branching_probability_calibration import BranchingProbabilityOptimization
from algorithms.information_gain_routing_accuracy_calculator import InformationGainRoutingAccuracyCalculator
from algorithms.threshold_optimization_algorithms.bayesian_clusterer import BayesianClusterer
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_networks import DeepQNetworks
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer import \
    DirectThresholdOptimizer
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer_entropy import \
    DirectThresholdOptimizerEntropy
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.fashion_net.fashion_cign_lite import FashionCignLite
from collections import Counter


class MixedBayesianOptimizer:
    train_accuracies = []
    test_accuracies = []

    @staticmethod
    def get_random_thresholds(cluster_count, network, kind):
        list_of_threshold_dicts = []
        for cluster_id in range(cluster_count):
            thresholds_dict = {}
            for node in network.innerNodes:
                child_nodes = network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
                child_count = len(child_nodes)
                if kind == "probabiliy":
                    max_bound = 1.0 / child_count
                    thresholds_dict[node.index] = np.random.uniform(low=0.0, high=max_bound, size=(1, child_count))
                elif kind == "entropy":
                    max_bound = -np.log(1.0 / child_count)
                    thresholds_dict[node.index] = np.random.uniform(low=0.0, high=max_bound)
                else:
                    raise NotImplementedError()
            list_of_threshold_dicts.append(thresholds_dict)
        return list_of_threshold_dicts

    @staticmethod
    def calculate_bounds(cluster_count, network, kind):
        pbounds = {}
        for cluster_id in range(cluster_count):
            for node in network.innerNodes:
                child_nodes = network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
                if kind == "probability":
                    max_bound = 1.0 / len(child_nodes)
                    for c_nd in child_nodes:
                        pbounds["c_{0}_t_{1}_{2}".format(cluster_id, node.index, c_nd.index)] = (0.0, max_bound)
                elif kind == "entropy":
                    max_bound = -np.log(1.0 / len(child_nodes))
                    pbounds["c_{0}_t_{1}".format(cluster_id, node.index)] = (0.0, max_bound)
                    # if node.isRoot:
                    #     interval_size = max_bound / cluster_count
                    #     pbounds["c_{0}_t_{1}".format(cluster_id, node.index)] = (cluster_id*interval_size,
                    #                                                             (cluster_id+1)*interval_size)
                    # else:
                    #     pbounds["c_{0}_t_{1}".format(cluster_id, node.index)] = (0.0, max_bound)
                else:
                    raise NotImplementedError()
        return pbounds

    @staticmethod
    def decode_bayesian_optimization_parameters(args_dict, network, cluster_count, kind):
        list_of_threshold_dicts = []
        for cluster_id in range(cluster_count):
            thrs_dict = {}
            for node in network.innerNodes:
                child_nodes = network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
                if kind == "probability":
                    thrs_arr = np.array([args_dict["c_{0}_t_{1}_{2}".format(cluster_id, node.index, c_nd.index)]
                                         for c_nd in child_nodes])
                    thrs_dict[node.index] = thrs_arr[np.newaxis, :]
                elif kind == "entropy":
                    thrs_dict[node.index] = args_dict["c_{0}_t_{1}".format(cluster_id, node.index)]
                else:
                    raise NotImplementedError()
            list_of_threshold_dicts.append(thrs_dict)
        return list_of_threshold_dicts

    @staticmethod
    def get_thresholding_results(sess, network, optimization_step, cluster_weights, cluster_count,
                                 threshold_optimizer, routing_data, indices,
                                 list_of_threshold_dicts, temperatures_dict, mixing_lambda):
        if optimization_step == 0:
            cluster_weights = np.ones_like(cluster_weights)
            cluster_weights = (1.0 / cluster_weights.shape[1]) * cluster_weights
        # Get thresholding results for all clusters
        correctness_results = []
        activation_costs = []
        score_vectors = []
        accuracies = []
        selection_tuples = []
        posteriors_list = []
        labels = routing_data.labelList[indices]
        for cluster_id in range(cluster_count):
            thrs_dict = list_of_threshold_dicts[cluster_id]
            optimizer_results = threshold_optimizer.run_threshold_calculator(sess=sess,
                                                                             routing_data=routing_data,
                                                                             indices=indices,
                                                                             mixing_lambda=mixing_lambda,
                                                                             temperatures_dict=temperatures_dict,
                                                                             thresholds_dict=thrs_dict)
            correctness_vector = optimizer_results["correctnessVector"]
            activation_costs_vector = optimizer_results["activationCostsArr"]
            accuracies.append(optimizer_results["accuracy"])
            selection_tuples.append(optimizer_results["selectionTuples"])
            posteriors_list.append(optimizer_results["posteriorsTensor"])
            correctness_results.append(correctness_vector)
            activation_costs.append(activation_costs_vector)
            score_vector = mixing_lambda * correctness_vector - (1.0 - mixing_lambda) * activation_costs_vector
            score_vectors.append(score_vector)

        # Let thresholds be spread out
        thresholds_arr = np.zeros(shape=(cluster_count, len(network.innerNodes)))
        for cluster_id in range(cluster_count):
            for idx, node in enumerate(network.innerNodes):
                thresholds_arr[cluster_id, idx] = list_of_threshold_dicts[cluster_id][node.index]
        cluster_arr = np.mean(cluster_weights, axis=0)
        weighted_thresholds_mean = cluster_arr[:, np.newaxis] * thresholds_arr
        expected_thresholds = weighted_thresholds_mean * thresholds_arr
        mean_threshold = np.sum(expected_thresholds, axis=0)
        diff_arr = thresholds_arr - mean_threshold[np.newaxis, :]
        euclidean_distances = np.sqrt(np.sum(np.square(diff_arr), axis=1))
        total_distance_to_mean = np.sum(euclidean_distances)
        print("cluster_arr={0}".format(cluster_arr))
        print("total_distance_to_mean={0}".format(total_distance_to_mean))
        print("X")
        argmax_weights = UtilityFuncs.get_argmax_matrix_from_weight_matrix(cluster_weights)
        selections_tensor = np.stack(selection_tuples, axis=-1)
        weighted_tensors = selections_tensor * np.expand_dims(argmax_weights, axis=1)
        selected_tuples = np.sum(weighted_tensors, axis=-1)

        def get_accumulated_metric(list_of_cluster_results):
            metric_matrix = np.stack(list_of_cluster_results, axis=-1)
            weighted_metric_matrix = cluster_weights * metric_matrix
            weighted_metric_vector = np.sum(weighted_metric_matrix, axis=-1)
            final_metric = np.mean(weighted_metric_vector)
            return final_metric, metric_matrix

        # Scores
        final_score, scores_matrix = get_accumulated_metric(list_of_cluster_results=score_vectors)
        # Accuracies
        final_accuracy, accuracy_matrix = get_accumulated_metric(list_of_cluster_results=correctness_results)
        # Calculation Costs
        final_cost, cost_matrix = get_accumulated_metric(list_of_cluster_results=activation_costs)
        # Check if posteriors do hold.
        selected_posteriors = posteriors_list[0] * np.expand_dims(selected_tuples, axis=1)
        final_posteriors = np.sum(selected_posteriors, axis=-1)
        score_v2 = np.mean(np.argmax(final_posteriors, axis=-1) == labels)

        results_dict = {"final_score": final_score, "scores_matrix": scores_matrix,
                        "final_accuracy": final_accuracy, "accuracy_matrix": accuracy_matrix,
                        "final_cost": final_cost, "cost_matrix": cost_matrix,
                        "total_distance_to_mean": total_distance_to_mean,
                        "selections_tensor": selections_tensor,
                        "selected_tuples": selected_tuples}
        return results_dict

    @staticmethod
    def get_cluster_weights(routing_data, sess, bc, train_indices, test_indices):
        training_weights = bc.get_cluster_scores(sess=sess,
                                                 features=routing_data.get_dict("pre_branch_feature")
                                                 [0][train_indices])
        test_weights = bc.get_cluster_scores(sess=sess,
                                             features=routing_data.get_dict("pre_branch_feature")[0][test_indices])
        cluster_weights = {"train": training_weights, "test": test_weights}
        return cluster_weights

    @staticmethod
    def write_results_to_db(run_id, experiment_id, seed, iteration_id, cluster_count, mixing_lambda, xi,
                            results_dict, kwargs, operation_type):
        row = [(run_id,
                experiment_id,
                seed,
                iteration_id,
                "BO Optimization",
                cluster_count,
                "x",
                mixing_lambda,
                0,
                results_dict["train"]["final_score"],
                results_dict["train"]["final_accuracy"],
                results_dict["train"]["final_cost"],
                results_dict["test"]["final_score"],
                results_dict["test"]["final_accuracy"],
                results_dict["test"]["final_cost"],
                operation_type,
                xi,
                "{0}".format(datetime.now()),
                np.asscalar(results_dict["train"]["total_distance_to_mean"]),
                np.asscalar(results_dict["test"]["total_distance_to_mean"])
                )]
        DbLogger.write_into_table(row, "threshold_optimization_clusters", col_count=None)
        rows = []
        for thrs_name, threshold_value in kwargs.items():
            rows.append((experiment_id, seed, iteration_id, thrs_name, threshold_value))
        # bo_iteration += 1
        DbLogger.write_into_table(rows, "threshold_optimization_thresholds", col_count=None)

    @staticmethod
    def optimize(optimization_iterations_count,
                 iteration,
                 xi,
                 mixing_lambda,
                 cluster_count,
                 fc_layers,
                 run_id,
                 network,
                 routing_data,
                 seed):
        experiment_id = DbLogger.get_run_id()
        DbLogger.write_into_table(rows=[(experiment_id, "BO Experiment")], table=DbLogger.runMetaData, col_count=2)
        train_indices = routing_data.trainingIndices
        test_indices = routing_data.testIndices
        # Learn the standard information gain based accuracies
        train_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=network,
                                                                               routing_data=routing_data,
                                                                               indices=train_indices)
        test_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=network,
                                                                              routing_data=routing_data,
                                                                              indices=test_indices)
        train_accuracies = []
        train_mean_distances = []
        test_accuracies = []
        test_mean_distances = []
        print("train_ig_accuracy={0}".format(train_ig_accuracy))
        print("test_ig_accuracy={0}".format(test_ig_accuracy))
        # Threshold Optimizer
        dto = DirectThresholdOptimizerEntropy(network=network, routing_data=routing_data, seed=seed)
        temperatures_dict = BranchingProbabilityOptimization.calibrate_branching_probabilities(
            network=network, routing_data=routing_data, run_id=run_id, iteration=iteration, indices=train_indices,
            seed=seed)
        dto.build_network()
        # Clusterer
        bc = BayesianClusterer(network=network, routing_data=routing_data, cluster_count=cluster_count,
                               fc_layers=fc_layers)
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # config = tf.ConfigProto(device_count={'GPU': 0})
        # sess = tf.Session(config=config)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        pbounds = MixedBayesianOptimizer.calculate_bounds(cluster_count=cluster_count, network=network, kind=dto.kind)
        list_of_best_thresholds = []
        # Two - Phase optimization iterations
        bo_results_file_name = "bo_results.sav"
        if not os.path.exists(bo_results_file_name):
            for iteration_id in range(optimization_iterations_count):
                bo_iteration = 0
                sample_indices = {"train": train_indices, "test": test_indices}
                cluster_weights = MixedBayesianOptimizer.get_cluster_weights(
                    routing_data=routing_data, sess=sess, bc=bc, train_indices=train_indices, test_indices=test_indices)

                # Loss Function
                def f_(**kwargs):
                    # Convert Bayesian Optimization space sample into usable thresholds
                    list_of_threshold_dicts = MixedBayesianOptimizer.decode_bayesian_optimization_parameters(
                        args_dict=kwargs, network=network, cluster_count=cluster_count, kind=dto.kind)
                    results_dict = {}
                    for data_type in ["train", "test"]:
                        results = MixedBayesianOptimizer.get_thresholding_results(
                            sess=sess,
                            network=network,
                            optimization_step=iteration_id,
                            cluster_weights=cluster_weights[data_type],
                            cluster_count=cluster_count,
                            threshold_optimizer=dto,
                            routing_data=routing_data,
                            indices=sample_indices[
                                data_type],
                            list_of_threshold_dicts=list_of_threshold_dicts,
                            temperatures_dict=temperatures_dict,
                            mixing_lambda=mixing_lambda)
                        results_dict[data_type] = results
                    print("Train Accuracy: {0} Train Computation Load:{1} Train Score:{2}".format(
                        results_dict["train"]["final_accuracy"],
                        results_dict["train"]["final_cost"], results_dict["train"]["final_score"]))
                    print("Test Accuracy: {0} Test Computation Load:{1} Test Score:{2}".format(
                        results_dict["test"]["final_accuracy"],
                        results_dict["test"]["final_cost"], results_dict["test"]["final_score"]))
                    MixedBayesianOptimizer.train_accuracies.append(results_dict["train"]["final_accuracy"])
                    MixedBayesianOptimizer.test_accuracies.append(results_dict["test"]["final_accuracy"])
                    # DB recording
                    train_accuracies.append(results_dict["train"]["final_accuracy"])
                    train_mean_distances.append(np.asscalar(results_dict["train"]["total_distance_to_mean"]))
                    test_accuracies.append(results_dict["test"]["final_accuracy"])
                    test_mean_distances.append(np.asscalar(results_dict["test"]["total_distance_to_mean"]))
                    MixedBayesianOptimizer.write_results_to_db(
                        run_id=run_id, experiment_id=experiment_id, seed=seed, iteration_id=iteration_id,
                        cluster_count=cluster_count, mixing_lambda=mixing_lambda, xi=xi, results_dict=results_dict,
                        kwargs=kwargs, operation_type="Inside BO")
                    return results_dict["train"]["final_score"]

                # Phase - 1 Bayesian Optimization of the thresholds
                optimizer = BayesianOptimization(
                    f=f_,
                    pbounds=pbounds,
                )
                optimizer.maximize(
                    init_points=100,
                    n_iter=150,
                    acq="ei",
                    xi=xi
                )
                best_params = optimizer.max["params"]
                print("After BO Result")
                f_(**best_params)
                # Phase - 2 Gradient Based Optimization of the Clusterer
                list_of_best_thresholds = MixedBayesianOptimizer.decode_bayesian_optimization_parameters(
                    args_dict=best_params, network=network, cluster_count=cluster_count, kind=dto.kind)
                res = MixedBayesianOptimizer.get_thresholding_results(
                    sess=sess,
                    network=network,
                    cluster_weights=cluster_weights["train"],
                    optimization_step=iteration_id,
                    cluster_count=cluster_count,
                    threshold_optimizer=dto,
                    routing_data=routing_data,
                    indices=train_indices,
                    list_of_threshold_dicts=list_of_best_thresholds,
                    temperatures_dict=temperatures_dict,
                    mixing_lambda=mixing_lambda)
                features = routing_data.get_dict("pre_branch_feature")[0][train_indices]
                scores_after_bo = res["scores_matrix"]
                accuracies_after_bo = res["accuracy_matrix"]
                assert features.shape[0] == scores_after_bo.shape[0]
                bc.optimize_clustering(sess=sess, features=features, scores=scores_after_bo,
                                       accuracies=accuracies_after_bo)
                cluster_weights = MixedBayesianOptimizer.get_cluster_weights(
                    routing_data=routing_data, sess=sess, bc=bc, train_indices=train_indices, test_indices=test_indices)
                results_dict = {}
                print("After Clustering Result Weighted")
                for data_type in ["train", "test"]:
                    print(Counter(np.argmax(cluster_weights[data_type], axis=1)))
                    results = MixedBayesianOptimizer.get_thresholding_results(
                        sess=sess,
                        network=network,
                        optimization_step=iteration_id + 1,
                        cluster_weights=cluster_weights[data_type],
                        cluster_count=cluster_count,
                        threshold_optimizer=dto,
                        routing_data=routing_data,
                        indices=sample_indices[
                            data_type],
                        list_of_threshold_dicts=list_of_best_thresholds,
                        temperatures_dict=temperatures_dict,
                        mixing_lambda=mixing_lambda)
                    results_dict[data_type] = results
                    print("After clustering {0} score:{1}".format(data_type, results["final_score"]))
                MixedBayesianOptimizer.write_results_to_db(
                    run_id=run_id, experiment_id=experiment_id, seed=seed, iteration_id=iteration_id,
                    cluster_count=cluster_count, mixing_lambda=mixing_lambda, xi=xi, results_dict=results_dict,
                    kwargs=best_params, operation_type="After Clustering {0} - Weighted".format(iteration_id))
                print("After Clustering Result Argmax")
                for data_type in ["train", "test"]:
                    selected_clusters = np.argmax(cluster_weights[data_type], axis=1)
                    argmax_weights = np.zeros_like(cluster_weights[data_type])
                    argmax_weights[np.arange(argmax_weights.shape[0]), selected_clusters] = 1.0
                    results = MixedBayesianOptimizer.get_thresholding_results(
                        sess=sess,
                        network=network,
                        optimization_step=iteration_id + 1,
                        cluster_weights=argmax_weights,
                        cluster_count=cluster_count,
                        threshold_optimizer=dto,
                        routing_data=routing_data,
                        indices=sample_indices[
                            data_type],
                        list_of_threshold_dicts=list_of_best_thresholds,
                        temperatures_dict=temperatures_dict,
                        mixing_lambda=mixing_lambda)
                    results_dict[data_type] = results
                    print("After clustering {0} score:{1}".format(data_type, results["final_score"]))
                MixedBayesianOptimizer.write_results_to_db(
                    run_id=run_id, experiment_id=experiment_id, seed=seed, iteration_id=iteration_id,
                    cluster_count=cluster_count, mixing_lambda=mixing_lambda, xi=xi, results_dict=results_dict,
                    kwargs=best_params, operation_type="After Clustering {0} - Argmax".format(iteration_id))
            print("X")
            all_cluster_weights = bc.get_cluster_scores(sess=sess,
                                                        features=routing_data.get_dict("pre_branch_feature")[0])
            final_results = MixedBayesianOptimizer.get_thresholding_results(
                sess=sess,
                network=network,
                optimization_step=optimization_iterations_count + 1,
                cluster_weights=all_cluster_weights,
                cluster_count=cluster_count,
                threshold_optimizer=dto,
                routing_data=routing_data,
                indices=np.arange(all_cluster_weights.shape[0]),
                list_of_threshold_dicts=list_of_best_thresholds,
                temperatures_dict=temperatures_dict,
                mixing_lambda=mixing_lambda)
            f = open(bo_results_file_name, "wb")
            pickle.dump(final_results, f)
            f.close()
        else:
            f = open(bo_results_file_name, "rb")
            final_results = pickle.load(f)
            f.close()
        MixedBayesianOptimizer.calculate_routing_weights_with_regression(
            network=network,
            routing_data=routing_data,
            selection_tuples=final_results["selected_tuples"],
            train_indices=train_indices,
            test_indices=test_indices)
        print("X")

    @staticmethod
    def calculate_routing_weights_with_regression(network, routing_data, selection_tuples, train_indices,
                                                  test_indices,
                                                  cut_off_value=5.0):
        X = routing_data.get_dict("pre_branch_feature")[0]
        y = routing_data.labelList
        X_formatted = UtilityFuncs.vectorize_with_gap(X)
        posteriors = [routing_data.get_dict("posterior_probs")[node.index] for node in network.leafNodes]
        posteriors = np.stack(posteriors, axis=-1)
        label_count = posteriors.shape[1]
        # Get optimal coefficients
        optimal_coefficients = []
        predicted_y = []
        for idx in range(y.shape[0]):
            y_one_hot = np.zeros((label_count,), dtype=np.float32)
            y_one_hot[y[idx]] = 1.0
            b = y_one_hot
            posterior_matrix = posteriors[idx]
            selection_of_posteriors = selection_tuples[idx]
            posteriors_with_selection = posterior_matrix * selection_of_posteriors[np.newaxis, :]
            coeffs = np.linalg.lstsq(posteriors_with_selection, b)[0]
            coeffs = coeffs * selection_of_posteriors
            coeffs = np.clip(coeffs, a_min=-cut_off_value, a_max=cut_off_value)
            optimal_coefficients.append(coeffs)
            y_hat = np.squeeze(np.dot(posteriors_with_selection, coeffs[:, np.newaxis]))
            y_hat = np.argmax(y_hat)
            predicted_y.append(y_hat)
        predicted_y = np.array(predicted_y)
        optimal_coefficients = np.stack(optimal_coefficients, axis=0)
        ideal_train_accuracy = np.mean(y[train_indices] == predicted_y[train_indices])
        ideal_test_accuracy = np.mean(y[test_indices] == predicted_y[test_indices])
        print("ideal_train_accuracy={0}".format(ideal_train_accuracy))
        print("ideal_test_accuracy={0}".format(ideal_test_accuracy))

        # Build the regression model
        X_train = X_formatted[train_indices]
        X_test = X_formatted[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        posteriors_train = posteriors[train_indices]
        posteriors_test = posteriors[test_indices]
        selections_train = selection_tuples[train_indices]
        selections_test = selection_tuples[test_indices]
        optimal_coefficients_train = optimal_coefficients[train_indices]
        optimal_coefficients_test = optimal_coefficients[test_indices]

        file_name = "routing_weights_regressor.sav"
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
                "mlp__alpha": [0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0],
                "mlp__max_iter": [10000],
                "mlp__early_stopping": [True],
                "mlp__n_iter_no_change": [100]
            }]
        search = GridSearchCV(pipe, param_grid, n_jobs=8, cv=10, verbose=10,
                              scoring=["neg_mean_squared_error", "r2"],
                              refit="neg_mean_squared_error")
        search.fit(X_train, optimal_coefficients_train)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        model = search.best_estimator_
        f = open(file_name, "wb")
        pickle.dump(model, f)
        f.close()
        print("X")

    @staticmethod
    def calculate_routing_cost(weights, X_, y_, posteriors, selections, lambda_coeff):
        A_ = posteriors
        W_shape = (A_.shape[-1], X_.shape[-1])
        label_count = A_.shape[1]

        W = weights

        # Squared differences
        sum_of_diffs = 0.0
        is_accurate = []
        N = X_.shape[0]
        for i in range(N):
            x_i = X_[i]
            y_i = np.zeros(shape=(label_count,), dtype=np.float32)
            label_truth = y_[i]
            y_i[label_truth] = 1.0
            A_i = A_[i]
            A_i = A_i * selections[i][np.newaxis, :]
            coeffs_i = np.dot(W, x_i[:, np.newaxis])
            y_i_predicted = np.dot(A_i, coeffs_i)
            y_i_predicted = np.squeeze(y_i_predicted)
            delta_i = y_i_predicted - y_i
            l2_norm_squared = np.linalg.norm(delta_i) ** 2.0
            sum_of_diffs += l2_norm_squared / (2.0 * N)
            label_predicted = np.argmax(y_i_predicted)
            is_accurate.append(label_predicted == label_truth)
        # Tikhonov Regularizer
        regularizer_cost = (lambda_coeff / 2.0) * (np.linalg.norm(W) ** 2.0)
        total_cost = sum_of_diffs + regularizer_cost
        # accuracy = np.mean(np.array(is_accurate))
        return total_cost, np.array(is_accurate)

    @staticmethod
    def fit_least_squares_routing(X_, y_, posteriors, selections, lambda_coeff):
        A_ = posteriors
        W_shape = (A_.shape[-1], X_.shape[-1])
        label_count = A_.shape[1]
        # Build as the least squares solution
        N = X_.shape[0]
        W_dim = np.asscalar(np.prod(W_shape))
        # Build the matrix A and vector b
        vector_b = np.zeros(shape=(W_dim, 1))
        matrix_A = np.zeros(shape=(W_dim, W_dim))
        for i in range(N):
            x_i = X_[i][:, np.newaxis]
            y_i = np.zeros(shape=(label_count,), dtype=np.float32)
            label_truth = y_[i]
            y_i[label_truth] = 1.0
            y_i = y_i[:, np.newaxis]
            A_i = A_[i]
            A_i = A_i * selections[i][np.newaxis, :]

            # Matrix A
            xx_T = np.dot(x_i, x_i.T)
            A_TA = np.dot(A_i.T, A_i)
            kron_i = np.kron(xx_T, A_TA)
            assert matrix_A.shape == kron_i.shape
            matrix_A = matrix_A + kron_i

            # Vector b
            A_ty = np.dot(A_i.T, y_i)
            A_tyx = np.dot(A_ty, x_i.T)
            vec_b = np.reshape(A_tyx, (A_tyx.shape[0] * A_tyx.shape[1], 1), order='F')
            vector_b = vector_b + vec_b
        regularizer_matrix = N * lambda_coeff * np.identity(n=W_dim, dtype=matrix_A.dtype)
        matrix_A = matrix_A + regularizer_matrix

        # Now we have the following linear system: matrix_A * Vec(W) = vector_b
        # Solve for Vec(W) and then reshape.
        vec_W = np.linalg.lstsq(matrix_A, np.squeeze(vector_b))[0]
        W_optimal = np.reshape(vec_W, W_shape, order='F')
        return W_optimal

    @staticmethod
    def calculate_mixing_coefficients_least_squares(network, routing_data, selection_tuples, train_indices,
                                                    test_indices,
                                                    cut_off_value=5.0):
        X = routing_data.get_dict("pre_branch_feature")[0]
        y = routing_data.labelList
        X_formatted = UtilityFuncs.vectorize_with_gap(X)
        posteriors = [routing_data.get_dict("posterior_probs")[node.index] for node in network.leafNodes]
        posteriors = np.stack(posteriors, axis=-1)
        label_count = posteriors.shape[1]
        W = np.random.normal(loc=0.0, scale=0.1, size=(posteriors.shape[-1], X_formatted.shape[-1]))
        cluster_count = 3
        X_train = X_formatted[train_indices]
        X_test = X_formatted[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        posteriors_train = posteriors[train_indices]
        posteriors_test = posteriors[test_indices]
        selections_train = selection_tuples[train_indices]
        selections_test = selection_tuples[test_indices]

        for cluster_count in range(2, 11):
            kmeans = KMeans(n_clusters=cluster_count).fit(X_train)

            training_clusters = kmeans.predict(X_train)
            test_clusters = kmeans.predict(X_test)
            print("Training Cluster Distribution:{0}".format(Counter(training_clusters)))
            print("Test Cluster Distribution:{0}".format(Counter(test_clusters)))

            for lambda_coeff in [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
                print("**********Cluster Count:{0} Lambda Coeff:{1}**********".format(cluster_count, lambda_coeff))
                cluster_accuracy_results_train = {}
                cluster_accuracy_results_test = {}
                for cluster_id in range(cluster_count):
                    cluster_mask = training_clusters == cluster_id
                    X_ = X_train[cluster_mask]
                    y_ = y_train[cluster_mask]
                    A_ = posteriors_train[cluster_mask]
                    S_ = selections_train[cluster_mask]
                    W_cluster = MixedBayesianOptimizer.fit_least_squares_routing(X_=X_,
                                                                                 y_=y_,
                                                                                 posteriors=A_,
                                                                                 selections=S_,
                                                                                 lambda_coeff=lambda_coeff)
                    # Train results
                    train_cost_opt, train_accuracies = MixedBayesianOptimizer.calculate_routing_cost(
                        weights=W_cluster,
                        X_=X_,
                        y_=y_,
                        posteriors=A_,
                        selections=S_,
                        lambda_coeff=lambda_coeff)
                    cluster_accuracy_results_train[cluster_id] = train_accuracies
                    # Test results
                    # Update the cluster mask
                    cluster_mask = test_clusters == cluster_id
                    test_cost_opt, test_accuracies = MixedBayesianOptimizer.calculate_routing_cost(
                        weights=W_cluster,
                        X_=X_test[cluster_mask],
                        y_=y_test[cluster_mask],
                        posteriors=posteriors_test[cluster_mask],
                        selections=selections_test[cluster_mask],
                        lambda_coeff=lambda_coeff)
                    cluster_accuracy_results_test[cluster_id] = test_accuracies

                train_results_arr = np.mean(np.concatenate(list(cluster_accuracy_results_train.values())))
                test_results_arr = np.mean(np.concatenate(list(cluster_accuracy_results_test.values())))
                print("train_results_arr={0}".format(train_results_arr))
                print("test_results_arr={0}".format(test_results_arr))

        # Get optimal coefficients
        # optimal_coefficients = []
        # predicted_y = []
        # for idx in range(y.shape[0]):
        #     y_one_hot = np.zeros((label_count,), dtype=np.float32)
        #     y_one_hot[y[idx]] = 1.0
        #     b = y_one_hot
        #     posterior_matrix = posteriors[idx]
        #     selection_of_posteriors = selection_tuples[idx]
        #     posteriors_with_selection = posterior_matrix * selection_of_posteriors[np.newaxis, :]
        #     coeffs = np.linalg.lstsq(posteriors_with_selection, b)[0]
        #     coeffs = coeffs * selection_of_posteriors
        #     coeffs = np.clip(coeffs, a_min=-cut_off_value, a_max=cut_off_value)
        #     optimal_coefficients.append(coeffs)
        #     y_hat = np.squeeze(np.dot(posteriors_with_selection, coeffs[:, np.newaxis]))
        #     y_hat = np.argmax(y_hat)
        #     predicted_y.append(y_hat)
        # predicted_y = np.array(predicted_y)
        # optimal_coefficients = np.stack(optimal_coefficients, axis=0)
        # ideal_train_accuracy = np.mean(y[train_indices] == predicted_y[train_indices])
        # ideal_test_accuracy = np.mean(y[test_indices] == predicted_y[test_indices])

        # print("ideal_train_accuracy={0}".format(ideal_train_accuracy))
        # print("ideal_test_accuracy={0}".format(ideal_test_accuracy))
        print("X")
