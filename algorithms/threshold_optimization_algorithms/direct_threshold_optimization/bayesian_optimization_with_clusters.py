import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import imblearn
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from algorithms.branching_probability_calibration import BranchingProbabilityOptimization
from algorithms.information_gain_routing_accuracy_calculator import InformationGainRoutingAccuracyCalculator
from algorithms.threshold_optimization_algorithms.bayesian_clusterer import BayesianClusterer
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_networks import DeepQNetworks
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer import \
    DirectThresholdOptimizer
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer_entropy import \
    DirectThresholdOptimizerEntropy
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.ig_clustered_bayesian_optimization import \
    IgBasedBayesianOptimization
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.mixed_bayesian_optimizer import \
    MixedBayesianOptimizer
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.fashion_net.fashion_cign_lite import FashionCignLite


class BayesianOptimizationWithClusters:
    train_accuracies = []
    test_accuracies = []

    @staticmethod
    def get_thresholding_results(sess, cluster_count, clusterer,
                                 threshold_optimizer, routing_data, indices,
                                 list_of_threshold_dicts, temperatures_dict, mixing_lambda):
        if clusterer is not None:
            # Get features for training samples
            X = IgBasedBayesianOptimization.get_formatted_input(routing_data=routing_data)[indices]
            cluster_weights = np.zeros(dtype=np.float32, shape=(indices.shape[0], cluster_count))
            y_hat = clusterer.predict(X)
            cluster_weights[np.arange(y_hat.shape[0]), y_hat] = 1.0
            # # Get cluster weights for training samples
            # cluster_weights = clusterer.get_cluster_scores(sess=sess, features=features)
            # c_ids = np.argmax(cluster_weights, axis=1)
            # print(Counter(c_ids))
            # if optimization_step == 0:
            #     cluster_weights = np.ones_like(cluster_weights)
            #     cluster_weights = (1.0 / cluster_weights.shape[1]) * cluster_weights
            # cluster_weights = np.zeros(dtype=np.float32, shape=(indices.shape[0], cluster_count))
        else:
            cluster_weights = np.ones(dtype=np.float32, shape=(indices.shape[0], cluster_count))
            cluster_weights = (1.0 / cluster_count) * cluster_weights
        # Get thresholding results for all clusters
        correctness_results = []
        activation_costs = []
        score_vectors = []
        accuracies = []
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
            correctness_results.append(correctness_vector)
            activation_costs.append(activation_costs_vector)
            score_vector = mixing_lambda * correctness_vector + (1.0 - mixing_lambda) * activation_costs_vector
            score_vectors.append(score_vector)

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
        accuracies_sum = np.sum(accuracy_matrix, axis=1)
        ideal_accuracy = np.mean(accuracies_sum > 0)
        results_dict = {"final_score": final_score, "scores_matrix": scores_matrix,
                        "final_accuracy": final_accuracy, "accuracy_matrix": accuracy_matrix,
                        "final_cost": final_cost, "cost_matrix": cost_matrix,
                        "ideal_accuracy": ideal_accuracy}
        return results_dict

    @staticmethod
    def resample_training_set(X, y, max_sample_count, up_sample_ratio):
        q_counter = Counter(y)
        under_sample_counts = {lbl: min(max_sample_count, cnt) for lbl, cnt in q_counter.items()}
        under_sampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=under_sample_counts)
        X_hat, y_hat = under_sampler.fit_resample(X=X, y=y)
        over_sample_counts = {lbl: min(int(up_sample_ratio * cnt), max_sample_count)
                              for lbl, cnt in under_sample_counts.items()}
        X_hat_2, y_hat_2 = None, None
        neighbor_count = 10
        while neighbor_count > 0:
            print("Trying with neighbor count:{0}".format(neighbor_count))
            try:
                over_sampler = imblearn.over_sampling.SMOTE(sampling_strategy=over_sample_counts,
                                                            n_jobs=1, k_neighbors=neighbor_count)
                X_hat_2, y_hat_2 = over_sampler.fit_resample(X=X_hat, y=y_hat)
            except:
                print("Failed with neighbor count:{0}".format(neighbor_count))
                neighbor_count = neighbor_count - 1
                continue
            break
        print("------->Fitted with neighbor count:{0}".format(neighbor_count))
        if X_hat_2 is None or y_hat_2 is None:
            X_hat_2 = X_hat
            y_hat_2 = y_hat
        return X_hat_2, y_hat_2

    @staticmethod
    def get_optimal_ids(matrix):
        optimal_list = []
        for idx in range(matrix.shape[0]):
            max_score = np.max(matrix[idx])
            max_indices = tuple(sorted(np.nonzero(matrix[idx] == max_score)[0]))
            optimal_list.append(max_indices)
        return optimal_list

    @staticmethod
    def resample_training_set_v2(X, score_matrix, accuracy_matrix, max_sample_count):
        accurate_counts = np.sum(accuracy_matrix, axis=1)
        max_score_ids = BayesianOptimizationWithClusters.get_optimal_ids(matrix=score_matrix)
        cluster_count = score_matrix.shape[1]
        X_arr = []
        y_arr = []
        selected_samples = set()
        sample_count_dict = {s_id: 0 for s_id in range(cluster_count)}
        # for sample_id, id_tpl in enumerate(max_score_ids):
        #     if len(id_tpl) == 1:
        #         cluster_id = id_tpl[0]
        #         X_arr.append(X[sample_id])
        #         y_arr.append(cluster_id)
        #         selected_samples.add(sample_id)
        # print("X")
        for cluster_id in range(cluster_count):
            # candidate_ids = np.array([idx for idx in range(score_matrix.shape[0]) if idx not in selected_samples])
            selected_ids = np.arange(cluster_id * max_sample_count, (cluster_id + 1) * max_sample_count)
            # selected_ids = np.random.choice(candidate_ids, max_sample_count)
            for sample_id in selected_ids:
                if cluster_id in max_score_ids[sample_id]:
                    X_arr.append(X[sample_id])
                    y_arr.append(cluster_id)
                    selected_samples.add(sample_id)
        X_hat = np.stack(X_arr, axis=0)
        y_hat = np.stack(y_arr, axis=0)
        return X_hat, y_hat

    @staticmethod
    def resample_training_set_v3(X, score_matrix, accuracy_matrix, max_sample_count):
        accurate_counts = np.sum(accuracy_matrix, axis=1)
        max_score_ids = BayesianOptimizationWithClusters.get_optimal_ids(matrix=score_matrix)
        cluster_count = score_matrix.shape[1]
        X_arr = []
        y_arr = []
        selected_samples = set()
        for sample_id, id_tpl in enumerate(max_score_ids):
            if len(id_tpl) == 1:
                cluster_id = id_tpl[0]
                X_arr.append(X[sample_id])
                y_arr.append(cluster_id)
                selected_samples.add(sample_id)
        X_hat = np.stack(X_arr, axis=0)
        y_hat = np.stack(y_arr, axis=0)
        return X_hat, y_hat

    @staticmethod
    def optimize(mixing_lambda, iteration,
                 cluster_count, fc_layers, run_id, network, routing_data, seed):
        train_indices = routing_data.trainingIndices
        test_indices = routing_data.testIndices
        # Learn the standard information gain based accuracies
        train_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=network,
                                                                               routing_data=routing_data,
                                                                               indices=train_indices)
        test_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=network,
                                                                              routing_data=routing_data,
                                                                              indices=test_indices)
        print("train_ig_accuracy={0}".format(train_ig_accuracy))
        print("test_ig_accuracy={0}".format(test_ig_accuracy))
        # Threshold Optimizer
        dto = DirectThresholdOptimizerEntropy(network=network, routing_data=routing_data, seed=seed)
        temperatures_dict = BranchingProbabilityOptimization.calibrate_branching_probabilities(
            network=network, routing_data=routing_data, run_id=run_id, iteration=iteration, indices=train_indices,
            seed=seed)
        dto.build_network()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        pbounds = MixedBayesianOptimizer.calculate_bounds(cluster_count=cluster_count, network=network, kind=dto.kind)

        # Bayesian Optimization Loss Function
        # Loss Function
        def f_(**kwargs):
            # Convert Bayesian Optimization space sample into usable thresholds
            list_of_threshold_dicts = MixedBayesianOptimizer.decode_bayesian_optimization_parameters(
                args_dict=kwargs, network=network, cluster_count=cluster_count, kind=dto.kind)
            results_dict = {}
            sample_indices = {"train": train_indices, "test": test_indices}
            for data_type in ["train", "test"]:
                results = BayesianOptimizationWithClusters.get_thresholding_results(
                    sess=sess,
                    clusterer=None,
                    cluster_count=cluster_count,
                    threshold_optimizer=dto,
                    routing_data=routing_data,
                    indices=sample_indices[
                        data_type],
                    list_of_threshold_dicts=list_of_threshold_dicts,
                    temperatures_dict=temperatures_dict,
                    mixing_lambda=mixing_lambda)
                results_dict[data_type] = results
            print("Train Accuracy: {0} Train Computation Load:{1} Train Score:{2} Ideal Accuracy:{3}".format(
                results_dict["train"]["final_accuracy"],
                results_dict["train"]["final_cost"], results_dict["train"]["final_score"],
                results_dict["train"]["ideal_accuracy"]))
            print("Test Accuracy: {0} Test Computation Load:{1} Test Score:{2} Ideal Accuracy:{3}".format(
                results_dict["test"]["final_accuracy"],
                results_dict["test"]["final_cost"], results_dict["test"]["final_score"],
                results_dict["test"]["ideal_accuracy"]))
            MixedBayesianOptimizer.train_accuracies.append(results_dict["train"]["final_accuracy"])
            MixedBayesianOptimizer.test_accuracies.append(results_dict["test"]["final_accuracy"])
            return results_dict["train"]["final_score"]

        # Phase - 1 Bayesian Optimization of the thresholds
        optimizer = BayesianOptimization(
            f=f_,
            pbounds=pbounds,
        )
        optimizer.maximize(
            init_points=100,
            n_iter=100,
            acq="ei",
            xi=0.0
        )
        best_params = optimizer.max["params"]

        # Phase 2 - Classification according to the best clusters
        list_of_best_thresholds = MixedBayesianOptimizer.decode_bayesian_optimization_parameters(
            args_dict=best_params, network=network, cluster_count=cluster_count, kind=dto.kind)
        results_dict = {}
        sample_indices = {"train": train_indices, "test": test_indices}
        for data_type in ["train", "test"]:
            results = BayesianOptimizationWithClusters.get_thresholding_results(
                sess=sess,
                clusterer=None,
                cluster_count=cluster_count,
                threshold_optimizer=dto,
                routing_data=routing_data,
                indices=sample_indices[
                    data_type],
                list_of_threshold_dicts=list_of_best_thresholds,
                temperatures_dict=temperatures_dict,
                mixing_lambda=mixing_lambda)
            results_dict[data_type] = results
        X = IgBasedBayesianOptimization.get_formatted_input(routing_data=routing_data)
        X_train = X[train_indices]
        X_test = X[test_indices]
        scores_train = results_dict["train"]["scores_matrix"]
        accuracies_train = results_dict["train"]["accuracy_matrix"]
        scores_test = results_dict["test"]["scores_matrix"]
        accuracies_test = results_dict["test"]["accuracy_matrix"]
        y_train = np.argmax(scores_train, axis=1)
        y_test = np.argmax(scores_test, axis=1)

        max_sample_count = 1000
        up_sample_ratio = 5.0
        X_hat, y_hat = BayesianOptimizationWithClusters.resample_training_set(X=X_train, y=y_train,
                                                                              max_sample_count=max_sample_count,
                                                                              up_sample_ratio=up_sample_ratio)
        # X_hat, y_hat = BayesianOptimizationWithClusters.resample_training_set_v3(X=X_train,
        #                                                                          accuracy_matrix=accuracies_train,
        #                                                                          score_matrix=scores_train,
        #                                                                          max_sample_count=max_sample_count)
        # q_counter = Counter(y_train)
        # under_sample_counts = {lbl: min(max_sample_count, cnt) for lbl, cnt in q_counter.items()}
        # under_sampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=under_sample_counts)
        # X_hat, y_hat = under_sampler.fit_resample(X=X_train, y=y_train)
        standard_scaler = StandardScaler()
        pca = PCA()
        # mlp = MLPClassifier()
        svm = SVC()
        pca = PCA()
        mlp = MLPClassifier()
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
                              scoring=["accuracy", "f1_weighted", "f1_micro", "f1_macro",
                                       "balanced_accuracy"],
                              refit="accuracy")
        search.fit(X_hat, y_hat)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        model = search.best_estimator_
        y_pred = {"training": model.predict(X_train),
                  "test": model.predict(X_test)}
        print("*************Training*************")
        print(classification_report(y_pred=y_pred["training"], y_true=y_train))
        print("*************Test*************")
        print(classification_report(y_pred=y_pred["test"], y_true=y_test))
        print("X")
        for data_type in ["train", "test"]:
            results = BayesianOptimizationWithClusters.get_thresholding_results(
                sess=sess,
                clusterer=model,
                cluster_count=cluster_count,
                threshold_optimizer=dto,
                routing_data=routing_data,
                indices=sample_indices[
                    data_type],
                list_of_threshold_dicts=list_of_best_thresholds,
                temperatures_dict=temperatures_dict,
                mixing_lambda=mixing_lambda)
            results_dict[data_type] = results
        print("X")