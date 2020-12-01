import numpy as np
import tensorflow as tf
import os
import pickle
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from sklearn import preprocessing
from collections import Counter

from sklearn.neural_network import MLPClassifier

from algorithms.branching_probability_calibration import BranchingProbabilityOptimization
from algorithms.information_gain_routing_accuracy_calculator import InformationGainRoutingAccuracyCalculator
from algorithms.threshold_optimization_algorithms.bayesian_clusterer import BayesianClusterer
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_networks import DeepQNetworks
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer import \
    DirectThresholdOptimizer
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer_entropy import \
    DirectThresholdOptimizerEntropy
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.mixed_bayesian_optimizer import \
    MixedBayesianOptimizer
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.fashion_net.fashion_cign_lite import FashionCignLite
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class IgBasedBayesianOptimization:
    def __init__(self):
        pass

    @staticmethod
    def get_formatted_input(routing_data):
        sess = tf.Session()
        batch_size = 10000
        X = routing_data.get_dict("pre_branch_feature")[0]
        all_indices = np.arange(X.shape[0])
        X_shape = list(X.shape)
        X_shape[0] = None
        x_input = tf.placeholder(dtype=tf.float32, shape=X_shape, name="x_input")
        net = x_input
        x_output = DeepQNetworks.global_average_pooling(net_input=net)
        X_arr = []
        for batch_idx in range(0, all_indices.shape[0], batch_size):
            X_batch = X[batch_idx: batch_idx + batch_size]
            X_formatted_batch = sess.run([x_output], feed_dict={x_input: X_batch})[0]
            X_arr.append(X_formatted_batch)
        X_formatted = np.concatenate(X_arr, axis=0)
        return X_formatted

    @staticmethod
    def get_thresholding_results(sess, dto, routing_data, indices,
                                 network, mixing_lambda, temperatures_dict, threshold_dict):
        optimizer_results = dto.run_threshold_calculator(sess=sess,
                                                         routing_data=routing_data,
                                                         indices=indices,
                                                         mixing_lambda=mixing_lambda,
                                                         temperatures_dict=temperatures_dict,
                                                         thresholds_dict=threshold_dict)
        return {"accuracy": optimizer_results["accuracy"],
                "meanActivationCost": optimizer_results["meanActivationCost"],
                "score": optimizer_results["score"]}

    @staticmethod
    def optimize(run_id, network, routing_data, seed, mixing_lambda, test_ratio):
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
        # Get IG based routing decisions
        ig_paths = \
            InformationGainRoutingAccuracyCalculator.get_max_likelihood_paths(network=network,
                                                                              routing_data=routing_data)
        ig_leaf_nodes = ig_paths[:, -1]
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(ig_leaf_nodes)
        # Get root node features
        X = IgBasedBayesianOptimization.get_formatted_input(routing_data=routing_data)
        # Classify root node features into labels with a MLP
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        file_name = "leaf_classifier_runid{0}_seed{1}.sav".format(run_id, seed)
        if os.path.exists(file_name):
            f = open(file_name, "rb")
            model = pickle.load(f)
            f.close()
        else:
            # Regression pipeline
            standard_scaler = StandardScaler()
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
            search.fit(X_train, y_train)
            print("Best parameter (CV score=%0.3f):" % search.best_score_)
            print(search.best_params_)
            model = search.best_estimator_
            f = open(file_name, "wb")
            pickle.dump(model, f)
            f.close()
        # Classify training and test sets
        print("*************Training*************")
        y_train_hat = model.predict(X_train)
        print(classification_report(y_pred=y_train_hat, y_true=y_train))
        print("*************Test*************")
        y_test_hat = model.predict(X_test)
        print(classification_report(y_pred=y_test_hat, y_true=y_test))
        print("X")

        # Apply network calibration to routing logits
        temperatures_dict = BranchingProbabilityOptimization.calibrate_branching_probabilities(
            network=network, routing_data=routing_data, run_id=run_id, iteration=0, indices=train_indices,
            seed=seed)

        # Threshold optimizer
        dto = DirectThresholdOptimizer(network=network, routing_data=routing_data, seed=seed)
        dto.build_network()
        sess = tf.Session()
        # Apply Bayesian optimization to every leaf cluster
        accuracies_dict = {}
        costs_dict = {}
        scores_dict = {}
        sample_sizes_dict = {}
        for leaf_node in network.leafNodes:
            leaf_label = le.transform([leaf_node.index])
            leaf_train_indices = train_indices[y_train_hat == leaf_label]
            leaf_test_indices = test_indices[y_test_hat == leaf_label]
            train_accuracies = []
            train_costs = []
            test_accuracies = []
            test_costs = []
            train_scores = []
            test_scores = []

            leaf_train_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=network,
                                                                                        routing_data=routing_data,
                                                                                        indices=leaf_train_indices)
            leaf_test_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=network,
                                                                                       routing_data=routing_data,
                                                                                       indices=leaf_test_indices)
            print("Leaf Node:{0} leaf_train_ig_accuracy={1}".format(leaf_node.index, leaf_train_ig_accuracy))
            print("Leaf Node:{0} test_ig_accuracy={1}".format(leaf_node.index, leaf_test_ig_accuracy))

            print("X")

            def bo_cost_function(**kwargs):
                threshold_dict = MixedBayesianOptimizer.decode_bayesian_optimization_parameters(
                    args_dict=kwargs, network=network, cluster_count=1, kind=dto.kind)[0]
                sample_indices = {"train": leaf_train_indices, "test": leaf_test_indices}
                results_dict = {}
                for data_type in ["train", "test"]:
                    optimizer_results = dto.run_threshold_calculator(sess=sess,
                                                                     routing_data=routing_data,
                                                                     indices=sample_indices[data_type],
                                                                     mixing_lambda=mixing_lambda,
                                                                     temperatures_dict=temperatures_dict,
                                                                     thresholds_dict=threshold_dict)
                    results_dict[data_type] = optimizer_results
                print("Train Accuracy:{0} Train Computation Load:{1} Train Score:{2}".format(
                    results_dict["train"]["accuracy"],
                    results_dict["train"]["meanActivationCost"],
                    results_dict["train"]["score"]))
                print("Test Accuracy:{0} Test Computation Load:{1} Test Score:{2}".format(
                    results_dict["test"]["accuracy"],
                    results_dict["test"]["meanActivationCost"],
                    results_dict["test"]["score"]))
                train_accuracies.append(results_dict["train"]["accuracy"])
                train_costs.append(results_dict["train"]["meanActivationCost"])
                train_scores.append(results_dict["train"]["score"])
                test_accuracies.append(results_dict["test"]["accuracy"])
                test_costs.append(results_dict["test"]["meanActivationCost"])
                test_scores.append(results_dict["test"]["score"])
                return results_dict["train"]["score"]

            pbounds = MixedBayesianOptimizer.calculate_bounds(cluster_count=1, network=network, kind=dto.kind)
            optimizer = BayesianOptimization(
                f=bo_cost_function,
                pbounds=pbounds,
            )
            optimizer.maximize(
                init_points=100,
                n_iter=500,
                acq="ei",
                xi=0.0
            )
            best_params = optimizer.max["params"]
            bo_cost_function(**best_params)
            accuracies_dict[leaf_node.index] = {"train": np.array(train_accuracies), "test": np.array(test_accuracies)}
            costs_dict[leaf_node.index] = {"train": np.array(train_costs), "test": np.array(test_costs)}
            scores_dict[leaf_node.index] = {"train": np.array(train_scores), "test": np.array(test_scores)}
            sample_sizes_dict[leaf_node.index] = {"train": leaf_train_indices.shape[0],
                                                  "test": leaf_test_indices.shape[0]}
            correlation = np.corrcoef(np.array(train_accuracies), np.array(test_accuracies))
            print("Correlation:{0}".format(correlation))

        accuracy_list = []
        cost_list = []
        for leaf_node in network.leafNodes:
            assert accuracies_dict[leaf_node.index]["train"].shape[0] == \
                   accuracies_dict[leaf_node.index]["test"].shape[0] == \
                   costs_dict[leaf_node.index]["train"].shape[0] == \
                   costs_dict[leaf_node.index]["test"].shape[0] == \
                   scores_dict[leaf_node.index]["train"].shape[0] == \
                   scores_dict[leaf_node.index]["test"].shape[0]
            max_indices = np.argsort(scores_dict[leaf_node.index]["train"])[::-1]

            best_train_accuracies = accuracies_dict[leaf_node.index]["train"][max_indices][0:10]
            best_test_accuracies = accuracies_dict[leaf_node.index]["test"][max_indices][0:10]
            best_train_costs = costs_dict[leaf_node.index]["train"][max_indices][0:10]
            best_test_costs = costs_dict[leaf_node.index]["test"][max_indices][0:10]
            leaf_weight = float(sample_sizes_dict[leaf_node.index]["test"]) / float(y_test.shape[0])
            accuracy_list.append(np.mean(best_test_accuracies) * leaf_weight)
            cost_list.append(np.mean(best_test_costs) * leaf_weight)
        final_test_accuracy = np.sum(accuracy_list)
        final_test_cost = np.sum(cost_list)
        print("final_test_accuracy:{0}".format(final_test_accuracy))
        print("final_test_cost:{0}".format(final_test_cost))


            # # Threshold Optimizer
        # dto = DirectThresholdOptimizerEntropy(network=network, routing_data=routing_data, seed=seed,
        #                                       train_indices=train_indices, test_indices=test_indices)
        # temperatures_dict = BranchingProbabilityOptimization.calibrate_branching_probabilities(
        #     network=network, routing_data=routing_data, run_id=run_id, iteration=iteration, indices=train_indices,
        #     seed=seed)
        # dto.build_network()
        # # Clusterer
        # bc = BayesianClusterer(network=network, routing_data=routing_data, cluster_count=cluster_count,
        #                        fc_layers=fc_layers)
        # # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # # config = tf.ConfigProto(device_count={'GPU': 0})
        # # sess = tf.Session(config=config)
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # mixing_lambda = 1.0
        #
        # # Loss Function
        # def f_(**kwargs):
        #     # Convert Bayesian Optimization space sample into usable thresholds
        #     list_of_threshold_dicts = MixedBayesianOptimizer.decode_bayesian_optimization_parameters(
        #         args_dict=kwargs, network=network, cluster_count=cluster_count, kind=dto.kind)
        #     results_dict = {}
        #     sample_indices = {"train": train_indices, "test": test_indices}
        #     for data_type in ["train", "test"]:
        #         results = MixedBayesianOptimizer.get_thresholding_results(
        #             sess=sess,
        #             clusterer=bc,
        #             cluster_count=cluster_count,
        #             threshold_optimizer=dto,
        #             routing_data=routing_data,
        #             indices=sample_indices[data_type],
        #             list_of_threshold_dicts=list_of_threshold_dicts,
        #             temperatures_dict=temperatures_dict,
        #             mixing_lambda=mixing_lambda)
        #         results_dict[data_type] = results
        #     print("Train Accuracy: {0} Train Computation Load:{1} Train Score:{2}".format(
        #         results_dict["train"]["final_accuracy"],
        #         results_dict["train"]["final_cost"], results_dict["train"]["final_score"]))
        #     print("Test Accuracy: {0} Test Computation Load:{1} Test Score:{2}".format(
        #         results_dict["test"]["final_accuracy"],
        #         results_dict["test"]["final_cost"], results_dict["test"]["final_score"]))
        #     MixedBayesianOptimizer.train_accuracies.append(results_dict["train"]["final_accuracy"])
        #     MixedBayesianOptimizer.test_accuracies.append(results_dict["test"]["final_accuracy"])
        #     return results_dict["train"]["final_score"]
        #
        # pbounds = MixedBayesianOptimizer.calculate_bounds(cluster_count=cluster_count, network=network, kind=dto.kind)
        #
        # # Two - Phase optimization iterations
        # for iteration_id in range(optimization_iterations_count):
        #     # Phase - 1 Bayesian Optimization of the thresholds
        #     optimizer = BayesianOptimization(
        #         f=f_,
        #         pbounds=pbounds,
        #     )
        #     optimizer.maximize(
        #         init_points=100,
        #         n_iter=100,
        #         acq="ei",
        #         xi=0.0
        #     )
        #     best_params = optimizer.max["params"]
        #     f_(**best_params)
        #     # Phase - 2 Gradient Based Optimization of the Clusterer
        #     list_of_best_thresholds = MixedBayesianOptimizer.decode_bayesian_optimization_parameters(
        #         args_dict=best_params, network=network, cluster_count=cluster_count, kind=dto.kind)
        #     res = MixedBayesianOptimizer.get_thresholding_results(
        #         sess=sess,
        #         clusterer=bc,
        #         cluster_count=cluster_count,
        #         threshold_optimizer=dto,
        #         routing_data=routing_data,
        #         indices=train_indices,
        #         list_of_threshold_dicts=list_of_best_thresholds,
        #         temperatures_dict=temperatures_dict,
        #         mixing_lambda=mixing_lambda)
        #     features = routing_data.get_dict("pre_branch_feature")[0][train_indices]
        #     scores_after_bo = res["scores_matrix"]
        #     assert features.shape[0] == scores_after_bo.shape[0]
        #     bc.optimize_clustering(sess=sess, features=features, scores=scores_after_bo)
        #     f_(**best_params)
        # print("X")
