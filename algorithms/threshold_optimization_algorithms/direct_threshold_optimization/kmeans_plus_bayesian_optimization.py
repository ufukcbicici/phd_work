import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

from algorithms.branching_probability_calibration import BranchingProbabilityOptimization
from algorithms.ensemble_bayesian_threshold_optimizer import EnsembleBayesianThresholdOptimizer
from auxillary.general_utility_funcs import UtilityFuncs
from algorithms.bayesian_threshold_optimizer import BayesianThresholdOptimizer


class KmeansPlusBayesianOptimization:
    def __init__(self):
        pass

    @staticmethod
    def optimize(cluster_count, network, routing_data, mixing_lambda, xi, seed, run_id, iteration):
        # Prepare required training - test data
        train_indices = routing_data.trainingIndices
        test_indices = routing_data.testIndices
        X = routing_data.get_dict("pre_branch_feature")[0]
        y = routing_data.labelList
        # X_formatted = UtilityFuncs.vectorize_with_gap(X)
        X_formatted = X
        posteriors = [routing_data.get_dict("posterior_probs")[node.index] for node in network.leafNodes]
        posteriors = np.stack(posteriors, axis=-1)
        label_count = posteriors.shape[1]
        X_train = X_formatted[train_indices]
        X_test = X_formatted[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        posteriors_train = posteriors[train_indices]
        posteriors_test = posteriors[test_indices]
        # Apply k-means if necessary to build clusters
        if cluster_count > 1:
            kmeans = KMeans(n_clusters=cluster_count).fit(X_train)
            training_clusters = kmeans.predict(X_train)
            test_clusters = kmeans.predict(X_test)
        else:
            training_clusters = np.zeros(shape=(X_train.shape[0],), dtype=np.int32)
            test_clusters = np.zeros(shape=(X_test.shape[0],), dtype=np.int32)
        # Calibrate the branching probabilities
        temperatures_dict = BranchingProbabilityOptimization.calibrate_branching_probabilities(
            network=network, routing_data=routing_data, run_id=run_id, iteration=iteration, indices=train_indices,
            seed=seed)
        # Create the Bayesian Optimization Object
        tf.reset_default_graph()
        sess = tf.Session()
        bayesian_optimizer = BayesianThresholdOptimizer(
            network=network,
            routing_data=routing_data,
            session=sess,
            temperatures_dict=temperatures_dict,
            seed=seed,
            threshold_kind="probability",
            mixing_lambda=mixing_lambda,
            run_id=run_id)

        # good_thresholds = {0: 0.04031152075333284,
        #                    1: 0.5072578562024834,
        #                    2: 0.3954713354264498}
        # experimental_result = bayesian_optimizer.optimize(init_points=100,
        #                                                   n_iter=150,
        #                                                   xi=0.0,
        #                                                   weight_bound_min=-5.0,
        #                                                   weight_bound_max=5.0,
        #                                                   use_these_thresholds=good_thresholds,
        #                                                   use_these_weights=np.ones(shape=(1, posteriors.shape[-1]),
        #                                                                             dtype=np.float32))
        # weights_optimization_result = bayesian_optimizer.optimize(init_points=100,
        #                                                           n_iter=250,
        #                                                           xi=0.01,
        #                                                           weight_bound_min=-2.0,
        #                                                           weight_bound_max=2.0,
        #                                                           use_these_thresholds=good_thresholds,
        #                                                           use_these_weights=None)
        thrs_optimization_result = bayesian_optimizer.optimize(init_points=50,
                                                               n_iter=100,
                                                               xi=xi,
                                                               weight_bound_min=-2.0,
                                                               weight_bound_max=2.0,
                                                               use_these_thresholds=None,
                                                               use_these_weights=np.ones(
                                                                   shape=(1, posteriors.shape[-1]),
                                                                   dtype=np.float32))
        # complete_optimization_result = bayesian_optimizer.optimize(init_points=100,
        #                                                            n_iter=250,
        #                                                            xi=0.01,
        #                                                            weight_bound_min=-2.0,
        #                                                            weight_bound_max=2.0,
        #                                                            use_these_thresholds=None,
        #                                                            use_these_weights=None)
        print("X")

    @staticmethod
    def optimize_ensemble(list_of_networks, list_of_routing_data, mixing_lambda, xi, seed, run_id, iteration):
        list_of_temperatures = []
        for network, routing_data in zip(list_of_networks, list_of_routing_data):
            temperatures_dict = BranchingProbabilityOptimization.calibrate_branching_probabilities(
                network=network,
                routing_data=routing_data,
                run_id=run_id,
                iteration=iteration,
                indices=routing_data.trainingIndices,
                seed=seed)
            list_of_temperatures.append(temperatures_dict)
        # Create the Bayesian Optimization Object
        tf.reset_default_graph()
        sess = tf.Session()
        bayesian_optimizer = EnsembleBayesianThresholdOptimizer(
            list_of_networks=list_of_networks,
            list_of_routing_data=list_of_routing_data,
            list_of_temperature_dicts=list_of_temperatures,
            session=sess,
            seed=seed,
            threshold_kind="entropy",
            mixing_lambda=mixing_lambda,
            run_id=run_id)
        thrs_optimization_result = \
            bayesian_optimizer.optimize(init_points=50,
                                        n_iter=100,
                                        xi=xi,
                                        weight_bound_min=-2.0,
                                        weight_bound_max=2.0,
                                        use_these_thresholds=None,
                                        use_these_weights=
                                        [np.ones(shape=(1, len(network.leafNodes)), dtype=np.float32)
                                         for network in list_of_networks])
        print("X")
