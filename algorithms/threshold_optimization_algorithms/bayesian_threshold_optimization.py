import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm
from scipy.optimize import minimize
import multiprocessing

from algorithms.bayesian_optimization import BayesianOptimizer
from algorithms.coordinate_ascent_optimizer import CoordinateAscentOptimizer
from algorithms.threshold_optimization_algorithms.bayesian_routing_weights_optimization import \
    RoutingWeightBayesianOptimizer
from algorithms.threshold_optimization_algorithms.routing_weight_calculator import RoutingWeightCalculator
from algorithms.threshold_optimization_algorithms.routing_weight_deep_classifier import RoutingWeightDeepClassifier
from algorithms.threshold_optimization_algorithms.routing_weight_deep_classifier_ensemble import \
    RoutingWeightDeepClassifierEnsemble
from algorithms.threshold_optimization_algorithms.routing_weight_non_deep_classifier import \
    RoutingWeightNonDeepClassifier
from algorithms.threshold_optimization_algorithms.routing_weights_deep_regressor import RoutingWeightDeepRegressor
from algorithms.threshold_optimization_algorithms.routing_weights_deep_softmax_regressor import \
    RoutingWeightDeepSoftmaxRegressor
from algorithms.threshold_optimization_algorithms.routing_weights_finder_with_least_squares import \
    RoutingWeightsFinderWithLeastSquares
from algorithms.threshold_optimization_algorithms.threshold_optimizer import ThresholdOptimizer
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs


class BayesianThresholdOptimizer(ThresholdOptimizer):
    def __init__(self, run_id, network, iteration, routing_data, multipath_score_calculator, balance_coefficient,
                 use_weighted_scoring, lock, verbose, xi=0.01, max_iter=10000, initial_sample_count=10000,
                 test_ratio=0.5):
        super().__init__(run_id, network, routing_data, multipath_score_calculator, balance_coefficient,
                         test_ratio, use_weighted_scoring, verbose)
        self.network = network
        self.initialSampleCount = initial_sample_count
        self.xi = xi
        self.noiseLevel = 1e-10  # 0.0
        self.coordAscentSampleCount = 10000
        self.maxIterations = max_iter
        self.thresholdBounds = []
        self.iteration = iteration
        self.lock = lock
        bounds_dict = {}
        for node in self.network.topologicalSortedNodes:
            if node.isLeaf:
                continue
            child_count = len(self.network.dagObject.children(node=node))
            max_threshold = 1.0 / float(child_count)
            bounds_dict[node.index] = ([0, max_threshold], child_count)
        indices_sorted = sorted(list(bounds_dict.keys()))
        for idx in indices_sorted:
            bounds = bounds_dict[idx][0]
            child_count = bounds_dict[idx][1]
            for _ in range(child_count):
                self.thresholdBounds.append(bounds)

        self.bayesianOptimizer = BayesianOptimizer(xi=self.xi, bounds=self.thresholdBounds,
                                                   random_state_func=self.random_state_sampler,
                                                   score_func_val=self.calculate_threshold_score_val,
                                                   score_func_test=self.calculate_threshold_score_test)

    def random_state_sampler(self):
        random_state = self.pick_fully_random_state(as_numpy=True)[0, :]
        return random_state

    def calculate_threshold_score_val(self, x):
        best_threshold_as_dict = self.numpy_to_threshold_dict(thresholds_arr=x)
        score, result_obj = \
            self.calculate_threshold_score(threshold_state=best_threshold_as_dict, routing_data=self.validationData)
        return score, result_obj

    def calculate_threshold_score_test(self, x):
        best_threshold_as_dict = self.numpy_to_threshold_dict(thresholds_arr=x)
        score, result_obj = \
            self.calculate_threshold_score(threshold_state=best_threshold_as_dict, routing_data=self.testData)
        return score, result_obj

    def run(self):
        best_result, all_results = \
            self.bayesianOptimizer.run(max_iterations=self.maxIterations, initial_sample_count=self.initialSampleCount)
        # val_routing_matrix = best_result.valResultObj.routingMatrix
        # test_routing_matrix = best_result.testResultObj.routingMatrix
        # routing_weight_calculator = \
        #     RoutingWeightBayesianOptimizer(network=self.network,
        #                                    validation_routing_matrix=val_routing_matrix,
        #                                    test_routing_matrix=test_routing_matrix,
        #                                    validation_data=self.validationData,
        #                                    test_data=self.testData, min_weight=0.0,
        #                                    max_weight=1.0,
        #                                    best_val_accuracy=best_result.valResultObj.accuracy,
        #                                    best_test_accuracy=best_result.testResultObj.accuracy)
        # routing_weight_calculator.run()

        # routing_weight_calculator = RoutingWeightDeepSoftmaxRegressor(network=self.network,
        #                                                               validation_routing_matrix=val_routing_matrix,
        #                                                               test_routing_matrix=test_routing_matrix,
        #                                                               validation_data=self.validationData,
        #                                                               test_data=self.testData,
        #                                                               layers=[200, 100],
        #                                                               l2_lambda=0.0001,
        #                                                               batch_size=64,
        #                                                               max_iteration=1000000,
        #                                                               use_multi_path_only=True,
        #                                                               use_sparse_softmax=True)
        # routing_weight_calculator.run()
        # print("X")
        self.write_to_db(results=all_results)
        # return all_results

    def write_to_db(self, results):
        timestamp = UtilityFuncs.get_timestamp()
        db_rows = []
        # result = (iteration_id, new_score, new_accuracy, new_computation_overload)
        for result in results:
            val_score = result.valScore
            val_accuracy = result.valResultObj.accuracy
            val_overload = result.valResultObj.computationOverload
            test_score = result.testScore
            test_accuracy = result.testResultObj.accuracy
            test_overload = result.testResultObj.computationOverload
            db_rows.append((self.runId, self.network.networkName,
                            self.iteration,
                            self.balanceCoefficient,
                            int(self.useWeightedScoring), val_score, val_accuracy, val_overload,
                            test_score, test_accuracy, test_overload,
                            "Bayesian Optimization", self.xi, timestamp))
        self.lock.acquire()
        DbLogger.write_into_table(rows=db_rows, table=DbLogger.threshold_optimization, col_count=14)
        self.lock.release()
