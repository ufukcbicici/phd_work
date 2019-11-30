import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm
from scipy.optimize import minimize
import multiprocessing
from algorithms.coordinate_ascent_optimizer import CoordinateAscentOptimizer
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


class BayesianOptimizer(ThresholdOptimizer):
    def __init__(self, run_id, network, iteration, routing_data, multipath_score_calculator, balance_coefficient,
                 use_weighted_scoring, lock, verbose, xi=0.01, max_iter=10000, initial_sample_count=10000,
                 test_ratio=0.5):
        super().__init__(run_id, network, routing_data, multipath_score_calculator, balance_coefficient,
                         test_ratio, use_weighted_scoring, verbose)
        self.network = network
        self.initialSampleCount = initial_sample_count
        self.gpKernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
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

    def run(self):
        print("Process:{0} starts.".format(multiprocessing.current_process()))
        # Step 1: Sample a certain amount of random thresholds in order to train the Gaussian Process Regressor.
        random_thresholds = []
        for idx in range(self.initialSampleCount):
            random_thresholds.append(self.pick_fully_random_state())
        initial_thresholds = self.thresholds_to_numpy(thresholds=random_thresholds)
        # Step 2: The selected thresholds constitute our hyperparameter samples. Get the performance metrics, the "t"
        # variables.
        outputs = []
        for threshold in random_thresholds:
            final_score, result_obj = \
                self.calculate_threshold_score(threshold_state=threshold, routing_data=self.validationData)
            final_accuracy = result_obj.accuracy
            final_computation_overload = result_obj.computationOverload
            outputs.append(np.array([final_score, final_accuracy, final_computation_overload]))
        # Step 3: Build the first dataset for the Gaussian Process Regressor.
        outputs = np.stack(outputs, axis=0)
        X = np.copy(initial_thresholds)
        y = np.copy(outputs[:, 0])
        # Step 4: The Bayesian optimization framework
        curr_max_score = np.max(y)
        all_results = []
        best_result = None
        print("Process:{0} GP iterations start.".format(multiprocessing.current_process()))
        for iteration_id in range(self.maxIterations):
            print("Process:{0} Iteration:{1}".format(multiprocessing.current_process(), iteration_id))
            gpr = GaussianProcessRegressor(kernel=self.gpKernel, alpha=self.noiseLevel, n_restarts_optimizer=10)
            gpr.fit(X, y)
            best_score, best_threshold = self.propose_thresholds(gpr=gpr, max_score=curr_max_score,
                                                                 number_of_trials=100)
            best_threshold_as_dict = self.numpy_to_threshold_dict(thresholds_arr=best_threshold)
            val_score, val_result_obj = \
                self.calculate_threshold_score(threshold_state=best_threshold_as_dict, routing_data=self.validationData)
            test_score, test_result_obj = \
                self.calculate_threshold_score(threshold_state=best_threshold_as_dict, routing_data=self.testData)
            result = (iteration_id,
                      val_score, val_result_obj.accuracy, val_result_obj.computationOverload,
                      test_score, test_result_obj.accuracy, test_result_obj.computationOverload,
                      val_result_obj, test_result_obj)
            all_results.append(result)
            # print("Bayesian Optimization Iteration Id:{0}".format(iteration_id))
            # print("Proposed thresholds:{0}".format(best_threshold))
            # print("Predicted score at the proposal:{0}".format(best_score))
            # print("Actual score at the proposal:{0}".format(new_score))
            # print("Actual computation load at the proposal:{0}".format(new_computation_overload))
            if val_score > curr_max_score:
                curr_max_score = val_score
                best_result = result
                print("Process Id:{0} Best result so far at iteration {1} - Validation: {2},{3},{4} Test: {5},{6},{7}"
                      .format(multiprocessing.current_process(), iteration_id,
                              best_result[1], best_result[2], best_result[3],
                              best_result[4], best_result[5], best_result[6]))
            X = np.vstack((X, np.expand_dims(best_threshold, axis=0)))
            y = np.concatenate([y, np.array([val_score])])
        print("Process:{0} ends.".format(multiprocessing.current_process()))
        # self.routing_performance_analysis(best_result=best_result)
        val_result_obj = best_result[-2]
        test_result_obj = best_result[-1]
        val_routing_matrix = val_result_obj.routingMatrix
        test_routing_matrix = test_result_obj.routingMatrix
        # routing_weight_calculator = RoutingWeightCalculator(network=self.network,
        #                                                     validation_routing_matrix=val_routing_matrix,
        #                                                     test_routing_matrix=test_routing_matrix,
        #                                                     validation_data=self.validationData,
        #                                                     test_data=self.testData)
        # for i in range(1, 1001, 1):
        #     print("Cluster Size:{0}".format(i + 1))
        #     routing_weight_calculator = RoutingWeightsFinderWithLeastSquares(network=self.network,
        #                                                                      validation_routing_matrix=val_routing_matrix,
        #                                                                      test_routing_matrix=test_routing_matrix,
        #                                                                      validation_data=self.validationData,
        #                                                                      test_data=self.testData,
        #                                                                      min_cluster_size=i + 1)
        #     routing_weight_calculator.run()
        # routing_weight_calculator = RoutingWeightDeepSoftmaxRegressor(network=self.network,
        #                                                               validation_routing_matrix=val_routing_matrix,
        #                                                               test_routing_matrix=test_routing_matrix,
        #                                                               validation_data=self.validationData,
        #                                                               test_data=self.testData,
        #                                                               layers=[200, 100],
        #                                                               l2_lambda=0.0,
        #                                                               batch_size=5000,
        #                                                               max_iteration=100000)
        # routing_weight_calculator.run()
        # l2_lambda_list = [0.00001*i for i in range(21)] * 5
        # l2_lambda_list = sorted(l2_lambda_list)
        # for l2_lambda in l2_lambda_list:
        # routing_weight_calculator = RoutingWeightDeepClassifier(network=self.network,
        #                                                         validation_routing_matrix=val_routing_matrix,
        #                                                         test_routing_matrix=test_routing_matrix,
        #                                                         validation_data=self.validationData,
        #                                                         test_data=self.testData,
        #                                                         layers=[400, 200],
        #                                                         l2_lambda=0.0,
        #                                                         batch_size=5000,
        #                                                         max_iteration=1000000,
        #                                                         use_multi_path_only=True)
        routing_weight_calculator = RoutingWeightDeepSoftmaxRegressor(network=self.network,
                                                                      validation_routing_matrix=val_routing_matrix,
                                                                      test_routing_matrix=test_routing_matrix,
                                                                      validation_data=self.validationData,
                                                                      test_data=self.testData,
                                                                      layers=[200, 100],
                                                                      l2_lambda=0.0,
                                                                      batch_size=5000,
                                                                      max_iteration=1000000,
                                                                      use_multi_path_only=True)
        routing_weight_calculator.run()
        # print("X")
        # self.write_to_db(results=all_results)
        # return all_results

    def expected_improvement(self, X, **kwargs):
        gpr = kwargs["gpr"]
        max_score = kwargs["max_score"]
        if len(X.shape) == 1:
            mu, sigma = gpr.predict(np.expand_dims(X, axis=0), return_std=True)
        else:
            mu, sigma = gpr.predict(X, return_std=True)
        with np.errstate(divide='warn'):
            imp = mu - max_score - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def gpr_score_predict(self, X, gpr, balance=1.0):
        if len(X.shape) == 1:
            mean, std = gpr.predict(np.expand_dims(X, axis=0), return_std=True)
        else:
            mean, std = gpr.predict(X, return_std=True)
        scores = balance * mean + (balance - 1.0) * std
        return scores

    def propose_thresholds(self, gpr, max_score, number_of_trials=10):
        proposals = []

        def obj_func(X):
            return -self.expected_improvement(X=X, gpr=gpr, max_score=max_score)

        best_score = -np.infty
        best_threshold = None
        for trial_id in range(number_of_trials):
            # Sample a new start location
            thr = self.pick_fully_random_state(as_numpy=True)[0, :]
            res = minimize(obj_func, x0=thr, bounds=self.thresholdBounds, method='L-BFGS-B')
            # new_max_score, new_best_thr = \
            #     CoordinateAscentOptimizer.maximizer(bounds=self.thresholdBounds, p0=thr,
            #                                         sample_count_per_coordinate=self.coordAscentSampleCount,
            #                                         func=self.expected_improvement, gpr=gpr,
            #                                         max_score=max_score, tol=1e-300)
            # new_max_score, new_best_thr = \
            #     CoordinateAscentOptimizer.maximizer(bounds=self.thresholdBounds, p0=thr,
            #                                         sample_count_per_coordinate=self.coordAscentSampleCount,
            #                                         func=self.gpr_score_predict, gpr=gpr, tol=1e-300)
            if -res.fun > best_score:
                best_score = -res.fun
                best_threshold = res.x
        #     proposals.append((0.0, new_best_thr))
        # sorted_proposals = sorted(proposals, key=lambda prop: prop[0], reverse=True)
        return best_score, best_threshold

    def write_to_db(self, results):
        timestamp = UtilityFuncs.get_timestamp()
        db_rows = []
        # result = (iteration_id, new_score, new_accuracy, new_computation_overload)
        for result in results:
            val_score = result[1]
            val_accuracy = result[2]
            val_overload = result[3]
            test_score = result[4]
            test_accuracy = result[5]
            test_overload = result[6]
            db_rows.append((self.runId, self.network.networkName,
                            self.iteration,
                            self.balanceCoefficient,
                            int(self.useWeightedScoring), val_score, val_accuracy, val_overload,
                            test_score, test_accuracy, test_overload,
                            "Bayesian Optimization", self.xi, timestamp))
        self.lock.acquire()
        DbLogger.write_into_table(rows=db_rows, table=DbLogger.threshold_optimization, col_count=14)
        self.lock.release()
