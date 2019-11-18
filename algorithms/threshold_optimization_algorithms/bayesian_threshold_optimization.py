import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm
from scipy.optimize import minimize
from algorithms.coordinate_ascent_optimizer import CoordinateAscentOptimizer
from algorithms.threshold_optimization_algorithms.threshold_optimizer import ThresholdOptimizer


class BayesianOptimizer(ThresholdOptimizer):
    def __init__(self, run_id, network, multipath_score_calculators, balance_coefficient,
                 use_weighted_scoring,
                 verbose, xi=0.01, max_iter=10000, initial_sample_count=10000):
        super().__init__(run_id, network, multipath_score_calculators, balance_coefficient,
                         use_weighted_scoring, verbose)
        self.network = network
        self.initialSampleCount = initial_sample_count
        self.gpKernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.xi = xi
        self.noiseLevel = 1e-10 # 0.0
        self.coordAscentSampleCount = 10000
        self.maxIterations = max_iter
        self.thresholdBounds = []
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
        # Step 1: Sample a certain amount of random thresholds in order to train the Gaussian Process Regressor.
        random_thresholds = []
        for idx in range(self.initialSampleCount):
            random_thresholds.append(self.pick_fully_random_state())
        initial_thresholds = self.thresholds_to_numpy(thresholds=random_thresholds)
        # Step 2: The selected thresholds constitute our hyperparameter samples. Get the performance metrics, the "t"
        # variables.
        outputs = []
        for threshold in random_thresholds:
            final_score, final_accuracy, final_computation_overload = \
                self.calculate_threshold_score(threshold_state=threshold)
            outputs.append(np.array([final_score, final_accuracy, final_computation_overload]))
        # Step 3: Build the first dataset for the Gaussian Process Regressor.
        outputs = np.stack(outputs, axis=0)
        X = np.copy(initial_thresholds)
        y = np.copy(outputs[:, 0])
        # Step 4: The Bayesian optimization framework
        curr_max_score = np.max(y)
        best_result = []
        for iteration_id in range(self.maxIterations):
            gpr = GaussianProcessRegressor(kernel=self.gpKernel, alpha=self.noiseLevel, n_restarts_optimizer=10)
            gpr.fit(X, y)
            best_score, best_threshold = self.propose_thresholds(gpr=gpr, max_score=curr_max_score,
                                                                 number_of_trials=100)
            best_threshold_as_dict = self.numpy_to_threshold_dict(thresholds_arr=best_threshold)
            new_score, new_accuracy, new_computation_overload = \
                self.calculate_threshold_score(threshold_state=best_threshold_as_dict)
            print("Bayesian Optimization Iteration Id:{0}".format(iteration_id))
            print("Proposed thresholds:{0}".format(best_threshold))
            print("Predicted score at the proposal:{0}".format(best_score))
            print("Actual score at the proposal:{0}".format(new_score))
            print("Actual computation load at the proposal:{0}".format(new_computation_overload))
            if new_score > curr_max_score:
                curr_max_score = new_score
                best_result = (new_score, new_accuracy, new_computation_overload)
                print("Best result so far:{0},{1},{2}".format(best_result[0], best_result[1], best_result[2]))
            X = np.vstack((X, np.expand_dims(best_threshold, axis=0)))
            y = np.concatenate([y, np.array([new_score])])
            print("X")
        print("X")

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
