import multiprocessing

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


class BayesianOptimizationResult:
    def __init__(self, iteration_id, val_score, test_score, val_result_obj, test_result_obj):
        self.iterationId = iteration_id
        self.valScore = val_score
        self.testScore = test_score
        self.valResultObj = val_result_obj
        self.testResultObj = test_result_obj


class BayesianOptimizer:
    def __init__(self, xi, bounds, random_state_func, score_func_val, score_func_test):
        self.xi = xi
        self.bounds = bounds
        self.randomStateFunc = random_state_func
        self.noiseLevel = 1e-10  # 0.0
        self.gpKernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.f_val = score_func_val
        self.f_test = score_func_test

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
        best_state = None
        for trial_id in range(number_of_trials):
            # Sample a new start location
            # thr = self.pick_fully_random_state(as_numpy=True)[0, :]
            x0 = self.randomStateFunc()
            res = minimize(obj_func, x0=x0, bounds=self.bounds, method='L-BFGS-B')
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
                best_state = res.x
        #     proposals.append((0.0, new_best_thr))
        # sorted_proposals = sorted(proposals, key=lambda prop: prop[0], reverse=True)
        return best_score, best_state

    def run(self, max_iterations, initial_sample_count):
        # Step 1: Sample a certain amount of random states in order to train the Gaussian Process Regressor.
        initial_X = []
        initial_y = []
        for idx in range(initial_sample_count):
            initial_X.append(self.randomStateFunc())
        # Step 2: The selected thresholds constitute our hyperparameter samples. Get the performance metrics, the "t"
        # variables.
        for _x in initial_X:
            _y, result_obj = self.f_val(x=_x)
            initial_y.append(_y)
        # Step 3: Build the first dataset for the Gaussian Process Regressor.
        X = np.concatenate(initial_X, axis=0)
        y = np.concatenate(initial_y, axis=0)
        curr_max_score = np.max(y)
        all_results = []
        best_result = None
        print("Process:{0} GP iterations start.".format(multiprocessing.current_process()))
        for iteration_id in range(max_iterations):
            print("Process:{0} Iteration:{1}".format(multiprocessing.current_process(), iteration_id))
            gpr = GaussianProcessRegressor(kernel=self.gpKernel, alpha=self.noiseLevel, n_restarts_optimizer=10)
            gpr.fit(X, y)
            best_score, best_x = self.propose_thresholds(gpr=gpr, max_score=curr_max_score, number_of_trials=100)
            val_score, val_result_obj = self.f_val(x=best_x)
            test_score, test_result_obj = self.f_test(x=best_x)
            result = BayesianOptimizationResult(iteration_id=iteration_id, val_score=val_score,
                                                test_score=test_score, val_result_obj=val_result_obj,
                                                test_result_obj=test_result_obj)
            all_results.append(result)
            if val_score > curr_max_score:
                curr_max_score = val_score
                best_result = result
            X = np.vstack((X, np.expand_dims(best_x, axis=0)))
            y = np.concatenate([y, np.array([val_score])])
        print("Process:{0} ends.".format(multiprocessing.current_process()))
        return best_result, all_results
