import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from algorithms.threshold_optimization_algorithms.threshold_optimizer import ThresholdOptimizer


class BayesianOptimizer(ThresholdOptimizer):
    def __init__(self, run_id, network, multipath_score_calculators, balance_coefficient,
                 use_weighted_scoring,
                 verbose, max_iter=1000, initial_sample_count=10000):
        super().__init__(run_id, network, multipath_score_calculators, balance_coefficient,
                         use_weighted_scoring, verbose)
        self.network = network
        self.initialSampleCount = initial_sample_count
        self.gpKernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.noiseLevel = 1e-10
        self.maxIterations = max_iter
        # self.sampleCount = sample_count
        # self.threadCount = thread_count
        # self.batchSize = batch_size

    def run(self):
        # Step 1: Sample a certain amount of random thresholds in order to train the Gaussian Process Regressor.
        random_thresholds = []
        for idx in range(self.initialSampleCount):
            random_thresholds.append(self.pick_fully_random_state())
        initial_thresholds = ThresholdOptimizer.thresholds_to_numpy(thresholds=random_thresholds)
        # Step 2: The selected thresholds constitute our hyperparameter samples. Get the performance metrics, the "t"
        # variables.
        scores = []
        for threshold in random_thresholds:
            final_score, final_accuracy, final_computation_overload = \
                self.calculate_threshold_score(threshold_state=threshold)
            scores.append(final_score)
        # Step 3: Build the first dataset for the Gaussian Process Regressor.
        X = np.copy(initial_thresholds)
        y = np.array(scores)
        # Step 4: The Bayesian optimization framework
        gaussian_process_regressor = GaussianProcessRegressor(kernel=self.gpKernel, alpha=self.noiseLevel,
                                                              n_restarts_optimizer=10)
        for iteration_id in range(self.maxIterations):
            print("X")
        print("X")
