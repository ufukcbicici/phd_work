import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from algorithms.threshold_optimization_algorithms.threshold_optimizer import ThresholdOptimizer


class BayesianOptimizer(ThresholdOptimizer):
    def __init__(self, run_id, network, multipath_score_calculators, balance_coefficient,
                 use_weighted_scoring,
                 verbose, initial_sample_count=10000):
        super().__init__(run_id, network, multipath_score_calculators, balance_coefficient,
                         use_weighted_scoring, verbose)
        self.network = network
        self.initialSampleCount = initial_sample_count
        self.gpKernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        # self.sampleCount = sample_count
        # self.threadCount = thread_count
        # self.batchSize = batch_size

    def run(self):
        # Step 1: Sample a certain amount of random thresholds in order to train the Gaussian Process Regressor.
        random_thresholds = []
        for idx in range(self.initialSampleCount):
            random_thresholds.append(self.pick_fully_random_state())
        initial_thresholds = ThresholdOptimizer.thresholds_to_numpy(thresholds=random_thresholds)
        print("X")
