import threading

import numpy as np

from algorithms.threshold_optimization_algorithms.threshold_optimizer import ThresholdOptimizer
from auxillary.general_utility_funcs import UtilityFuncs


class BruteForceOptimizer(ThresholdOptimizer):
    class ScoreCalculator(threading.Thread):
        def __init__(self, thread_id, threshold_list, optimizer):
            super().__init__()
            self.threadId = thread_id
            self.thresholdList = threshold_list
            self.optimizer = optimizer
            self.bestResult = None

        def run(self):
            for threshold_state in self.thresholdList:
                score, accuracy, computation_overload = \
                    self.optimizer.calculate_threshold_score(threshold_state=threshold_state)
                if self.bestResult is None or score >= self.bestResult[1]:
                    self.bestResult = (threshold_state, score, accuracy, computation_overload)

    def __init__(self, network, sample_count, multipath_score_calculators, balance_coefficient, use_weighted_scoring,
                 verbose, thread_count=1, batch_size=10000):
        super().__init__(network, multipath_score_calculators, balance_coefficient, use_weighted_scoring, verbose)
        self.network = network
        self.sampleCount = sample_count
        self.threadCount = thread_count
        self.batchSize = batch_size

    def sample_random_states(self):
        generated_samples = 0
        while True:
            threshold_states = [self.pick_fully_random_state() for _ in range(self.batchSize)]
            yield threshold_states
            generated_samples += self.batchSize
            if generated_samples >= self.sampleCount:
                break

    def run(self):
        for sampled_thresholds in self.sample_random_states():
            thread_to_thresholds_dict = UtilityFuncs.distribute_evenly_to_threads(num_of_threads=self.threadCount,
                                                                                  list_to_distribute=sampled_thresholds)

            threads_dict = {}
            for thread_id in range(self.threadCount):
                threads_dict[thread_id] = BruteForceOptimizer.ScoreCalculator(thread_id=thread_id,
                                                                              threshold_list=thread_to_thresholds_dict[
                                                                                  thread_id],
                                                                              optimizer=self)
                threads_dict[thread_id].start()
            all_results = []
            for thread in threads_dict.values():
                thread.join()
            for thread in threads_dict.values():
                all_results.extend(thread.bestResult)

