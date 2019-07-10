import threading

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
            for idx, threshold_state in enumerate(self.thresholdList):
                score, accuracy, computation_overload = \
                    self.optimizer.calculate_threshold_score(threshold_state=threshold_state)
                if self.bestResult is None or score >= self.bestResult[1]:
                    self.bestResult = (threshold_state, score, accuracy, computation_overload)
                if self.optimizer.verbose and (idx + 1) % 100 == 0:
                    print("Thread ID:{0} threshold_state:{1} score:{2} accuracy:{3} computation_overload:{4}".format(
                        self.threadId, self.bestResult[0], self.bestResult[1], self.bestResult[2], self.bestResult[3],
                        self.bestResult[4]))
            print("Thread ID:{0} threshold_state:{1} score:{2} accuracy:{3} computation_overload:{4}".format(
                self.threadId, self.bestResult[0], self.bestResult[1], self.bestResult[2], self.bestResult[3],
                self.bestResult[4]))

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
        all_results = []
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
            for thread in threads_dict.values():
                thread.join()
            for thread in threads_dict.values():
                all_results.append(thread.bestResult)
        for result in all_results:
            print("threshold_state:{0} score:{1} accuracy:{2} computation_overload:{3}".format(
                result[0], result[1], result[2], result[3], result[4]))
