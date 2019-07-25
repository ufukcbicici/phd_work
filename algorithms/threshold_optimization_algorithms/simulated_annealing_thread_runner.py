import threading

from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs


class SimulatedAnnealingThreadRunner:
    class SimulatedAnnealingWorker(threading.Thread):
        def __init__(self, thread_id, sa_optimizers):
            super().__init__()
            self.threadId = thread_id
            self.saOptimizers = sa_optimizers
            self.results = []

        def run(self):
            for sa_optimizer in self.saOptimizers:
                best_state, best_score, best_accuracy, best_computation_overload = sa_optimizer.run()
                result_tuple = (sa_optimizer.runId, sa_optimizer.network.networkName, sa_optimizer.iterations,
                                sa_optimizer.balanceCoefficient, int(sa_optimizer.useWeightedScoring),
                                best_score, best_accuracy, best_computation_overload)
                self.results.append(result_tuple)
                print("ThreadID:{0} Results:{1}".format(self.threadId, result_tuple))

    def __init__(self, sa_optimizers, thread_count):
        self.saOptimizers = sa_optimizers
        self.threadCount = thread_count

    def run(self):
        thread_to_thresholds_dict = UtilityFuncs.distribute_evenly_to_threads(num_of_threads=self.threadCount,
                                                                              list_to_distribute=self.saOptimizers)
        threads_dict = {}
        for thread_id in range(self.threadCount):
            threads_dict[thread_id] = \
                SimulatedAnnealingThreadRunner.SimulatedAnnealingWorker(
                    thread_id=thread_id,
                    sa_optimizers=thread_to_thresholds_dict[thread_id])
            threads_dict[thread_id].start()
        for thread in threads_dict.values():
            thread.join()
        results = []
        for thread in threads_dict.values():
            results.append(thread.results)
        DbLogger.write_into_table(rows=results, table=DbLogger.threshold_optimization, col_count=8)