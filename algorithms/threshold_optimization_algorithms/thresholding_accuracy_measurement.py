import numpy as np

from auxillary.db_logger import DbLogger


class ThresholdAccuracyMeasurement:
    def __init__(self):
        pass

    @staticmethod
    def calculate_accuracy(run_id, iteration, max_overload, max_limit):
        time_stamp_query = "SELECT TimeStamp FROM threshold_optimization " \
                           "WHERE RunID = {0} AND Iterations = \"{1}\" GROUP BY TimeStamp".format(run_id, iteration)
        time_stamps = DbLogger.read_query(query=time_stamp_query)

        xi_dict = {}
        xi_matrix_dict = {}
        for time_stamp in time_stamps:
            accuracy_query = "SELECT * FROM threshold_optimization " \
                             "WHERE TimeStamp = \"{0}\" ORDER BY ValScore DESC LIMIT {1}".format(time_stamp[0],
                                                                                                 max_limit)
            accuracy_rows = DbLogger.read_query(query=accuracy_query)
            for accuracy_row in accuracy_rows:
                val_accuracy = accuracy_row[6]
                val_computation_overload = accuracy_row[7]
                if val_computation_overload > max_overload:
                    continue
                test_accuracy = accuracy_row[9]
                test_computation_overload = accuracy_row[10]
                if test_accuracy is None or test_computation_overload is None:
                    continue
                xi_val = accuracy_row[-2]
                result_arr = np.array([val_accuracy, val_computation_overload, test_accuracy, test_computation_overload])
                if xi_val not in xi_dict:
                    xi_dict[xi_val] = []
                xi_dict[xi_val].append(result_arr)
        for xi_val, results in xi_dict.items():
            xi_matrix_dict[xi_val] = np.stack(results, axis=0)
            print("xi_val={0}".format(xi_val))
            print(np.mean(xi_matrix_dict[xi_val], axis=0))
