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

        params_dict = {}
        param_matrix_dict = {}
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
                balance_coeff = accuracy_row[3]
                xi_val = accuracy_row[-2]
                result_arr = np.array([val_accuracy, val_computation_overload, test_accuracy, test_computation_overload])
                if (balance_coeff, xi_val) not in params_dict:
                    params_dict[(balance_coeff, xi_val)] = []
                params_dict[(balance_coeff, xi_val)].append(result_arr)
        for param_tpl, results in params_dict.items():
            param_matrix_dict[param_tpl] = np.stack(results, axis=0)
            print("lambda_val={0} xi_val={1}".format(param_tpl[0], param_tpl[1]))
            print(np.mean(param_matrix_dict[param_tpl], axis=0))
