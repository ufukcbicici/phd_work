import numpy as np
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants

min_run_id = 998
max_run_id = 1072
iteration = 48000
confusion_matrices = {}
accuracies_dict = {}

for run_id in range(min_run_id, max_run_id + 1):
    print("Processing RunId:{0}".format(run_id))
    cm3 = DbLogger.read_confusion_matrix(run_id=run_id, dataset=2, iteration=iteration, num_of_labels=10, leaf_id=3)
    cm4 = DbLogger.read_confusion_matrix(run_id=run_id, dataset=2, iteration=iteration, num_of_labels=10, leaf_id=4)
    cm5 = DbLogger.read_confusion_matrix(run_id=run_id, dataset=2, iteration=iteration, num_of_labels=10, leaf_id=5)
    cm6 = DbLogger.read_confusion_matrix(run_id=run_id, dataset=2, iteration=iteration, num_of_labels=10, leaf_id=6)
    cm_list = [cm3, cm4, cm5, cm6]
    mode_counts = []
    for cm in cm_list:
        total = np.sum(cm)
        label_counts = np.sum(cm, axis=1)
        label_distribution = label_counts / total
        label_distribution_dict = {}
        for i in range(len(label_distribution)):
            label_distribution_dict[i] = label_distribution[i]
        modes = UtilityFuncs.get_modes_from_distribution(distribution=label_distribution_dict,
                                                         percentile_threshold=GlobalConstants.PERCENTILE_THRESHOLD)
        mode_counts.append(len(modes))
    test_accuracy = DbLogger.read_test_accuracy(run_id=run_id, type="\"Regular\"")
    mode_counts_sorted = tuple(sorted(mode_counts, key=lambda cnt: cnt, reverse=True))
    total_mode_count = sum(mode_counts_sorted)
    if total_mode_count != 10:
        print("X")
    if mode_counts_sorted not in accuracies_dict:
        accuracies_dict[mode_counts_sorted] = []
    accuracies_dict[mode_counts_sorted].append(test_accuracy)
mean_accuracies_dict = {}
for k, v in accuracies_dict.items():
    mean_accuracies_dict[k] = sum(v) / float(len(v))
print("X")