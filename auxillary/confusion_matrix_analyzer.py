import numpy as np

from auxillary.confusion_matrix_grapher import ConfusionMatrixGrapher
from auxillary.db_logger import DbLogger


class ConfusionMatrixAnalyzer:
    @staticmethod
    def analyze_confusion_matrix(cm, threshold_percentile_for_modes):
        # Determine True Label Distribution
        total = np.sum(cm)
        label_counts = np.sum(cm, axis=1)
        label_distribution = label_counts / total
        label_distribution_dict = {}
        for i in range(label_distribution.shape[0]):
            label_distribution_dict[i] = label_distribution[i]
        # Overall leaf performance
        total_correct = np.trace(cm)
        overall_accuracy = total_correct / total
        print("Overall accuracy:{0} {1}/{2}".format(overall_accuracy, total_correct, total))
        # Determine modes
        cumulative_prob = 0.0
        modes = set()
        sorted_distribution = sorted(label_distribution_dict.items(), key=lambda tpl: tpl[1], reverse=True)
        for tpl in sorted_distribution:
            if cumulative_prob < threshold_percentile_for_modes:
                modes.add(tpl[0])
                cumulative_prob += tpl[1]
        # The accuracy on samples, which are predicted as modes.
        total_prediction_frequencies = np.sum(cm, axis=0)
        total_mode_prediction_count = np.sum(total_prediction_frequencies[list(modes)])
        total_modes_count = np.sum(label_counts[list(modes)])
        correct_modes = 0
        for mode in modes:
            correct_modes += cm[mode, mode]
        modes_accuracy = correct_modes / total_modes_count
        print("Modes (Dominant Labels):{0}".format(modes))
        print("Modes accuracy:{0} {1}/{2}".format(modes_accuracy, correct_modes, total_modes_count))
        print("Accuracy on samples which are predicted as modes:{0}".format(correct_modes / total_mode_prediction_count))
        # Determine non-modes
        total_non_modes_count = total - total_modes_count
        correct_non_modes = 0
        for i in range(label_distribution.shape[0]):
            if i in modes:
                continue
            correct_non_modes += cm[i, i]
        non_modes_accuracy = correct_non_modes / total_non_modes_count
        print("Non Modes accuracy:{0} {1}/{2}".format(non_modes_accuracy, correct_non_modes, total_non_modes_count))
        res_dict = {}
        res_dict["total_modes_count"] = total_modes_count
        res_dict["correct_modes"] = correct_modes
        res_dict["total_non_modes_count"] = total_non_modes_count
        res_dict["correct_non_modes"] = correct_non_modes
        res_dict["total_mode_prediction_count"] = total_mode_prediction_count
        return res_dict

run_id = 923
iteration = 40000
print("\nLeaf 3")
cm3 = DbLogger.read_confusion_matrix(run_id=run_id, dataset=2, iteration=iteration, num_of_labels=10, leaf_id=3)
res_dict_3 = ConfusionMatrixAnalyzer.analyze_confusion_matrix(cm=cm3, threshold_percentile_for_modes=0.85)

print("\nLeaf 4")
cm4 = DbLogger.read_confusion_matrix(run_id=run_id, dataset=2, iteration=iteration, num_of_labels=10, leaf_id=4)
res_dict_4 = ConfusionMatrixAnalyzer.analyze_confusion_matrix(cm=cm4, threshold_percentile_for_modes=0.85)

print("\nLeaf 5")
cm5 = DbLogger.read_confusion_matrix(run_id=run_id, dataset=2, iteration=iteration, num_of_labels=10, leaf_id=5)
res_dict_5 = ConfusionMatrixAnalyzer.analyze_confusion_matrix(cm=cm5, threshold_percentile_for_modes=0.85)

print("\nLeaf 6")
cm6 = DbLogger.read_confusion_matrix(run_id=run_id, dataset=2, iteration=iteration, num_of_labels=10, leaf_id=6)
res_dict_6 = ConfusionMatrixAnalyzer.analyze_confusion_matrix(cm=cm6, threshold_percentile_for_modes=0.85)

overall_modes_count = res_dict_3["total_modes_count"] + res_dict_4["total_modes_count"] + \
                      res_dict_5["total_modes_count"] + res_dict_6["total_modes_count"]
overall_correct_modes_count = res_dict_3["correct_modes"] + res_dict_4["correct_modes"] + \
                              res_dict_5["correct_modes"] + res_dict_6["correct_modes"]
overall_non_modes_count = res_dict_3["total_non_modes_count"] + res_dict_4["total_non_modes_count"] + \
                          res_dict_5["total_non_modes_count"] + res_dict_6["total_non_modes_count"]
overall_correct_non_modes_count = res_dict_3["correct_non_modes"] + res_dict_4["correct_non_modes"] + \
                                  res_dict_5["correct_non_modes"] + res_dict_6["correct_non_modes"]
overall_mode_prediction_count = res_dict_3["total_mode_prediction_count"] + res_dict_4["total_mode_prediction_count"] + \
                                  res_dict_5["total_mode_prediction_count"] + res_dict_6["total_mode_prediction_count"]

print("\nOverall Mode Labels Accuracy={0} Mode Count={1}".format(overall_correct_modes_count / overall_modes_count,
                                                                 overall_modes_count))
print("Overall Non Mode Labels Accuracy={0} Non Mode Count={1}".
      format(overall_correct_non_modes_count / overall_non_modes_count,
             overall_non_modes_count))
print("Overall Tree Accuracy={0}".format((overall_correct_modes_count + overall_correct_non_modes_count) /
                                         (overall_modes_count + overall_non_modes_count)))
print("Overall Count of Samples Predicted As Modes:{0}".format(overall_mode_prediction_count))
estimated_tree_accuracy = overall_correct_modes_count / overall_mode_prediction_count
print("Overall Accuracy On Samples Predicted As Modes:{0}".format(estimated_tree_accuracy))
estimated_final_network_accuracy = 0.99*(10000.0 - overall_mode_prediction_count) + \
                                   estimated_tree_accuracy*overall_mode_prediction_count

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
ConfusionMatrixGrapher.plot_confusion_matrix(cm=cm3, classes=classes, title="Leaf 3 Label Frequencies")
ConfusionMatrixGrapher.plot_confusion_matrix(cm=cm4, classes=classes, title="Leaf 4 Label Frequencies")
ConfusionMatrixGrapher.plot_confusion_matrix(cm=cm5, classes=classes, title="Leaf 5 Label Frequencies")
ConfusionMatrixGrapher.plot_confusion_matrix(cm=cm6, classes=classes, title="Leaf 6 Label Frequencies")
print("X")
