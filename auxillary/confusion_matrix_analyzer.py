import numpy as np

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
        total_modes_count = np.sum(label_counts[list(modes)])
        correct_modes = 0
        for mode in modes:
            correct_modes += cm[mode, mode]
        modes_accuracy = correct_modes / total_modes_count
        print("Modes accuracy:{0} {1}/{2}".format(modes_accuracy, correct_modes, total_modes_count))
        # Determine non-modes
        total_non_modes_count = total - total_modes_count
        correct_non_modes = 0
        for i in range(label_distribution.shape[0]):
            if i in modes:
                continue
            correct_non_modes += cm[i, i]
        non_modes_accuracy = correct_non_modes / total_non_modes_count
        print("Non Modes accuracy:{0} {1}/{2}".format(non_modes_accuracy, correct_non_modes, total_non_modes_count))



print("\nLeaf 3")
cm3 = DbLogger.read_confusion_matrix(run_id=1003, dataset=2, iteration=60000, num_of_labels=10, leaf_id=3)
ConfusionMatrixAnalyzer.analyze_confusion_matrix(cm=cm3, threshold_percentile_for_modes=0.85)

print("\nLeaf 4")
cm4 = DbLogger.read_confusion_matrix(run_id=1003, dataset=2, iteration=60000, num_of_labels=10, leaf_id=4)
ConfusionMatrixAnalyzer.analyze_confusion_matrix(cm=cm4, threshold_percentile_for_modes=0.85)

print("\nLeaf 5")
cm5 = DbLogger.read_confusion_matrix(run_id=1003, dataset=2, iteration=60000, num_of_labels=10, leaf_id=5)
ConfusionMatrixAnalyzer.analyze_confusion_matrix(cm=cm5, threshold_percentile_for_modes=0.85)

print("\nLeaf 6")
cm6 = DbLogger.read_confusion_matrix(run_id=1003, dataset=2, iteration=60000, num_of_labels=10, leaf_id=6)
ConfusionMatrixAnalyzer.analyze_confusion_matrix(cm=cm6, threshold_percentile_for_modes=0.85)

print("X")