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
        print("Overall accuracy:{0}".format(overall_accuracy))
        # Determine modes
        cumulative_prob = 0.0
        modes = []
        sorted_distribution = sorted(label_distribution_dict.items(), key=lambda tpl: tpl[1], reverse=True)
        for tpl in sorted_distribution:
            if cumulative_prob < threshold_percentile_for_modes:
                modes.append(tpl[0])
                cumulative_prob += tpl[1]
        total_modes = np.sum(label_counts[modes])
        correct_modes = 0
        for mode in modes:
            correct_modes += cm[mode, mode]
        modes_accuracy = correct_modes / total_modes
        print("Modes accuracy:{0}".format(modes_accuracy))


cm3 = DbLogger.read_confusion_matrix(run_id=797, dataset=1, iteration=60000, num_of_labels=10, leaf_id=3)
ConfusionMatrixAnalyzer.analyze_confusion_matrix(cm=cm3, threshold_percentile_for_modes=0.85)
cm4 = DbLogger.read_confusion_matrix(run_id=797, dataset=1, iteration=60000, num_of_labels=10, leaf_id=4)
ConfusionMatrixAnalyzer.analyze_confusion_matrix(cm=cm4, threshold_percentile_for_modes=0.85)
cm5 = DbLogger.read_confusion_matrix(run_id=797, dataset=1, iteration=60000, num_of_labels=10, leaf_id=5)
ConfusionMatrixAnalyzer.analyze_confusion_matrix(cm=cm5, threshold_percentile_for_modes=0.85)
cm6 = DbLogger.read_confusion_matrix(run_id=797, dataset=1, iteration=60000, num_of_labels=10, leaf_id=6)
ConfusionMatrixAnalyzer.analyze_confusion_matrix(cm=cm6, threshold_percentile_for_modes=0.85)