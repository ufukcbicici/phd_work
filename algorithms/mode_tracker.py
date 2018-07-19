import numpy as np

from auxillary.constants import DatasetTypes
from simple_tf.global_params import GlobalConstants, ModeTrackingStrategy


class ModeTracker:
    def __init__(self, network):
        self.network = network
        self.unchangedEpochCount = 0
        self.modesHistory = []
        self.isCompressed = False

    def reset(self):
        self.unchangedEpochCount = 0
        self.modesHistory = []
        self.isCompressed = False

    def get_modes(self):
        return self.modesHistory[-1]

    def modes_changed(self, prev_modes, curr_modes):
        for node in self.network.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            if prev_modes[node.index] != curr_modes[node.index]:
                return True
        return False

    def calculate_modes(self, leaf_true_labels_dict, dataset, dataset_type, kv_rows, run_id, iteration):
        # Measure overall label distribution in leaves, get modes
        total_mode_count = 0
        modes_per_leaves = {}
        for node in self.network.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            if node.index not in leaf_true_labels_dict:
                continue
            true_labels = leaf_true_labels_dict[node.index]
            frequencies = {}
            label_distribution = {}
            distribution_str = ""
            total_sample_count = true_labels.shape[0]
            for label in range(dataset.get_label_count()):
                frequencies[label] = np.sum(true_labels == label)
                label_distribution[label] = frequencies[label] / float(total_sample_count)
                distribution_str += "{0}:{1:.3f} ".format(label, label_distribution[label])
            # Get modes
            if dataset_type == DatasetTypes.training:
                cumulative_prob = 0.0
                sorted_distribution = sorted(label_distribution.items(), key=lambda lbl: lbl[1], reverse=True)
                modes_per_leaves[node.index] = set()
                mode_txt = ""
                for tpl in sorted_distribution:
                    if cumulative_prob < GlobalConstants.PERCENTILE_THRESHOLD:
                        modes_per_leaves[node.index].add(tpl[0])
                        mode_txt += str(tpl[0]) + ","
                        cumulative_prob += tpl[1]
                mode_txt = mode_txt[0:len(mode_txt) - 1]
                total_mode_count += len(modes_per_leaves[node.index])
                kv_rows.append((run_id, iteration, "Leaf {0} Modes".format(node.index), mode_txt))
            print("Node{0} Label Distribution: {1}".format(node.index, distribution_str))
        # if dataset_type == DatasetTypes.training and total_mode_count != GlobalConstants.NUM_LABELS:
        #     raise Exception("total_mode_count != GlobalConstants.NUM_LABELS")
        # Measure overall information gain
        if dataset_type == DatasetTypes.training:
            kv_rows.append((run_id, iteration, "Total Mode Count", total_mode_count))
            self.modesHistory.append(modes_per_leaves)

    def check_for_compression_start(self, dataset, epoch):
        if GlobalConstants.MODE_TRACKING_STRATEGY == ModeTrackingStrategy.wait_for_convergence:
            label_count = dataset.get_label_count()
            total_mode_count = 0
            # for modes in last_modes.values():
            #     total_mode_count += len(modes)
            if len(self.modesHistory) == 0:
                raise Exception("Mode history can't be zero.")
            elif len(self.modesHistory) == 1:
                self.unchangedEpochCount = 1
            else:
                prev_modes = self.modesHistory[-2]
                curr_modes = self.modesHistory[-1]
                if self.modes_changed(prev_modes=prev_modes, curr_modes=curr_modes):
                    self.unchangedEpochCount = 1
                else:
                    self.unchangedEpochCount += 1
            if self.unchangedEpochCount != GlobalConstants.MODE_WAIT_EPOCHS:
                return False
            if GlobalConstants.CONSTRAIN_WITH_COMPRESSION_LABEL_COUNT:
                curr_modes = self.modesHistory[-1]
                for v in curr_modes.values():
                    total_mode_count += len(v)
                if total_mode_count != label_count:
                    return False
            if self.isCompressed:
                return False
            self.isCompressed = True
            return True
        elif GlobalConstants.MODE_TRACKING_STRATEGY == ModeTrackingStrategy.wait_for_fixed_epochs:
            if epoch == GlobalConstants.MODE_WAIT_EPOCHS:
                self.isCompressed = True
                return True
            else:
                return False





