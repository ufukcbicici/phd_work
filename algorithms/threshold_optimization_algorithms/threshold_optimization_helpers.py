import numpy as np
from sklearn.model_selection import train_test_split


class BranchingInfo:
    def __init__(self, branching_probs, routing_matrix, path_probs):
        self.branchingProbs = branching_probs
        self.routingMatrix = routing_matrix
        self.pathProbabilities = path_probs


class MultipathResult:
    def __init__(self, result_tuple):
        self.methodType = result_tuple[0]
        self.thresholdsDict = result_tuple[1]
        self.accuracy = result_tuple[2]
        self.totalLeavesEvaluated = result_tuple[3]
        self.computationOverload = result_tuple[4]
        # res_method_0 = (0, thresholds_dict, accuracy_simple_avg, total_leaves_evaluated, computation_overload)
        # res_method_1 = (1, thresholds_dict, accuracy_weighted_avg, total_leaves_evaluated, computation_overload)


class RoutingDataset:
    def __init__(self, labels_dict_for_leaves, label_list, branch_probs, activations, posterior_probs):
        self.labelsDict = labels_dict_for_leaves
        self.labelList = label_list
        self.branchProbs = branch_probs
        self.activations = activations
        self.posteriorProbs = posterior_probs

    def apply_validation_test_split(self, test_ratio):
        indices = np.array(range(self.labelList.shape[0]))
        val_indices, test_indices = train_test_split(indices, test_size=test_ratio)
        split_sets = []

        def get_subset_of_dict(data_dict, _i):
            subset_dict = {}
            for node_id, arr in data_dict.items():
                new_arr = np.copy(arr[_i])
                subset_dict[node_id] = new_arr
            return subset_dict

        for idx in [val_indices, test_indices]:
            labels_dict = get_subset_of_dict(data_dict=self.labelsDict, _i=idx)
            labels_list = np.copy(self.labelList[idx])
            branch_probs = get_subset_of_dict(data_dict=self.branchProbs, _i=idx)
            activations = get_subset_of_dict(data_dict=self.activations, _i=idx)
            posterior_probs = get_subset_of_dict(data_dict=self.posteriorProbs, _i=idx)
            routing_data = RoutingDataset(labels_dict_for_leaves=labels_dict, label_list=labels_list,
                                          branch_probs=branch_probs, activations=activations,
                                          posterior_probs=posterior_probs)
            split_sets.append(routing_data)

        val_data = split_sets[0]
        test_data = split_sets[1]
        return val_data, test_data




