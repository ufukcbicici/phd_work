import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


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
        self.routingMatrix = result_tuple[5]
        self.posteriors = result_tuple[6]
        # res_method_0 = (0, thresholds_dict, accuracy_simple_avg, total_leaves_evaluated, computation_overload)
        # res_method_1 = (1, thresholds_dict, accuracy_weighted_avg, total_leaves_evaluated, computation_overload)


class RoutingDataset:
    def __init__(self, label_list, dict_of_data_dicts):
        self.labelList = label_list
        self.dictionaryOfRoutingData = dict_of_data_dicts

    def __eq__(self, other):
        if not np.array_equal(self.labelList, other.labelList):
            return False
        if not len(self.dictionaryOfRoutingData) == len(self.dictionaryOfRoutingData):
            return False
        for k, dict_arr in self.dictionaryOfRoutingData.items():
            if k not in other.dictionaryOfRoutingData:
                return False
            other_dict_arr = other.dictionaryOfRoutingData[k]
            if not len(dict_arr) == len(other_dict_arr):
                return False
            for _k, _v in dict_arr.items():
                if _k not in other_dict_arr:
                    return False
                if not np.array_equal(_v, other_dict_arr[_k]):
                    return False
        return True

    def obtain_splits(self, k_fold):
        kf = KFold(n_splits=k_fold, shuffle=True)
        training_index_list = []
        test_index_list = []
        indices = np.array(range(self.labelList.shape[0]))
        for train_indices, test_indices in kf.split(indices):
            training_index_list.append(train_indices)
            test_index_list.append(test_indices)
        return training_index_list, test_index_list

    def split_dataset_with_indices(self, training_indices, test_indices):
        def get_subset_of_dict(data_dict_, _i):
            subset_dict_ = {}
            for node_id, arr in data_dict_.items():
                new_arr = np.copy(arr[_i])
                subset_dict_[node_id] = new_arr
            return subset_dict_

        split_sets = []
        for idx in [training_indices, test_indices]:
            split_labels_list = np.copy(self.labelList[idx])
            dict_of_split_data = {}
            for data_name, data_dict in self.dictionaryOfRoutingData.items():
                if "Costs" in data_name:
                    dict_of_split_data[data_name] = data_dict
                    continue
                assert data_dict.__class__ == dict().__class__
                subset_dict = get_subset_of_dict(data_dict_=data_dict, _i=idx)
                dict_of_split_data[data_name] = subset_dict
            routing_data = RoutingDataset(label_list=split_labels_list, dict_of_data_dicts=dict_of_split_data)
            split_sets.append(routing_data)
        val_data = split_sets[0]
        test_data = split_sets[1]
        return val_data, test_data

    def apply_validation_test_split(self, test_ratio):
        indices = np.array(range(self.labelList.shape[0]))
        val_indices, test_indices = train_test_split(indices, test_size=test_ratio)
        val_data, test_data = self.split_dataset_with_indices(training_indices=val_indices, test_indices=test_indices)
        return val_data, test_data

    def get_dict(self, output_name):
        if output_name not in self.dictionaryOfRoutingData:
            return None
        return self.dictionaryOfRoutingData[output_name]
