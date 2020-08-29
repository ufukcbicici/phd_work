import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from auxillary.general_utility_funcs import UtilityFuncs


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
    def __init__(self, label_list, dict_of_data_dicts, index_multiplier=1):
        self.labelList = label_list
        self.dictionaryOfRoutingData = dict_of_data_dicts
        self.indexMultiplier = index_multiplier
        # Assert the integrity of augmented samples
        for idx in range(0, self.labelList.shape[0], self.indexMultiplier):
            sub_list = self.labelList[idx:idx + self.indexMultiplier].tolist()
            assert len(set(sub_list)) == 1

    # OK for data augmentation
    def __eq__(self, other):
        if not np.array_equal(self.labelList, other.labelList):
            return False
        if not len(self.dictionaryOfRoutingData) == len(self.dictionaryOfRoutingData):
            return False
        if self.indexMultiplier != other.indexMultiplier:
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

    # OK for data augmentation
    def obtain_splits(self, k_fold):
        kf = KFold(n_splits=k_fold, shuffle=True)
        training_index_list = []
        test_index_list = []
        indices = np.array(range(int(self.labelList.shape[0] / self.indexMultiplier)))
        for train_indices, test_indices in kf.split(indices):
            training_index_list.append(train_indices)
            test_index_list.append(test_indices)
        return training_index_list, test_index_list

    # OK for data augmentation
    def split_dataset_with_indices(self, training_indices, test_indices):
        assert self.indexMultiplier * (len(training_indices) + len(test_indices)) == self.labelList.shape[0]

        def get_subset_of_dict(data_dict_, _i):
            subset_dict_ = {}
            for node_id, arr in data_dict_.items():
                new_arr = np.copy(arr[_i])
                subset_dict_[node_id] = new_arr
            return subset_dict_

        expanded_training_indices = UtilityFuncs.expand_index_list(indices=training_indices,
                                                                   index_multiplier=self.indexMultiplier)
        expanded_test_indices = UtilityFuncs.expand_index_list(indices=test_indices,
                                                               index_multiplier=self.indexMultiplier)
        if self.indexMultiplier == 1:
            assert np.array_equal(training_indices, expanded_test_indices)
            assert np.array_equal(test_indices, expanded_test_indices)

        split_sets = []
        for idx in [expanded_training_indices, expanded_test_indices]:
            split_labels_list = np.copy(self.labelList[idx])
            dict_of_split_data = {}
            for data_name, data_dict in self.dictionaryOfRoutingData.items():
                if "Costs" in data_name:
                    dict_of_split_data[data_name] = data_dict
                    continue
                assert data_dict.__class__ == dict().__class__
                print(data_dict.keys())
                print(data_dict[0].shape)
                print(data_dict[1].shape)
                print(data_dict[2].shape)
                subset_dict = get_subset_of_dict(data_dict_=data_dict, _i=idx)
                dict_of_split_data[data_name] = subset_dict
            routing_data = RoutingDataset(label_list=split_labels_list, dict_of_data_dicts=dict_of_split_data,
                                          index_multiplier=self.indexMultiplier)
            split_sets.append(routing_data)
        val_data = split_sets[0]
        test_data = split_sets[1]
        return val_data, test_data

    # OK for data augmentation
    def apply_validation_test_split(self, test_ratio):
        indices = np.array(range(int(self.labelList.shape[0] / self.indexMultiplier)))
        val_indices, test_indices = train_test_split(indices, test_size=test_ratio)
        val_data, test_data = self.split_dataset_with_indices(training_indices=val_indices, test_indices=test_indices)
        return val_data, test_data

    # OK for data augmentation
    def get_dict(self, output_name):
        if output_name not in self.dictionaryOfRoutingData:
            return None
        return self.dictionaryOfRoutingData[output_name]


class MultiIterationRoutingDataset(RoutingDataset):
    def __init__(self, dict_of_routing_datasets, sample_linkage_info, test_iterations):
        self.iterations = sorted(list(dict_of_routing_datasets.keys()))
        self.dictOfDatasets = dict_of_routing_datasets
        # Node costs
        assert len(set([len(d.dictionaryOfRoutingData["nodeCosts"]) for d in self.dictOfDatasets.values()])) == 1
        for idx in range(len(self.iterations) - 1):
            iteration_t = self.iterations[idx]
            iteration_t_plus_1 = self.iterations[idx + 1]
            assert set(self.dictOfDatasets[iteration_t].dictionaryOfRoutingData["nodeCosts"].keys()) == \
                   set(self.dictOfDatasets[iteration_t_plus_1].dictionaryOfRoutingData["nodeCosts"].keys())
            assert all([self.dictOfDatasets[iteration_t].dictionaryOfRoutingData["nodeCosts"][k] ==
                    self.dictOfDatasets[iteration_t_plus_1].dictionaryOfRoutingData["nodeCosts"][k]
                    for k in self.dictOfDatasets[iteration_t].dictionaryOfRoutingData["nodeCosts"].keys()])
        self.nodeCosts = self.dictOfDatasets[self.iterations[0]].dictionaryOfRoutingData["nodeCosts"]
        # Linkage Information
        self.linkageInfo = {}
        for tpl in sample_linkage_info:
            # SampleId, Iteration, SampleIdForIteration, COUNT(1)
            sample_id = tpl[0]
            iteration = tpl[1]
            sample_id_for_iteration = tpl[2]
            self.linkageInfo[(sample_id, iteration)] = sample_id_for_iteration
        self.testIterations = test_iterations
        self.trainingIndices = None
        self.testIndices = None
        super().__init__(self.dictOfDatasets[self.iterations[0]].labelList,
                         self.dictOfDatasets[self.iterations[0]].dictionaryOfRoutingData)

    def split_dataset_with_indices(self, training_indices, test_indices):
        pass

    def apply_validation_test_split(self, test_ratio):
        indices = np.array(range(int(self.labelList.shape[0])))
        self.trainingIndices, self.testIndices = train_test_split(indices, test_size=test_ratio)

    def get_dict_of_iteration(self, output_name, iteration):
        if output_name not in self.dictOfDatasets[iteration].dictionaryOfRoutingData:
            return None
        return self.dictOfDatasets[iteration].dictionaryOfRoutingData[output_name]
