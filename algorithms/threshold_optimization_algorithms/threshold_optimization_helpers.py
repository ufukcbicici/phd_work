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


# RoutingDataset
class RoutingDatasetMultiIteration(RoutingDataset):
    def __init__(self, iterations, data_dict, sample_link_map):
        # self.iterations = iterations
        # self.minIteration = np.min(iterations)
        # self.dataDict = data_dict
        # self.sampleLinkMap = sample_link_map
        # Merged Label List
        super().__init__([], {})
        # 000000 = {tuple} <
        #
        # class 'tuple'>: (0, 43680, 0, 3)
        #
        # 000001 = {tuple} <
        #
        # class 'tuple'>: (0, 44160, 7087, 3)
        #
        # 000002 = {tuple} <
        #
        # class 'tuple'>: (0, 44640, 1645, 3)
        #
        # 000003 = {tuple} <
        #
        # class 'tuple'>: (0, 45120, 5743, 3)
        #
        # 000004 = {tuple} <
        #
        # class 'tuple'>: (0, 45600, 3109, 3)
        #
        # 000005 = {tuple} <
        #
        # class 'tuple'>: (0, 46080, 8697, 3)
        #
        # 000006 = {tuple} <
        #
        # class 'tuple'>: (0, 46560, 8593, 3)
        #
        # 000007 = {tuple} <
        #
        # class 'tuple'>: (0, 47040, 6166, 3)
        #
        # 00000
        # 8 = {tuple} <
        #
        # class 'tuple'>: (0, 47520, 6858, 3)
        #
        # 00000
        # 9 = {tuple} <
        #
        # class 'tuple'>: (0, 48000, 9256, 3)
        #
        # 000010 = {tuple} <
        #
        # class 'tuple'>: (1, 43680, 1, 3)
        #
        # 000011 = {tuple} <
        #
        # class 'tuple'>: (1, 44160, 9661, 3)
        #
        # 000012 = {tuple} <
        #
        # class 'tuple'>: (1, 44640, 9558, 3)
        #
        # 000013 = {tuple} <
        #
        # class 'tuple'>: (1, 45120, 7724, 3)
        #
        # 000014 = {tuple} <
        #
        # class 'tuple'>: (1, 45600, 2563, 3)
        #
        # 000015 = {tuple} <
        #
        # class 'tuple'>: (1, 46080, 1660, 3)
        #
        # 000016 = {tuple} <
        #
        # class 'tuple'>: (1, 46560, 3385, 3)
        #
        # 000017 = {tuple} <
        #
        # class 'tuple'>: (1, 47040, 9452, 3)
        #
        # 00001
        # 8 = {tuple} <
        #
        # class 'tuple'>: (1, 47520, 2974, 3)
        #
        # 00001
        # 9 = {tuple} <
        #
        # class 'tuple'>: (1, 48000, 6420, 3)
        #
        # 000020 = {tuple} <
        #
        # class 'tuple'>: (2, 43680, 2, 3)
        for tpl in sample_link_map:
            sample_id = tpl[0]
            iteration = tpl[1]
            sample_id_in_iteration = tpl[2]
            self.labelList.append(data_dict[iteration].labelList[sample_id_in_iteration])
            for k in data_dict[iteration].keys():
                if k not in self.dictionaryOfRoutingData:
                    self.dictionaryOfRoutingData[k] = []
                self.dictionaryOfRoutingData[k].append(data_dict[iteration][k][sample_id_in_iteration])




        # self.primaryRoutingDataset = data_dict[self.minIteration]
        # super().__init__(self.primaryRoutingDataset.labelList, self.primaryRoutingDataset.dictionaryOfRoutingData)

