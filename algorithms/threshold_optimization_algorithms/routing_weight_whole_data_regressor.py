import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from algorithms.threshold_optimization_algorithms.routing_weight_calculator import RoutingWeightCalculator
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants


class RoutingWeightWholeDataRegressor(RoutingWeightCalculator):
    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data):
        super().__init__(network, validation_routing_matrix, test_routing_matrix, validation_data, test_data)
        self.routingCombinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(self.leafNodes))
        self.routingCombinations = [np.array(route_vec) for route_vec in self.routingCombinations]
        self.modelsDict = {}

    def run(self):
        # Create Feature Sets from Activation and Posterior Vectors
        # features_dict = {}
        # targets_dict = {}
        # single_path_correct_counts_dict = {}
        # multi_path_indices_dict = {}
        # posterior_tensors_dict = {}
        # activation_tensors_dict = {}
        # data_objects_dict = {"validation": self.validationData, "test": self.testData}
        # routing_matrices_dict = {"validation": self.validationRoutingMatrix, "test": self.testRoutingMatrix}
        all_feature_names = list(GlobalConstants.INNER_NODE_OUTPUTS_TO_COLLECT)
        all_feature_names.extend(GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT)
        formatted_data_validation = RoutingWeightCalculator.format_routing_data(routing_data=self.validationData)
        formatted_data_test = RoutingWeightCalculator.format_routing_data(routing_data=self.testData)
        features_dict = {"validation": formatted_data_validation, "test": formatted_data_test}
        for route_vector in self.routingCombinations:
            if np.sum(route_vector) == 1:
                continue
            for data, data_type in zip([self.validationData, self.testData], ["validation", "test"]):
                posteriors_tensor = features_dict[data_type]["posterior_probs"]
                activations_tensor = features_dict[data_type]["activations"]
                posteriors_tensor_sparse = posteriors_tensor * np.expand_dims(
                    np.expand_dims(route_vector, axis=0), axis=0)
                x_dict = {feature_name: [] for feature_name in all_feature_names}
                y_list = []
                for idx in range(posteriors_tensor.shape[0]):
                    sparse_posteriors_idx = posteriors_tensor_sparse[idx, :, :]
                    true_label = data.labelList[idx]
                    label_vector = RoutingWeightCalculator.get_one_hot_label_vector(label=true_label,
                                                                                    dim=sparse_posteriors_idx.shape[0])
                    res = np.linalg.lstsq(sparse_posteriors_idx, label_vector, rcond=None)
                    alpha_lst_squares = res[0]
                    y_list.append(alpha_lst_squares)

