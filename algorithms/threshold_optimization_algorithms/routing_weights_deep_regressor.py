import numpy as np

import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from algorithms.threshold_optimization_algorithms.routing_weight_calculator import RoutingWeightCalculator
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants

feature_names = ["posterior_probs"]


class RoutingWeightDeepRegressor(RoutingWeightCalculator):
    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data):
        super().__init__(network, validation_routing_matrix, test_routing_matrix, validation_data, test_data)
        self.routingCombinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(self.leafNodes))
        self.routingCombinations = [np.array(route_vec) for route_vec in self.routingCombinations]
        self.modelsDict = {}
        self.validation_X, self.validation_Y, self.test_X, self.test_Y = \
            self.build_data_sets(selected_features=GlobalConstants.SELECTED_FEATURES_FOR_WEIGHT_REGRESSION)

    def run(self):
        print("X")