import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from algorithms.bayesian_optimization import BayesianOptimizer
from algorithms.threshold_optimization_algorithms.routing_weight_calculator import RoutingWeightCalculator
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants

feature_names = ["posterior_probs"]


class RoutingWeightBayesianOptimizer(RoutingWeightCalculator):
    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data,
                 layers, l2_lambda, batch_size, max_iteration, leaf_index=None):
        super().__init__(network, validation_routing_matrix, test_routing_matrix, validation_data, test_data)
        self.routingCombinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(self.leafNodes))
        self.routingCombinations = [np.array(route_vec) for route_vec in self.routingCombinations]
        self.optimalWeightsDict = {}

    def run(self):
        for route_vec in self.routingCombinations:
            route_vector = np.array(route_vec)
            if np.sum(route_vector) <= 1:
                continue
            route_vector_as_tuple = tuple(route_vector.tolist())
            element_wise_compliance = self.validationRoutingMatrix == route_vector
            valid_samples_indicator_vector = np.all(element_wise_compliance, axis=1)
            if np.sum(valid_samples_indicator_vector) == 0:
                self.optimalWeightsDict[route_vector_as_tuple] = (1.0 / np.sum(route_vector)) * route_vector
                continue
