import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.routing_weights_deep_regressor import RoutingWeightDeepRegressor
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants


class RoutingWeightDeepSoftmaxRegressor(RoutingWeightDeepRegressor):
    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data, layers,
                 l2_lambda, batch_size, max_iteration):
        super().__init__(network, validation_routing_matrix, test_routing_matrix, validation_data, test_data, layers,
                         l2_lambda, batch_size, max_iteration, leaf_index=None)
        posterior_dim = validation_data.get_dict("posterior_probs")[0].shape[1]
        leaf_count = len(self.leafNodes)
        self.input_posteriors = tf.placeholder(dtype=tf.float32, shape=[leaf_count, posterior_dim],
                                               name='input_posteriors')
