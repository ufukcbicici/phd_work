from algorithms.threshold_optimization_algorithms.routing_weights_deep_softmax_regressor import \
    RoutingWeightDeepSoftmaxRegressor
import tensorflow as tf
import numpy as np

from auxillary.db_logger import DbLogger


class RoutingWeightSparseMoERegressor(RoutingWeightDeepSoftmaxRegressor):
    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data, layers,
                 l2_lambda, batch_size, max_iteration, use_multi_path_only=False):
        super().__init__(network, validation_routing_matrix, test_routing_matrix, validation_data, test_data, layers,
                         l2_lambda, batch_size, max_iteration, use_multi_path_only)
        self.logits = None
        self.classCount = self.fullDataDict["validation"].y_one_hot.shape[1]
        self.ceVector = None
        self.ceLoss = None
        self.finalPosterior = None
        self.labelsVector = tf.placeholder(dtype=tf.int64, shape=[None, ], name='labelsVector')
        self.runId = None
        self.iteration = 0