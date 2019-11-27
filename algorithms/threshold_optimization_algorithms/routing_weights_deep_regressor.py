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
    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data,
                 layers, l2_lambda):
        super().__init__(network, validation_routing_matrix, test_routing_matrix, validation_data, test_data)
        self.routingCombinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(self.leafNodes))
        self.routingCombinations = [np.array(route_vec) for route_vec in self.routingCombinations]
        self.modelsDict = {}
        self.validation_X, self.validation_Y, self.test_X, self.test_Y = \
            self.build_data_sets(selected_features=GlobalConstants.SELECTED_FEATURES_FOR_WEIGHT_REGRESSION)
        # Network entry points
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.validation_X.shape[1]], name='input_x')
        self.input_t = tf.placeholder(dtype=tf.float32, shape=[None, self.validation_Y.shape[1]], name='input_t')
        self.predicted_t = None
        self.layers = layers
        self.regressionMeanSquaredError = None
        self.l2Loss = None
        self.totalLoss = None
        self.l2Lambda = l2_lambda
        self.paramL2Norms = {}
        self.globalStep = None
        self.optimizer = None

    def build_network(self):
        x = self.input_x
        for layer_id, hidden_dim in enumerate(self.layers):
            with tf.variable_scope('layer_{0}'.format(layer_id)):
                net_shape = x.get_shape().as_list()
                fc_w = tf.get_variable('fc_w', shape=[net_shape[-1], hidden_dim], dtype=tf.float32)
                fc_b = tf.get_variable('fc_b', shape=[hidden_dim], dtype=tf.float32)
                x = tf.matmul(x, fc_w) + fc_b
                x = tf.nn.relu(x)
        # Output layer
        with tf.variable_scope('output_layer'):
            output_dim = self.validation_Y.shape[1]
            net_shape = x.get_shape().as_list()
            fc_w = tf.get_variable('fc_w', shape=[net_shape[-1], output_dim], dtype=tf.float32)
            fc_b = tf.get_variable('fc_b', shape=[output_dim], dtype=tf.float32)
            self.predicted_t = tf.matmul(x, fc_w) + fc_b

    def build_loss(self):
        with tf.name_scope("loss"):
            # MSE Loss
            squared_diff = tf.squared_difference(self.predicted_t, self.input_t)
            sample_wise_sum = tf.reduce_sum(squared_diff, axis=1)
            self.regressionMeanSquaredError = tf.reduce_mean(sample_wise_sum)
            # L2 Loss
            tvars = tf.trainable_variables()
            self.l2Loss = tf.constant(0.0)
            for tv in tvars:
                if 'fc_w' in tv.name:
                    self.l2Loss += self.l2Lambda * tf.nn.l2_loss(tv)
                self.paramL2Norms[tv.name] = tf.nn.l2_loss(tv)
            self.totalLoss = self.regressionMeanSquaredError + self.l2Loss

    def build_optimizer(self):
        with tf.variable_scope("optimizer"):
            self.globalStep = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.totalLoss, global_step=self.globalStep)

    def run(self):
        print("X")
