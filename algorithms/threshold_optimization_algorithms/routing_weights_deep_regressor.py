import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from algorithms.threshold_optimization_algorithms.routing_weight_calculator import RoutingWeightCalculator
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants

feature_names = ["posterior_probs"]


class RoutingWeightDeepRegressor(RoutingWeightCalculator):
    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data,
                 layers, l2_lambda, batch_size, max_iteration, leaf_index=None):
        super().__init__(network, validation_routing_matrix, test_routing_matrix, validation_data, test_data)
        self.routingCombinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(self.leafNodes))
        self.routingCombinations = [np.array(route_vec) for route_vec in self.routingCombinations]
        self.modelsDict = {}
        self.validation_X, self.validation_Y, self.test_X, self.test_Y = \
            self.build_data_sets(selected_features=GlobalConstants.SELECTED_FEATURES_FOR_WEIGHT_REGRESSION)
        # if leaf_index is not None:
        #     self.validation_Y = np.expand_dims(self.validation_Y[:, leaf_index], axis=1)
        #     self.test_Y = np.expand_dims(self.test_Y[:, leaf_index], axis=1)
        #     z_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        #     z_scaler.fit(self.validation_Y)
        #     self.validation_Y = z_scaler.transform(self.validation_Y)
        #     self.test_Y = z_scaler.transform(self.test_Y)
        #     print("X")
        # Network entry points
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.validation_X.shape[1]], name='input_x')
        self.input_t = tf.placeholder(dtype=tf.float32, shape=[None, self.validation_Y.shape[1]], name='input_t')
        self.networkOutput = None
        self.predicted_t = None
        self.layers = layers
        self.sampleWiseErrors = None
        self.regressionMeanSquaredError = None
        self.l2Loss = None
        self.totalLoss = None
        self.l2Lambda = l2_lambda
        self.batchSize = batch_size
        self.maxIteration = max_iteration
        self.paramL2Norms = {}
        self.globalStep = None
        self.optimizer = None
        self.learningRate = None
        self.sess = tf.Session()

    def build_network(self):
        x = self.input_x
        for layer_id, hidden_dim in enumerate(self.layers):
            with tf.variable_scope('layer_{0}'.format(layer_id)):
                net_shape = x.get_shape().as_list()
                fc_w = tf.get_variable('fc_w', shape=[net_shape[-1], hidden_dim], dtype=tf.float32)
                fc_b = tf.get_variable('fc_b', shape=[hidden_dim], dtype=tf.float32)
                x = tf.matmul(x, fc_w) + fc_b
                x = tf.nn.leaky_relu(x)
        # Output layer
        with tf.variable_scope('output_layer'):
            output_dim = self.validation_Y.shape[1]
            net_shape = x.get_shape().as_list()
            fc_w = tf.get_variable('fc_w', shape=[net_shape[-1], output_dim], dtype=tf.float32)
            fc_b = tf.get_variable('fc_b', shape=[output_dim], dtype=tf.float32)
            self.networkOutput = tf.matmul(x, fc_w) + fc_b

    def get_l2_loss(self):
        # L2 Loss
        tvars = tf.trainable_variables()
        self.l2Loss = tf.constant(0.0)
        for tv in tvars:
            if 'fc_w' in tv.name:
                self.l2Loss += self.l2Lambda * tf.nn.l2_loss(tv)
            self.paramL2Norms[tv.name] = tf.nn.l2_loss(tv)

    def build_loss(self):
        with tf.name_scope("loss"):
            # MSE Loss
            self.predicted_t = self.networkOutput
            squared_diff = tf.squared_difference(self.predicted_t, self.input_t)
            sample_wise_sum = tf.reduce_sum(squared_diff, axis=1)
            self.sampleWiseErrors = sample_wise_sum
            self.regressionMeanSquaredError = tf.reduce_mean(self.sampleWiseErrors)
            self.get_l2_loss()
            self.totalLoss = self.regressionMeanSquaredError + self.l2Loss

    def build_optimizer(self):
        with tf.variable_scope("optimizer"):
            self.globalStep = tf.Variable(0, name='global_step', trainable=False)
            # self.optimizer = tf.train.AdamOptimizer().minimize(self.totalLoss, global_step=self.globalStep)
            boundaries = [100000, 200000, 300000]
            values = [0.001, 0.0001, 0.00001, 0.000001]
            self.learningRate = tf.train.piecewise_constant(self.globalStep, boundaries, values)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(
                self.totalLoss, global_step=self.globalStep)

    def get_train_batch(self):
        iter_count = 0
        while True:
            chosen_indices = np.random.choice(self.validation_X.shape[0], self.batchSize, replace=False)
            chosen_X = self.validation_X[chosen_indices]
            chosen_Y = self.validation_Y[chosen_indices]
            yield (chosen_X, chosen_Y)
            iter_count += 1
            if iter_count == self.maxIteration:
                break

    def get_eval_batch(self, X_, Y_):
        sample_count = 0
        while True:
            chosen_X = X_[sample_count:sample_count + self.batchSize]
            chosen_Y = Y_[sample_count:sample_count + self.batchSize]
            yield (chosen_X, chosen_Y)
            sample_count += self.batchSize
            if sample_count >= self.test_X.shape[0]:
                break

    def train(self):
        data_generator = self.get_train_batch()
        self.sess.run(tf.global_variables_initializer())
        iteration = 0
        losses = []
        val_mse = self.eval_mse(X_=self.validation_X, Y_=self.validation_Y)
        test_mse = self.eval_mse(X_=self.test_X, Y_=self.test_Y)
        print("Beginning")
        print("val_mse:{0}".format(val_mse))
        print("test_mse:{0}".format(test_mse))
        for x_, y_ in data_generator:
            # print("******************************************************************")
            feed_dict = {self.input_x: x_, self.input_t: y_}
            run_ops = [self.optimizer, self.totalLoss]
            results = self.sess.run(run_ops, feed_dict=feed_dict)
            # print(results[1])
            iteration += 1
            losses.append(results[1])
            if (iteration + 1) % 10 == 0:
                print("Iteration:{0} Main_loss={1}".format(iteration, np.mean(np.array(losses))))
                losses = []
            if (iteration + 1) % 500 == 0:
                print("Iteration:{0}".format(iteration))
                val_mse = self.eval_mse(X_=self.validation_X, Y_=self.validation_Y)
                test_mse = self.eval_mse(X_=self.test_X, Y_=self.test_Y)
                print("val_mse:{0}".format(val_mse))
                print("test_mse:{0}".format(test_mse))
        val_mse = self.eval_mse(X_=self.validation_X, Y_=self.validation_Y)
        test_mse = self.eval_mse(X_=self.test_X, Y_=self.test_Y)
        print("val_mse:{0}".format(val_mse))
        print("test_mse:{0}".format(test_mse))

    def eval_mse(self, X_, Y_):
        feed_dict = {self.input_x: X_, self.input_t: Y_}
        results = self.sess.run([self.regressionMeanSquaredError, self.sampleWiseErrors, self.predicted_t],
                                feed_dict=feed_dict)
        mse = results[0]
        return mse

    def run(self):
        self.build_network()
        self.build_loss()
        self.build_optimizer()
        self.train()
