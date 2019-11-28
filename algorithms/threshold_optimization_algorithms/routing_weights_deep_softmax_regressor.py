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
        self.posteriorDim = validation_data.get_dict("posterior_probs")[0].shape[1]
        leaf_count = len(self.leafNodes)
        self.inputPosteriors = tf.placeholder(dtype=tf.float32, shape=[None, self.posteriorDim, leaf_count],
                                              name='inputPosteriors')
        self.inputRouteMatrix = tf.placeholder(dtype=tf.int32, shape=[None, leaf_count],
                                               name='inputRouteMatrix')
        self.labelMatrix = tf.placeholder(dtype=tf.float32, shape=[None, self.posteriorDim], name='labelVector')
        self.weights = None
        self.weightedPosteriors = None
        self.finalPosterior = None
        self.squaredDiff = None
        self.sampleWiseSum = None
        # self.logQ = None
        # self.crossEntropyMatrix = None

    def build_loss(self):
        with tf.name_scope("loss"):
            self.weights = self.networkOutput
            self.weightedPosteriors = tf.expand_dims(self.weights, axis=1) * self.inputPosteriors
            self.finalPosterior = tf.reduce_sum(self.weightedPosteriors, axis=2)
            self.squaredDiff = tf.squared_difference(self.finalPosterior, self.labelMatrix)
            self.sampleWiseSum = tf.reduce_sum(self.squaredDiff, axis=1)
            self.regressionMeanSquaredError = tf.reduce_mean(self.sampleWiseSum)
            self.get_l2_loss()
            self.totalLoss = self.regressionMeanSquaredError + self.l2Loss

    def get_train_batch(self):
        while True:
            chosen_indices = np.random.choice(self.validation_X.shape[0], self.batchSize, replace=False)
            chosen_X = self.validation_X[chosen_indices]
            chosen_labels = self.validationData.labelList[chosen_indices]
            one_hot_labels = np.zeros(shape=(self.batchSize, self.posteriorDim))
            one_hot_labels[np.arange(self.batchSize), chosen_labels] = 1.0
            route_matrix = self.validationRoutingMatrix[chosen_indices]
            chosen_sparse_posteriors = self.validationSparsePosteriors[chosen_indices]
            yield (chosen_X, one_hot_labels, route_matrix, chosen_sparse_posteriors)

    def get_eval_batch(self, **kwargs):
        sample_count = 0
        while True:
            X = kwargs["X"]
            route_matrix = kwargs["route_matrix"]
            labels = kwargs["labels"]
            sparse_posteriors = kwargs["sparse_posteriors"]
            chosen_X = X[sample_count:sample_count + self.batchSize]
            chosen_route_matrix = route_matrix[sample_count:sample_count + self.batchSize]
            chosen_sparse_posteriors = sparse_posteriors[sample_count:sample_count + self.batchSize]
            chosen_labels = labels[sample_count:sample_count + self.batchSize]
            one_hot_labels = np.zeros(shape=(self.batchSize, self.posteriorDim))
            one_hot_labels[np.arange(self.batchSize), chosen_labels] = 1.0
            yield (chosen_X, one_hot_labels, chosen_route_matrix, chosen_sparse_posteriors)
            sample_count += self.batchSize
            if sample_count >= self.test_X.shape[0]:
                break

    def eval_mse(self, **kwargs):
        X = kwargs["X"]
        route_matrix = kwargs["route_matrix"]
        labels = kwargs["labels"]
        sparse_posteriors = kwargs["sparse_posteriors"]
        one_hot_labels = np.zeros(shape=(self.batchSize, self.posteriorDim))
        one_hot_labels[np.arange(self.batchSize), labels] = 1.0
        feed_dict = {self.input_x: X,
                     self.inputPosteriors: sparse_posteriors,
                     self.labelMatrix: one_hot_labels,
                     self.inputRouteMatrix: route_matrix}

        self.weights = self.networkOutput
        self.weightedPosteriors = tf.expand_dims(self.weights, axis=1) * self.inputPosteriors
        self.finalPosterior = tf.reduce_sum(self.weightedPosteriors, axis=2)
        self.squaredDiff = tf.squared_difference(self.finalPosterior, self.labelMatrix)
        self.sampleWiseSum = tf.reduce_sum(self.squaredDiff, axis=1)
        self.regressionMeanSquaredError = tf.reduce_mean(self.sampleWiseSum)
        results = self.sess.run([self.regressionMeanSquaredError,
                                 self.weights,
                                 self.weightedPosteriors,
                                 self.finalPosterior,
                                 self.squaredDiff,
                                 self.sampleWiseSum],
                                feed_dict=feed_dict)
        mse = results[0]
        return mse

    def eval_datasets(self):
        val_mse = self.eval_mse(X=self.validation_X, route_matrix=self.validationRoutingMatrix,
                                labels=self.validationData.labelList, sparse_posteriors=self.validationSparsePosteriors)
        test_mse = self.eval_mse(X=self.test_X, route_matrix=self.testRoutingMatrix,
                                 labels=self.testData.labelList, sparse_posteriors=self.testSparsePosteriors)
        print("val_mse:{0}".format(val_mse))
        print("test_mse:{0}".format(test_mse))

    def train(self):
        data_generator = self.get_train_batch()
        self.sess.run(tf.global_variables_initializer())
        iteration = 0
        losses = []
        self.eval_datasets()
        for chosen_X, one_hot_labels, route_matrix, chosen_sparse_posteriors in data_generator:
            # print("******************************************************************")
            feed_dict = {self.input_x: chosen_X,
                         self.inputPosteriors: chosen_sparse_posteriors,
                         self.labelMatrix: one_hot_labels,
                         self.inputRouteMatrix: route_matrix}
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
                self.eval_datasets()
        self.eval_datasets()
