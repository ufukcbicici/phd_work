import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.routing_weights_deep_regressor import RoutingWeightDeepRegressor
from sklearn.decomposition import PCA
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants


class NetworkInput:
    def __init__(self, _x, _y, routing_matrix, sparse_posteriors):
        self.X = _x
        self.y = _y
        self.routingMatrix = routing_matrix
        self.sparsePosteriors = sparse_posteriors
        self.y_one_hot = np.zeros(shape=(_y.shape[0], self.sparsePosteriors.shape[1]))
        self.y_one_hot[np.arange(_y.shape[0]), self.y] = 1.0


class RoutingWeightDeepSoftmaxRegressor(RoutingWeightDeepRegressor):
    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data, layers,
                 l2_lambda, batch_size, max_iteration, use_multi_path_only=False):
        super().__init__(network, validation_routing_matrix, test_routing_matrix, validation_data, test_data, layers,
                         l2_lambda, batch_size, max_iteration, leaf_index=None)
        self.posteriorDim = self.validationSparsePosteriors.shape[1]
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
        self.useMultiPathOnly = use_multi_path_only
        self.fullDataDict = {"validation": NetworkInput(_x=self.validation_X, _y=self.validationData.labelList,
                                                        routing_matrix=self.validationRoutingMatrix,
                                                        sparse_posteriors=self.validationSparsePosteriors),
                             "test": NetworkInput(_x=self.test_X, _y=self.testData.labelList,
                                                  routing_matrix=self.testRoutingMatrix,
                                                  sparse_posteriors=self.testSparsePosteriors)}
        self.multiPathDataDict = {}
        self.singlePathCorrectCounts = {}
        # Create multi path data
        for data_type in ["validation", "test"]:
            data = self.fullDataDict[data_type]
            leaf_eval_counts = np.sum(data.routingMatrix, axis=1)
            single_path_indices = np.nonzero(leaf_eval_counts == 1)[0]
            multi_path_indices = np.nonzero(leaf_eval_counts > 1)[0]
            self.multiPathDataDict[data_type] = NetworkInput(
                _x=data.X[multi_path_indices],
                _y=data.y[multi_path_indices],
                routing_matrix=data.routingMatrix[multi_path_indices],
                sparse_posteriors=data.sparsePosteriors[multi_path_indices])
            single_path_posteriors = data.sparsePosteriors[single_path_indices, :]
            single_path_posteriors = np.sum(single_path_posteriors, axis=2)
            single_path_predicted_labels = np.argmax(single_path_posteriors, axis=1)
            single_path_labels = data.y[single_path_indices]
            single_path_correct_count = np.sum(single_path_predicted_labels == single_path_labels)
            self.singlePathCorrectCounts[data_type] = single_path_correct_count

    def preprocess_data(self):
        pass
        # pca = PCA(n_components=self.validation_X.shape[1])
        # if not self.useMultiPathOnly:
        #     data_dict = self.fullDataDict
        # else:
        #     data_dict = self.multiPathDataDict
        # pca.fit(data_dict["validation"].X)
        # data_dict["validation"].X = pca.transform(data_dict["validation"].X)
        # data_dict["test"].X = pca.transform(data_dict["test"].X)

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
        if not self.useMultiPathOnly:
            data = self.fullDataDict["validation"]
        else:
            data = self.multiPathDataDict["validation"]
        batch_size = min(data.X.shape[0], self.batchSize)
        while True:
            chosen_indices = np.random.choice(data.X.shape[0], batch_size, replace=False)
            chosen_X = data.X[chosen_indices]
            one_hot_labels = data.y_one_hot[chosen_indices]
            route_matrix = data.routingMatrix[chosen_indices]
            chosen_sparse_posteriors = data.sparsePosteriors[chosen_indices]
            yield (chosen_X, one_hot_labels, route_matrix, chosen_sparse_posteriors)

    def eval_performance(self, data_type):
        data = self.multiPathDataDict[data_type]
        assert np.array_equal(data.routingMatrix, (np.sum(data.sparsePosteriors, axis=1) != 0.0).astype(np.float32))
        feed_dict = {self.input_x: data.X,
                     self.inputPosteriors: data.sparsePosteriors,
                     self.labelMatrix: data.y_one_hot,
                     self.inputRouteMatrix: data.routingMatrix}
        results = self.sess.run([self.regressionMeanSquaredError,
                                 self.weights,
                                 self.weightedPosteriors,
                                 self.finalPosterior,
                                 self.squaredDiff,
                                 self.sampleWiseSum],
                                feed_dict=feed_dict)
        mse = results[0]

        multi_path_predicted_posteriors = results[3]
        predicted_multi_path_labels = np.argmax(multi_path_predicted_posteriors, axis=1)
        multi_path_correct_count = np.sum(data.y == predicted_multi_path_labels)
        accuracy = np.sum(self.singlePathCorrectCounts[data_type] + multi_path_correct_count) / \
                   self.fullDataDict[data_type].X.shape[0]
        return mse, accuracy

    def eval_datasets(self):
        val_mse, val_accuracy = self.eval_performance(data_type="validation")
        test_mse, test_accuracy = self.eval_performance(data_type="test")
        print("val_mse:{0} val_accuracy:{1}".format(val_mse, val_accuracy))
        print("test_mse:{0} test_accuracy:{1}".format(test_mse, test_accuracy))

    def get_ideal_performances(self, data_type):
        data = self.multiPathDataDict[data_type]
        ideal_posteriors_arr = []
        for idx in range(data.sparsePosteriors.shape[0]):
            A = data.sparsePosteriors[idx, :]
            b = data.y_one_hot[idx, :]
            w = np.linalg.lstsq(A, b, rcond=None)[0]
            p = A @ w
            ideal_posteriors_arr.append(p)
        P = np.stack(ideal_posteriors_arr, axis=0)
        assert data.y_one_hot.shape == P.shape
        diff_arr = data.y_one_hot - P
        squared_diff_arr = np.square(diff_arr)
        se = np.sum(squared_diff_arr, axis=1)
        ideal_mse = np.mean(se)
        predicted_labels = np.argmax(P, axis=1)
        comparison_vector = data.y == predicted_labels
        ideal_accuracy = (self.singlePathCorrectCounts[data_type] + np.sum(comparison_vector)) / \
                         self.fullDataDict[data_type].X.shape[0]
        return ideal_mse, ideal_accuracy

    def train(self):
        self.preprocess_data()
        data_generator = self.get_train_batch()
        self.sess.run(tf.global_variables_initializer())
        iteration = 0
        losses = []
        # Ideal
        ideal_val_mse, ideal_val_accuracy = self.get_ideal_performances(data_type="validation")
        ideal_test_mse, ideal_test_accuracy = self.get_ideal_performances(data_type="test")
        print("ideal_val_mse={0} ideal_val_accuracy={1}".format(ideal_val_mse, ideal_val_accuracy))
        print("ideal_test_mse={0} ideal_test_accuracy={1}".format(ideal_test_mse, ideal_test_accuracy))
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
            if (iteration + 1) % 100 == 0:
                print("Iteration:{0} Main_loss={1}".format(iteration, np.mean(np.array(losses))))
                losses = []
            if (iteration + 1) % 500 == 0:
                print("Iteration:{0}".format(iteration))
                self.eval_datasets()
        self.eval_datasets()
