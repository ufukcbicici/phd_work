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
        # Validation
        leaf_eval_counts_val = np.sum(self.validationRoutingMatrix, axis=1)
        single_path_indices_val = np.nonzero(leaf_eval_counts_val == 1)[0]
        multi_path_indices_val = np.nonzero(leaf_eval_counts_val > 1)[0]
        self.multiPathDataDict["validation"] = NetworkInput(
            _x=self.validation_X[multi_path_indices_val],
            _y=self.validationData.labelList[multi_path_indices_val],
            routing_matrix=self.validationRoutingMatrix[multi_path_indices_val],
            sparse_posteriors=self.validationSparsePosteriors[multi_path_indices_val])
        # Test
        leaf_eval_counts_test = np.sum(self.testRoutingMatrix, axis=1)
        single_path_indices_test = np.nonzero(leaf_eval_counts_test == 1)[0]
        multi_path_indices_test = np.nonzero(leaf_eval_counts_test > 1)[0]
        self.multiPathDataDict["test"] = NetworkInput(
            _x=self.test_X[multi_path_indices_test],
            _y=self.testData.labelList[multi_path_indices_test],
            routing_matrix=self.testRoutingMatrix[multi_path_indices_test],
            sparse_posteriors=self.testSparsePosteriors[multi_path_indices_test])
        self.fullDataCount = {"validation": self.validation_X.shape[0], "test": self.test_X.shape[0]}


        # if self.useMultiPathOnly:
        #     # Validation
        #     self.validation_X = self.validation_X[multi_path_indices_val]
        #     self.validationLabels = self.validationData.labelList[multi_path_indices_val]
        #     self.validationRoutingMatrix = self.validationRoutingMatrix[multi_path_indices_val]
        #     self.validationSparsePosteriors = self.validationSparsePosteriors[multi_path_indices_val]
        #     # Test
        #     self.test_X = self.test_X[multi_path_indices_test]
        #     self.testLabels = self.testData.labelList[multi_path_indices_test]
        #     self.testRoutingMatrix = self.testRoutingMatrix[multi_path_indices_test]
        #     self.testSparsePosteriors = self.testSparsePosteriors[multi_path_indices_test]
        # else:
        #     self.validationLabels = self.validationData.labelList
        #     self.testLabels = self.testData.labelList
        # # Calculate single route samples' performance only at the beginning.
        # predicted_posteriors = results[3]
        # leaf_eval_counts = np.sum(route_matrix, axis=1)
        # single_path_indices = np.nonzero(leaf_eval_counts == 1)[0]
        # multi_path_indices = np.nonzero(leaf_eval_counts > 1)[0]
        # assert single_path_indices.shape[0] + multi_path_indices.shape[0] == labels.shape[0]
        # single_path_posteriors = sparse_posteriors[single_path_indices, :]
        # single_path_posteriors = np.sum(single_path_posteriors, axis=2)
        # single_path_predicted_labels = np.argmax(single_path_posteriors, axis=1)
        # single_path_labels = labels[single_path_indices]
        # single_path_correct_count = np.sum(single_path_predicted_labels == single_path_labels)

        # self.logQ = None
        # self.crossEntropyMatrix = None

    def preprocess_data(self):
        pca = PCA(n_components=self.validation_X.shape[1])
        if not self.useMultiPathOnly:
            pca.fit(self.validation_X)
        else:
            leaf_eval_counts = np.sum(self.validationRoutingMatrix, axis=1)
            multi_path_indices = np.nonzero(leaf_eval_counts > 1)[0]
            pca.fit(self.validation_X[multi_path_indices])
        self.validation_X = pca.transform(self.validation_X)
        self.test_X = pca.transform(self.test_X)

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
            if not self.useMultiPathOnly:
                chosen_indices = np.random.choice(self.validation_X.shape[0], self.batchSize, replace=False)
                batch_size = self.batchSize
            else:
                leaf_eval_counts = np.sum(self.validationRoutingMatrix, axis=1)
                multi_path_indices = np.nonzero(leaf_eval_counts > 1)[0]
                batch_size = min(multi_path_indices.shape[0], self.batchSize)
                chosen_indices = np.random.choice(multi_path_indices, batch_size, replace=False)
            chosen_X = self.validation_X[chosen_indices]
            chosen_labels = self.validationData.labelList[chosen_indices]
            one_hot_labels = np.zeros(shape=(batch_size, self.posteriorDim))
            one_hot_labels[np.arange(batch_size), chosen_labels] = 1.0
            route_matrix = self.validationRoutingMatrix[chosen_indices]
            chosen_sparse_posteriors = self.validationSparsePosteriors[chosen_indices]
            yield (chosen_X, one_hot_labels, route_matrix, chosen_sparse_posteriors)

    def eval_performance(self, **kwargs):
        X = kwargs["X"]
        route_matrix = kwargs["route_matrix"]
        labels = kwargs["labels"]
        sparse_posteriors = kwargs["sparse_posteriors"]
        one_hot_labels = np.zeros(shape=(X.shape[0], self.posteriorDim))
        one_hot_labels[np.arange(X.shape[0]), labels] = 1.0
        assert np.array_equal(route_matrix, (np.sum(sparse_posteriors, axis=1) != 0.0).astype(np.float32))
        feed_dict = {self.input_x: X,
                     self.inputPosteriors: sparse_posteriors,
                     self.labelMatrix: one_hot_labels,
                     self.inputRouteMatrix: route_matrix}
        results = self.sess.run([self.regressionMeanSquaredError,
                                 self.weights,
                                 self.weightedPosteriors,
                                 self.finalPosterior,
                                 self.squaredDiff,
                                 self.sampleWiseSum],
                                feed_dict=feed_dict)
        mse = results[0]

        predicted_posteriors = results[3]
        leaf_eval_counts = np.sum(route_matrix, axis=1)
        single_path_indices = np.nonzero(leaf_eval_counts == 1)[0]
        multi_path_indices = np.nonzero(leaf_eval_counts > 1)[0]
        assert single_path_indices.shape[0] + multi_path_indices.shape[0] == labels.shape[0]
        single_path_posteriors = sparse_posteriors[single_path_indices, :]
        single_path_posteriors = np.sum(single_path_posteriors, axis=2)
        single_path_predicted_labels = np.argmax(single_path_posteriors, axis=1)
        single_path_labels = labels[single_path_indices]
        single_path_correct_count = np.sum(single_path_predicted_labels == single_path_labels)

        multi_path_predicted_posteriors = predicted_posteriors[multi_path_indices, :]
        predicted_multi_path_labels = np.argmax(multi_path_predicted_posteriors, axis=1)
        multi_path_labels = labels[multi_path_indices]
        multi_path_correct_count = np.sum(multi_path_labels == predicted_multi_path_labels)
        accuracy = np.sum(single_path_correct_count + multi_path_correct_count) / X.shape[0]
        return mse, accuracy

    def eval_datasets(self):
        val_mse, val_accuracy = self.eval_performance(X=self.validation_X, route_matrix=self.validationRoutingMatrix,
                                                      labels=self.validationData.labelList,
                                                      sparse_posteriors=self.validationSparsePosteriors)
        test_mse, test_accuracy = self.eval_performance(X=self.test_X, route_matrix=self.testRoutingMatrix,
                                                        labels=self.testData.labelList,
                                                        sparse_posteriors=self.testSparsePosteriors)
        print("val_mse:{0} val_accuracy:{1}".format(val_mse, val_accuracy))
        print("test_mse:{0} test_accuracy:{1}".format(test_mse, test_accuracy))

    def get_ideal_performances(self, sparse_posteriors_tensor, ideal_weights, labels):
        label_matrix = np.zeros(shape=(sparse_posteriors_tensor.shape[0], self.posteriorDim))
        label_matrix[np.arange(sparse_posteriors_tensor.shape[0]), labels] = 1.0
        ideal_posteriors_arr = []
        for idx in range(sparse_posteriors_tensor.shape[0]):
            A = sparse_posteriors_tensor[idx, :]
            b = ideal_weights[idx, :]
            p = A @ b
            ideal_posteriors_arr.append(p)
        P = np.stack(ideal_posteriors_arr, axis=0)
        diff_arr = label_matrix - P
        squared_diff_arr = np.square(diff_arr)
        se = np.sum(squared_diff_arr, axis=1)
        ideal_mse = np.mean(se)
        predicted_labels = np.argmax(P, axis=1)
        comparison_vector = labels == predicted_labels
        ideal_accuracy = np.sum(comparison_vector) / labels.shape[0]
        return ideal_mse, ideal_accuracy

    def train(self):
        self.preprocess_data()
        data_generator = self.get_train_batch()
        self.sess.run(tf.global_variables_initializer())
        iteration = 0
        losses = []
        ideal_val_mse, ideal_val_accuracy = self.get_ideal_performances(
            sparse_posteriors_tensor=self.validationSparsePosteriors,
            ideal_weights=self.validation_Y, labels=self.validationData.labelList)
        ideal_test_mse, ideal_test_accuracy = self.get_ideal_performances(
            sparse_posteriors_tensor=self.testSparsePosteriors,
            ideal_weights=self.test_Y, labels=self.testData.labelList)
        print("ideal_val_mse={0} ideal_val_accuracy={1}".format(ideal_val_mse, ideal_val_accuracy))
        print("ideal_test_mse={0} ideal_test_accuracy={1}".format(ideal_test_mse, ideal_test_accuracy))
        # Ideal
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
