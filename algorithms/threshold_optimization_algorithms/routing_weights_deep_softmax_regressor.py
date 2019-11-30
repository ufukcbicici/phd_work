import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.routing_weights_deep_regressor import RoutingWeightDeepRegressor
from sklearn.decomposition import PCA

from auxillary.db_logger import DbLogger
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
        self.labelMatrix = tf.placeholder(dtype=tf.float32, shape=[None, self.posteriorDim], name='labelMatrix')
        self.weights = None
        self.weightedPosteriors = None
        self.finalPosterior = None
        self.squaredDiff = None
        self.sampleWiseSum = None
        self.runId = None
        self.iteration = 0
        self.dbRows = []
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

    def get_simple_average_results(self, data_type):
        data = self.multiPathDataDict[data_type]
        sum_posteriors = np.sum(data.sparsePosteriors, axis=2)
        leaf_counts = np.sum(data.routingMatrix, axis=1)
        reciprocal_leaf_counts = np.expand_dims(np.reciprocal(leaf_counts), axis=1)
        mean_posteriors = sum_posteriors * reciprocal_leaf_counts
        simple_average_predicted_labels = np.argmax(mean_posteriors, axis=1)
        simple_average_correct_count = np.sum(data.y == simple_average_predicted_labels)
        simple_average_accuracy = (self.singlePathCorrectCounts[data_type] + simple_average_correct_count) / \
                                  self.fullDataDict[data_type].X.shape[0]
        return simple_average_accuracy, mean_posteriors

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
            chosen_Y = data.y[chosen_indices]
            one_hot_labels = data.y_one_hot[chosen_indices]
            route_matrix = data.routingMatrix[chosen_indices]
            chosen_sparse_posteriors = data.sparsePosteriors[chosen_indices]
            yield (chosen_X, one_hot_labels, route_matrix, chosen_sparse_posteriors, chosen_Y)

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
        # Convert to probability
        min_scores = np.min(multi_path_predicted_posteriors, axis=1, keepdims=True)
        multi_path_predicted_posteriors = multi_path_predicted_posteriors + np.abs(min_scores)
        sums = np.sum(multi_path_predicted_posteriors, axis=1, keepdims=True)
        multi_path_predicted_posteriors = multi_path_predicted_posteriors / sums
        predicted_multi_path_labels = np.argmax(multi_path_predicted_posteriors, axis=1)
        multi_path_correct_count = np.sum(data.y == predicted_multi_path_labels)
        regressor_accuracy = (self.singlePathCorrectCounts[data_type] + multi_path_correct_count) / \
                   self.fullDataDict[data_type].X.shape[0]

        # Simple Average Accuracy
        simple_average_accuracy, simple_average_mean_posteriors = self.get_simple_average_results(data_type=data_type)

        # Ensemble of the two
        ensemble_posteriors = np.stack([multi_path_predicted_posteriors, simple_average_mean_posteriors], axis=2)
        ensemble_posteriors = np.mean(ensemble_posteriors, axis=2)
        ensemble_predicted_labels = np.argmax(ensemble_posteriors, axis=1)
        ensemble_correct_count = np.sum(data.y == ensemble_predicted_labels)
        ensemble_average_accuracy = (self.singlePathCorrectCounts[data_type] + ensemble_correct_count) / \
                                    self.fullDataDict[data_type].X.shape[0]
        return mse, regressor_accuracy, simple_average_accuracy, ensemble_average_accuracy

    def eval_datasets(self):
        val_mse, val_accuracy_regressor, val_accuracy_simple_avg, val_accuracy_ensemble = \
            self.eval_performance(data_type="validation")
        test_mse, test_accuracy_regressor, test_accuracy_simple_avg, test_accuracy_ensemble = \
            self.eval_performance(data_type="test")
        print("val_mse:{0} val_accuracy_regressor:{1}, val_accuracy_simple_avg:{2} val_accuracy_ensemble:{3}"
              .format(val_mse, val_accuracy_regressor, val_accuracy_simple_avg, val_accuracy_ensemble))
        print("test_mse:{0} test_accuracy_regressor:{1}, test_accuracy_simple_avg:{2} test_accuracy_ensemble:{3}"
              .format(test_mse, test_accuracy_regressor, test_accuracy_simple_avg, test_accuracy_ensemble))
        result_dict = \
            {"val_mse": val_mse,
             "val_accuracy_regressor": val_accuracy_regressor,
             "val_accuracy_simple_avg": val_accuracy_simple_avg,
             "val_accuracy_ensemble": val_accuracy_ensemble,
             "test_mse": test_mse,
             "test_accuracy_regressor": test_accuracy_regressor,
             "test_accuracy_simple_avg": test_accuracy_simple_avg,
             "test_accuracy_ensemble": test_accuracy_ensemble}
        row = (self.runId, self.iteration, val_accuracy_regressor, val_accuracy_simple_avg,
               val_accuracy_ensemble, np.asscalar(val_mse), test_accuracy_regressor, test_accuracy_simple_avg,
               test_accuracy_ensemble, np.asscalar(test_mse), self.l2Lambda)
        self.dbRows.append(row)
        if len(self.dbRows) >= 1000:
            DbLogger.write_into_table(rows=self.dbRows, table="multipath_regression",
                                      col_count=len(self.dbRows[0]))
            self.dbRows = []
        return result_dict

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

    def get_explanation(self):
        exp_string = ""
        exp_string += "l2_lambda={0}\n".format(self.l2Lambda)
        exp_string += "layers={0}\n".format(self.layers)
        exp_string += "batch_size={0}\n".format(self.batchSize)
        return exp_string

    def build_optimizer(self):
        with tf.variable_scope("optimizer"):
            self.globalStep = tf.Variable(0, name='global_step', trainable=False)
            # self.optimizer = tf.train.AdamOptimizer().minimize(self.totalLoss, global_step=self.globalStep)
            # boundaries = [15000, 30000, 45000]
            # values = [0.01, 0.001, 0.0001, 0.00001]
            # self.learningRate = tf.train.piecewise_constant(self.globalStep, boundaries, values)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(
                self.totalLoss, global_step=self.globalStep)

    def train(self):
        self.runId = DbLogger.get_run_id()
        exp_string = self.get_explanation()
        DbLogger.write_into_table(rows=[(self.runId, exp_string)], table=DbLogger.runMetaData, col_count=2)
        self.preprocess_data()
        data_generator = self.get_train_batch()
        self.sess.run(tf.global_variables_initializer())
        losses = []
        self.eval_datasets()
        for chosen_X, one_hot_labels, route_matrix, chosen_sparse_posteriors, chosen_Y in data_generator:
            # print("******************************************************************")
            feed_dict = {self.input_x: chosen_X,
                         self.inputPosteriors: chosen_sparse_posteriors,
                         self.labelMatrix: one_hot_labels,
                         self.inputRouteMatrix: route_matrix}
            run_ops = [self.optimizer, self.totalLoss]
            results = self.sess.run(run_ops, feed_dict=feed_dict)
            # print(results[1])
            self.iteration += 1
            losses.append(results[1])
            if (self.iteration + 1) % 1 == 0:
                print("Iteration:{0} Main_loss={1}".format(self.iteration, np.mean(np.array(losses))))
                losses = []
            if (self.iteration + 1) % 1 == 0:
                print("Iteration:{0}".format(self.iteration))
                self.eval_datasets()
            if self.iteration == self.maxIteration:
                break
        self.eval_datasets()
        tf.reset_default_graph()
