from algorithms.threshold_optimization_algorithms.routing_weights_deep_softmax_regressor import \
    RoutingWeightDeepSoftmaxRegressor
import tensorflow as tf
import numpy as np

from auxillary.db_logger import DbLogger


class RoutingWeightDeepClassifier(RoutingWeightDeepSoftmaxRegressor):
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
            net_shape = x.get_shape().as_list()
            fc_w = tf.get_variable('fc_w', shape=[net_shape[-1], self.classCount], dtype=tf.float32)
            fc_b = tf.get_variable('fc_b', shape=[self.classCount], dtype=tf.float32)
            self.networkOutput = tf.matmul(x, fc_w) + fc_b

    def build_loss(self):
        with tf.name_scope("loss"):
            self.logits = self.networkOutput
            self.finalPosterior = tf.nn.softmax(self.logits)
            self.ceVector = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labelsVector, logits=self.logits)
            self.ceLoss = tf.reduce_mean(self.ceVector)
            self.get_l2_loss()
            self.totalLoss = self.ceLoss + self.l2Loss

    def eval_performance(self, data_type):
        data = self.multiPathDataDict[data_type]
        assert np.array_equal(data.routingMatrix, (np.sum(data.sparsePosteriors, axis=1) != 0.0).astype(np.float32))
        feed_dict = {self.input_x: data.X,
                     self.inputPosteriors: data.sparsePosteriors,
                     self.labelMatrix: data.y_one_hot,
                     self.inputRouteMatrix: data.routingMatrix,
                     self.labelsVector: data.y}
        results = self.sess.run([self.ceLoss,
                                 self.logits,
                                 self.finalPosterior,
                                 self.totalLoss],
                                feed_dict=feed_dict)
        ce_loss = results[0]
        logits_ = results[1]
        # Specific expert classifier for multi path
        multi_path_predicted_posteriors = results[2]
        total_loss = results[3]
        predicted_multi_path_labels = np.argmax(multi_path_predicted_posteriors, axis=1)
        multi_path_correct_count = np.sum(data.y == predicted_multi_path_labels)
        classifier_accuracy = (self.singlePathCorrectCounts[data_type] + multi_path_correct_count) / \
                              self.fullDataDict[data_type].X.shape[0]
        # Accuracy of the simple average result
        simple_average_accuracy, mean_posteriors = self.get_simple_average_results(data_type=data_type)
        # Ensemble of the two
        ensemble_posteriors = np.stack([multi_path_predicted_posteriors, mean_posteriors], axis=2)
        ensemble_posteriors = np.mean(ensemble_posteriors, axis=2)
        enemble_predicted_labels = np.argmax(ensemble_posteriors, axis=1)
        ensemble_correct_count = np.sum(data.y == enemble_predicted_labels)
        ensemble_average_accuracy = (self.singlePathCorrectCounts[data_type] + ensemble_correct_count) / \
                                    self.fullDataDict[data_type].X.shape[0]
        return ce_loss, classifier_accuracy, simple_average_accuracy, ensemble_average_accuracy

    def eval_datasets(self):
        val_cee, val_accuracy_classifier, val_accuracy_simple_avg, val_accuracy_ensemble = \
            self.eval_performance(data_type="validation")
        test_cee, test_accuracy_classifier, test_accuracy_simple_avg, test_accuracy_ensemble = \
            self.eval_performance(data_type="test")
        print("val_cee:{0} val_accuracy_classifier:{1}, val_accuracy_simple_avg:{2} val_accuracy_ensemble:{3}"
              .format(val_cee, val_accuracy_classifier, val_accuracy_simple_avg, val_accuracy_ensemble))
        print("test_cee:{0} test_accuracy_classifier:{1}, test_accuracy_simple_avg:{2} test_accuracy_ensemble:{3}"
              .format(test_cee, test_accuracy_classifier, test_accuracy_simple_avg, test_accuracy_ensemble))
        result_dict = \
            {"val_cee": val_cee,
             "val_accuracy_classifier": val_accuracy_classifier,
             "val_accuracy_simple_avg": val_accuracy_simple_avg,
             "val_accuracy_ensemble": val_accuracy_ensemble,
             "test_cee": test_cee,
             "test_accuracy_classifier": test_accuracy_classifier,
             "test_accuracy_simple_avg": test_accuracy_simple_avg,
             "test_accuracy_ensemble": test_accuracy_ensemble}
        row = (self.runId, self.iteration, val_accuracy_classifier, val_accuracy_simple_avg,
               val_accuracy_ensemble, np.asscalar(val_cee), test_accuracy_classifier, test_accuracy_simple_avg,
               test_accuracy_ensemble, np.asscalar(test_cee), self.l2Lambda)
        self.dbRows.append(row)
        if len(self.dbRows) >= 1000:
            DbLogger.write_into_table(rows=self.dbRows, table="multipath_classification",
                                      col_count=len(self.dbRows[0]))
            self.dbRows = []
        return result_dict

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
                         self.inputRouteMatrix: route_matrix,
                         self.labelsVector: chosen_Y}
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