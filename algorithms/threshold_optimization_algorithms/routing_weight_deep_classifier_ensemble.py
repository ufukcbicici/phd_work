from algorithms.threshold_optimization_algorithms.routing_weights_deep_softmax_regressor import \
    RoutingWeightDeepSoftmaxRegressor
import tensorflow as tf
import numpy as np

from auxillary.db_logger import DbLogger


class RoutingWeightDeepClassifier(RoutingWeightDeepSoftmaxRegressor):
    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data, layers,
                 l2_lambda, batch_size, max_iteration, ensemble_count, use_multi_path_only=False):
        super().__init__(network, validation_routing_matrix, test_routing_matrix, validation_data, test_data, layers,
                         l2_lambda, batch_size, max_iteration, use_multi_path_only)
        self.ensembleCount = ensemble_count
        self.logits = None
        self.classCount = self.fullDataDict["validation"].y_one_hot.shape[1]
        self.ceVector = None
        self.ceLoss = None
        self.finalPosterior = None
        self.xTensors = [tf.placeholder(dtype=tf.float32,
                                        shape=[None, self.validation_X.shape[1]], name='input_x_{0}'.format(i))
                         for i in range(self.ensembleCount)]
        self.yTensors = [tf.placeholder(dtype=tf.int64, shape=[None, ], name='labelsVector_{0}'.format(i))
                         for i in range(self.ensembleCount)]
        self.networkOutputs = []
        self.logitsList = []
        self.ceLosses = []
        self.sumCeLoss = None
        self.finalPosteriors = []
        self.runId = None
        self.iteration = 0

    def build_network(self):
        for network_id in range(self.ensembleCount):
            x = self.xTensors[network_id]
            for layer_id, hidden_dim in enumerate(self.layers):
                with tf.variable_scope('network_{0}_layer_{1}'.format(network_id, layer_id)):
                    net_shape = x.get_shape().as_list()
                    fc_w = tf.get_variable('fc_w', shape=[net_shape[-1], hidden_dim], dtype=tf.float32)
                    fc_b = tf.get_variable('fc_b', shape=[hidden_dim], dtype=tf.float32)
                    x = tf.matmul(x, fc_w) + fc_b
                    x = tf.nn.leaky_relu(x)
            # Output layer
            with tf.variable_scope('network_{0}_output_layer'.format(network_id)):
                net_shape = x.get_shape().as_list()
                fc_w = tf.get_variable('fc_w', shape=[net_shape[-1], self.classCount], dtype=tf.float32)
                fc_b = tf.get_variable('fc_b', shape=[self.classCount], dtype=tf.float32)
                network_output = tf.matmul(x, fc_w) + fc_b
                self.networkOutputs.append(network_output)

    def build_loss(self):
        for network_id in range(self.ensembleCount):
            with tf.name_scope("loss"):
                logits = self.networkOutputs[network_id]
                final_posterior = tf.nn.softmax(logits)
                ce_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.yTensors[network_id],
                                                                           logits=logits)
                ce_loss = tf.reduce_mean(ce_vector)
                self.logitsList.append(logits)
                self.finalPosteriors.append(final_posterior)
                self.ceLosses.append(ce_loss)
        self.get_l2_loss()
        self.sumCeLoss = tf.add_n(self.ceLosses)
        self.totalLoss = self.sumCeLoss + self.l2Loss

    def eval_performance(self, data_type):
        data = self.multiPathDataDict[data_type]
        assert np.array_equal(data.routingMatrix, (np.sum(data.sparsePosteriors, axis=1) != 0.0).astype(np.float32))
        feed_dict = {}
        for network_id in range(self.ensembleCount):
            feed_dict[self.xTensors[network_id]] = data.X
            feed_dict[self.yTensors[network_id]] = data.y
        results = self.sess.run([self.sumCeLoss,
                                 self.logitsList,
                                 self.finalPosteriors,
                                 self.totalLoss],
                                feed_dict=feed_dict)
        sum_ce_loss = results[0]
        mean_ce_loss = sum_ce_loss / self.ensembleCount
        # Specific expert classifier for multi path
        list_of_predicted_posteriors = results[2]
        predicted_posteriors_tensor = np.stack(list_of_predicted_posteriors, axis=2)
        predicted_ensemble_posterior = np.mean(predicted_posteriors_tensor, axis=2)
        predicted_multi_path_labels = np.argmax(predicted_ensemble_posterior, axis=1)
        multi_path_correct_count = np.sum(data.y == predicted_multi_path_labels)
        classifier_accuracy = (self.singlePathCorrectCounts[data_type] + multi_path_correct_count) / \
                              self.fullDataDict[data_type].X.shape[0]
        # Accuracy of the simple average result
        simple_average_accuracy, mean_posteriors = self.get_simple_average_results(data_type=data_type)
        # Ensemble of the two
        ensemble_posteriors = np.stack([predicted_ensemble_posterior, mean_posteriors], axis=2)
        ensemble_posteriors = np.mean(ensemble_posteriors, axis=2)
        ensemble_predicted_labels = np.argmax(ensemble_posteriors, axis=1)
        ensemble_correct_count = np.sum(data.y == ensemble_predicted_labels)
        ensemble_average_accuracy = (self.singlePathCorrectCounts[data_type] + ensemble_correct_count) / \
                                    self.fullDataDict[data_type].X.shape[0]
        return mean_ce_loss, classifier_accuracy, simple_average_accuracy, ensemble_average_accuracy

    def get_explanation(self):
        exp_string = ""
        exp_string += "l2_lambda={0}\n".format(self.l2Lambda)
        exp_string += "layers={0}\n".format(self.layers)
        exp_string += "batch_size={0}\n".format(self.batchSize)
        exp_string += "ensemble_count={0}\n".format(self.ensembleCount)
        return exp_string

    def get_train_batch(self):
        if not self.useMultiPathOnly:
            data = self.fullDataDict["validation"]
        else:
            data = self.multiPathDataDict["validation"]
        batch_size = min(data.X.shape[0], self.batchSize)
        while True:
            X_list = []
            y_list = []
            for network_id in range(self.ensembleCount):
                # Bootstrapping effect
                chosen_indices = np.random.choice(data.X.shape[0], batch_size, replace=True)
                chosen_X = data.X[chosen_indices]
                chosen_Y = data.y[chosen_indices]
                # one_hot_labels = data.y_one_hot[chosen_indices]
                # route_matrix = data.routingMatrix[chosen_indices]
                # chosen_sparse_posteriors = data.sparsePosteriors[chosen_indices]
                X_list.append(chosen_X)
                y_list.append(chosen_Y)
            yield (X_list, y_list)

    def build_optimizer(self):
        with tf.variable_scope("optimizer"):
            self.globalStep = tf.Variable(0, name='global_step', trainable=False)
            # self.optimizer = tf.train.AdamOptimizer().minimize(self.totalLoss, global_step=self.globalStep)
            boundaries = [100000, 200000, 300000]
            values = [0.001, 0.0001, 0.00001, 0.000001]
            self.learningRate = tf.train.piecewise_constant(self.globalStep, boundaries, values)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(
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
        for X_list, y_list in data_generator:
            # print("******************************************************************")
            feed_dict = {}
            for network_id in range(self.ensembleCount):
                feed_dict[self.xTensors[network_id]] = X_list[network_id]
                feed_dict[self.yTensors[network_id]] = y_list[network_id]
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

