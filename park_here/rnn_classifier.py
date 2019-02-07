import tensorflow as tf
import numpy as np

from auxillary.db_logger import DbLogger
from auxillary.constants import DatasetTypes
from park_here.constants import Constants
from simple_tf.global_params import GlobalConstants


class RnnClassifier:
    def __init__(self, dataset):
        self.dataset = dataset
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.input = tf.placeholder(dtype=tf.int32,
                                    shape=[None, self.dataset.maxLength, Constants.ORIGINAL_DATA_DIMENSION],
                                    name='input')
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None], name='input_y')
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        self.embeddings = None
        self.initial_state = None
        self.initial_state_fw = None
        self.initial_state_bw = None
        self.finalLstmState = None
        self.finalState = None
        self.outputs = None
        self.stateObject = None
        self.attentionMechanismInput = None
        self.contextVector = None
        self.alpha = None
        self.temps = []
        self.l2_loss = None
        self.logits = None
        self.predictions = None
        self.mainLoss = None
        self.correctPredictions = None
        self.numOfCorrectPredictions = None
        self.accuracy = None
        self.globalStep = None
        self.optimizer = None

    def build_classifier(self):
        self.get_embeddings()
        self.get_classifier_structure()
        self.get_softmax_layer()
        self.get_loss()
        self.get_accuracy()
        self.get_optimizer()
        # self.sess = tf.Session()

    def get_embeddings(self):
        if Constants.EMBEDDING_DIM == 0:
            return self.input
        else:
            return tf.layers.dense(self.input, Constants.EMBEDDING_DIM, activation=None)

    @staticmethod
    def get_stacked_lstm_cells(hidden_dimension, num_layers):
        cell_list = [tf.contrib.rnn.LSTMCell(hidden_dimension,
                                             forget_bias=1.0,
                                             state_is_tuple=True) for _ in range(num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)
        return cell

    def get_classifier_structure(self):
        net = self.get_embeddings()
        if not Constants.USE_BIDIRECTIONAL:
            cell = RnnClassifier.get_stacked_lstm_cells(hidden_dimension=Constants.LSTM_HIDDEN_DIM,
                                                        num_layers=Constants.NUM_OF_LSTM_LAYERS)
            # Add dropout to cell output
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            self.initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            # Dynamic LSTM
            with tf.variable_scope('LSTM'):
                self.outputs, state = tf.nn.dynamic_rnn(cell,
                                                        inputs=net,
                                                        initial_state=self.initial_state,
                                                        sequence_length=self.sequence_length)
            self.stateObject = state
            final_state = state
            self.finalLstmState = final_state[Constants.NUM_OF_LSTM_LAYERS - 1].h
        else:
            cell_fw = RnnClassifier.get_stacked_lstm_cells(hidden_dimension=Constants.LSTM_HIDDEN_DIM,
                                                           num_layers=Constants.NUM_OF_LSTM_LAYERS)
            cell_bw = RnnClassifier.get_stacked_lstm_cells(hidden_dimension=Constants.LSTM_HIDDEN_DIM,
                                                           num_layers=Constants.NUM_OF_LSTM_LAYERS)
            # Add dropout to cell output
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)
            # Init states
            self.initial_state_fw = cell_fw.zero_state(self.batch_size, dtype=tf.float32)
            self.initial_state_bw = cell_bw.zero_state(self.batch_size, dtype=tf.float32)
            # Dynamic Bi-LSTM
            with tf.variable_scope('Bi-LSTM'):
                self.outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                      cell_bw,
                                                                      inputs=net,
                                                                      initial_state_fw=self.initial_state_fw,
                                                                      initial_state_bw=self.initial_state_bw,
                                                                      sequence_length=self.sequence_length)
                self.stateObject = state
                final_state_fw = state[0][Constants.NUM_OF_LSTM_LAYERS - 1]
                final_state_bw = state[1][Constants.NUM_OF_LSTM_LAYERS - 1]
                self.finalLstmState = tf.concat([final_state_fw.h, final_state_bw.h], 1)
        self.finalState = self.finalLstmState
        if Constants.USE_ATTENTION_MECHANISM:
            if Constants.USE_BIDIRECTIONAL:
                forward_rnn_outputs = self.outputs[0]
                backward_rnn_outputs = self.outputs[1]
                self.attentionMechanismInput = tf.concat([forward_rnn_outputs, backward_rnn_outputs], axis=2)
            else:
                self.attentionMechanismInput = self.outputs
            with tf.variable_scope('Attention-Model'):
                hidden_state_length = self.attentionMechanismInput.get_shape().as_list()[-1]
                self.contextVector = tf.Variable(tf.random_normal([hidden_state_length], stddev=0.1))
                w = self.contextVector
                H = self.attentionMechanismInput
                M = tf.tanh(H)
                M = tf.reshape(M, [-1, hidden_state_length])
                w = tf.reshape(w, [-1, 1])
                pre_softmax = tf.reshape(tf.matmul(M, w), [-1, self.dataset.maxLength])
                zero_mask = tf.equal(pre_softmax, 0.0)
                replacement_tensor = tf.fill([Constants.BATCH_SIZE, self.dataset.maxLength], -1e100)
                masked_pre_softmax = tf.where(zero_mask, replacement_tensor, pre_softmax)
                self.alpha = tf.nn.softmax(masked_pre_softmax)
                r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                              tf.reshape(self.alpha, [-1, self.dataset.maxLength, 1]))
                r = tf.squeeze(r)
                h_star = tf.tanh(r)
                h_drop = tf.nn.dropout(h_star, self.keep_prob)
                self.finalState = h_drop
                self.temps.append(pre_softmax)
                self.temps.append(zero_mask)
                self.temps.append(masked_pre_softmax)

    def get_softmax_layer(self):
        hidden_layer_size = Constants.LSTM_HIDDEN_DIM
        num_of_classes = self.dataset.get_label_count()
        # Softmax output layer
        with tf.name_scope('softmax'):
            if not Constants.USE_BIDIRECTIONAL:
                softmax_w = tf.get_variable('softmax_w', shape=[hidden_layer_size, num_of_classes], dtype=tf.float32)
            elif Constants.USE_BIDIRECTIONAL:
                softmax_w = tf.get_variable('softmax_w', shape=[2 * hidden_layer_size, num_of_classes],
                                            dtype=tf.float32)
            else:
                raise NotImplementedError()
            softmax_b = tf.get_variable('softmax_b', shape=[num_of_classes], dtype=tf.float32)
            self.l2_loss += tf.nn.l2_loss(softmax_w)
            # self.l2_loss += tf.nn.l2_loss(softmax_b)
            self.logits = tf.matmul(self.finalState, softmax_w) + softmax_b
            predictions = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(predictions, 1, name='predictions')

    def get_loss(self):
        # Loss
        with tf.name_scope('loss'):
            tvars = tf.trainable_variables()
            # L2 regularization for LSTM weights
            for tv in tvars:
                if 'kernel' in tv.name:
                    self.l2_loss += tf.nn.l2_loss(tv)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                    logits=self.logits)
            self.mainLoss = tf.reduce_mean(losses) + Constants.L2_LAMBDA_COEFFICENT * self.l2_loss

    def get_accuracy(self):
        with tf.name_scope('accuracy'):
            self.correctPredictions = tf.equal(self.predictions, self.input_y)
            self.numOfCorrectPredictions = tf.reduce_sum(tf.cast(self.correctPredictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(self.correctPredictions, tf.float32), name='accuracy')

    def get_optimizer(self):
        # Train procedure
        self.globalStep = tf.Variable(0, name='global_step', trainable=False)
        # Learning rate decay
        starter_learning_rate = Constants.INITIAL_LR
        # learning_rate = tf.train.exponential_decay(starter_learning_rate,
        #                                            self.globalStep,
        #                                            GlobalConstants.DECAY_PERIOD_LSTM,
        #                                            GlobalConstants.DECAY_RATE_LSTM,
        #                                            staircase=True)
        self.optimizer = tf.train.AdamOptimizer(starter_learning_rate).minimize(self.mainLoss,
                                                                                global_step=self.globalStep)

    # def experiment(self):
    #     self.corpus.set_current_dataset_type(dataset_type=DatasetType.Training)
    #     self.sess.run(tf.global_variables_initializer())
    #     tf.assign(self.embeddings, self.wordEmbeddings).eval(session=self.sess)
    #     data, labels, lengths, document_ids = \
    #         self.corpus.get_next_training_batch(batch_size=GlobalConstants.BATCH_SIZE)
    #     lengths[0] = 10
    #     lengths[1:] = 15
    #     feed_dict = {self.batch_size: GlobalConstants.BATCH_SIZE,
    #                  self.input_word_codes: data,
    #                  self.input_y: labels,
    #                  self.keep_prob: GlobalConstants.DROPOUT_KEEP_PROB,
    #                  self.sequence_length: lengths}
    #     run_ops = [self.mainLoss, self.outputs, self.finalLstmState, self.stateObject, self.attentionMechanismInput,
    #                self.contextVector, self.alpha, self.finalState, self.temps]
    #     results = self.sess.run(run_ops, feed_dict=feed_dict)
    #     print("X")

    def train(self):
        sess = tf.Session()
        run_id = DbLogger.get_run_id()
        explanation = str(Constants)
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData, col_count=2)
        sess.run(tf.global_variables_initializer())
        # trainable_var_dict = {v.name: v for v in tf.trainable_variables()}
        # saver = tf.train.Saver(trainable_var_dict)
        # tf.assign(self.embeddings, self.wordEmbeddings).eval(session=self.sess)
        losses = []
        iteration = 0
        for epoch_id in range(Constants.TOTAL_EPOCH_COUNT):
            print("*************Epoch {0}*************".format(epoch_id))
            self.dataset.set_current_dataset_type(dataset_type=DatasetTypes.Training)
            while True:
                minibatch = self.dataset.get_next_batch(batch_size=Constants.BATCH_SIZE)
                feed_dict = {self.batch_size: Constants.BATCH_SIZE,
                             self.input: minibatch.sequences,
                             self.input_y: minibatch.labels,
                             self.keep_prob: Constants.DROPOUT_KEEP_PROB,
                             self.sequence_length: minibatch.lengths}
                # sequences, labels, lengths
                run_ops = [self.optimizer, self.mainLoss]
                results = sess.run(run_ops, feed_dict=feed_dict)
                losses.append(results[1])
                iteration += 1
                if iteration % 100 == 0:
                    avg_loss = np.mean(np.array(losses))
                    print("Iteration:{0} Avg. Loss:{1}".format(iteration, avg_loss))
                    losses = []
                if self.dataset.isNewEpoch:
                    print("Original results")
                    training_accuracy, doc_training_accuracy = self.test(dataset_type=DatasetTypes.Training)
                    test_accuracy, doc_test_accuracy = self.test(dataset_type=DatasetTypes.Validation)
                    DbLogger.write_into_table(rows=[(run_id,
                                                     iteration,
                                                     epoch_id,
                                                     training_accuracy,
                                                     test_accuracy,
                                                     test_accuracy,
                                                     0.0,
                                                     0.0,
                                                     "-")],
                                              table=DbLogger.logsTable, col_count=9)
                    break

    def test(self, dataset_type):
        confusion_dict = {}
        batch_size = GlobalConstants.BATCH_SIZE
        if dataset_type == DatasetType.Validation:
            self.corpus.set_current_dataset_type(dataset_type=DatasetType.Validation)
        else:
            self.corpus.set_current_dataset_type(dataset_type=DatasetType.Training)
        total_correct_count = 0
        total_count = 0
        total_correct_document_count = 0
        total_document_count = 0
        document_predictions = {}
        document_correct_labels = {}
        while True:
            data, labels, lengths, document_ids = \
                self.corpus.get_next_training_batch(batch_size=batch_size, wrap_around=False)
            feed_dict = {self.batch_size: batch_size,
                         self.input_word_codes: data,
                         self.input_y: labels,
                         self.keep_prob: 1.0,
                         self.sequence_length: lengths,
                         self.isTrainingFlag: False}
            run_ops = [self.correctPredictions, self.predictions]
            results = self.sess.run(run_ops, feed_dict=feed_dict)
            for sample_id, document_id in enumerate(document_ids.tolist()):
                if lengths[sample_id] == 0:
                    break
                is_sample_correct = results[0][sample_id]
                sample_prediction = results[1][sample_id]
                total_correct_count += is_sample_correct
                total_count += 1
                if document_id not in document_predictions:
                    document_predictions[document_id] = []
                if document_id not in document_correct_labels:
                    document_correct_labels[document_id] = []
                correct_label = labels[sample_id]
                document_predictions[document_id].append(sample_prediction)
                document_correct_labels[document_id].append(correct_label)
            if self.corpus.isNewEpoch:
                break
        accuracy = float(total_correct_count) / float(total_count)
        print("Dataset:{0} Accuracy:{1}".format(dataset_type, accuracy))
        # Check document correctness
        for k, v in document_correct_labels.items():
            label_set = set(v)
            assert len(label_set) == 1
            document_label = list(label_set)[0]
            prediction_list = document_predictions[k]
            tpl = Counter(prediction_list).most_common(1)
            predicted_label = tpl[0][0]
            if predicted_label == document_label:
                total_correct_document_count += 1
            if (document_label, predicted_label) not in confusion_dict:
                confusion_dict[(document_label, predicted_label)] = 0
            confusion_dict[(document_label, predicted_label)] += 1
            total_document_count += 1
        document_wise_accuracy = float(total_correct_document_count) / float(total_document_count)
        print("Dataset:{0} Document-Wise Accuracy:{1}".format(dataset_type, document_wise_accuracy))
        print("Confusion Matrix:")
        print(confusion_dict)
        return accuracy, document_wise_accuracy