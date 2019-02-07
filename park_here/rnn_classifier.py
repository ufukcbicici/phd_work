import tensorflow as tf
import numpy as np


class RnnClassifier:
    def __init__(self, dataset):
        self.dataset = dataset
        self.input_word_codes = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_word_codes')
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

    def get_embeddings(self):
        self.inputs = tf.layers.dense(self.inputs, 256, activation=tf.nn.relu)
        if GlobalConstants.USE_INPUT_DROPOUT:
            self.inputs = tf.nn.dropout(self.inputs, keep_prob=self.keep_prob)

    @staticmethod
    def get_stacked_lstm_cells(hidden_dimension, num_layers):
        cell_list = [tf.contrib.rnn.LSTMCell(hidden_dimension,
                                             forget_bias=1.0,
                                             state_is_tuple=True) for _ in range(num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)
        return cell

    def get_classifier_structure(self):
        num_layers = GlobalConstants.NUM_OF_LSTM_LAYERS
        if not GlobalConstants.USE_BIDIRECTIONAL_LSTM:
            cell = RnnClassifier.get_stacked_lstm_cells(hidden_dimension=GlobalConstants.LSTM_HIDDEN_LAYER_SIZE,
                                                        num_layers=num_layers)
            # Add dropout to cell output
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            self.initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            # Dynamic LSTM
            with tf.variable_scope('LSTM'):
                self.outputs, state = tf.nn.dynamic_rnn(cell,
                                                        inputs=self.inputs,
                                                        initial_state=self.initial_state,
                                                        sequence_length=self.sequence_length)
            self.stateObject = state
            final_state = state
            self.finalLstmState = final_state[num_layers - 1].h
        else:
            cell_fw = RnnClassifier.get_stacked_lstm_cells(hidden_dimension=GlobalConstants.LSTM_HIDDEN_LAYER_SIZE,
                                                           num_layers=num_layers)
            cell_bw = RnnClassifier.get_stacked_lstm_cells(hidden_dimension=GlobalConstants.LSTM_HIDDEN_LAYER_SIZE,
                                                           num_layers=num_layers)
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
                                                                      inputs=self.inputs,
                                                                      initial_state_fw=self.initial_state_fw,
                                                                      initial_state_bw=self.initial_state_bw,
                                                                      sequence_length=self.sequence_length)
                self.stateObject = state
                final_state_fw = state[0][num_layers - 1]
                final_state_bw = state[1][num_layers - 1]
                self.finalLstmState = tf.concat([final_state_fw.h, final_state_bw.h], 1)
        self.finalState = self.finalLstmState
        if GlobalConstants.USE_ATTENTION_MECHANISM:
            if GlobalConstants.USE_BIDIRECTIONAL_LSTM:
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
                pre_softmax = tf.reshape(tf.matmul(M, w), [-1, GlobalConstants.MAX_SEQUENCE_LENGTH])
                zero_mask = tf.equal(pre_softmax, 0.0)
                replacement_tensor = tf.fill([GlobalConstants.BATCH_SIZE, GlobalConstants.MAX_SEQUENCE_LENGTH], -1e100)
                masked_pre_softmax = tf.where(zero_mask, replacement_tensor, pre_softmax)
                self.alpha = tf.nn.softmax(masked_pre_softmax)
                r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                              tf.reshape(self.alpha, [-1, GlobalConstants.MAX_SEQUENCE_LENGTH, 1]))
                r = tf.squeeze(r)
                h_star = tf.tanh(r)
                h_drop = tf.nn.dropout(h_star, self.keep_prob)
                self.finalState = h_drop
                self.temps.append(pre_softmax)
                self.temps.append(zero_mask)
                self.temps.append(masked_pre_softmax)

    def get_softmax_layer(self):
        hidden_layer_size = GlobalConstants.LSTM_HIDDEN_LAYER_SIZE
        num_of_classes = self.corpus.get_num_of_classes()
        # Softmax output layer
        with tf.name_scope('softmax'):
            if not GlobalConstants.USE_BIDIRECTIONAL_LSTM:
                softmax_w = tf.get_variable('softmax_w', shape=[hidden_layer_size, num_of_classes], dtype=tf.float32)
            elif GlobalConstants.USE_BIDIRECTIONAL_LSTM:
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

    def experiment(self):
        self.corpus.set_current_dataset_type(dataset_type=DatasetType.Training)
        self.sess.run(tf.global_variables_initializer())
        tf.assign(self.embeddings, self.wordEmbeddings).eval(session=self.sess)
        data, labels, lengths, document_ids = \
            self.corpus.get_next_training_batch(batch_size=GlobalConstants.BATCH_SIZE)
        lengths[0] = 10
        lengths[1:] = 15
        feed_dict = {self.batch_size: GlobalConstants.BATCH_SIZE,
                     self.input_word_codes: data,
                     self.input_y: labels,
                     self.keep_prob: GlobalConstants.DROPOUT_KEEP_PROB,
                     self.sequence_length: lengths}
        run_ops = [self.mainLoss, self.outputs, self.finalLstmState, self.stateObject, self.attentionMechanismInput,
                   self.contextVector, self.alpha, self.finalState, self.temps]
        results = self.sess.run(run_ops, feed_dict=feed_dict)
        print("X")

    def train(self):
        run_id = DbLogger.get_run_id()
        explanation = RnnClassifier.get_explanation()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData, col_count=2)
        self.sess.run(tf.global_variables_initializer())
        trainable_var_dict = {v.name: v for v in tf.trainable_variables()}
        saver = tf.train.Saver(trainable_var_dict)
        tf.assign(self.embeddings, self.wordEmbeddings).eval(session=self.sess)
        losses = []
        iteration = 0
        for epoch_id in range(GlobalConstants.EPOCH_COUNT_CLASSIFIER):
            print("*************Epoch {0}*************".format(epoch_id))
            self.corpus.set_current_dataset_type(dataset_type=DatasetType.Training)
            while True:
                data, labels, lengths, document_ids = \
                    self.corpus.get_next_training_batch(batch_size=GlobalConstants.BATCH_SIZE)
                feed_dict = {self.batch_size: GlobalConstants.BATCH_SIZE,
                             self.input_word_codes: data,
                             self.input_y: labels,
                             self.keep_prob: GlobalConstants.DROPOUT_KEEP_PROB,
                             self.sequence_length: lengths}
                run_ops = [self.optimizer, self.mainLoss]
                results = self.sess.run(run_ops, feed_dict=feed_dict)
                losses.append(results[1])
                iteration += 1
                if iteration % 100 == 0:
                    avg_loss = np.mean(np.array(losses))
                    print("Iteration:{0} Avg. Loss:{1}".format(iteration, avg_loss))
                    losses = []
                if self.corpus.isNewEpoch:
                    # Save the model
                    saver.save(self.sess, "LSTM_Models//lstm{0}_epoch{1}.ckpt".format(run_id, epoch_id))
                    print("Original results")
                    training_accuracy, doc_training_accuracy = self.test(dataset_type=DatasetType.Training)
                    test_accuracy, doc_test_accuracy = self.test(dataset_type=DatasetType.Validation)
                    print("Unified results")
                    # training_accuracy_unified, doc_training_accuracy_unified = \
                    #     self.test_compare_with_set(dataset_type=DatasetType.Training)
                    # test_accuracy_unified, doc_test_accuracy_unified = \
                    #     self.test_compare_with_set(dataset_type=DatasetType.Validation)
                    DbLogger.write_into_table(rows=[(run_id, epoch_id, training_accuracy, test_accuracy,
                                                     doc_training_accuracy, doc_test_accuracy)],
                                              table=DbLogger.logsTable, col_count=6)
                    # DbLogger.write_into_table(rows=[(run_id, epoch_id, training_accuracy_unified, test_accuracy_unified,
                    #                                  doc_training_accuracy_unified, doc_test_accuracy_unified)],
                    #                           table="logs_table_unified", col_count=6)
                    break
