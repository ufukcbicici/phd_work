import numpy as np
import tensorflow as tf
import os
from collections import deque
import re


class NetworkCalibrationWithTemperatureScaling:
    def __init__(self, logits, labels):
        self.logitsArr = logits
        self.labelsArr = labels
        # assert logits.shape == labels.shape
        self.logitsTf = tf.placeholder(name="logitsArr", shape=[None, self.logitsArr.shape[1]], dtype=tf.float32)
        self.labelsTf = tf.placeholder(name="labelsArr", shape=[None], dtype=tf.int32)
        self.temperature = tf.Variable(name="temperature", dtype=tf.float32, initial_value=tf.constant(1.0))
        self.newTemperatureValue = tf.placeholder(name="newTemperatureValue", dtype=tf.float32)
        self.temperatureAssignOp = tf.assign(self.temperature, self.newTemperatureValue)
        self.scaledLogits = None
        self.ceLossTensor = None
        self.ceLoss = None
        self.temperatureGrad = None
        self.standardPredictions = None
        self.calibratedPredictions = None
        self.sess = tf.Session()
        self.build_network()

    def build_network(self):
        self.scaledLogits = self.logitsTf / self.temperature
        self.ceLossTensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labelsTf,
                                                                           logits=self.scaledLogits)
        self.ceLoss = tf.reduce_mean(self.ceLossTensor)
        self.temperatureGrad = tf.gradients(self.ceLoss, self.temperature)
        self.standardPredictions = tf.nn.softmax(tf.identity(self.logitsTf))
        self.calibratedPredictions = tf.nn.softmax(self.scaledLogits)

    # A basic RPROP scheme. This would be best solved with a convex optimizer, since it is a convex
    # optimization problem of a single variable!
    def train(self, **kwargs):
        alpha = 1.2
        beta = 0.5
        lr = 0.001
        min_lr = 1e-10
        loss_history = deque(maxlen=10000)
        grad_history = deque(maxlen=10000)
        self.sess.run(tf.global_variables_initializer())
        for iteration_id in range(1000000):
            results = self.sess.run([self.ceLoss, self.temperatureGrad, self.temperature],
                                    feed_dict={self.logitsTf: self.logitsArr, self.labelsTf: self.labelsArr})
            curr_loss = results[0]
            delta = results[1][0]
            curr_temperature = results[2]
            loss_history.append(curr_loss)
            grad_history.append(delta)
            print("curr_loss={0}".format(curr_loss))
            print("curr_temperature={0}".format(curr_temperature))
            if len(grad_history) > 1:
                sgn_t = grad_history[-1] / abs(grad_history[-1])
                sgn_t_minus_1 = grad_history[-2] / abs(grad_history[-2])
                if (sgn_t * sgn_t_minus_1) > 0:
                    lr = min(lr * alpha, 1.0)
                elif (sgn_t * sgn_t_minus_1) < 0:
                    lr = max(lr * beta, min_lr)
                else:
                    return curr_temperature
            new_temperature = curr_temperature - lr * (delta / abs(delta))
            self.sess.run([self.temperatureAssignOp], feed_dict={self.newTemperatureValue: new_temperature})
            if lr == min_lr or np.allclose(new_temperature, curr_temperature):
                break
        return self.sess.run([self.temperature])[0]

    def draw_reliability_diagram(self, bin_count=10, **kwargs):
        # Obtain predictions
        temperature = kwargs["temperature"]
        true_labels = self.labelsArr
        standard_predictions = self.sess.run([self.standardPredictions], feed_dict={self.logitsTf: self.logitsArr})[0]
        calibrated_predictions = self.sess.run([self.calibratedPredictions],
                                               feed_dict={self.logitsTf: self.logitsArr,
                                                          self.temperature: temperature})[0]
        bin_length = 1.0 / float(bin_count)
        for predictions in [standard_predictions, calibrated_predictions]:
            # Accuracy:
            predicted_labels = np.argmax(predictions, axis=1)
            prediction_probabilities = np.max(predictions, axis=1)
            truth_vector = predicted_labels == true_labels
            accuracy = np.sum(truth_vector) / self.labelsArr.shape[0]
            print("accuracy={0}".format(accuracy))
            assert predicted_labels.shape[0] == prediction_probabilities.shape[0]
            prediction_bins = [[] for _ in range(bin_count)]
            accuracy_bins = [[] for _ in range(bin_count)]
            for idx in range(predictions.shape[0]):
                p = prediction_probabilities[idx]
                l = true_labels[idx]
                l_hat = predicted_labels[idx]
                p_bin = np.asscalar(np.clip(p / bin_length, a_min=0, a_max=bin_count - 1).astype(np.int32))
                prediction_bins[p_bin].append(p)
                accuracy_bins[p_bin].append(float(l == l_hat))
            prediction_list = np.array([np.mean(np.array(arr)) for arr in prediction_bins])
            accuracy_list = np.array([np.mean(np.array(arr)) for arr in accuracy_bins])
            dif_arr = np.abs(accuracy_list - prediction_list)
            bin_sizes = np.array([len(arr) for arr in prediction_bins])
            bin_sizes = bin_sizes[np.logical_not((np.isnan(dif_arr)))]
            weights = bin_sizes / np.sum(bin_sizes)
            dif_arr = dif_arr[np.logical_not((np.isnan(dif_arr)))]
            estimated_calibration_error = np.sum(weights * dif_arr)
            print("estimated_calibration_error={0}".format(estimated_calibration_error))

    @staticmethod
    def read_test_result(file_name):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.abspath(os.path.join(curr_path, file_name))
        logits_arr = []
        labels_arr = []
        with open(file_path) as output_file:
            lines = output_file.readlines()
            for line in lines:
                logits_str = line[line.index("[") + 1:line.index("]")]
                logits_str_list = np.array([float(sub_s) for sub_s in logits_str.split(",")])
                logits_arr.append(logits_str_list)
                line = line[(line.index("]") + 1):]
                labels_str = line[line.index("[") + 1:line.index("]")]
                labels_str_list = np.array([int(sub_s) for sub_s in labels_str.split(",")])
                labels_arr.append(labels_str_list)
        logits_arr = np.stack(logits_arr, axis=0)
        labels_arr = np.stack(labels_arr, axis=0)
        return logits_arr, labels_arr

    @staticmethod
    def read_data(gt_indices_file_path, pred_indices_file_path, logits_file_path):
        # Logits
        logits_arr = []
        with open(logits_file_path) as logits_file:
            lines = logits_file.readlines()
            for line in lines:
                logits = np.array([float(logit) for logit in line.split(" ")])
                logits_arr.append(logits)
        logits_arr = np.stack(logits_arr, axis=0)
        indices_dict = {}
        for i_type, path_type in zip(["gt", "pred"], [gt_indices_file_path, pred_indices_file_path]):
            idx_arr = []
            with open(path_type) as idx_file:
                lines = idx_file.readlines()
                for line in lines:
                    index = float(line)
                    idx_arr.append(index)
                idx_arr = np.array(idx_arr).astype(np.int32)
                indices_dict[i_type] = idx_arr
        assert np.array_equal(np.argmax(logits_arr, axis=-1), indices_dict["pred"])
        return logits_arr, indices_dict["gt"]
