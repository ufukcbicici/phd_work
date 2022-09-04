from collections import deque

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_2_cign.utilities.utilities import Utilities


class EntropyVarianceCalculator(tf.keras.Model):
    def __init__(self, routing_activations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = tf.Variable(tf.ones(shape=(), dtype=tf.float64), trainable=True)
        # self.routingActivations = tf.convert_to_tensor(routing_activations)

    def call(self, inputs, **kwargs):
        routing_activations = tf.cast(inputs, dtype=tf.float64)
        routing_arrs_for_block_tempered = routing_activations / self.temperature
        routing_probs = tf.nn.softmax(routing_arrs_for_block_tempered)
        log_prob = tf.math.log(routing_probs + Utilities.INFO_GAIN_LOG_EPSILON)
        prob_log_prob = routing_probs * log_prob
        entropies = -1.0 * tf.reduce_sum(prob_log_prob, axis=1)
        variance = tf.math.reduce_variance(entropies)
        return -variance


class SoftmaxTemperatureOptimizer(object):
    def __init__(self, multi_path_object):
        self.multiPathObject = multi_path_object
        self.maxEntropies = []

    def run(self, block_id):
        # routing_arrs_for_block = []
        # n_past_decisions = sum(self.multiPathObject.pathCounts[1:][:block_id])
        # for k, v in self.multiPathObject.past_decisions_routing_activations_dict.items():
        #     if len(k) == n_past_decisions:
        #         routing_arrs_for_block.append(v)
        activations_dict = self.multiPathObject.get_routing_activations_for_block(block_id)
        routing_arrs_for_block = list(activations_dict.values())
        routing_arrs_for_block = np.concatenate(routing_arrs_for_block, axis=0).astype(np.float64)
        # Prepare a keras model
        entropy_variance_calculator = EntropyVarianceCalculator(routing_activations=routing_arrs_for_block)
        # A basic RPROP scheme.
        alpha = 1.2
        beta = 0.5
        lr = 0.001
        min_lr = 1e-10
        loss_history = deque(maxlen=10000)
        grad_history = deque(maxlen=10000)
        for iteration_id in range(1000000):
            with tf.GradientTape() as tape:
                entropies_variance = entropy_variance_calculator(routing_arrs_for_block,
                                                                 training=True)
            temperature_grad = tape.gradient(entropies_variance,
                                             entropy_variance_calculator.temperature)
            print("X")
            curr_loss = entropies_variance
            delta = temperature_grad
            curr_temperature = entropy_variance_calculator.temperature.numpy()
            loss_history.append(curr_loss.numpy())
            grad_history.append(delta.numpy())
            print("curr_loss={0}".format(curr_loss.numpy()))
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
            entropy_variance_calculator.temperature.assign(value=new_temperature)
            if lr == min_lr or np.allclose(new_temperature, curr_temperature):
                break
        return entropy_variance_calculator.temperature.numpy()
