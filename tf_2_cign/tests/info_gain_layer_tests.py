import time
import unittest

import numpy as np
import tensorflow as tf

from algorithms.info_gain import InfoGainLoss
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class InfoGainLayerTests(unittest.TestCase):

    def test_info_gain_layer(self):
        bs = 125
        dim = 128
        momentum = 0.9
        target_val = 5.0
        classes = 10
        nd = 2
        balance = 2.0
        temp = 1.5
        activation_layer = CignDenseLayer(output_dim=nd, activation=None,
                                          node=None, use_bias=True, name="fc_op_decision")

        # MODEL 1
        ig_layer = InfoGainLayer(class_count=classes)
        # IG with weights
        x_input = tf.keras.Input(shape=(dim,), name="x_input")
        b_ = tf.keras.Input(shape=(), name="balance")
        t_ = tf.keras.Input(shape=(), name="temperature")
        l_ = tf.keras.Input(shape=(), name="labels", dtype=tf.int32)
        w_ = tf.keras.Input(shape=(), name="weight_vector", dtype=tf.int32)

        act = activation_layer(x_input)
        ig_value = ig_layer([act, l_, t_, b_, w_])

        ig_model_1 = tf.keras.Model(inputs=[x_input, t_, l_, w_, b_], outputs={"ig_value": ig_value})

        # MODEL 2
        x_input_2 = tf.keras.Input(shape=(dim,), name="x_input_2")
        t_2 = tf.keras.Input(shape=(), name="t_2")
        l_2 = tf.keras.Input(shape=(), name="l_2", dtype=tf.int32)
        w_2 = tf.keras.Input(shape=(), name="w_2", dtype=tf.int32)

        # Routing temperatures
        masked_x = tf.boolean_mask(x_input_2, tf.cast(w_2, tf.bool))
        masked_labels = tf.boolean_mask(l_2, tf.cast(w_2, tf.bool))
        act_2 = activation_layer(masked_x)
        activations_with_temperature = act_2 / t_2
        p_n_given_x = tf.nn.softmax(activations_with_temperature)
        p_c_given_x = tf.one_hot(masked_labels, classes)
        ig_value_2 = InfoGainLoss.get_loss(p_n_given_x_2d=p_n_given_x, p_c_given_x_2d=p_c_given_x,
                                           balance_coefficient=balance)
        ig_model_2 = tf.keras.Model(inputs=[x_input_2, t_2, l_2, w_2], outputs={"ig_value": ig_value_2})

        times_ig1 = []
        times_ig2 = []
        for i in range(10000):
            x = np.random.uniform(low=-1.0, high=1.0, size=(bs, dim))
            mask_vector = np.random.randint(low=0, high=2, size=(bs,))
            class_labels = np.random.randint(low=0, high=classes, size=(bs,))

            t0 = time.time()
            outputs_dict1 = ig_model_1([x, temp, class_labels, mask_vector, balance], training=True)
            t1 = time.time()
            outputs_dict2 = ig_model_2([x, temp, class_labels, mask_vector], training=True)
            t2 = time.time()
            print("Iter:{0} Time IG1={1} Time IG2={2} IG1={3} IG2={4}".format(
                i, t1 - t0, t2 - t1, outputs_dict1["ig_value"], outputs_dict2["ig_value"]))
            times_ig1.append(t1 - t0)
            times_ig2.append(t2 - t1)
            self.assertTrue(np.allclose(outputs_dict1["ig_value"], outputs_dict2["ig_value"]))
        print("IG 1 Mean Time:{0}".format(np.mean(np.array(times_ig1))))
        print("IG 2 Mean Time:{0}".format(np.mean(np.array(times_ig2))))


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    unittest.main()

    # activation_layer = CignDenseLayer(output_dim=2, activation=None,
    #                                   node=None, use_bias=True, name="fc_op_decision")
    # # MODEL 1
    # ig_layer = InfoGainLayer(class_count=10)
    # # IG with weights
    # x_input = tf.keras.Input(shape=(125,), name="x_input")
    # b_ = tf.keras.Input(shape=(), name="balance")
    # t_ = tf.keras.Input(shape=(), name="temperature")
    # l_ = tf.keras.Input(shape=(), name="labels", dtype=tf.int32)
    # w_ = tf.keras.Input(shape=(), name="weight_vector", dtype=tf.int32)
    #
    # act = activation_layer(x_input)
    #
    # batch_size = tf.shape(act)[0]
    # node_degree = tf.shape(act)[1]
    #
    # joint_distribution = tf.ones(shape=(batch_size, 10, node_degree), dtype=act.dtype)
