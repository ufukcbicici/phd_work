import time
import unittest

import numpy as np
import tensorflow as tf

from algorithms.info_gain import InfoGainLoss
from tf_2_cign.cigt.custom_layers.cigt_gumbel_softmax import CigtGumbelSoftmax
from tf_2_cign.cigt.custom_layers.cigt_gumbel_softmax_decision_layer import CigtGumbelSoftmaxDecisionLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer
from tf_2_cign.utilities.utilities import Utilities


class GumbelSoftmaxTests(unittest.TestCase):

    def test_rounding_property(self):
        sample_count = 1000000
        route_counts = [2, 3, 4, 5, 10]
        temperatures = [0.1, 0.5, 1.0, 5.0, 10.0]
        cartesian_product = Utilities.get_cartesian_product(list_of_lists=[route_counts,
                                                                           temperatures,
                                                                           [True, False]])

        for tpl in cartesian_product:
            route_count = tpl[0]
            temperature = tpl[1]
            is_train = tpl[2]
            print("Gumbel Softmax Testing route_count:{0} temperature:{1} is_train:{2}".format(
                route_count, temperature, is_train))
            gs_layer = CigtGumbelSoftmax()
            logits = np.random.uniform(low=-10.0, high=10.0, size=(125, route_count))
            logits_soft_plus = tf.nn.softplus(logits)
            z_samples = gs_layer([logits_soft_plus, temperature, sample_count], training=is_train)
            logits_soft_plus_normalized = (logits_soft_plus / tf.reduce_sum(logits_soft_plus, axis=1,
                                                                            keepdims=True)).numpy()
            arg_max_indices = tf.argmax(z_samples, axis=1).numpy()
            z_hard = tf.one_hot(arg_max_indices, axis=1, depth=route_count).numpy()

            # This piece of code checks if one hot works correctly.
            # it = np.nditer(arg_max_indices, flags=["multi_index"])
            # for x_ in it:
            #     vec = np.zeros(shape=(route_count,), dtype=z_hard.dtype)
            #     vec[x_] = 1
            #     self.assertTrue(np.array_equal(vec, z_hard[it.multi_index[0], :, it.multi_index[1]]))

            z_distribution = np.mean(z_hard, axis=-1)
            z_max = np.max(z_distribution, axis=1)
            p_max = np.max(logits_soft_plus_normalized, axis=1)
            res = np.allclose(z_max, p_max, rtol=0.01)
            self.assertTrue(res)


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
