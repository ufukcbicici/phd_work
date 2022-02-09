import unittest
import numpy as np
import tensorflow as tf
import time

from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class CigtMaskingAlgorithmTests(unittest.TestCase):

    @unittest.skip
    def test_masking_layer_cnn(self):
        with tf.device("GPU"):
            batch_size = 125
            dim = 64
            momentum = 0.9
            target_val = 5.0
            error_tol_ratio = 0.025
            iter_count = 1000
            route_count = 4
            route_width = dim // 4

            i_x = tf.keras.Input(shape=(14, 14, dim))
            routing_matrix = tf.keras.Input(shape=(route_count,), dtype=tf.int32)
            net = tf.keras.layers.Conv2D(filters=dim,
                                         kernel_size=3,
                                         activation=tf.nn.relu,
                                         strides=1,
                                         padding="same",
                                         use_bias=True,
                                         name="conv")(i_x)
            masking_layer = CigtMaskingLayer()
            net_output = masking_layer([net, routing_matrix])

            model = tf.keras.Model(inputs=[i_x, routing_matrix],
                                   outputs={"masked_output": net_output,
                                            "unmasked_output": net})

            for i in range(iter_count):
                print("Iteration:{0}".format(i))
                input_to_model = np.random.uniform(size=(batch_size, 14, 14, dim))
                input_routing_matrix = np.zeros(shape=(batch_size, route_count), dtype=np.int32)
                route_indices = np.random.randint(low=0, high=route_count, size=(batch_size,))
                input_routing_matrix[np.arange(batch_size), route_indices] = 1
                full_path_indices = np.random.choice(batch_size, replace=False, size=(20,))
                input_routing_matrix[full_path_indices] = np.ones_like(input_routing_matrix[full_path_indices])
                output_dict = model([input_to_model, input_routing_matrix])
                masked_output = output_dict["masked_output"]
                unmasked_output = output_dict["unmasked_output"]
                for sample_id in range(batch_size):
                    for route_id in range(route_count):
                        masked_arr = masked_output[sample_id, ...,
                                     route_id * route_width:(route_id + 1) * route_width]
                        unmasked_arr = unmasked_output[sample_id, ...,
                                       route_id * route_width:(route_id + 1) * route_width]
                        is_route_open = input_routing_matrix[sample_id, route_id]
                        if is_route_open:
                            self.assertTrue(np.array_equal(masked_arr, unmasked_arr))
                        else:
                            self.assertTrue(np.array_equal(masked_arr, np.zeros_like(masked_arr)))

    # @unittest.skip
    def test_masking_layer_dense(self):
        with tf.device("GPU"):
            batch_size = 125
            dim = 128
            iter_count = 1000
            route_count = 4
            route_width = dim // 4

            i_x = tf.keras.Input(shape=(dim, ))
            routing_matrix = tf.keras.Input(shape=(route_count,), dtype=tf.int32)
            net = tf.keras.layers.Dense(dim, activation=tf.nn.relu)(i_x)

            masking_layer = CigtMaskingLayer()
            net_output = masking_layer([net, routing_matrix])

            model = tf.keras.Model(inputs=[i_x, routing_matrix],
                                   outputs={"masked_output": net_output,
                                            "unmasked_output": net})

            for i in range(iter_count):
                print("Iteration:{0}".format(i))
                input_to_model = np.random.uniform(size=(batch_size, dim))
                input_routing_matrix = np.zeros(shape=(batch_size, route_count), dtype=np.int32)
                route_indices = np.random.randint(low=0, high=route_count, size=(batch_size,))
                input_routing_matrix[np.arange(batch_size), route_indices] = 1
                full_path_indices = np.random.choice(batch_size, replace=False, size=(20,))
                input_routing_matrix[full_path_indices] = np.ones_like(
                    input_routing_matrix[full_path_indices])
                output_dict = model([input_to_model, input_routing_matrix])
                masked_output = output_dict["masked_output"]
                unmasked_output = output_dict["unmasked_output"]
                for sample_id in range(batch_size):
                    for route_id in range(route_count):
                        masked_arr = masked_output[sample_id, route_id * route_width:(route_id + 1) * route_width]
                        unmasked_arr = unmasked_output[sample_id, route_id * route_width:(route_id + 1) * route_width]
                        is_route_open = input_routing_matrix[sample_id, route_id]
                        if is_route_open:
                            self.assertTrue(np.array_equal(masked_arr, unmasked_arr))
                        else:
                            self.assertTrue(np.array_equal(masked_arr, np.zeros_like(masked_arr)))


if __name__ == '__main__':
    # gpus = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    unittest.main()
