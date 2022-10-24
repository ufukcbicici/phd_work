import unittest

import numpy as np
import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_batch_normalization import CigtBatchNormalization
from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer
from tf_2_cign.cigt.custom_layers.cigt_route_averaging_layer import CigtRouteAveragingLayer

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class CigtRouteAveragingTests(unittest.TestCase):
    BATCH_SIZE = 125
    ROUTE_COUNT = None
    ABSOLUTE_ERROR_TOLERANCE = 1e-3
    RELATIVE_ERROR_TOLERANCE = 1e-5
    ITER_COUNT = 1000
    CONV_DIM = 16
    DENSE_DIM = 128
    BATCH_DIM = None
    ROUTE_WIDTH = None
    CONV_FILTER_COUNT = 32
    DENSE_LATENT_DIM = 64
    LARGEST_DIFF = 0.0
    LARGEST_DIFF_PAIR = None

    def create_mock_input(self, batch_dim, batch_size, route_count, matrix_type, ratio_of_all_one_routes=0.2):
        x = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, *batch_dim))
        if matrix_type == "one_hot":
            # Create mock routing matrix
            open_route_indices = np.random.randint(low=0, high=route_count, size=(batch_size,))
            route_matrix = np.zeros(shape=(batch_size, route_count), dtype=np.int32)
            route_matrix[np.arange(batch_size), open_route_indices] = 1
            # Create fully selected routes
            fully_route_indices = np.random.choice(batch_size, int(batch_size * ratio_of_all_one_routes), replace=False)
            route_matrix[fully_route_indices] = np.ones_like(route_matrix[fully_route_indices])
        else:
            route_activations = np.random.uniform(size=(batch_size, route_count))
            normalizing_constants = np.sum(route_activations, axis=1)
            route_matrix = route_activations * np.reciprocal(normalizing_constants)[:, np.newaxis]
        return x, route_matrix

    def build_test_model(self, input_shape, model_type):
        i_x = tf.keras.Input(shape=input_shape)
        routing_matrix = tf.keras.Input(shape=(self.ROUTE_COUNT,))

        # CNN Output
        if model_type == "conv":
            op_layer_output = tf.keras.layers.Conv2D(filters=self.CONV_FILTER_COUNT,
                                                     kernel_size=3,
                                                     activation=tf.nn.relu,
                                                     strides=1,
                                                     padding="same",
                                                     use_bias=True,
                                                     name="conv")(i_x)
        else:
            op_layer_output = tf.keras.layers.Dense(self.DENSE_LATENT_DIM, activation=tf.nn.relu)(i_x)
        self.ROUTE_WIDTH = op_layer_output.shape[-1] // self.ROUTE_COUNT
        cigt_route_averaging_layer = CigtRouteAveragingLayer()
        averaged_array = cigt_route_averaging_layer([op_layer_output, routing_matrix])
        model = tf.keras.Model(inputs=[i_x, routing_matrix],
                               outputs={"op_layer_output": op_layer_output, "averaged_array": averaged_array})
        return model

    def test_cigt_route_averaging_layer(self):
        with tf.device("GPU"):
            # momentum = 0.9
            # epsilon = 1e-5
            # target_val = 5.0
            iter_count = self.ITER_COUNT

            for model_type in ["conv", "dense"]:
                for num_of_routes in [4, 1]:
                    for matrix_type in ["one_hot", "probability"]:
                        self.ROUTE_COUNT = num_of_routes
                        if model_type == "conv":
                            self.BATCH_DIM = [32, 32, self.CONV_DIM]
                        else:
                            self.BATCH_DIM = [self.DENSE_DIM]

                        model = self.build_test_model(model_type=model_type, input_shape=self.BATCH_DIM)

                        for i in range(iter_count):
                            print("Model Type:{0} Num of Routes:{1} Matrix Type:{2} Iteration:{3}".format(
                                model_type, num_of_routes, matrix_type, i))
                            x, route_matrix = self.create_mock_input(batch_dim=self.BATCH_DIM,
                                                                     batch_size=self.BATCH_SIZE,
                                                                     route_count=self.ROUTE_COUNT,
                                                                     matrix_type=matrix_type)
                            output_dict = model([x, route_matrix], training=True)
                            # Check if the route averaging works as intended.
                            averaged_array = output_dict["averaged_array"]
                            op_layer_output = output_dict["op_layer_output"]
                            coefficients = np.reciprocal(np.sum(route_matrix, axis=1, dtype=np.float32))
                            for i in range(len(x.shape) - 1):
                                coefficients = tf.expand_dims(coefficients, axis=1)

                            route_parts = []
                            for rid in range(self.ROUTE_COUNT):
                                x_part = op_layer_output[..., rid * self.ROUTE_WIDTH:(rid + 1) * self.ROUTE_WIDTH]
                                route_parts.append(x_part)
                            accum_arr = np.zeros_like(route_parts[rid])
                            for route_part in route_parts:
                                accum_arr += route_part
                            averaged_array_np = coefficients * accum_arr
                            self.assertTrue(np.allclose(averaged_array_np, averaged_array.numpy()))


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # unittest.main()
