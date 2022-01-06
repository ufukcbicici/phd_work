import unittest
import numpy as np
import tensorflow as tf
import time

from algorithms.info_gain import InfoGainLoss
from auxillary.db_logger import DbLogger
from tf_2_cign.custom_layers.cign_binary_action_generator_layer import CignBinaryActionGeneratorLayer
from tf_2_cign.custom_layers.cign_binary_action_result_generator_layer import CignBinaryActionResultGeneratorLayer
from tf_2_cign.custom_layers.cign_binary_rl_routing_layer import CignBinaryRlRoutingLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer
from tf_2_cign.custom_layers.masked_batch_norm import MaskedBatchNormalization
from tf_2_cign.custom_layers.weighted_batch_norm import WeightedBatchNormalization
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.fashion_net.fashion_cign_binary_rl import FashionRlBinaryRouting
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from collections import Counter


class WeightedBatchNormTests(unittest.TestCase):

    def batch_norm_test(self, model, iter_count, batch_size, error_tol_ratio, batch_dim):
        wb_population_mean = [v for v in model.variables if "weighted" in v.name and "popMean" in v.name]
        wb_population_var = [v for v in model.variables if "weighted" in v.name and "popVar" in v.name]
        wb_gamma = [v for v in model.variables if "weighted" in v.name and "gamma" in v.name]
        wb_beta = [v for v in model.variables if "weighted" in v.name and "beta" in v.name]

        assert len(wb_population_mean) == 1 and \
               len(wb_population_var) == 1 and len(wb_gamma) == 1 and len(wb_beta) == 1
        wb_population_mean = wb_population_mean[0]
        wb_population_var = wb_population_var[0]
        wb_gamma = wb_gamma[0]
        wb_beta = wb_beta[0]

        errors_dict = dict()
        optimizer = tf.keras.optimizers.Adam()
        loss_tracker = tf.keras.metrics.Mean(name="total_loss")

        for i in range(iter_count):
            # x1 = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 32, 32, dim))
            x1 = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, *batch_dim))
            mask_vector = np.random.randint(low=0, high=2, size=(batch_size,))

            if i % 500 == 0:
                print("Iteration {0}".format(i))
            with tf.GradientTape() as tape:
                outputs_dict = model([x1, mask_vector], training=True)
                loss_tracker.update_state(values=outputs_dict["total_loss"])
            if i >= iter_count // 2:
                # Assert that both masked batch norm and weighted batch norms are equal.
                if not np.allclose(outputs_dict["norm_result_mb"].numpy(),
                                   outputs_dict["norm_result_wb"].numpy()):
                    errors_dict[("norm_results", i)] = (outputs_dict["norm_result_mb"].numpy(),
                                                        outputs_dict["norm_result_wb"].numpy())
                # Compare population means
                if not np.allclose(model.variables[4].numpy(), model.variables[9].numpy()):
                    errors_dict[("pop_means", i)] = (model.variables[4].numpy(), model.variables[9].numpy())

                # Compare population variances
                if not np.allclose(model.variables[5].numpy(), model.variables[10].numpy()):
                    errors_dict[("pop_vars", i)] = (model.variables[5].numpy(), model.variables[10].numpy())
            grads = tape.gradient(outputs_dict["total_loss"], model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print("{0} Loss:{1}".format(i, loss_tracker.result().numpy()))

            results = model([x1, mask_vector], training=False)
            # Check that population statistics are correctly used during the inference phase.
            net_x = results["net"]
            normed_x = tf.nn.batch_normalization(x=net_x,
                                                 mean=wb_population_mean,
                                                 variance=wb_population_var,
                                                 offset=wb_beta,
                                                 scale=wb_gamma,
                                                 variance_epsilon=1e-5)
            self.assertTrue(np.allclose(normed_x, results["norm_result_wb"]))

        num_of_errors_in_norm_results = 0
        num_of_errors_in_pop_means = 0
        num_errors_in_pop_vars = 0
        for tpl in errors_dict.keys():
            if tpl[0] == "norm_results":
                num_of_errors_in_norm_results += 1
            elif tpl[0] == "pop_means":
                num_of_errors_in_pop_means += 1
            elif tpl[0] == "pop_vars":
                num_errors_in_pop_vars += 1
            else:
                raise NotImplementedError()
        print("num_of_errors_in_norm_results / iter_count={0}".format(num_of_errors_in_norm_results / iter_count))
        print("num_of_errors_in_pop_means / iter_count={0}".format(num_of_errors_in_pop_means / iter_count))
        print("num_errors_in_pop_vars / iter_count={0}".format(num_errors_in_pop_vars / iter_count))

        self.assertTrue(num_of_errors_in_norm_results / iter_count < error_tol_ratio)
        self.assertTrue(num_of_errors_in_pop_means / iter_count < error_tol_ratio)
        self.assertTrue(num_errors_in_pop_vars / iter_count < error_tol_ratio)

    # @unittest.skip
    def test_weighted_batch_norm_layer_with_conv(self):
        with tf.device("GPU"):
            batch_size = 125
            dim = 16
            momentum = 0.9
            target_val = 5.0
            error_tol_ratio = 0.025
            iter_count = 20000

            # CNN Output
            i_x = tf.keras.Input(shape=(32, 32, dim))

            mb_norm = MaskedBatchNormalization(momentum=momentum)
            wb_norm = WeightedBatchNormalization(momentum=momentum)

            mask_vector_tf = tf.keras.Input(shape=())

            # CNN Output
            net = tf.keras.layers.Conv2D(filters=16,
                                         kernel_size=3,
                                         activation=tf.nn.relu,
                                         strides=1,
                                         padding="same",
                                         use_bias=True,
                                         name="conv")(i_x)

            masked_net = tf.boolean_mask(net, mask_vector_tf)
            norm_result_mb = mb_norm([net, masked_net])
            norm_result_wb = wb_norm([net, mask_vector_tf])

            # mb_dense = tf.keras.layers.Flatten()(norm_result_mb)
            # wb_dense = tf.keras.layers.Flatten()(norm_result_wb)

            mb_dense = tf.keras.layers.GlobalAveragePooling2D()(norm_result_mb)
            wb_dense = tf.keras.layers.GlobalAveragePooling2D()(norm_result_wb)

            y_hat_mb = tf.reduce_mean(mb_dense, axis=-1)
            y_hat_wb = tf.reduce_mean(wb_dense, axis=-1)

            y_gt = target_val * tf.ones_like(y_hat_mb)

            reconstruction_loss_mb = tf.keras.losses.mean_squared_error(y_gt, y_hat_mb)
            reconstruction_loss_wb = tf.keras.losses.mean_squared_error(y_gt, y_hat_wb)
            total_loss = 0.5 * (reconstruction_loss_mb + reconstruction_loss_wb)
            model = tf.keras.Model(inputs=[i_x, mask_vector_tf],
                                   outputs={"masked_net": masked_net,
                                            "net": net,
                                            "norm_result_mb": norm_result_mb,
                                            "norm_result_wb": norm_result_wb,
                                            "y_hat_mb": y_hat_mb,
                                            "y_hat_wb": y_hat_wb,
                                            "y_gt": y_gt,
                                            "total_loss": total_loss})
            self.batch_norm_test(model=model,
                                 iter_count=iter_count,
                                 batch_size=batch_size,
                                 error_tol_ratio=error_tol_ratio,
                                 batch_dim=[32, 32, dim])

    # @unittest.skip
    def test_weighted_batch_norm_layer_with_dense(self):
        with tf.device("GPU"):
            batch_size = 125
            dim = 64
            momentum = 0.9
            target_val = 5.0
            error_tol_ratio = 0.025
            iter_count = 20000

            i_x = tf.keras.Input(shape=(dim,))

            mb_norm = MaskedBatchNormalization(momentum=momentum)
            wb_norm = WeightedBatchNormalization(momentum=momentum)

            mask_vector_tf = tf.keras.Input(shape=())

            # Dense Output
            net = tf.keras.layers.Dense(dim / 4, activation=tf.nn.relu)(i_x)

            masked_net = tf.boolean_mask(net, mask_vector_tf)
            mb_dense = mb_norm([net, masked_net])
            wb_dense = wb_norm([net, mask_vector_tf])

            y_hat_mb = tf.reduce_mean(mb_dense, axis=-1)
            y_hat_wb = tf.reduce_mean(wb_dense, axis=-1)

            y_gt = target_val * tf.ones_like(y_hat_mb)

            reconstruction_loss_mb = tf.keras.losses.mean_squared_error(y_gt, y_hat_mb)
            reconstruction_loss_wb = tf.keras.losses.mean_squared_error(y_gt, y_hat_wb)
            total_loss = 0.5 * (reconstruction_loss_mb + reconstruction_loss_wb)
            model = tf.keras.Model(inputs=[i_x, mask_vector_tf],
                                   outputs={"masked_net": masked_net,
                                            "net": net,
                                            "norm_result_mb": mb_dense,
                                            "norm_result_wb": wb_dense,
                                            "y_hat_mb": y_hat_mb,
                                            "y_hat_wb": y_hat_wb,
                                            "y_gt": y_gt,
                                            "total_loss": total_loss})
            self.batch_norm_test(model=model,
                                 iter_count=iter_count,
                                 batch_size=batch_size,
                                 error_tol_ratio=error_tol_ratio,
                                 batch_dim=[dim])


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    unittest.main()
