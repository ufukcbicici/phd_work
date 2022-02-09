import unittest

import numpy as np
import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_batch_normalization import CigtBatchNormalization

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class CigtBatchNormTests(unittest.TestCase):
    BATCH_SIZE = 125
    ROUTE_COUNT = None
    ERROR_TOL_RATIO = 0.0001
    ITER_COUNT = 1000
    CONV_DIM = 16
    DENSE_DIM = 128
    BATCH_DIM = None

    def check_cigt_batch_norm_correctness(self, model, cigt_batch_norm_output, net_output, route_matrix):
        # Manually calculate the batch normalization
        batch_size = net_output.shape[0]
        channel_count = net_output.shape[-1]
        channel_width = net_output.shape[-1] // route_matrix.shape[-1]
        batch_entries = []
        for channel_id in range(channel_count):
            batch_entries.append([])
        batch_mean = np.zeros(shape=(net_output.shape[-1],))
        # batch_entry_summations = np.zeros(shape=(batch_size,))
        # batch_entry_counts = np.zeros(shape=(batch_size,))
        batch_var = np.zeros(shape=(net_output.shape[-1],))

        for sample_id in range(batch_size):
            for channel_id in range(channel_count):
                route_index = channel_id // channel_width
                is_route_open = route_matrix[sample_id, route_index]
                sample_channel_feature = net_output[sample_id, ..., channel_id]
                if is_route_open == 0:
                    continue
                flat_feature = np.reshape(sample_channel_feature, -1)
                batch_entries[channel_id].append(flat_feature)

        for channel_id in range(channel_count):
            all_entries = np.concatenate(batch_entries[channel_id])
            batch_mean[channel_id] = np.mean(all_entries)
            batch_var[channel_id] = np.var(all_entries)

        beta = [v for v in model.trainable_variables if "beta" in v.name and "cigt" in v.name]
        assert len(beta) == 1
        beta = beta[0]

        gamma = [v for v in model.trainable_variables if "gamma" in v.name and "cigt" in v.name]
        assert len(gamma) == 1
        gamma = gamma[0]

        normed_x = tf.nn.batch_normalization(x=net_output,
                                             mean=tf.cast(batch_mean, dtype=tf.float32),
                                             variance=tf.cast(batch_var, dtype=tf.float32),
                                             offset=beta,
                                             scale=gamma,
                                             variance_epsilon=1e-5)

        diff_matrix = np.abs(cigt_batch_norm_output.numpy() - normed_x.numpy())
        self.assertTrue(np.max(diff_matrix) < self.ERROR_TOL_RATIO)

    # def check_single_channel_batch_norm(self, model, cigt_batch_norm_output,
    #                                       net_output, route_matrix, error_tol_ratio):

    def batch_norm_test(self, model, batch_dim):
        wb_population_mean = [v for v in model.variables if "popMean" in v.name and "cigt" in v.name]
        wb_population_var = [v for v in model.variables if "popVar" in v.name and "cigt" in v.name]
        wb_gamma = [v for v in model.variables if "gamma" in v.name and "cigt" in v.name]
        wb_beta = [v for v in model.variables if "beta" in v.name and "cigt" in v.name]

        assert len(wb_population_mean) == 1 and \
               len(wb_population_var) == 1 and len(wb_gamma) == 1 and len(wb_beta) == 1
        wb_population_mean = wb_population_mean[0]
        wb_population_var = wb_population_var[0]
        wb_gamma = wb_gamma[0]
        wb_beta = wb_beta[0]

        errors_dict = dict()
        optimizer = tf.keras.optimizers.Adam()
        loss_tracker = tf.keras.metrics.Mean(name="loss")

        iter_count = self.ITER_COUNT
        batch_size = self.BATCH_SIZE
        route_count = self.ROUTE_COUNT
        error_tol_ratio = self.ERROR_TOL_RATIO

        for i in range(iter_count):
            # Create mock input
            x = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, *batch_dim))
            # Create mock routing matrix
            open_route_indices = np.random.randint(low=0, high=route_count, size=(batch_size,))
            route_matrix = np.zeros(shape=(batch_size, route_count), dtype=np.int32)
            route_matrix[np.arange(batch_size), open_route_indices] = 1
            # Create fully selected routes
            fully_route_indices = np.random.choice(batch_size, 25, replace=False)
            route_matrix[fully_route_indices] = np.ones_like(route_matrix[fully_route_indices])

            with tf.GradientTape() as tape:
                output_dict = model([x, route_matrix], training=True)
                loss_tracker.update_state(values=output_dict["loss"])
                self.check_cigt_batch_norm_correctness(model=model,
                                                       cigt_batch_norm_output=output_dict["normed_net"],
                                                       net_output=output_dict["net_without_norm"],
                                                       route_matrix=route_matrix)

            grads = tape.gradient(output_dict["loss"], model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print("{0} Loss:{1}".format(i, loss_tracker.result().numpy()))

            # (self, model, cigt_batch_norm_output, net_output, route_matrix)

            # Ensure that inference mode also works correctly.
            inference_output = model([x, route_matrix], training=False)
            # Check that population statistics are correctly used during the inference phase.
            net_x = inference_output["net_without_norm"]
            normed_custom_x = inference_output["normed_net"]
            normed_x = tf.nn.batch_normalization(x=net_x,
                                                 mean=wb_population_mean,
                                                 variance=wb_population_var,
                                                 offset=wb_beta,
                                                 scale=wb_gamma,
                                                 variance_epsilon=1e-5)
            self.assertTrue(np.allclose(normed_custom_x, normed_x))

    def create_mock_input(self, batch_dim, batch_size, route_count, ratio_of_all_one_routes=0.2):
        x = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, *batch_dim))
        # Create mock routing matrix
        open_route_indices = np.random.randint(low=0, high=route_count, size=(batch_size,))
        route_matrix = np.zeros(shape=(batch_size, route_count), dtype=np.int32)
        route_matrix[np.arange(batch_size), open_route_indices] = 1
        # Create fully selected routes
        fully_route_indices = np.random.choice(batch_size, int(batch_size * ratio_of_all_one_routes), replace=False)
        route_matrix[fully_route_indices] = np.ones_like(route_matrix[fully_route_indices])
        return x, route_matrix

    # @unittest.skip
    def test_cigt_batch_norm_layer_with_conv(self):
        with tf.device("GPU"):
            momentum = 0.9
            target_val = 5.0

            for num_of_routes in [4, 1]:
                self.ROUTE_COUNT = num_of_routes
                i_x = tf.keras.Input(shape=(32, 32, self.CONV_DIM))
                self.BATCH_DIM = [32, 32, self.CONV_DIM]
                self.ROUTE_DIM = [32, 32, self.CONV_DIM // self.ROUTE_COUNT]
                routing_matrix = tf.keras.Input(shape=(self.ROUTE_COUNT,), dtype=tf.int32)

                # CNN Output
                conv_layer_output = tf.keras.layers.Conv2D(filters=16,
                                                           kernel_size=3,
                                                           activation=tf.nn.relu,
                                                           strides=1,
                                                           padding="same",
                                                           use_bias=True,
                                                           name="conv")(i_x)
                # Cigt route
                cigt_batch_normalization = CigtBatchNormalization(momentum=momentum, name="cigt")
                cigt_normed_net = cigt_batch_normalization([conv_layer_output, routing_matrix])
                # A separate batch normalization operation for every route
                list_of_conventional_batch_norms = []
                list_of_non_zero_indices = []
                list_of_route_x = []
                list_of_conventional_batch_norms_after_scatter_nd = []
                route_width = self.CONV_DIM // self.ROUTE_COUNT
                for rid in range(self.ROUTE_COUNT):
                    conventional_batch_norm = tf.keras.layers.BatchNormalization()
                    route_input = conv_layer_output[:, :, :, rid * route_width:(rid + 1) * route_width]
                    mask_vector = routing_matrix[:, rid]
                    non_zero_indices = tf.where(mask_vector)
                    list_of_non_zero_indices.append(non_zero_indices)
                    route_x = tf.gather_nd(route_input, non_zero_indices)
                    list_of_route_x.append(route_x)
                    conventional_normed_route_x = conventional_batch_norm(route_x)
                    list_of_conventional_batch_norms.append(conventional_normed_route_x)
                    conventional_normed_route_x_after_scatter_nd = tf.scatter_nd(non_zero_indices,
                                                                                 conventional_normed_route_x,
                                                                                 shape=[self.BATCH_SIZE,
                                                                                        *self.ROUTE_DIM])
                    list_of_conventional_batch_norms_after_scatter_nd.append(
                        conventional_normed_route_x_after_scatter_nd)

                    # # Mask the input
                    # masked_input = tf.boolean_mask(route_input, mask_vector)
                    # conventional_normed_route = conventional_batch_norm(masked_input)
                    # conventional_normed_route = tf.scatter_nd()
                    # list_of_conventional_batch_norms.append(conventional_normed_route)
                # Concatenate all conventional batch normed outputs
                # conventional_normed_net = tf.concat(list_of_conventional_batch_norms, axis=-1)

                normed_1d_layer = tf.keras.layers.GlobalAveragePooling2D()(cigt_normed_net)
                # conventional_normed_1d_layer = tf.keras.layers.GlobalAveragePooling2D()(conventional_normed_net)
                # final_layer = 0.5 * (normed_1d_layer + conventional_normed_1d_layer)
                final_layer = normed_1d_layer
                y_hat = tf.reduce_mean(final_layer, axis=-1)
                y_gt = target_val * tf.ones_like(y_hat)
                loss = tf.keras.losses.mean_squared_error(y_gt, y_hat)

                model = tf.keras.Model(inputs=[i_x, routing_matrix],
                                       outputs={
                                           "conv_layer_output": conv_layer_output,
                                           "normed_net": cigt_normed_net,
                                           "list_of_route_x": list_of_route_x,
                                           "list_of_conventional_batch_norms_after_scatter_nd":
                                               list_of_conventional_batch_norms_after_scatter_nd,
                                           # "conventional_normed_net": conventional_normed_net,
                                           "list_of_conventional_batch_norms": list_of_conventional_batch_norms,
                                           "list_of_non_zero_indices": list_of_non_zero_indices,
                                           "normed_1d_layer": normed_1d_layer,
                                           # "conventional_normed_1d_layer": conventional_normed_1d_layer,
                                           "final_layer": final_layer,
                                           "y_hat": y_hat,
                                           "loss": loss})

                errors_dict = dict()
                optimizer = tf.keras.optimizers.Adam()
                loss_tracker = tf.keras.metrics.Mean(name="loss")

                iter_count = self.ITER_COUNT
                # batch_size = self.BATCH_SIZE
                # route_count = self.ROUTE_COUNT
                # error_tol_ratio = self.ERROR_TOL_RATIO

                for i in range(iter_count):
                    x, route_matrix = self.create_mock_input(batch_dim=self.BATCH_DIM,
                                                             batch_size=self.BATCH_SIZE,
                                                             route_count=self.ROUTE_COUNT)
                    with tf.GradientTape() as tape:
                        output_dict = model([x, route_matrix], training=True)
                        loss_tracker.update_state(values=output_dict["loss"])
                        print("X")

                    x_ = output_dict["conv_layer_output"]
                    for rid in range(self.ROUTE_COUNT):
                        # Check if gather_nd has worked correctly.
                        x_route = x_[:, :, :, rid * route_width:(rid + 1) * route_width]
                        x_route_masked = x_route[route_matrix[:, rid] == 1]
                        non_zero_indices_np = np.nonzero(route_matrix[:, rid])[0]
                        res_gather = np.array_equal(x_route_masked.numpy(), output_dict["list_of_route_x"][rid].numpy())
                        self.assertTrue(res_gather)
                        # Check if scatter_nd has worked correctly.
                        scatter_res = output_dict["list_of_conventional_batch_norms"][rid].numpy()
                        scatter_arr_np = np.zeros(shape=[self.BATCH_SIZE, *self.ROUTE_DIM], dtype=scatter_res.dtype)
                        scatter_arr_np[non_zero_indices_np] = scatter_res
                        scatter_arr_tf = output_dict["list_of_conventional_batch_norms_after_scatter_nd"][rid].numpy()
                        res_scatter = np.array_equal(scatter_arr_np, scatter_arr_tf)
                        self.assertTrue(res_scatter)



            # self.batch_norm_test(model=model, iter_count=iter_count, batch_size=batch_size,
            #                      error_tol_ratio=error_tol_ratio, batch_dim=[32, 32, dim],
            #                      route_count=route_count)

    @unittest.skip
    def test_cigt_batch_norm_layer_with_dense(self):
        with tf.device("GPU"):
            momentum = 0.9
            target_val = 5.0

            i_x = tf.keras.Input(shape=(dim,))
            routing_matrix = tf.keras.Input(shape=(route_count,), dtype=tf.int32)
            net = tf.keras.layers.Dense(dim / 4, activation=tf.nn.relu)(i_x)
            # CNN Output

            cigt_batch_normalization = CigtBatchNormalization(momentum=momentum, name="cigt")
            conventional_batch_norm = tf.keras.layers.BatchNormalization()
            normed_net = cigt_batch_normalization([net, routing_matrix])
            conventional_normed_net = conventional_batch_norm(net)

            y_hat = tf.reduce_mean(normed_net, axis=-1)
            y_gt = target_val * tf.ones_like(y_hat)
            loss = tf.keras.losses.mean_squared_error(y_gt, y_hat)

            model = tf.keras.Model(inputs=[i_x, routing_matrix],
                                   outputs={"net_without_norm": net,
                                            "normed_net": normed_net,
                                            "conventional_normed_net": conventional_normed_net,
                                            "y_hat": y_hat,
                                            "loss": loss})

            self.batch_norm_test(model=model, iter_count=iter_count, batch_size=batch_size,
                                 error_tol_ratio=error_tol_ratio, batch_dim=[dim],
                                 route_count=route_count)


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    unittest.main()
