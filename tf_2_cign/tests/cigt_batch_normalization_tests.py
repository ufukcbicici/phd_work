import unittest

import numpy as np
import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_batch_normalization import CigtBatchNormalization
from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer
from tf_2_cign.cigt.custom_layers.cigt_probabilistic_batch_normalization import CigtProbabilisticBatchNormalization
from tf_2_cign.utilities.utilities import Utilities
from tqdm import tqdm

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class CigtBatchNormTests(unittest.TestCase):
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
    LARGEST_MOVING_MEAN_DIFF = 0
    LARGEST_MOVING_MEAN_DIFF_PAIR = None
    LARGEST_MOVING_VAR_DIFF = 0
    LARGEST_MOVING_VAR_DIFF_PAIR = None

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

    def build_test_model(self, momentum, epsilon, input_shape, model_type, target_val=5.0):
        i_x = tf.keras.Input(shape=input_shape)
        routing_matrix = tf.keras.Input(shape=(self.ROUTE_COUNT,), dtype=tf.int32)

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

        # Cigt route
        cigt_batch_normalization = CigtBatchNormalization(momentum=momentum, epsilon=epsilon, name="cigt",
                                                          start_moving_averages_from_zero=True)
        cigt_normed_net = cigt_batch_normalization([op_layer_output, routing_matrix])
        cigt_masking_layer = CigtMaskingLayer()
        cigt_normed_net = cigt_masking_layer([cigt_normed_net, routing_matrix])
        # Probabilistic Cigt route
        cigt_probabilistic_batch_normalization = CigtProbabilisticBatchNormalization(
            momentum=momentum, epsilon=epsilon, name="cigt_probabilistic", start_moving_averages_from_zero=True)
        cigt_probabilistic_normed_net, joint_probability = \
            cigt_probabilistic_batch_normalization([op_layer_output, routing_matrix])
        cigt_probabilistic_masking_layer = CigtMaskingLayer()
        cigt_probabilistic_normed_net = cigt_probabilistic_masking_layer([cigt_probabilistic_normed_net,
                                                                          routing_matrix])

        # A separate batch normalization operation for every route
        list_of_conventional_batch_norms = []
        list_of_non_zero_indices = []
        list_of_route_x = []
        list_of_conventional_batch_norms_after_scatter_nd = []
        list_of_route_means = []
        list_of_route_variances = []
        list_of_conventional_manual_batch_norms = []
        list_of_conventional_manual_batch_norms_after_scatter_nd = []
        for rid in range(self.ROUTE_COUNT):
            conventional_batch_norm = \
                tf.keras.layers.BatchNormalization(momentum=momentum,
                                                   epsilon=epsilon,
                                                   name="conventional_batch_norm_{0}".format(rid))
            route_input = op_layer_output[..., rid * self.ROUTE_WIDTH:(rid + 1) * self.ROUTE_WIDTH]
            mask_vector = routing_matrix[:, rid]
            non_zero_indices = tf.where(mask_vector)
            list_of_non_zero_indices.append(non_zero_indices)
            route_x = tf.gather_nd(route_input, non_zero_indices)
            list_of_route_x.append(route_x)
            conventional_normed_route_x = conventional_batch_norm(route_x)
            list_of_conventional_batch_norms.append(conventional_normed_route_x)
            route_dim = conventional_normed_route_x.shape[1:]
            conventional_normed_route_x_after_scatter_nd = tf.scatter_nd(non_zero_indices,
                                                                         conventional_normed_route_x,
                                                                         shape=[self.BATCH_SIZE,
                                                                                *route_dim])
            list_of_conventional_batch_norms_after_scatter_nd.append(
                conventional_normed_route_x_after_scatter_nd)

        # Concatenate all conventional batch normed outputs
        conventional_normed_net = tf.concat(list_of_conventional_batch_norms_after_scatter_nd, axis=-1)

        if model_type == "conv":
            normed_1d_layer = tf.keras.layers.GlobalAveragePooling2D()(cigt_normed_net)
            normed_1d_probabilistic_layer = tf.keras.layers.GlobalAveragePooling2D()(cigt_probabilistic_normed_net)
            conventional_normed_1d_layer = tf.keras.layers.GlobalAveragePooling2D()(conventional_normed_net)
        else:
            normed_1d_layer = tf.identity(cigt_normed_net)
            normed_1d_probabilistic_layer = tf.identity(cigt_probabilistic_normed_net)
            conventional_normed_1d_layer = tf.identity(conventional_normed_net)

        final_layer = (1.0 / 3.0) * (normed_1d_layer + normed_1d_probabilistic_layer + conventional_normed_1d_layer)
        y_hat = tf.reduce_mean(final_layer, axis=-1)
        y_gt = target_val * tf.ones_like(y_hat)
        loss = tf.keras.losses.mean_squared_error(y_gt, y_hat)

        model = tf.keras.Model(inputs=[i_x, routing_matrix],
                               outputs={
                                   "op_layer_output": op_layer_output,
                                   "cigt_normed_net": cigt_normed_net,
                                   "cigt_probabilistic_normed_net": cigt_probabilistic_normed_net,
                                   "list_of_route_x": list_of_route_x,
                                   "list_of_conventional_batch_norms_after_scatter_nd":
                                       list_of_conventional_batch_norms_after_scatter_nd,
                                   # "list_of_conventional_manual_batch_norms_after_scatter_nd":
                                   #     list_of_conventional_manual_batch_norms_after_scatter_nd,
                                   "conventional_normed_net": conventional_normed_net,
                                   # "conventional_manual_normed_net": conventional_manual_normed_net,
                                   "list_of_conventional_batch_norms": list_of_conventional_batch_norms,
                                   # "list_of_conventional_manual_batch_norms":
                                   #    list_of_conventional_manual_batch_norms,
                                   "list_of_non_zero_indices": list_of_non_zero_indices,
                                   "normed_1d_layer": normed_1d_layer,
                                   "conventional_normed_1d_layer": conventional_normed_1d_layer,
                                   # "conventional_manual_normed_1d_layer": conventional_manual_normed_1d_layer,
                                   "list_of_route_means": list_of_route_means,
                                   "list_of_route_variances": list_of_route_variances,
                                   "final_layer": final_layer,
                                   "y_hat": y_hat,
                                   "loss": loss})
        return model

    def get_cigt_variable(self, model, var_name, return_numpy=True):
        cigt_var = [v for v in model.variables if var_name in v.name]
        assert len(cigt_var) == 1
        if return_numpy:
            cigt_var = cigt_var[0].numpy()
        else:
            cigt_var = cigt_var[0]
        return cigt_var

    def get_conventional_batch_norm_variable(self, model, var_name):
        conventional_vars = [v for v in model.variables if var_name in v.name and "conventional_batch_norm" in v.name]
        conventional_vars = sorted(conventional_vars, key=lambda v: v.name)
        conventional_vars = [arr.numpy() for arr in conventional_vars]
        assert len(conventional_vars) == self.ROUTE_COUNT
        conventional_var = np.concatenate(conventional_vars)
        return conventional_var

    def compare_cigt_output_with_conventional_bn_output(self, output_dict):
        cigt_normed_net_res = output_dict["cigt_normed_net"].numpy()
        cigt_probabilistic_net_res = output_dict["cigt_probabilistic_normed_net"].numpy()
        conventional_normed_net_res = output_dict["conventional_normed_net"].numpy()
        diff_matrix_1 = np.abs(cigt_normed_net_res - conventional_normed_net_res)
        max_diff = np.max(diff_matrix_1)
        max_index = np.unravel_index(np.argmax(diff_matrix_1), shape=diff_matrix_1.shape)
        a_val = cigt_normed_net_res[max_index]
        b_val = conventional_normed_net_res[max_index]
        print("a_val={0} b_val={1} max_diff={2}".format(a_val, b_val, max_diff))
        if max_diff > self.LARGEST_DIFF:
            self.LARGEST_DIFF = max_diff
            self.LARGEST_DIFF_PAIR = (a_val, b_val)
        is_close_matrix = np.isclose(cigt_normed_net_res, conventional_normed_net_res,
                                     atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                     rtol=self.RELATIVE_ERROR_TOLERANCE)
        is_close_with_probabilistic_version_matrix = np.isclose(cigt_normed_net_res, cigt_probabilistic_net_res,
                                                                atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                                                rtol=self.RELATIVE_ERROR_TOLERANCE)
        self.assertTrue(np.all(is_close_matrix))
        self.assertTrue(np.all(is_close_with_probabilistic_version_matrix))

    def update_batch_norm_parameters(self, cigt_value, cigt_probabilistic_value, conventional_value,
                                     model, parameter_name):
        shared_value = (1.0 / 3.0) * (cigt_value + cigt_probabilistic_value + conventional_value)
        if parameter_name == "gamma":
            cigt_variable = self.get_cigt_variable(model=model, var_name="cigt/wb_gamma", return_numpy=False)
            cigt_probabilistic_variable = self.get_cigt_variable(model=model,
                                                                 var_name="cigt_probabilistic/wb_gamma",
                                                                 return_numpy=False)
        elif parameter_name == "beta":
            cigt_variable = self.get_cigt_variable(model=model, var_name="cigt/wb_beta", return_numpy=False)
            cigt_probabilistic_variable = self.get_cigt_variable(model=model,
                                                                 var_name="cigt_probabilistic/wb_beta",
                                                                 return_numpy=False)
        else:
            raise NotImplementedError()
        cigt_variable.assign(shared_value)
        cigt_probabilistic_variable.assign(shared_value)

        conventional_vars = [v for v in model.variables if parameter_name
                             in v.name and "conventional_batch_norm" in v.name]
        conventional_vars = sorted(conventional_vars, key=lambda v: v.name)

        for rid in range(self.ROUTE_COUNT):
            arr_route = shared_value[rid * self.ROUTE_WIDTH:(rid + 1) * self.ROUTE_WIDTH]
            conventional_vars[rid].assign(arr_route)

    # @unittest.skip
    def test_cigt_batch_norm_layer(self):
        with tf.device("GPU"):
            momentum = 0.9
            epsilon = 1e-5
            target_val = 5.0
            iter_count = self.ITER_COUNT

            for model_type in ["conv", "dense"]:
                for num_of_routes in [4, 1]:
                    self.ROUTE_COUNT = num_of_routes
                    if model_type == "conv":
                        self.BATCH_DIM = [32, 32, self.CONV_DIM]
                    else:
                        self.BATCH_DIM = [self.DENSE_DIM]

                    model = self.build_test_model(momentum=momentum,
                                                  epsilon=epsilon,
                                                  target_val=target_val,
                                                  model_type=model_type,
                                                  input_shape=self.BATCH_DIM)
                    optimizer = tf.keras.optimizers.Adam()
                    loss_tracker = tf.keras.metrics.Mean(name="loss")

                    for i in range(iter_count):
                        x, route_matrix = self.create_mock_input(batch_dim=self.BATCH_DIM,
                                                                 batch_size=self.BATCH_SIZE,
                                                                 route_count=self.ROUTE_COUNT)
                        with tf.GradientTape() as tape:
                            output_dict = model([x, route_matrix], training=True)
                            loss_tracker.update_state(values=output_dict["loss"])

                        grads = tape.gradient(output_dict["loss"], model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))
                        print("*****************Iteration:{0} Loss:{1}*****************".format(
                            i, loss_tracker.result().numpy()))

                        x_ = output_dict["op_layer_output"]
                        for rid in range(self.ROUTE_COUNT):
                            # Check if gather_nd has worked correctly.
                            x_route = x_[..., rid * self.ROUTE_WIDTH:(rid + 1) * self.ROUTE_WIDTH]
                            x_route_masked = x_route[route_matrix[:, rid] == 1]
                            non_zero_indices_np = np.nonzero(route_matrix[:, rid])[0]
                            res_gather = np.array_equal(x_route_masked.numpy(),
                                                        output_dict["list_of_route_x"][rid].numpy())
                            self.assertTrue(res_gather)
                            # Check if scatter_nd has worked correctly.
                            scatter_res = output_dict["list_of_conventional_batch_norms"][rid].numpy()
                            route_dim = x_route.shape[1:]
                            scatter_arr_np = np.zeros(shape=[self.BATCH_SIZE, *route_dim], dtype=scatter_res.dtype)
                            scatter_arr_np[non_zero_indices_np] = scatter_res
                            scatter_arr_tf = output_dict[
                                "list_of_conventional_batch_norms_after_scatter_nd"][rid].numpy()
                            res_scatter = np.array_equal(scatter_arr_np, scatter_arr_tf)
                            self.assertTrue(res_scatter)
                        # Compare results of different batch normalization operations
                        # Check if masked output of CIGT batch normalization matches the independent outputs of the
                        # separate route channels, using the conventional batch normalization.
                        print("Training")
                        self.compare_cigt_output_with_conventional_bn_output(output_dict=output_dict)

                        # Compare scale and location parameters between cigt batch norm
                        # and conventional batch norm: gamma and beta
                        cigt_gamma = self.get_cigt_variable(model=model, var_name="cigt/wb_gamma")
                        cigt_beta = self.get_cigt_variable(model=model, var_name="cigt/wb_beta")
                        cigt_probabilistic_gamma = self.get_cigt_variable(model=model,
                                                                          var_name="cigt_probabilistic/wb_gamma")
                        cigt_probabilistic_beta = self.get_cigt_variable(model=model,
                                                                         var_name="cigt_probabilistic/wb_beta")

                        conventional_gamma = self.get_conventional_batch_norm_variable(model=model, var_name="gamma")
                        conventional_beta = self.get_conventional_batch_norm_variable(model=model, var_name="beta")
                        self.assertTrue(np.allclose(cigt_gamma, conventional_gamma,
                                                    atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                                    rtol=self.RELATIVE_ERROR_TOLERANCE))
                        self.assertTrue(np.allclose(cigt_beta, conventional_beta,
                                                    atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                                    rtol=self.RELATIVE_ERROR_TOLERANCE))
                        self.assertTrue(np.allclose(cigt_probabilistic_gamma, conventional_gamma,
                                                    atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                                    rtol=self.RELATIVE_ERROR_TOLERANCE))
                        self.assertTrue(np.allclose(cigt_probabilistic_beta, conventional_beta,
                                                    atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                                    rtol=self.RELATIVE_ERROR_TOLERANCE))
                        self.assertTrue(np.allclose(cigt_probabilistic_gamma, cigt_gamma,
                                                    atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                                    rtol=self.RELATIVE_ERROR_TOLERANCE))
                        self.assertTrue(np.allclose(cigt_probabilistic_beta, cigt_beta,
                                                    atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                                    rtol=self.RELATIVE_ERROR_TOLERANCE))

                        # Compare population statistics
                        cigt_moving_mean = self.get_cigt_variable(model=model, var_name="cigt/popMean")
                        cigt_moving_var = self.get_cigt_variable(model=model, var_name="cigt/popVar")
                        cigt_probabilistic_moving_mean = self.get_cigt_variable(model=model,
                                                                                var_name="cigt_probabilistic/popMean")
                        cigt_probabilistic_moving_var = self.get_cigt_variable(model=model,
                                                                               var_name="cigt_probabilistic/popVar")
                        conventional_moving_mean = self.get_conventional_batch_norm_variable(model=model,
                                                                                             var_name="moving_mean")
                        conventional_moving_var = self.get_conventional_batch_norm_variable(model=model,
                                                                                            var_name="moving_var")
                        self.assertTrue(np.allclose(cigt_moving_mean, conventional_moving_mean,
                                                    atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                                    rtol=self.RELATIVE_ERROR_TOLERANCE))
                        self.assertTrue(np.allclose(cigt_moving_mean, cigt_probabilistic_moving_mean,
                                                    atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                                    rtol=self.RELATIVE_ERROR_TOLERANCE))
                        self.assertTrue(np.allclose(cigt_probabilistic_moving_mean, conventional_moving_mean,
                                                    atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                                    rtol=self.RELATIVE_ERROR_TOLERANCE))

                        max_diff_mean = \
                            np.max([
                                np.max(np.abs(cigt_moving_mean - conventional_moving_mean)),
                                np.max(np.abs(cigt_moving_mean - cigt_probabilistic_moving_mean)),
                                np.max(np.abs(cigt_probabilistic_moving_mean - conventional_moving_mean))])
                        if max_diff_mean > self.LARGEST_MOVING_MEAN_DIFF:
                            self.LARGEST_MOVING_MEAN_DIFF = max_diff_mean
                            # self.LARGEST_MOVING_MEAN_DIFF_PAIR = (cigt_moving_mean, conventional_moving_mean)
                            print("Iteration:{0} LARGEST_MOVING_MEAN_DIFF:{1}".format(i, max_diff_mean))
                        max_diff_var = \
                            np.max([
                                np.max(np.abs(cigt_moving_var - conventional_moving_var)),
                                np.max(np.abs(cigt_moving_var - cigt_probabilistic_moving_var)),
                                np.max(np.abs(cigt_probabilistic_moving_var - conventional_moving_var))])
                        if max_diff_var > self.LARGEST_MOVING_VAR_DIFF:
                            self.LARGEST_MOVING_VAR_DIFF = max_diff_var
                            # self.LARGEST_MOVING_VAR_DIFF_PAIR = (cigt_moving_var, conventional_moving_var)
                            print("Iteration:{0} LARGEST_MOVING_VAR_DIFF:{1}".format(i, max_diff_var))

                        self.assertTrue(np.allclose(cigt_moving_var, conventional_moving_var,
                                                    atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                                    rtol=self.RELATIVE_ERROR_TOLERANCE))
                        self.assertTrue(np.allclose(cigt_moving_var, cigt_probabilistic_moving_var,
                                                    atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                                    rtol=self.RELATIVE_ERROR_TOLERANCE))
                        self.assertTrue(np.allclose(cigt_probabilistic_moving_var, conventional_moving_var,
                                                    atol=self.ABSOLUTE_ERROR_TOLERANCE,
                                                    rtol=self.RELATIVE_ERROR_TOLERANCE))

                        # Set cigt and conventional moving statistics to the same value.
                        self.update_batch_norm_parameters(cigt_value=cigt_gamma,
                                                          cigt_probabilistic_value=cigt_probabilistic_gamma,
                                                          conventional_value=conventional_gamma,
                                                          model=model,
                                                          parameter_name="gamma")
                        self.update_batch_norm_parameters(cigt_value=cigt_beta,
                                                          cigt_probabilistic_value=cigt_probabilistic_beta,
                                                          conventional_value=conventional_beta,
                                                          model=model,
                                                          parameter_name="beta")
                        # Test inference outputs
                        output_dict_inference = model([x, route_matrix], training=False)
                        print("Inference")
                        self.compare_cigt_output_with_conventional_bn_output(output_dict=output_dict_inference)

                    print("Largest Diff:{0} A:{1} B:{2}".format(self.LARGEST_DIFF,
                                                                self.LARGEST_DIFF_PAIR[0],
                                                                self.LARGEST_DIFF_PAIR[1]))
                    self.LARGEST_DIFF = 0.0
                    self.LARGEST_MOVING_MEAN_DIFF = 0.0
                    self.LARGEST_MOVING_MEAN_DIFF_PAIR = None
                    self.LARGEST_MOVING_VAR_DIFF = 0.0
                    self.LARGEST_MOVING_VAR_DIFF_PAIR = None

    def compare_arrays(self, arr_A, arr_B, comparison_name, errors_dict):
        diff_arr = np.abs(arr_A - arr_B)
        max_diff = np.max(diff_arr)
        argmax_index = np.unravel_index(np.argmax(diff_arr), shape=diff_arr.shape)
        elem_A = arr_A[argmax_index]
        elem_B = arr_B[argmax_index]
        is_true = np.allclose(arr_A, arr_B, atol=self.ABSOLUTE_ERROR_TOLERANCE, rtol=self.RELATIVE_ERROR_TOLERANCE)
        if not is_true:
            print("False!")
        if comparison_name not in errors_dict:
            errors_dict[comparison_name] = (0.0, 0.0, 0.0)
        error_tpl = errors_dict[comparison_name]
        curr_max_diff = error_tpl[0]
        if max_diff > curr_max_diff:
            errors_dict[comparison_name] = (max_diff, elem_A, elem_B)

    # @unittest.skip
    def test_joint_probability_calculation_and_probabilistic_cigt_batch_norm(self):
        momentum = 0.9
        epsilon = 1e-5
        cigt_probabilistic_batch_normalization_layer = CigtProbabilisticBatchNormalization(
            momentum=momentum, epsilon=epsilon,
            start_moving_averages_from_zero=False)

        experiment_count = 1000
        input_shapes = [(128, 4, 4, 64), (128, 16, 16, 64),
                        (125, 7, 7, 128), (120, 1, 1, 512), (125, 64), (128, 256)]
        route_counts = [1, 2, 4, 16, 32]
        matrix_types = ["one_hot", "probability"]
        cartesian_product = Utilities.get_cartesian_product(list_of_lists=[input_shapes, route_counts, matrix_types])
        # pbar = tqdm(cartesian_product)
        for tpl in cartesian_product:
            input_shape = tpl[0]
            route_count = tpl[1]
            matrix_type = tpl[2]
            errors_dict = {}
            print("Current Tuple: ({0},{1},{2})".format(input_shape, route_count, matrix_type))
            # pbar.set_postfix({'Current Tuple': "({0},{1},{2})".format(input_shape, route_count, matrix_type)})
            batch_size = input_shape[0]
            channel_count = input_shape[-1]
            channels_per_route = channel_count // route_count
            i_x = tf.keras.Input(shape=input_shape[1:])
            tf_routing_matrix = tf.keras.Input(shape=(route_count,))
            cigt_probabilistic_batch_normalization = CigtProbabilisticBatchNormalization(
                momentum=momentum, epsilon=epsilon, name="cigt_probabilistic")
            normed_x, tf_p_s_c_given_ch_r = cigt_probabilistic_batch_normalization([i_x, tf_routing_matrix])
            model = tf.keras.Model(inputs=[i_x, tf_routing_matrix], outputs={
                "normed_x": normed_x, "tf_p_s_c_given_ch_r": tf_p_s_c_given_ch_r})
            moving_mean = np.zeros(shape=(input_shape[-1],))
            moving_var = np.zeros(shape=(input_shape[-1],))
            for exp_id in tqdm(range(experiment_count)):
                x_ = np.random.uniform(size=input_shape)
                route_activations = np.random.uniform(size=(batch_size, route_count))
                # Assure that every route is selected always once, at least.
                for ri in range(route_count):
                    route_activations[ri, ri] = 2.0 * np.max(route_activations[ri])
                # Convert activations to a routing matrix according to the matrix type.
                if matrix_type == "one_hot":
                    arg_max_indices = np.argmax(route_activations, axis=1)
                    routing_matrix = np.zeros_like(route_activations)
                    routing_matrix[np.arange(routing_matrix.shape[0]), arg_max_indices] = 1.0
                else:
                    normalizing_constants = np.sum(route_activations, axis=1)
                    routing_matrix = route_activations * np.reciprocal(normalizing_constants)[:, np.newaxis]

                route_selection_counts = np.sum(routing_matrix, axis=0)
                assert np.all(route_selection_counts > 0.0)

                model_output = model([x_, routing_matrix], training=True)
                tf_x_hat = model_output["normed_x"].numpy()
                # Joint probability calculation
                routing_matrix_extended = np.repeat(routing_matrix, repeats=[channels_per_route] * route_count, axis=-1)
                for _ in range(len(input_shape) - 2):
                    routing_matrix_extended = np.expand_dims(routing_matrix_extended, axis=1)
                p_s = (1.0 / batch_size) * np.ones_like(x_)
                p_r_given_s = routing_matrix_extended
                p_ch = (1.0 / channels_per_route) * np.ones_like(x_)
                p_c = (1.0 / np.prod(input_shape[1:-1])) * np.ones_like(x_)
                p_s_r_ch_c = p_s * p_r_given_s * p_ch * p_c
                p_ch_r = np.sum(p_s_r_ch_c, axis=tuple([i_ for i_ in range(len(input_shape) - 1)]))

                for _ in range(len(input_shape) - 1):
                    p_ch_r = np.expand_dims(p_ch_r, axis=0)
                p_s_c_given_ch_r = p_s_r_ch_c * np.reciprocal(p_ch_r)
                self.compare_arrays(arr_A=model_output["tf_p_s_c_given_ch_r"].numpy(),
                                    arr_B=p_s_r_ch_c, errors_dict=errors_dict,
                                    comparison_name="joint_probability")

                mean_manual = np.sum(p_s_c_given_ch_r * x_, axis=tuple([i_ for i_ in range(len(input_shape) - 1)]))
                tf_mean_x = self.get_cigt_variable(model=model, var_name="batchMean")
                self.compare_arrays(arr_A=tf_mean_x, arr_B=mean_manual, errors_dict=errors_dict,
                                    comparison_name="training_mean")
                for _ in range(len(input_shape) - 1):
                    mean_manual = np.expand_dims(mean_manual, axis=0)

                x_mean_zero = x_ - mean_manual
                # self.assertTrue(np.allclose(model_output["zero_meaned"].numpy(), x_mean_zero))
                var_manual = np.sum(p_s_c_given_ch_r * np.square(x_mean_zero),
                                    axis=tuple([i_ for i_ in range(len(input_shape) - 1)]))
                tf_variance_x = self.get_cigt_variable(model=model, var_name="batchVar")
                self.compare_arrays(arr_A=tf_variance_x, arr_B=var_manual, errors_dict=errors_dict,
                                    comparison_name="training_var")
                for _ in range(len(input_shape) - 1):
                    var_manual = np.expand_dims(var_manual, axis=0)

                x_hat = x_mean_zero * np.reciprocal(np.sqrt(var_manual + 1e-5))
                self.compare_arrays(arr_A=tf_x_hat, arr_B=x_hat,
                                    errors_dict=errors_dict, comparison_name="x_hat_comparison_training")
                if exp_id == 0:
                    moving_mean = mean_manual
                    moving_var = var_manual
                else:
                    moving_mean = momentum * moving_mean + (1.0 - momentum) * mean_manual
                    moving_var = momentum * moving_var + (1.0 - momentum) * var_manual

                # Inference mode
                model_output = model([x_, routing_matrix], training=False)
                x_hat_inference = (x_ - moving_mean) * np.reciprocal(np.sqrt(moving_var + 1e-5))
                tf_x_hat_inference = model_output["normed_x"].numpy()
                mu = np.squeeze(moving_mean)
                sigma = np.squeeze(moving_var)
                tf_mu = self.get_cigt_variable(model=model, var_name="popMean")
                tf_sigma = self.get_cigt_variable(model=model, var_name="popVar")
                self.compare_arrays(arr_A=mu, arr_B=tf_mu, errors_dict=errors_dict, comparison_name="popMean")
                self.compare_arrays(arr_A=sigma, arr_B=tf_sigma, errors_dict=errors_dict, comparison_name="popVar")
                self.compare_arrays(arr_A=x_hat_inference, arr_B=tf_x_hat_inference,
                                    errors_dict=errors_dict, comparison_name="x_hat_comparison_inference")
            for k, v_tpl in errors_dict.items():
                print("Variable:{0} Max Diff={1} Element A={2} Element B={3}".format(k, v_tpl[0], v_tpl[1], v_tpl[2]))


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    unittest.main()
