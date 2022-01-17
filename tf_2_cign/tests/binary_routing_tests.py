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
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.fashion_net.fashion_cign_binary_rl import FashionRlBinaryRouting
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from tf_2_cign.utilities.utilities import Utilities
from collections import Counter


class BinaryRoutingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fashionMnist = FashionMnist(batch_size=FashionNetConstants.batch_size,
                                        validation_size=5000,
                                        validation_source="training")
        cls.softmaxDecayController = StepWiseDecayAlgorithm(
            decay_name="Stepwise",
            initial_value=FashionNetConstants.softmax_decay_initial,
            decay_coefficient=FashionNetConstants.softmax_decay_coefficient,
            decay_period=FashionNetConstants.softmax_decay_period,
            decay_min_limit=FashionNetConstants.softmax_decay_min_limit)

        with tf.device("GPU"):
            cls.cign = FashionRlBinaryRouting(valid_prediction_reward=FashionNetConstants.valid_prediction_reward,
                                              invalid_prediction_penalty=FashionNetConstants.invalid_prediction_penalty,
                                              include_ig_in_reward_calculations=True,
                                              lambda_mac_cost=FashionNetConstants.lambda_mac_cost,
                                              q_net_params=FashionNetConstants.q_net_params,
                                              batch_size=FashionNetConstants.batch_size,
                                              input_dims=FashionNetConstants.input_dims,
                                              node_degrees=FashionNetConstants.degree_list,
                                              filter_counts=FashionNetConstants.filter_counts,
                                              kernel_sizes=FashionNetConstants.kernel_sizes,
                                              hidden_layers=FashionNetConstants.hidden_layers,
                                              decision_drop_probability=FashionNetConstants.decision_drop_probability,
                                              classification_drop_probability=FashionNetConstants.drop_probability,
                                              decision_wd=FashionNetConstants.decision_wd,
                                              classification_wd=FashionNetConstants.classification_wd,
                                              decision_dimensions=FashionNetConstants.decision_dimensions,
                                              class_count=10,
                                              information_gain_balance_coeff=1.0,
                                              softmax_decay_controller=cls.softmaxDecayController,
                                              learning_rate_schedule=FashionNetConstants.learning_rate_calculator,
                                              decision_loss_coeff=1.0,
                                              warm_up_period=FashionNetConstants.warm_up_period,
                                              cign_rl_train_period=FashionNetConstants.rl_cign_iteration_period,
                                              q_net_coeff=1.0,
                                              epsilon_decay_rate=FashionNetConstants.epsilon_decay_rate,
                                              epsilon_step=FashionNetConstants.epsilon_step,
                                              reward_type="Zero Rewards")
            # run_id = DbLogger.get_run_id()
            cls.cign.init()
            # explanation = cls.cign.get_explanation_string()
            # DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
            cls.actionResultGeneratorLayers = []
            for level in range(cls.cign.get_max_trajectory_length()):
                cls.actionResultGeneratorLayers.append(CignBinaryActionResultGeneratorLayer(
                    level=level, network=cls.cign))

    # OK
    # @unittest.skip
    def test_action_result_generator_layer(self):
        experiment_count = 1000
        batch_size = FashionNetConstants.batch_size

        for exp_id in range(experiment_count):
            if (exp_id + 1) % 100 == 0:
                print("test_action_space_generator_layer Experiment:{0}".format(exp_id + 1))
            comparisons = []
            sc_routing_tensor_curr_level = tf.expand_dims(tf.ones(shape=[batch_size], dtype=tf.int32), axis=-1)
            ig_activations_dict = self.cign.create_mock_ig_activations(batch_size=batch_size)
            for level in range(self.cign.get_max_trajectory_length()):
                ig_activations_list = []
                # Generate mock information gain activations
                for node in self.cign.orderedNodesPerLevel[level]:
                    ig_activations = ig_activations_dict[node.index]
                    ig_activations_list.append(ig_activations)
                ig_activations = tf.stack(ig_activations_list, axis=-1)
                sc_routing_matrix_gt_0, sc_routing_matrix_gt_1 = \
                    self.cign.calculate_next_level_action_results_manual(
                        level=level,
                        ig_activations=ig_activations,
                        sc_routing_matrix_curr_level=sc_routing_tensor_curr_level)
                sc_routing_matrix_tf_0, sc_routing_matrix_tf_1 = self.actionResultGeneratorLayers[level]([
                    ig_activations, sc_routing_tensor_curr_level])
                action_0_equal = np.array_equal(sc_routing_matrix_tf_0.numpy(), sc_routing_matrix_gt_0)
                action_1_equal = np.array_equal(sc_routing_matrix_tf_1.numpy(), sc_routing_matrix_gt_1)
                self.assertTrue(action_0_equal and action_1_equal)
                # assert action_0_equal and action_1_equal
                actions = tf.cast(tf.random.uniform([batch_size, 1], dtype=tf.int32, minval=0, maxval=2),
                                  dtype=tf.bool)
                sc_routing_tensor_curr_level = tf.where(actions, sc_routing_matrix_tf_1, sc_routing_matrix_tf_0)
        print("Passed test_action_space_generator_layer!!!")

    # OK
    # @unittest.skip
    def test_calculate_sample_action_space(self):
        experiment_count = 1000
        batch_size = FashionNetConstants.batch_size

        for exp_id in range(experiment_count):
            if (exp_id + 1) % 100 == 0:
                print("test_calculate_sample_action_space Experiment:{0}".format(exp_id + 1))
            ig_activations_dict = self.cign.create_mock_ig_activations(batch_size=batch_size)
            action_space_manual = self.cign.calculate_sample_action_results_manual(
                ig_activations_dict=ig_activations_dict)
            action_space_auto = self.cign.calculate_sample_action_results(ig_activations_dict=ig_activations_dict)
            self.assertTrue(len(action_space_manual) == len(action_space_auto))
            for level in range(len(action_space_auto)):
                self.assertTrue(np.array_equal(action_space_manual[level], action_space_auto[level]))

        print("Passed test_calculate_sample_action_space!!!")

    # OK
    # @unittest.skip
    def test_action_generator_layer(self):
        experiment_count = 10
        sample_count = 25000
        batch_size = FashionNetConstants.batch_size

        # Action generator
        action_generator_layer = CignBinaryActionGeneratorLayer(network=self.cign)
        # Action generator in a tf.Model
        q_net_output_simulated = tf.keras.Input(shape=[2], name="x_", dtype=tf.float32)
        f_actions, xplr_vs_xplo, xplr_actions, xplo_actions = action_generator_layer(
            q_net_output_simulated)
        model = tf.keras.Model(inputs=q_net_output_simulated,
                               outputs=[f_actions, xplr_vs_xplo, xplr_actions, xplo_actions])
        for exp_id in range(experiment_count):
            if (exp_id + 1) % 100 == 0:
                print("test_action_generator_layer Experiment:{0}".format(exp_id + 1))
            step_count = np.random.randint(low=5000, high=50000)
            print("step_count={0}".format(step_count))
            self.cign.globalStep.assign(value=step_count)
            eps_prob = self.cign.exploreExploitEpsilon(step_count)
            print("eps_prob={0}".format(eps_prob))
            combinations = Utilities.get_cartesian_product(list_of_lists=[
                ["Without tf.model", "With tf.model"],
                ["training", "test"]])
            results_dict = {}
            for tpl in combinations:
                op_type = tpl[0]
                phase = tpl[1]
                if op_type not in results_dict:
                    results_dict[op_type] = dict()
                results_dict[op_type][phase] = dict()
                results_dict[op_type][phase]["action_counter"] = Counter()
                results_dict[op_type][phase]["final_actions"] = []
                results_dict[op_type][phase]["explore_exploit_vec"] = []
                results_dict[op_type][phase]["explore_actions"] = []
                results_dict[op_type][phase]["exploit_actions"] = []

            for sample_id in range(sample_count):
                if (sample_id + 1) % 100 == 0:
                    print("test_action_generator_layer Experiment:{0} Sample:{1}".format(exp_id + 1, sample_id + 1))
                # Mock q_net output
                q_net_output = tf.random.uniform(shape=[batch_size, 2], dtype=tf.float32, minval=-1.0, maxval=1.0)
                for tpl in [("Without tf.model", action_generator_layer), ("With tf.model", model)]:
                    op_type = tpl[0]
                    op = tpl[1]
                    for phase in ["training", "test"]:
                        is_training = phase == "training"
                        final_actions, explore_exploit_vec, explore_actions, exploit_actions = op(q_net_output,
                                                                                                  training=is_training)
                        res_vec = tf.where(explore_exploit_vec,
                                           tf.convert_to_tensor(["explore"] * batch_size),
                                           tf.convert_to_tensor(["exploit"] * batch_size))
                        results_dict[op_type][phase]["action_counter"].update(Counter(res_vec.numpy()))
                        results_dict[op_type][phase]["final_actions"].append(final_actions.numpy())
                        results_dict[op_type][phase]["explore_exploit_vec"].append(explore_exploit_vec.numpy())
                        results_dict[op_type][phase]["explore_actions"].append(explore_actions.numpy())
                        results_dict[op_type][phase]["exploit_actions"].append(exploit_actions.numpy())

            # Concatenate all arrays
            for tpl in combinations:
                op_type = tpl[0]
                phase = tpl[1]
                results_dict[op_type][phase]["final_actions"] = \
                    np.concatenate(results_dict[op_type][phase]["final_actions"])
                results_dict[op_type][phase]["explore_exploit_vec"] = \
                    np.concatenate(results_dict[op_type][phase]["explore_exploit_vec"])
                results_dict[op_type][phase]["explore_actions"] = \
                    np.concatenate(results_dict[op_type][phase]["explore_actions"])
                results_dict[op_type][phase]["exploit_actions"] = \
                    np.concatenate(results_dict[op_type][phase]["exploit_actions"])

            # Check if exploit arrays are equal
            # for arr_type in ["explore_exploit_vec", "explore_actions", "exploit_actions"]:
            arr_type = "exploit_actions"
            ref_array = results_dict["Without tf.model"]["training"][arr_type]
            for tpl in combinations:
                op_type = tpl[0]
                phase = tpl[1]
                print("Checking array:{0}".format(arr_type))
                self.assertTrue(np.array_equal(ref_array,
                                               results_dict[op_type][phase][arr_type]))
            # Check if final_actions are DIFFERENT among training and test phases, for all model types.
            # This test will assert that TF 2.3.0 error in if branches are dealt with.
            for op_type in ["Without tf.model", "With tf.model"]:
                print("Op Type:{0}".format(op_type))
                self.assertFalse(np.array_equal(
                    results_dict[op_type]["training"]["final_actions"],
                    results_dict[op_type]["test"]["final_actions"]))
            # Check if during the training phase "final_actions" does reflect "explore_exploit_vec".
            # This test will also assert that the TF 2.3.0 error is handled.
            for op_type in ["Without tf.model", "With tf.model"]:
                final_actions_manual = []
                explore_exploit_vec = results_dict[op_type]["training"]["explore_exploit_vec"]
                explore_actions = results_dict[op_type]["training"]["explore_actions"]
                exploit_actions = results_dict[op_type]["training"]["exploit_actions"]
                final_actions = results_dict[op_type]["training"]["final_actions"]
                for idx in range(explore_exploit_vec.shape[0]):
                    if explore_exploit_vec[idx]:
                        final_actions_manual.append(explore_actions[idx])
                    else:
                        final_actions_manual.append(exploit_actions[idx])
                final_actions_manual = np.array(final_actions_manual)
                self.assertTrue(np.array_equal(final_actions, final_actions_manual))
            # Finally, check if explore_exploit_vec has an acceptable distribution.
            for tpl in combinations:
                op_type = tpl[0]
                phase = tpl[1]
                print("Op Type:{0}".format(tpl[0]))
                print("Phase:{0}".format(phase))
                print("eps_prob:{0}".format(eps_prob))

                action_counter = results_dict[op_type][phase]["action_counter"]
                monte_carlo_prob = action_counter[b"explore"] / \
                                   (action_counter[b"explore"] + action_counter[b"exploit"])

                print("monte_carlo_prob:{0}".format(monte_carlo_prob))
                # self.assertTrue(eps_prob.numpy() * (1.0 - 0.05) <= monte_carlo_prob <= eps_prob.numpy() * (1.0 + 0.05))
                self.assertTrue(np.abs(eps_prob.numpy() - monte_carlo_prob) <= 0.01)

    # @unittest.skip
    def test_binary_rl_routing_layer(self):
        experiment_count = 1000
        batch_size = FashionNetConstants.batch_size

        for exp_id in range(experiment_count):
            if (exp_id + 1) % 100 == 0:
                print("test_binary_rl_routing_layer Experiment:{0}".format(exp_id + 1))
            # Mock IG activations
            ig_activations_dict = self.cign.create_mock_ig_activations(batch_size=batch_size)
            curr_sc_routing_matrix = tf.expand_dims(tf.ones(shape=[batch_size], dtype=tf.int32), axis=-1)
            for level in range(self.cign.get_max_trajectory_length()):
                ig_activations = tf.stack(
                    [ig_activations_dict[nd.index] for nd in self.cign.orderedNodesPerLevel[level]], axis=-1)
                # Mock actions
                predicted_actions = tf.random.uniform([batch_size, ], dtype=tf.int32, minval=0, maxval=2)

                routing_calculation_layer = CignBinaryRlRoutingLayer(level=level, network=self.cign)
                secondary_routing_matrix_cign_output = routing_calculation_layer(
                    [ig_activations, curr_sc_routing_matrix, predicted_actions])
                # Now, the manual result.
                next_level_config_action_0, next_level_config_action_1 = \
                    self.cign.calculate_next_level_action_results_manual(
                        level=level,
                        ig_activations=ig_activations,
                        sc_routing_matrix_curr_level=curr_sc_routing_matrix)
                secondary_routing_matrix_cign_output_manual = []
                for idx in range(batch_size):
                    if predicted_actions.numpy()[idx] == 1:
                        secondary_routing_matrix_cign_output_manual.append(next_level_config_action_1[idx])
                    else:
                        secondary_routing_matrix_cign_output_manual.append(next_level_config_action_0[idx])
                secondary_routing_matrix_cign_output_manual = tf.stack(secondary_routing_matrix_cign_output_manual,
                                                                       axis=0)
                self.assertTrue(np.array_equal(secondary_routing_matrix_cign_output.numpy(),
                                               secondary_routing_matrix_cign_output_manual.numpy()))
                curr_sc_routing_matrix = secondary_routing_matrix_cign_output

    # @unittest.skip
    def test_calculate_optimal_q_tables(self):
        experiment_count = 100
        batch_size = FashionNetConstants.batch_size
        mean_accuracy = 0.922
        std_accuracy = 0.03
        for experiment_id in range(experiment_count):
            print("Experiment:{0}".format(experiment_id))
            true_labels, ig_activations_dict, posterior_arrays_dict, actions_predicted = self.cign. \
                create_complete_output_mock_data(batch_size=batch_size, mean_accuracy=mean_accuracy,
                                                 std_accuracy=std_accuracy)
            # true_labels = np.random.randint(low=0, high=10, size=(batch_size,))
            # ig_activations_dict = self.cign.create_mock_ig_activations(batch_size=batch_size)
            # # Create mock posterior
            # posterior_arrays_dict = self.cign.create_mock_posteriors(true_labels=true_labels,
            #                                                          batch_size=batch_size,
            #                                                          mean_accuracy=mean_accuracy,
            #                                                          std_accuracy=std_accuracy)
            optimal_q_tables_auto = self.cign.calculate_optimal_q_tables(
                true_labels=true_labels,
                posteriors_dict=posterior_arrays_dict,
                ig_activations_dict=ig_activations_dict)
            optimal_q_tables_manual = self.cign.calculate_optimal_q_tables_manual(
                true_labels=true_labels,
                posteriors_dict=posterior_arrays_dict,
                ig_activations_dict=ig_activations_dict)

            keys_1 = set(optimal_q_tables_auto.keys())
            keys_2 = set(optimal_q_tables_manual.keys())
            self.assertTrue(keys_1 == keys_2)
            for k_ in optimal_q_tables_auto.keys():
                self.assertTrue(np.array_equal(optimal_q_tables_auto[k_], optimal_q_tables_manual[k_]))
        print("test_calculate_optimal_q_tables works!!!")

    # @unittest.skip
    def test_calculate_q_tables_from_network_outputs(self):
        experiment_count = 100
        batch_size = FashionNetConstants.batch_size
        mean_accuracy = 0.922
        std_accuracy = 0.03
        for experiment_id in range(experiment_count):
            print("Experiment:{0}".format(experiment_id))
            true_labels, ig_activations_dict, posterior_arrays_dict, actions_predicted = self.cign. \
                create_complete_output_mock_data(batch_size=batch_size, mean_accuracy=mean_accuracy,
                                                 std_accuracy=std_accuracy)
            posterior_arrays_dict = {k: tf.convert_to_tensor(v) for k, v in posterior_arrays_dict.items()}
            model_outputs = {"posteriors_dict": posterior_arrays_dict, "actions_predicted": actions_predicted,
                             "ig_activations_dict": ig_activations_dict}

            regression_q_targets_auto, optimal_q_values = self.cign.calculate_q_tables_from_network_outputs(
                true_labels=true_labels, model_outputs=model_outputs)
            regression_q_targets_manual = self.cign.calculate_q_tables_from_network_outputs_manual(
                true_labels=true_labels,
                posteriors_dict=posterior_arrays_dict,
                ig_activations_dict=ig_activations_dict,
                actions_predicted=actions_predicted,
                ig_masks_dict=None)
            self.assertTrue(len(regression_q_targets_auto) == len(regression_q_targets_manual))
            for level in range(len(regression_q_targets_auto)):
                self.assertTrue(np.array_equal(regression_q_targets_auto[level], regression_q_targets_manual[level]))


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    unittest.main()
