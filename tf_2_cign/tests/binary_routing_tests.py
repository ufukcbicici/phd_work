import unittest
import numpy as np
import tensorflow as tf

from auxillary.db_logger import DbLogger
from tf_2_cign.custom_layers.cign_binary_action_space_generator_layer import CignBinaryActionSpaceGeneratorLayer
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.fashion_net.fashion_cign_binary_rl import FashionRlBinaryRouting
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants


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
                                              epsilon_step=FashionNetConstants.epsilon_step)
            # run_id = DbLogger.get_run_id()
            cls.cign.init()
            # explanation = cls.cign.get_explanation_string()
            # DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
            cls.actionSpaceGeneratorLayers = []
            for level in range(cls.cign.get_max_trajectory_length()):
                cls.actionSpaceGeneratorLayers.append(CignBinaryActionSpaceGeneratorLayer(
                    level=level, network=cls.cign))

    # @unittest.skip
    def test_action_space_generator_layer(self):
        experiment_count = 1000
        batch_size = FashionNetConstants.batch_size

        for exp_id in range(experiment_count):
            if (exp_id + 1) % 100 == 0:
                print("test_action_space_generator_layer Experiment:{0}".format(exp_id + 1))
            comparisons = []
            sc_routing_tensor_curr_level = tf.expand_dims(tf.ones(shape=[batch_size], dtype=tf.int32), axis=-1)
            for level in range(self.cign.get_max_trajectory_length()):
                ig_activations_list = []
                # Generate mock information gain activations
                for node in self.cign.orderedNodesPerLevel[level]:
                    ig_activations = tf.random.uniform(
                        shape=[batch_size, len(self.cign.dagObject.children(node=node))], dtype=tf.float32)
                    ig_activations_list.append(ig_activations)

                ig_activations = tf.stack(ig_activations_list, axis=-1)
                sc_routing_matrix_gt_0, sc_routing_matrix_gt_1 = \
                    self.cign.calculate_next_level_configurations_manuel(
                        level=level,
                        ig_activations=ig_activations,
                        sc_routing_matrix_curr_level=sc_routing_tensor_curr_level)
                sc_routing_matrix_tf_0, sc_routing_matrix_tf_1 = self.actionSpaceGeneratorLayers[level]([
                    ig_activations, sc_routing_tensor_curr_level])
                action_0_equal = np.array_equal(sc_routing_matrix_tf_0.numpy(), sc_routing_matrix_gt_0)
                action_1_equal = np.array_equal(sc_routing_matrix_tf_1.numpy(), sc_routing_matrix_gt_1)
                self.assertTrue(action_0_equal and action_1_equal)
                # assert action_0_equal and action_1_equal
                actions = tf.cast(tf.random.uniform([batch_size, 1], dtype=tf.int32, minval=0, maxval=2),
                                  dtype=tf.bool)
                sc_routing_tensor_curr_level = tf.where(actions, sc_routing_matrix_tf_1, sc_routing_matrix_tf_0)
        print("Passed test_action_space_generator_layer!!!")

    # @unittest.skip
    def test_calculate_sample_action_space(self):
        experiment_count = 1000
        batch_size = FashionNetConstants.batch_size

        for exp_id in range(experiment_count):
            if (exp_id + 1) % 100 == 0:
                print("test_calculate_sample_action_space Experiment:{0}".format(exp_id + 1))

            ig_activations_dict = {}
            # Mock IG activations
            for node in self.cign.topologicalSortedNodes:
                if node.isLeaf:
                    continue
                ig_arr = tf.random.uniform(
                    shape=[batch_size, len(self.cign.dagObject.children(node=node))], dtype=tf.float32)
                ig_activations_dict[node.index] = ig_arr
            action_space_manuel = self.cign.calculate_sample_action_space_manuel(
                ig_activations_dict=ig_activations_dict)
            action_space_auto = self.cign.calculate_sample_action_space(ig_activations_dict=ig_activations_dict)
            self.assertTrue(len(action_space_manuel) == len(action_space_auto))
            for level in range(len(action_space_auto)):
                self.assertTrue(np.array_equal(action_space_manuel[level], action_space_auto[level]))

        print("Passed test_calculate_sample_action_space!!!")

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    unittest.main()
