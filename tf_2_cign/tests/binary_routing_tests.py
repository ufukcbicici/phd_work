import unittest
import tensorflow as tf

from auxillary.db_logger import DbLogger
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
            run_id = DbLogger.get_run_id()
            cls.cign.init()
            explanation = cls.cign.get_explanation_string()
            DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

    def test1(self):
        """One"""
        print("X")
        self.assertTrue(True)


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    unittest.main()
