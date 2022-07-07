import tensorflow as tf
import numpy as np
import os

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.bayesian_optimizers.multipath_threshold_optimizer import MultipathThresholdOptimizer
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from tf_2_cign.utilities.utilities import Utilities


class FashionMnistLenetThresholdOptimizer(MultipathThresholdOptimizer):
    def __init__(self, xi, init_points, n_iter, accuracy_mac_balance_coeff, model_id, val_ratio):
        super().__init__(xi, init_points, n_iter, accuracy_mac_balance_coeff, model_id, val_ratio)

    def get_model_outputs(self):
        X = 0.15
        Y = 3.7233209862205525  # kwargs["information_gain_balance_coefficient"]
        Z = 0.7564802988471849  # kwargs["decision_loss_coefficient"]
        W = 0.01

        FashionNetConstants.softmax_decay_initial = 25.0
        FashionNetConstants.softmax_decay_coefficient = 0.9999
        FashionNetConstants.softmax_decay_period = 1
        FashionNetConstants.softmax_decay_min_limit = 0.1

        fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size, validation_size=0)
        softmax_decay_controller = StepWiseDecayAlgorithm(
            decay_name="Stepwise",
            initial_value=FashionNetConstants.softmax_decay_initial,
            decay_coefficient=FashionNetConstants.softmax_decay_coefficient,
            decay_period=FashionNetConstants.softmax_decay_period,
            decay_min_limit=FashionNetConstants.softmax_decay_min_limit)

        learning_rate_calculator = DiscreteParameter(name="lr_calculator",
                                                     value=W,
                                                     schedule=[(15000 + 12000, (1.0 / 2.0) * W),
                                                               (30000 + 12000, (1.0 / 4.0) * W),
                                                               (40000 + 12000, (1.0 / 40.0) * W)])
        print(learning_rate_calculator)

        with tf.device("GPU"):
            run_id = DbLogger.get_run_id()
            self.model = LenetCigt(batch_size=125,
                                   input_dims=(28, 28, 1),
                                   filter_counts=[32, 64, 128],
                                   kernel_sizes=[5, 5, 1],
                                   hidden_layers=[512, 256],
                                   decision_drop_probability=0.0,
                                   classification_drop_probability=X,
                                   decision_wd=0.0,
                                   classification_wd=0.0,
                                   decision_dimensions=[128, 128],
                                   class_count=10,
                                   information_gain_balance_coeff=Y,
                                   softmax_decay_controller=softmax_decay_controller,
                                   learning_rate_schedule=learning_rate_calculator,
                                   decision_loss_coeff=Z,
                                   path_counts=[2, 4],
                                   bn_momentum=0.9,
                                   warm_up_period=25,
                                   routing_strategy_name="Enforced_Routing",
                                   run_id=run_id,
                                   evaluation_period=10,
                                   measurement_start=25,
                                   use_straight_through=True,
                                   optimizer_type="SGD",
                                   decision_non_linearity="Softmax",
                                   save_model=True,
                                   model_definition="Enforced Routing")
            weights_folder_path = os.path.join(os.path.dirname(__file__), "..", "saved_models",
                                               "weights_{0}".format(self.modelId))
            self.model.load_weights(filepath=os.path.join(weights_folder_path, "fully_trained_weights"))
            self.model.isInWarmUp = False

            # Check that we have the correct accuracy
            training_accuracy, training_info_gain_list = self.model.evaluate(
                x=fashion_mnist.trainDataTf, epoch_id=0, dataset_type="training")
            print("Training Accuracy:{0}".format(training_accuracy))
            test_accuracy, test_info_gain_list = self.model.evaluate(
                x=fashion_mnist.testDataTf, epoch_id=0, dataset_type="test")
            print("Test Accuracy:{0}".format(test_accuracy))

            decision_arrays = [[0, 1] for _ in range(len(self.model.pathCounts) - 1)]
            decision_combinations = Utilities.get_cartesian_product(list_of_lists=decision_arrays)
            # decision_combinations = set([tuple(sorted(arr)) for arr in decision_combinations])

            combinations_y_dict = {}
            combinations_y_hat_dict = {}
            combinations_routing_probabilities_dict = {}
            combinations_routing_entropies_dict = {}

            for decision_combination in decision_combinations:
                enforced_decision_arr = np.zeros(shape=(self.model.batchSize, len(self.model.pathCounts) - 1),
                                                 dtype=np.int32)
                combinations_y_dict[decision_combination] = []
                combinations_y_hat_dict[decision_combination] = []
                combinations_routing_probabilities_dict[decision_combination] = []
                combinations_routing_entropies_dict[decision_combination] = []

                for _ in range(len(self.model.pathCounts) - 1):
                    combinations_routing_probabilities_dict[decision_combination].append([])

                for idx, val in enumerate(decision_combination):
                    enforced_decision_arr[:, idx] = val
                self.model.enforcedRoutingDecisions.assign(enforced_decision_arr)
                for x_, y_ in fashion_mnist.testDataTf:
                    results_dict = self.model.call(inputs=[x_, y_,
                                                           tf.convert_to_tensor(1.0),
                                                           tf.convert_to_tensor(False)], training=False)
                    combinations_y_dict[decision_combination].append(y_.numpy())
                    combinations_y_hat_dict[decision_combination].append(results_dict["logits"].numpy())
                    for i_, arr in enumerate(results_dict["routing_probabilities"]):
                        combinations_routing_probabilities_dict[decision_combination][i_].append(arr.numpy())

                combinations_y_dict[decision_combination] = np.concatenate(combinations_y_dict[decision_combination],
                                                                           axis=0)
                combinations_y_hat_dict[decision_combination] = np.concatenate(combinations_y_hat_dict[
                                                                                   decision_combination], axis=0)
                for i_ in range(len(combinations_routing_probabilities_dict[decision_combination])):
                    combinations_routing_probabilities_dict[decision_combination][i_] = \
                        np.concatenate(combinations_routing_probabilities_dict[decision_combination][i_], axis=0)
                    combinations_routing_entropies_dict[decision_combination].append(
                        Utilities.calculate_entropies(combinations_routing_probabilities_dict[decision_combination][i_])
                    )

            y_matrix = np.stack(list(combinations_y_dict.values()), axis=1)
            y_avg = np.mean(y_matrix, axis=1).astype(dtype=y_matrix.dtype)
            y_diff = y_matrix - y_avg[:, np.newaxis]
            assert np.all(y_diff == 0)
            y_hat_matrix = np.stack(list(combinations_y_hat_dict.values()), axis=2)
            y_hat_matrix = np.argmax(y_hat_matrix, axis=1)
            equals_matrix = np.equal(y_hat_matrix, y_avg[:, np.newaxis])
            correct_vec = np.sum(equals_matrix, axis=1)
            best_accuracy = np.mean(correct_vec > 0.0)
            print("best_accuracy={0}".format(best_accuracy))

        res_tuple = (combinations_routing_probabilities_dict, combinations_routing_entropies_dict,
                     combinations_y_hat_dict, combinations_y_dict, training_accuracy, test_accuracy)
        return res_tuple
