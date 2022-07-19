import tensorflow as tf
import numpy as np
import os

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from tf_2_cign.utilities.utilities import Utilities

# def prepare_routing_configurations_score_table():
INFO_GAIN_LOG_EPSILON = 1e-30


def calculate_entropies(prob_distributions):
    log_prob = np.log(prob_distributions + INFO_GAIN_LOG_EPSILON)
    # is_inf = tf.is_inf(log_prob)
    # zero_tensor = tf.zeros_like(log_prob)
    # log_prob = tf.where(is_inf, x=zero_tensor, y=log_prob)
    prob_log_prob = prob_distributions * log_prob
    entropies = -1.0 * np.sum(prob_log_prob, axis=1)
    return entropies


def get_model(routing_method, model_id):
    X = 0.15
    Y = 3.7233209862205525  # kwargs["information_gain_balance_coefficient"]
    Z = 0.7564802988471849  # kwargs["decision_loss_coefficient"]
    W = 0.01

    FashionNetConstants.softmax_decay_initial = 25.0
    FashionNetConstants.softmax_decay_coefficient = 0.9999
    FashionNetConstants.softmax_decay_period = 1
    FashionNetConstants.softmax_decay_min_limit = 0.1
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
        fashion_cigt = LenetCigt(batch_size=125,
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
                                 routing_strategy_name=routing_method,
                                 run_id=run_id,
                                 evaluation_period=10,
                                 measurement_start=25,
                                 use_straight_through=True,
                                 optimizer_type="SGD",
                                 decision_non_linearity="Softmax",
                                 save_model=True,
                                 model_definition="Multipath Capacitiy with {0}".format(routing_method))
        weights_folder_path = os.path.join(os.path.dirname(__file__), "..", "saved_models",
                                           "weights_{0}".format(model_id))
        fashion_cigt.load_weights(filepath=os.path.join(weights_folder_path, "fully_trained_weights"))
        fashion_cigt.isInWarmUp = False
        weights_folder_path = os.path.join(os.path.dirname(__file__), "..", "saved_models",
                                           "weights_{0}".format(model_id))
        fashion_cigt.load_weights(filepath=os.path.join(weights_folder_path, "fully_trained_weights"))
        fashion_cigt.isInWarmUp = False
        return fashion_cigt


def measure_model_accuracy(model, dataset):
    training_accuracy, training_info_gain_list = model.evaluate(
        x=dataset.trainDataTf, epoch_id=0, dataset_type="training")
    test_accuracy, test_info_gain_list = model.evaluate(
        x=dataset.testDataTf, epoch_id=0, dataset_type="test")
    print("training_accuracy={0}".format(training_accuracy))
    print("test_accuracy={0}".format(test_accuracy))


def run():
    fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size, validation_size=0)
    fashion_cigt = get_model(routing_method="Enforced_Routing", model_id=424)
    measure_model_accuracy(model=fashion_cigt, dataset=fashion_mnist)
    # Check that we have the correct accuracy
    training_accuracy, training_info_gain_list = fashion_cigt.evaluate(
        x=fashion_mnist.trainDataTf, epoch_id=0, dataset_type="training")
    test_accuracy, test_info_gain_list = fashion_cigt.evaluate(
        x=fashion_mnist.testDataTf, epoch_id=0, dataset_type="test")
    print("training_accuracy={0}".format(training_accuracy))
    print("test_accuracy={0}".format(test_accuracy))

    decision_arrays = [[0, 1] for _ in range(len(fashion_cigt.pathCounts) - 1)]
    decision_combinations = Utilities.get_cartesian_product(list_of_lists=decision_arrays)
    # decision_combinations = set([tuple(sorted(arr)) for arr in decision_combinations])

    combinations_y_dict = {}
    combinations_y_hat_dict = {}
    combinations_routing_probabilities_dict = {}
    combinations_routing_entropies_dict = {}

    for decision_combination in decision_combinations:
        enforced_decision_arr = np.zeros(shape=(fashion_cigt.batchSize, len(fashion_cigt.pathCounts) - 1),
                                         dtype=np.int32)
        combinations_y_dict[decision_combination] = []
        combinations_y_hat_dict[decision_combination] = []
        combinations_routing_probabilities_dict[decision_combination] = []
        combinations_routing_entropies_dict[decision_combination] = []

        for _ in range(len(fashion_cigt.pathCounts) - 1):
            combinations_routing_probabilities_dict[decision_combination].append([])

        for idx, val in enumerate(decision_combination):
            enforced_decision_arr[:, idx] = val
        fashion_cigt.enforcedRoutingDecisions.assign(enforced_decision_arr)
        for x_, y_ in fashion_mnist.testDataTf:
            results_dict = fashion_cigt.call(inputs=[x_, y_,
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
                calculate_entropies(combinations_routing_probabilities_dict[decision_combination][i_])
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

    # Assert for consistency of routing distributions.
    for block_id in range(len(fashion_cigt.pathCounts) - 1):
        all_previous_combinations = Utilities.get_cartesian_product(
            [[0, 1] for _ in range(block_id)])
        for previous_combination in all_previous_combinations:
            valid_combinations = []
            for combination in decision_combinations:
                if combination[0:block_id] == previous_combination:
                    valid_combinations.append(combination)
            valid_arrays = []
            for valid_combination in valid_combinations:
                arr = combinations_routing_probabilities_dict[valid_combination][block_id]
                valid_arrays.append(arr)
            for i_ in range(len(valid_arrays) - 1):
                assert np.allclose(valid_arrays[i_], valid_arrays[i_ + 1])


def run_probability_threshold_capacity_calculator():
    fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size, validation_size=0)
    fashion_cigt = get_model(routing_method="Probability_Thresholds", model_id=424)
    measure_model_accuracy(model=fashion_cigt, dataset=fashion_mnist)
    # Check that we have the correct accuracy
    training_accuracy, training_info_gain_list = fashion_cigt.evaluate(
        x=fashion_mnist.trainDataTf, epoch_id=0, dataset_type="training")
    test_accuracy, test_info_gain_list = fashion_cigt.evaluate(
        x=fashion_mnist.testDataTf, epoch_id=0, dataset_type="test")
    print("training_accuracy={0}".format(training_accuracy))
    print("test_accuracy={0}".format(test_accuracy))


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    DbLogger.log_db_path = DbLogger.home_asus
    run()
