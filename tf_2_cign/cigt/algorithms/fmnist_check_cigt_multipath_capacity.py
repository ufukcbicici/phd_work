import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.data_classes.multipath_routing_info import MultipathCombinationInfo
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
                                 model_definition="Multipath Capacity with {0}".format(routing_method))
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

    decision_arrays = [[0, 1] for _ in range(len(fashion_cigt.pathCounts) - 1)]
    decision_combinations = Utilities.get_cartesian_product(list_of_lists=decision_arrays)
    # decision_combinations = set([tuple(sorted(arr)) for arr in decision_combinations])

    multipath_info_obj = MultipathCombinationInfo(batch_size=fashion_cigt.batchSize,
                                                  path_counts=fashion_cigt.pathCounts)
    for decision_combination in decision_combinations:
        enforced_decision_arr = np.zeros(shape=(fashion_cigt.batchSize, len(fashion_cigt.pathCounts) - 1),
                                         dtype=np.int32)
        multipath_info_obj.add_new_combination(decision_combination=decision_combination)

        for idx, val in enumerate(decision_combination):
            enforced_decision_arr[:, idx] = val
        fashion_cigt.enforcedRoutingDecisions.assign(enforced_decision_arr)

        multipath_info_obj.fill_data_buffers_for_combination(cigt=fashion_cigt, dataset=fashion_mnist.testDataTf,
                                                             decision_combination=decision_combination)
    # Assert routing probability integrities
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
                arr = multipath_info_obj.combinations_routing_probabilities_dict[valid_combination][block_id]
                valid_arrays.append(arr)
            for i_ in range(len(valid_arrays) - 1):
                assert np.allclose(valid_arrays[i_], valid_arrays[i_ + 1])

    multipath_info_obj.assess_accuracy()


def run_probability_threshold_capacity_calculator():
    fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size, validation_size=0)
    fashion_cigt = get_model(routing_method="Probability_Thresholds", model_id=424)
    measure_model_accuracy(model=fashion_cigt, dataset=fashion_mnist)

    decisions_per_level = []
    for path_count in fashion_cigt.pathCounts[1:]:
        decision_arrays = [[0, 1] for _ in range(path_count)]
        decision_combinations = Utilities.get_cartesian_product(list_of_lists=decision_arrays)
        decision_combinations = [tpl for tpl in decision_combinations if sum(tpl) > 0]
        decisions_per_level.append(decision_combinations)

    decision_combinations_per_level = Utilities.get_cartesian_product(list_of_lists=decisions_per_level)
    decision_combinations_per_level = [tuple(np.concatenate(dc)) for dc in decision_combinations_per_level]
    multipath_info_obj = MultipathCombinationInfo(batch_size=fashion_cigt.batchSize,
                                                  path_counts=fashion_cigt.pathCounts)
    for decision_combination in tqdm(decision_combinations_per_level):
        multipath_info_obj.add_new_combination(decision_combination=decision_combination)

        thresholds_list = np.logical_not(np.array(decision_combination)).astype(np.float)
        thresholds_list = 3.0 * thresholds_list - 1.5
        prob_thresholds_arr = np.stack([thresholds_list] * fashion_cigt.batchSize, axis=0)
        fashion_cigt.routingProbabilityThresholds.assign(prob_thresholds_arr)

        multipath_info_obj.fill_data_buffers_for_combination(cigt=fashion_cigt,
                                                             dataset=fashion_mnist.testDataTf,
                                                             decision_combination=decision_combination)

    # Assert routing probability integrities
    past_sum = 0
    for block_id, route_count in enumerate(fashion_cigt.pathCounts[1:]):
        routing_probabilities_dict = {}
        for decision_combination in tqdm(decision_combinations_per_level):
            past_combination = decision_combination[0:past_sum]
            if past_combination not in routing_probabilities_dict:
                routing_probabilities_dict[past_combination] = []
            routing_probabilities_dict[past_combination].append(
                multipath_info_obj.combinations_routing_probabilities_dict[decision_combination][block_id])
        for k, arr in routing_probabilities_dict.items():
            for i_ in range(len(arr) - 1):
                assert np.allclose(arr[i_], arr[i_ + 1])
        past_sum += route_count


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    DbLogger.log_db_path = DbLogger.blackshark_desktop
    run_probability_threshold_capacity_calculator()
