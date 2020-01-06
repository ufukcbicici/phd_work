from algorithms.multipath_calculator_early_exit import MultipathCalculatorEarlyExit
from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from algorithms.threshold_optimization_algorithms.bayesian_threshold_optimization import BayesianOptimizer, \
    BayesianThresholdOptimizer
from algorithms.threshold_optimization_algorithms.brute_force_threshold_optimizer import BruteForceOptimizer
from algorithms.threshold_optimization_algorithms.simulated_annealing_thread_runner import \
    SimulatedAnnealingThreadRunner
from algorithms.threshold_optimization_algorithms.simulated_annealing_uniform_optimizer import \
    SimulatedAnnealingUniformOptimizer
from algorithms.threshold_optimization_algorithms.thresholding_accuracy_measurement import ThresholdAccuracyMeasurement
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DecayingParameter
from simple_tf.cign.fast_tree import FastTreeNetwork
from multiprocessing import Pool, Lock

# Initial Operations
from simple_tf.global_params import GlobalConstants

# run_id = 1613
# # network_name = "Cifar100_CIGN_Sampling"
network_name = "FashionNet_Lite"
# iteration = 48000
# routing_data_dict = {}
# max_num_of_iterations = 10000
# balance_coefficient = 1.0
# sa_sample_count = 100
#
# use_weighted_scoring = False
# brute_force_sample_count = 1000000
# # node_costs = {i: 1 for i in range(7)}
# # node_costs = {0: 67391424.0, 2: 16754176.0, 6: 3735040.0, 5: 3735040.0, 1: 16754176.0, 4: 3735040.0, 3: 3735040.0}
# node_costs = {0: 465152.0, 2: 2564352.0, 6: 124544.0, 5: 124544.0, 1: 2564352.0, 4: 124544.0, 3: 124544.0}
#
# # GlobalConstants.INNER_NODE_OUTPUTS_TO_COLLECT = ["branch_probs", "activations"]
# # GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT = ["posterior_probs", "label_tensor"]
#
# # GlobalConstants.INNER_NODE_OUTPUTS_TO_COLLECT = ["branch_probs", "activations", "branching_feature"]
# # GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT = ["posterior_probs", "label_tensor", "final_feature_final", "logits"]
#
# GlobalConstants.INNER_NODE_OUTPUTS_TO_COLLECT = ["branch_probs", "activations", "branching_feature"]
# GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT = ["posterior_probs", "label_tensor", "final_feature_final", "logits",
#                                                 "logits_late_exit", "posterior_probs_late", "final_feature_late_exit"]
#
# network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name, node_costs=node_costs)
# routing_data = FastTreeNetwork.load_routing_info(network=network, run_id=run_id, iteration=iteration, data_type="test")
# multipath_calculator = MultipathCalculatorV2(thresholds_list=None, network=network)
#
multiprocess_lock = Lock()


def bayesian_process_runner(param_tpl):
    run_id = param_tpl[0]
    iteration = param_tpl[1]
    xi = param_tpl[2]
    use_weighted = param_tpl[3]
    accuracy_computation_balance = param_tpl[4]
    network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
    routing_data = network.load_routing_info(run_id=run_id, iteration=iteration, data_type="")
    multipath_calculator = MultipathCalculatorV2(network=network, routing_data=routing_data)
    bayesian_optimizer = BayesianThresholdOptimizer(
        run_id=run_id, network=network, iteration=iteration, routing_data=routing_data,
        multipath_score_calculator=multipath_calculator,
        balance_coefficient=accuracy_computation_balance, lock=multiprocess_lock, xi=xi,
        use_weighted_scoring=use_weighted, initial_sample_count=10,
        test_ratio=0.5, max_iter=50, verbose=True)
    bayesian_optimizer.run()


def main():
    # run_id = 1235
    # # network_name = "Cifar100_CIGN_Sampling"
    # network_name = "FashionNet_Lite"
    # iteration = 43680
    # routing_data_dict = {}
    # max_num_of_iterations = 10000
    # balance_coefficient = 1.0
    # sa_sample_count = 100
    #
    # use_weighted_scoring = False
    # brute_force_sample_count = 1000000
    # # node_costs = {i: 1 for i in range(7)}
    # # node_costs = {0: 67391424.0, 2: 16754176.0, 6: 3735040.0, 5: 3735040.0, 1: 16754176.0, 4: 3735040.0, 3: 3735040.0}
    # node_costs = {0: 465152.0, 2: 2564352.0, 6: 124544.0, 5: 124544.0, 1: 2564352.0, 4: 124544.0, 3: 124544.0}
    #
    # GlobalConstants.INNER_NODE_OUTPUTS_TO_COLLECT = ["branch_probs", "activations"]
    # GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT = ["posterior_probs", "label_tensor"]
    #
    # # GlobalConstants.INNER_NODE_OUTPUTS_TO_COLLECT = ["branch_probs", "activations", "branching_feature"]
    # # GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT = ["posterior_probs", "label_tensor", "final_feature_final", "logits"]
    #
    # network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name, node_costs=node_costs)
    # routing_data = FastTreeNetwork.load_routing_info(network=network, run_id=run_id, iteration=iteration)
    # multipath_calculator = MultipathCalculatorV2(thresholds_list=None, network=network)

    # xi_list = [0.01, 0.02, 0.05, 0.1, 0.001, 0.005, 0.0001, 0.0] * 120
    # weighted_score_list = [False]
    # balance_list = [1.0]

    run_ids = [67]
    iterations = [119100, 119200, 119300, 119400, 119500, 119600, 119700, 119800, 119900, 120000]
    xi_list = [0.01, 0.02, 0.05, 0.1, 0.001, 0.005, 0.0001, 0.0] * 10
    weighted_score_list = [False]
    balance_list = [1.0, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[run_ids, iterations,
                                                                          xi_list, weighted_score_list, balance_list])
    # bayesian_process_runner(cartesian_product)

    pool = Pool(processes=5)
    pool.map(bayesian_process_runner, cartesian_product)

    # ThresholdAccuracyMeasurement.calculate_accuracy(run_id=67, iteration=119100, max_overload=10.0, max_limit=1)
    print("X")
    # for db_rows in all_results:
    #     DbLogger.write_into_table(rows=db_rows, table=DbLogger.threshold_optimization, col_count=11)
    # bayesian_process_runner(cartesian_product[0])
    # print("X")
