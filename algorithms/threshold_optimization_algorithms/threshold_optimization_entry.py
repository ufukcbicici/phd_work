from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from algorithms.threshold_optimization_algorithms.bayesian_threshold_optimization import BayesianOptimizer
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
run_id = 67
# network_name = "Cifar100_CIGN_Sampling"
network_name = "Cifar100_CIGN_BatchSize500_[64,64,64]"
iterations = [119100]
routing_data_dict = {}
max_num_of_iterations = 10000
balance_coefficient = 1.0
sa_sample_count = 100

use_weighted_scoring = False
brute_force_sample_count = 1000000
# node_costs = {i: 1 for i in range(7)}
node_costs = {0: 67391424.0, 2: 16754176.0, 6: 3735040.0, 5: 3735040.0, 1: 16754176.0, 4: 3735040.0, 3: 3735040.0}

network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name, node_costs=node_costs)
multipath_calculators = {}
for iteration in iterations:
    routing_data = FastTreeNetwork.load_routing_info(network=network, run_id=run_id, iteration=iteration)
    routing_data_dict[iteration] = routing_data
    multipath_calculator = MultipathCalculatorV2(thresholds_list=None, network=network)
    multipath_calculators[iteration] = multipath_calculator

multiprocess_lock = Lock()


def bayesian_process_runner(param_tpl):
    xi = param_tpl[0]
    use_weighted = param_tpl[1]
    accuracy_computation_balance = param_tpl[2]
    bayesian_optimizer = BayesianOptimizer(
        run_id=run_id, network=network, routing_data_dict=routing_data_dict,
        multipath_score_calculators=multipath_calculators,
        balance_coefficient=accuracy_computation_balance, lock=multiprocess_lock, xi=xi,
        use_weighted_scoring=use_weighted, initial_sample_count=10,
        test_ratio=0.5, max_iter=125, verbose=True)
    bayesian_optimizer.run()


def main():
    # sa_optimizers = []
    # for _ in range(sa_sample_count):
    #     annealing_schedule = DecayingParameter(name="Temperature", value=0.75, decay=0.999, decay_period=1)
    #     sa_optimizer = SimulatedAnnealingUniformOptimizer(run_id=run_id,
    #                                                       network=tree, max_num_of_iterations=max_num_of_iterations,
    #                                                       annealing_schedule=annealing_schedule,
    #                                                       balance_coefficient=balance_coefficient,
    #                                                       use_weighted_scoring=use_weighted_scoring,
    #                                                       multipath_score_calculators=multipath_calculators,
    #                                                       verbose=False, neighbor_volume_ratio=0.1)
    #     sa_optimizers.append(sa_optimizer)
    #
    # sa_algorithm_runner = SimulatedAnnealingThreadRunner(sa_optimizers=sa_optimizers, thread_count=10)
    # sa_algorithm_runner.run()
    # sa_optimizer.run()
    # bf_optimizer = BruteForceOptimizer(run_id=run_id, network=tree, sample_count=brute_force_sample_count,
    #                                    multipath_score_calculators=multipath_calculators,
    #                                    balance_coefficient=balance_coefficient,
    #                                    use_weighted_scoring=use_weighted_scoring,
    #                                    thread_count=1, verbose=True, batch_size=10000)
    # bf_optimizer.run()

    # xi_list = [0.01, 0.02, 0.05, 0.1, 0.001, 0.005, 0.0001, 0.0] * 120
    # weighted_score_list = [False]
    # balance_list = [1.0]

    xi_list = [0.01]
    weighted_score_list = [False]
    balance_list = [1.0]
    cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[xi_list, weighted_score_list, balance_list])
    bayesian_process_runner(cartesian_product[0])

    # pool = Pool(processes=3)
    # pool.map(bayesian_process_runner, cartesian_product)

    # ThresholdAccuracyMeasurement.calculate_accuracy(run_id=67, iteration=119100, max_overload=10.0, max_limit=1)
    print("X")
    # for db_rows in all_results:
    #     DbLogger.write_into_table(rows=db_rows, table=DbLogger.threshold_optimization, col_count=11)
    # bayesian_process_runner(cartesian_product[0])
    # print("X")
