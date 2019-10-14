from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from algorithms.threshold_optimization_algorithms.brute_force_threshold_optimizer import BruteForceOptimizer
from algorithms.threshold_optimization_algorithms.simulated_annealing_thread_runner import \
    SimulatedAnnealingThreadRunner
from algorithms.threshold_optimization_algorithms.simulated_annealing_uniform_optimizer import \
    SimulatedAnnealingUniformOptimizer
from auxillary.parameters import DecayingParameter
from simple_tf.cign.fast_tree import FastTreeNetwork

run_id = 67
# network_name = "Cifar100_CIGN_Sampling"
network_name = "None"
iterations = [119100]
max_num_of_iterations = 10000
balance_coefficient = 1.0
sa_sample_count = 100

use_weighted_scoring = False
brute_force_sample_count = 10000
# node_costs = {i: 1 for i in range(7)}
node_costs = {0: 67391424.0, 2: 16754176.0, 6: 3735040.0, 5: 3735040.0, 1: 16754176.0, 4: 3735040.0, 3: 3735040.0}


def main():
    tree = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name, node_costs=node_costs)
    multipath_calculators = {}
    for iteration in iterations:
        leaf_true_labels_dict, branch_probs_dict, posterior_probs_dict, activations_dict = \
            FastTreeNetwork.load_routing_info(network=tree, run_id=run_id, iteration=iteration)
        label_list = list(leaf_true_labels_dict.values())[0]
        sample_count = label_list.shape[0]
        multipath_calculator = MultipathCalculatorV2(thresholds_list=None, network=tree,
                                                     sample_count=sample_count,
                                                     label_list=label_list, branch_probs=branch_probs_dict,
                                                     activations=activations_dict, posterior_probs=posterior_probs_dict)
        multipath_calculators[iteration] = multipath_calculator

    sa_optimizers = []
    for _ in range(sa_sample_count):
        annealing_schedule = DecayingParameter(name="Temperature", value=0.75, decay=0.999, decay_period=1)
        sa_optimizer = SimulatedAnnealingUniformOptimizer(run_id=run_id,
                                                          network=tree, max_num_of_iterations=max_num_of_iterations,
                                                          annealing_schedule=annealing_schedule,
                                                          balance_coefficient=balance_coefficient,
                                                          use_weighted_scoring=use_weighted_scoring,
                                                          multipath_score_calculators=multipath_calculators,
                                                          verbose=False, neighbor_volume_ratio=0.1)
        sa_optimizers.append(sa_optimizer)

    sa_algorithm_runner = SimulatedAnnealingThreadRunner(sa_optimizers=sa_optimizers, thread_count=10)
    sa_algorithm_runner.run()
    # sa_optimizer.run()
    # bf_optimizer = BruteForceOptimizer(run_id=run_id, network=tree, sample_count=brute_force_sample_count,
    #                                    multipath_score_calculators=multipath_calculators,
    #                                    balance_coefficient=balance_coefficient,
    #                                    use_weighted_scoring=use_weighted_scoring,
    #                                    thread_count=10, verbose=True, batch_size=100)
    # bf_optimizer.run()
    print("X")
