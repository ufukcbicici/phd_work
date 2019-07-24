from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from algorithms.simple_accuracy_calculator import SimpleAccuracyCalculator
from algorithms.threshold_optimization_algorithms.brute_force_threshold_optimizer import BruteForceOptimizer
from algorithms.threshold_optimization_algorithms.simulated_annealing_uniform_optimizer import \
    SimulatedAnnealingUniformOptimizer
from auxillary.parameters import DecayingParameter
from simple_tf.cign.fast_tree import FastTreeNetwork

run_id = 0
# network_name = "Cifar100_CIGN_Sampling"
network_name = "Cifar100_CIGN_Single_GPU"
iterations = [118200]
max_num_of_iterations = 10
annealing_schedule = DecayingParameter(name="Temperature", value=100.0, decay=0.9999, decay_period=1)
balance_coefficient = 0.95
use_weighted_scoring = False
brute_force_sample_count = 100000
node_costs = {i: 1 for i in range(7)}


def main():
    tree = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name, node_costs=node_costs)
    multipath_calculators = {}
    for iteration in iterations:
        leaf_true_labels_dict, branch_probs_dict, posterior_probs_dict, activations_dict = \
            SimpleAccuracyCalculator.load_routing_info(network=tree, run_id=run_id, iteration=iteration)
        label_list = list(leaf_true_labels_dict.values())[0]
        sample_count = label_list.shape[0]
        multipath_calculator = MultipathCalculatorV2(thresholds_list=None, network=tree,
                                                     sample_count=sample_count,
                                                     label_list=label_list, branch_probs=branch_probs_dict,
                                                     activations=activations_dict, posterior_probs=posterior_probs_dict)
        multipath_calculators[iteration] = multipath_calculator
    sa_optimizer = SimulatedAnnealingUniformOptimizer(run_id=run_id,
                                                      network=tree, max_num_of_iterations=max_num_of_iterations,
                                                      annealing_schedule=annealing_schedule,
                                                      balance_coefficient=balance_coefficient,
                                                      use_weighted_scoring=use_weighted_scoring,
                                                      multipath_score_calculators=multipath_calculators,
                                                      verbose=True, neighbor_volume_ratio=0.1)
    sa_optimizer.run()

    # bf_optimizer = BruteForceOptimizer(run_id=run_id, network=tree, sample_count=brute_force_sample_count,
    #                                    multipath_score_calculators=multipath_calculators,
    #                                    balance_coefficient=balance_coefficient,
    #                                    use_weighted_scoring=use_weighted_scoring,
    #                                    thread_count=10, verbose=True, batch_size=100)
    # bf_optimizer.run()
    print("X")
