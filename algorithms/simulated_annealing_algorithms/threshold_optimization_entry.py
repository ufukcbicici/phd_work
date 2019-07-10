from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from algorithms.simple_accuracy_calculator import SimpleAccuracyCalculator
from algorithms.simulated_annealing_algorithms.simulated_annealing_uniform_optimizer import \
    SimulatedAnnealingUniformOptimizer
from auxillary.parameters import DecayingParameter
from simple_tf.cign.fast_tree import FastTreeNetwork

run_id = 789
iterations = [43680 + i * 480 for i in range(10)]
max_num_of_iterations = 1000000
annealing_schedule = DecayingParameter(name="Temperature", value=100.0, decay=0.99999, decay_period=1)
balance_coefficient = 0.5
use_weighted_scoring = False


def main():
    tree = FastTreeNetwork.get_mock_tree(degree_list=[2, 2])
    multipath_calculators = []
    for iteration in iterations:
        leaf_true_labels_dict, branch_probs_dict, posterior_probs_dict, activations_dict = \
            SimpleAccuracyCalculator.load_routing_info(network=tree, run_id=run_id, iteration=iteration)
        label_list = list(leaf_true_labels_dict.values())[0]
        sample_count = label_list.shape[0]
        multipath_calculator = MultipathCalculatorV2(thresholds_list=None, network=tree,
                                                     sample_count=sample_count,
                                                     label_list=label_list, branch_probs=branch_probs_dict,
                                                     activations=activations_dict, posterior_probs=posterior_probs_dict)
        multipath_calculators.append(multipath_calculator)
    sa_optimizer = SimulatedAnnealingUniformOptimizer(network=tree, max_num_of_iterations=max_num_of_iterations,
                                                      annealing_schedule=annealing_schedule,
                                                      balance_coefficient=balance_coefficient,
                                                      use_weighted_scoring=use_weighted_scoring,
                                                      multipath_score_calculators=multipath_calculators,
                                                      verbose=True, neighbor_volume_ratio=0.1)
    sa_optimizer.run()
    print("X")
