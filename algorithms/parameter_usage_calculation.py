import numpy as np
from algorithms.cign_activation_cost_calculator import CignActivationCostCalculator
from algorithms.dataset_linking_algorithm import DatasetLinkingAlgorithm
from simple_tf.cign.fast_tree import FastTreeNetwork

network_id = 453
network_name = "FashionNet_Lite"
iteration = 47520
list_of_l2_coeffs = [0.0, 0.00001, 0.000025, 0.00005, 0.000075, 0.0001,
                     0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.005, 0.01]
list_of_seeds = [67, 112, 42, 594713, 87, 1111, 484, 8779, 32999, 55123]
network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
routing_data = DatasetLinkingAlgorithm.link_dataset_v3(network_name_="FashionNet_Lite", run_id_=453,
                                                       degree_list_=[2, 2],
                                                       test_iterations_=[43680, 44160, 44640, 45120, 45600,
                                                                         46080, 46560, 47040, 47520, 48000])

parameter_count_dict = {0: 1322, 1: 8157, 2: 8157, 3: 18660, 4: 18660, 5: 18660, 6: 18660}
network_activation_costs, network_activation_costs_dict = \
    CignActivationCostCalculator.calculate_mac_cost(
        network=network,
        node_costs=routing_data.dictOfDatasets[43680].get_dict("nodeCosts"))
network_parameter_costs, network_parameter_costs_dict = \
    CignActivationCostCalculator.calculate_mac_cost(
        network=network,
        node_costs=parameter_count_dict)


sample_count = 5000
# 1.066 * 10^6
activation_ids = np.zeros((sample_count, ), dtype=np.int32)
activation_ids[0:900] = 2
activation_ids[1000:1550] = 4
activation_ids[2000:2500] = 12
activation_ids[3000:3210] = 14
activation_ids[4000:4500] = 10
mean_cost = np.mean([network_activation_costs[a_id] for a_id in activation_ids])
total_activation_cost = (1.0 + mean_cost) * 1057.0
base_parameter_cost = parameter_count_dict[0] + parameter_count_dict[1] + parameter_count_dict[3]
mean_parameter_cost = np.mean([network_parameter_costs[a_id] for a_id in activation_ids])
total_parameter_cost = (1.0 + mean_parameter_cost) * base_parameter_cost
print("X")
# average_parameter_usage = CignActivationCostCalculator.calculate_average_parameter_count(
#     network=network, node_costs=parameter_count_dict, selections_list=activation_ids,
#     selection_tuples=network_activation_costs_dict)
# print("X")
