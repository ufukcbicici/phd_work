import numpy as np
import matplotlib.pyplot as plt

from simple_tf.cign.fast_tree import FastTreeNetwork


class RoutingVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def visualize_routing(network_name, run_id, iteration, degree_list):
        network = FastTreeNetwork.get_mock_tree(degree_list=degree_list, network_name=network_name)
        routing_data = network.load_routing_info(run_id=run_id, iteration=iteration, data_type="test",
                                                 include_original_samples=True)
        branch_probs = routing_data.dictionaryOfRoutingData["branch_probs"]
        sample_images = routing_data.dictionaryOfRoutingData["original_samples"]

        # Calculate the total routing entropies for each sample
        total_entropies = []
        for node_id, branch_probs_for_node in branch_probs.items():
            probs = branch_probs_for_node
            log_probs = np.log(probs)
            p_log_p = -1.0 * (probs * log_probs)
            entropies = np.sum(p_log_p, axis=1)
            total_entropies[node_id] = entropies









# distributions = np.random.uniform(size=(3, 2))
# distributions = distributions / np.sum(distributions, axis=1, keepdims=True)
# node_labels = ["Node1", "Node2"]
#
# labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# men_means = [20, 34, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]
#
# x = np.arange(len(node_labels))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, distributions[0], width, label='Routing Probabilities')
# # rects2 = ax.bar(x + width/2, women_means, width, label='Women')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Routing Probabilities')
# ax.set_xticks(x - width/2)
# ax.set_xticklabels(node_labels)
# ax.legend()
#
#
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{0:.3f}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#
# autolabel(rects1)
# fig.tight_layout()
# plt.show()
# print("X")