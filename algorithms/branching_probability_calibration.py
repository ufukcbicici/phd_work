import numpy as np
import tensorflow as tf
import os
import pickle
from collections import Counter

from algorithms.network_calibration import NetworkCalibrationWithTemperatureScaling


class BranchingProbabilityOptimization:
    def __init__(self):
        pass

    @staticmethod
    def calibrate_branching_probabilities(network, routing_data, run_id, iteration, indices, seed):
        temperatures_dict = {}
        file_name = "network{0}_iteration{1}_seed{2}.sav".format(run_id, iteration, seed)
        if os.path.exists(file_name):
            f = open(file_name, "rb")
            temperatures_dict = pickle.load(f)
            f.close()
        else:
            indices_dict = {network.innerNodes[0].index: indices}
            label_count = len(Counter(routing_data.labelList[indices]))
            for node in network.innerNodes:
                # Determine clusters, map labels to the most likely children
                node_indices = indices_dict[node.index]
                logits = routing_data.get_dict("activations")[node.index][node_indices]
                labels = routing_data.labelList[node_indices]
                child_nodes = network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda n: n.index)
                selected_branches = np.argmax(logits, axis=-1)
                siblings_dict = {sibling_node.index: order_index for order_index, sibling_node in
                                 enumerate(child_nodes)}
                counters_dict = {}
                for child_node in child_nodes:
                    child_labels = labels[selected_branches == siblings_dict[child_node.index]]
                    label_counter = Counter(child_labels)
                    counters_dict[child_node.index] = label_counter
                    indices_dict[child_node.index] = node_indices[selected_branches == siblings_dict[child_node.index]]
                label_mapping = {}
                for label_id in range(label_count):
                    branch_distribution = [(nd.index, counters_dict[nd.index][label_id])
                                           for nd in child_nodes if label_id in counters_dict[nd.index]]
                    if len(branch_distribution) == 0:
                        continue
                    mode_tpl = sorted(branch_distribution, key=lambda tpl: tpl[1], reverse=True)[0]
                    label_mapping[label_id] = siblings_dict[mode_tpl[0]]
                mapped_labels = [label_mapping[l_id] for l_id in labels]
                network_calibration = NetworkCalibrationWithTemperatureScaling(logits=logits, labels=mapped_labels)
                temperature = network_calibration.train()
                temperatures_dict[node.index] = temperature
                tf.reset_default_graph()
            f = open(file_name, "wb")
            pickle.dump(temperatures_dict, f)
            f.close()
        return temperatures_dict
