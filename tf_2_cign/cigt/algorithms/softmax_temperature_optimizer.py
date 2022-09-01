import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_2_cign.utilities.utilities import Utilities


class SoftmaxTemperatureOptimizer(object):
    def __init__(self, multi_path_object):
        self.multiPathObject = multi_path_object
        self.maxEntropies = []
        self.sortedOriginalEntropies = self.get_sorted_entropy_lists()

    def get_sorted_entropy_lists(self):
        sorted_entropies_list = []
        for block_id in range(len(self.multiPathObject.pathCounts) - 1):
            next_block_path_count = self.multiPathObject.pathCounts[block_id + 1]
            max_entropy = np.asscalar(-np.log(1.0 / next_block_path_count))
            ents = []
            for list_of_entropies in self.multiPathObject.combinations_routing_entropies_dict.values():
                entropy_list = list_of_entropies[block_id].tolist()
                ents.extend(entropy_list)
            ents.append(max_entropy)
            self.maxEntropies.append(max_entropy)
            ents = list(set(ents))
            entropies_sorted = sorted(ents)
            sorted_entropies_list.append(entropies_sorted)
        return sorted_entropies_list

    def plot_entropy_histogram_with_temperature(self, temperature, block_id):
        routing_arrs_for_block = []
        n_past_decisions = sum(self.multiPathObject.pathCounts[1:][:block_id])
        for k, v in self.multiPathObject.past_decisions_routing_activations_dict.items():
            if len(k) == n_past_decisions:
                routing_arrs_for_block.append(v)
        routing_arrs_for_block = np.concatenate(routing_arrs_for_block, axis=0)
        routing_arrs_for_block_tempered = routing_arrs_for_block / temperature
        routing_probs = tf.nn.softmax(routing_arrs_for_block_tempered).numpy()
        # routing_probs = np.exp(
        #     routing_arrs_for_block_tempered) / np.sum(np.exp(routing_arrs_for_block_tempered), axis=1)[:, np.newaxis]
        entropies = Utilities.calculate_entropies(routing_probs)

        fig, ax = plt.subplots(1, 1)
        ax.set_title("Block {0} entropy distribution with temperature {1}".format(block_id, temperature))
        ax.hist(entropies, density=False, histtype='stepfilled', alpha=1.0, bins=100, range=(
            0, self.maxEntropies[block_id]))
        ax.legend(loc='best', frameon=False)
        plt.tight_layout()
        plt.show()
        plt.close()

        print("X")

    def run(self, block_id):
        routing_arrs_for_block = []
        n_past_decisions = sum(self.multiPathObject.pathCounts[1:][:block_id])
        for k, v in self.multiPathObject.past_decisions_routing_activations_dict.items():
            if len(k) == n_past_decisions:
                routing_arrs_for_block.append(v)
        routing_arrs_for_block = np.concatenate(routing_arrs_for_block, axis=0)


