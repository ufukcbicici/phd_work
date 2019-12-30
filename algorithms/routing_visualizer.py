import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from auxillary.general_utility_funcs import UtilityFuncs
from data_handling.cifar_dataset import CifarDataSet
from simple_tf.cign.fast_tree import FastTreeNetwork


class RoutingVisualizer:
    def __init__(self, network_name, run_id, iteration, degree_list, original_images, augmented_images,
                          output_names):
        self.network = FastTreeNetwork.get_mock_tree(degree_list=degree_list, network_name=network_name)
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        self.routingData = self.network.load_routing_info(run_id=run_id, iteration=iteration, data_type="test",
                                                          output_names=output_names)
        self.branchProbs = self.routingData.get_dict("branch_probs")
        self.posteriors = self.routingData.get_dict("posterior_probs")
        self.labelList = self.routingData.labelList
        self.routedSamples = self.routingData.dictionaryOfRoutingData["original_samples"][0]
        self.originalImages = original_images
        self.augmentedImages = augmented_images

    @staticmethod
    def get_original_image(routed_image, original_images, augmented_images):
        routed_sample = routed_image
        reshaped_routed_sample = np.expand_dims(
            np.reshape(routed_sample, newshape=(np.prod(routed_sample.shape),)), axis=0)
        reshaped_augmented_images = np.reshape(augmented_images,
                                               newshape=(
                                                   augmented_images.shape[0], np.prod(augmented_images.shape[1:])))
        difs_with_augmented_images = reshaped_routed_sample - reshaped_augmented_images
        difs_with_augmented_images = np.sum(np.square(difs_with_augmented_images), axis=1)
        min_index = np.argmin(difs_with_augmented_images)
        return original_images[min_index]

    def prepare_routing_information(self):
        total_path_entropies = []
        for idx in range(self.routedSamples.shape[0]):
            curr_node = self.network.topologicalSortedNodes[0]
            total_path_entropy = 0
            while True:
                if curr_node.isLeaf:
                    total_path_entropies.append(total_path_entropy)
                    break
                routing_distribution = self.branchProbs[curr_node.index][idx]
                entropy = UtilityFuncs.calculate_distribution_entropy(distribution=routing_distribution)
                total_path_entropy += entropy
                arg_max_child_index = np.argmax(routing_distribution)
                child_nodes = self.network.dagObject.children(node=curr_node)
                child_nodes = sorted(child_nodes, key=lambda c_node: c_node.index)
                curr_node = child_nodes[arg_max_child_index]
        total_path_entropies = np.array(total_path_entropies)
        print("X")
        # for i in range(routed_samples.shape[0]):
        #     routed_sample = routed_samples[i, :]
        #     original_image = RoutingVisualizer.get_original_image(routed_image=routed_sample,
        #                                                           original_images=original_images,
        #                                                           augmented_images=augmented_images)
        #     print("X")

        # Calculate the total routing entropies for each sample
        # total_entropies = {}
        # for node_id, branch_probs_for_node in branch_probs.items():
        #     probs = branch_probs_for_node
        #     log_probs = np.log(probs + 1e-10)
        #     p_log_p = -1.0 * (probs * log_probs)
        #     entropies = np.sum(p_log_p, axis=1)
        #     total_entropies[node_id] = entropies
        # stacked_entropies = np.stack(list(total_entropies.values()), axis=-1)
        # total_entropies = np.sum(stacked_entropies, axis=-1)
        # sorted_sample_indices = np.argsort(total_entropies)
        # RoutingVisualizer.draw_routing_probabilities(
        #     network=network, sample_idx=sorted_sample_indices[0], branch_probs=branch_probs,
        #     routed_samples=routed_samples, original_images=original_images, augmented_images=augmented_images)
        # print("X")

    @staticmethod
    def draw_routing_probabilities(network, sample_idx, branch_probs,
                                   routed_samples, original_images, augmented_images):
        routed_image = routed_samples[sample_idx]
        original_image = RoutingVisualizer.get_original_image(routed_image=routed_image,
                                                              original_images=original_images,
                                                              augmented_images=augmented_images)
        inner_nodes = [node for node in network.topologicalSortedNodes if not node.isLeaf]
        inner_nodes = sorted(inner_nodes, key=lambda node: node.index)
        fig, axes = plt.subplots(1, len(inner_nodes) + 1)
        axes[0].imshow(original_image)
        plt.show()

    @staticmethod
    def whiten_dataset(dataset):
        data_tensor = tf.placeholder(tf.float32,
                                     shape=(dataset.get_image_size(),
                                            dataset.get_image_size(),
                                            dataset.get_num_of_channels()),
                                     name="dataTensor")
        image = tf.image.per_image_standardization(data_tensor)
        sess = tf.Session()
        whitened_images = []
        for i in range(dataset.testSamples.shape[0]):
            whitened_image = sess.run([image], feed_dict={data_tensor: dataset.testSamples[i]})[0]
            whitened_images.append(whitened_image)
        whitened_images = np.stack(whitened_images, axis=0)
        return whitened_images


def main():
    run_id = 715
    # network_name = "Cifar100_CIGN_Sampling"
    network_name = "Cifar100_CIGN_MultiGpuSingleLateExit"
    iteration = 119100
    # node_costs = {0: 67391424.0, 2: 16754176.0, 6: 3735040.0, 5: 3735040.0, 1: 16754176.0, 4: 3735040.0, 3: 3735040.0}
    # pickle.dump(node_costs, open("nodeCosts.sav", "wb"))
    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "original_samples"]
    dataset = CifarDataSet(session=None, validation_sample_count=0, load_validation_from=None)
    whitened_images = RoutingVisualizer.whiten_dataset(dataset=dataset)
    routing_visualizer = RoutingVisualizer(network_name=network_name, run_id=run_id,
                                           original_images=dataset.testSamples, augmented_images=whitened_images,
                                           iteration=iteration, output_names=output_names, degree_list=[2, 2])
    routing_visualizer.prepare_routing_information()
    # RoutingVisualizer.visualize_routing(network_name=network_name, run_id=run_id,
    #                                     original_images=dataset.testSamples,
    #                                     augmented_images=whitened_images,
    #                                     iteration=iteration, output_names=output_names, degree_list=[2, 2])
    print("X")

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
