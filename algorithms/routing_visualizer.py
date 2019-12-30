import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_handling.cifar_dataset import CifarDataSet
from simple_tf.cign.fast_tree import FastTreeNetwork


class RoutingVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def visualize_routing(network_name, run_id, iteration, degree_list, original_images, augmented_images,
                          output_names):
        network = FastTreeNetwork.get_mock_tree(degree_list=degree_list, network_name=network_name)
        routing_data = network.load_routing_info(run_id=run_id, iteration=iteration, data_type="test",
                                                 output_names=output_names)
        branch_probs = routing_data.dictionaryOfRoutingData["branch_probs"]
        routed_samples = routing_data.dictionaryOfRoutingData["original_samples"][0]
        for i in range(routed_samples.shape[0]):
            routed_sample = routed_samples[i, :]
            reshaped_routed_sample = np.expand_dims(
                np.reshape(routed_sample, newshape=(np.prod(routed_sample.shape),)), axis=0)
            reshaped_augmented_images = np.reshape(augmented_images,
                                                   newshape=(
                                                       augmented_images.shape[0], np.prod(augmented_images.shape[1:])))
            difs_with_augmented_images = reshaped_routed_sample - reshaped_augmented_images
            print("X")

        # sample_images = routing_data.dictionaryOfRoutingData["original_samples"]
        # plt.imshow(sample_images[0][0, :])
        # plt.show()

        # Calculate the total routing entropies for each sample
        total_entropies = []
        for node_id, branch_probs_for_node in branch_probs.items():
            probs = branch_probs_for_node
            log_probs = np.log(probs)
            p_log_p = -1.0 * (probs * log_probs)
            entropies = np.sum(p_log_p, axis=1)
            total_entropies[node_id] = entropies

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
    RoutingVisualizer.visualize_routing(network_name=network_name, run_id=run_id,
                                        original_images=dataset.testSamples,
                                        augmented_images=whitened_images,
                                        iteration=iteration, output_names=output_names, degree_list=[2, 2])
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
