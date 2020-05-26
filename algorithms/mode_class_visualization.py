import numpy as np
import tensorflow as tf
from auxillary.constants import DatasetTypes
import os
import matplotlib.pyplot as plt
import cv2
from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from data_handling.cifar_dataset import CifarDataSet
from data_handling.fashion_mnist import FashionMnistDataSet
from data_handling.mnist_data_set import MnistDataSet
from simple_tf.cign.fast_tree import FastTreeNetwork
from collections import Counter
import pickle
from mpl_toolkits.axes_grid1 import ImageGrid


class ModeVisualizer:
    MAX_CLASS_COUNT_IN_LEAF_DISTRIBUTION = 25

    def __init__(self, network, dataset, run_id, iteration, data_type, output_names):
        self.network = network
        self.dataset = dataset
        routing_data = self.network.load_routing_info(run_id=run_id, iteration=iteration,
                                                      data_type=data_type, output_names=output_names)
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        print("X")
        labels_list = routing_data.labelList
        sample_count = labels_list.shape[0]
        self.multipathCalculator = MultipathCalculatorV2(network=self.network, routing_data=routing_data)

    def get_sample_distribution_visual(self, dataset, mode_threshold=0.8, sample_count_per_class=5):
        threshold_state = {}
        for node in self.network.topologicalSortedNodes:
            if node.isLeaf:
                continue
            child_count = len(self.network.dagObject.children(node=node))
            max_threshold = 1.0 / float(child_count)
            threshold_state[node.index] = max_threshold * np.ones(shape=(child_count,))
        leaf_reachability_dict = \
            self.multipathCalculator.get_sample_distributions_on_leaf_nodes(thresholds_dict=threshold_state)
        label_list = self.multipathCalculator.labelList
        num_of_labels = len(set(label_list.tolist()))
        min_leaf_node_id = min([l_n.index for l_n in self.leafNodes])
        leaf_distribution_matrix = np.zeros(shape=(num_of_labels, len(self.leafNodes)))
        # Calculate mode distributions
        for node in self.network.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            reached_labels = label_list[leaf_reachability_dict[node.index]]
            counter = Counter(reached_labels)
            for k, v in counter.items():
                leaf_distribution_matrix[k, node.index - min_leaf_node_id] += v
            label_freq_pairs = [(label, float(count) / float(reached_labels.shape[0]))
                                for label, count in counter.items()]
            label_freq_pairs = sorted(label_freq_pairs, key=lambda tpl: tpl[1], reverse=True)
            cut_off_idx = 0
            cumulative_probability = 0
            while True:
                new_cumul_prob = cumulative_probability + label_freq_pairs[cut_off_idx][1]
                cut_off_idx += 1
                if new_cumul_prob >= mode_threshold:
                    break
                cumulative_probability = new_cumul_prob
            mode_labels = label_freq_pairs[0: cut_off_idx]
            self.plot_mode_images_v3(dataset=dataset, node=node, mode_labels=mode_labels,
                                     sample_count_per_class=sample_count_per_class)
        self.plot_leaf_distribution(leaf_distribution_matrix=leaf_distribution_matrix)
        print("X")

    def plot_leaf_distribution(self, leaf_distribution_matrix):
        max_class_count = ModeVisualizer.MAX_CLASS_COUNT_IN_LEAF_DISTRIBUTION
        for idx in range(0, len(leaf_distribution_matrix), max_class_count):
            mat = leaf_distribution_matrix[idx:idx + max_class_count, :]
            # label_ids = ["Class #{0}".format(idx + jdx) for jdx in range(mat.shape[0])]
            label_ids = ["T-Shirt", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
            node_ids = ["Node #{0}".format(l_n.index) for l_n in self.leafNodes]

            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            im = ax.imshow(mat)
            ax.set_xticks(np.arange(len(node_ids)))
            ax.set_yticks(np.arange(len(label_ids)))
            ax.set_xticklabels(node_ids)
            ax.set_yticklabels(label_ids)
            ax.tick_params(labelsize=15)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            for i in range(len(label_ids)):
                for j in range(len(node_ids)):
                    text = ax.text(j, i, int(mat[i, j]),
                                   ha="center", va="center", color="w", fontsize=15)

            ax.set_title("Class distribution on leaf nodes", fontsize=18)
            fig.tight_layout()
            plt.show()
            # fig.savefig("Leaf_distribution_Ids_({0}-{1}).png".format(idx, idx + max_class_count))
            fig.savefig("Leaf_distribution_Ids_FashionMNIST.png")
            # plt.savefig("Leaf_distribution_Ids_({0}-{1}).png".format(idx, idx + max_class_count))
            print("X")

    def plot_mode_images_v3(self, dataset, node, mode_labels, sample_count_per_class, column_count=1):
        extent_size = 4
        column_margin = 32
        _w = dataset.testSamples.shape[2]
        _h = dataset.testSamples.shape[1]
        class_nums_per_column = [int(len(mode_labels) / column_count)] * column_count
        left_over = len(mode_labels) % column_count
        for i in range(left_over):
            class_nums_per_column[i] += 1
        assert sum(class_nums_per_column) == len(mode_labels)
        column_width = (sample_count_per_class + 3) * extent_size + (sample_count_per_class + 2) * _w
        column_height = (max(class_nums_per_column) + 1) * extent_size + max(class_nums_per_column) * _h
        img_width = column_count * column_width + column_count * column_margin
        img_height = column_height
        canvas = np.ones(shape=(img_height, img_width, 3), dtype=np.uint8)
        canvas[:] = 255
        max_probability_mass = mode_labels[0][1]
        curr_class_idx = 0
        for col_idx, col_class_count in enumerate(class_nums_per_column):
            for idx in range(col_class_count):
                label = mode_labels[curr_class_idx][0]
                probability_mass = mode_labels[curr_class_idx][1]
                top = (idx + 1) * extent_size + idx * _h
                col_left = col_idx * (column_width + column_margin) + extent_size
                curr_class_idx += 1
                cv2.putText(canvas, "#{0}".format(label), (col_left, top + _h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color=(0, 0, 0), thickness=0, lineType=cv2.LINE_AA)
                sample_indices_with_correct_label = np.nonzero(dataset.testLabels == label)[0]
                indices = np.random.choice(sample_indices_with_correct_label, sample_count_per_class, replace=False)
                for img_col in range(1, sample_count_per_class + 2, 1):
                    left = col_left + (img_col + 1) * extent_size + img_col * _w
                    # Draw sample images
                    if img_col < sample_count_per_class + 1:
                        img = dataset.testSamples[indices[img_col - 1]]
                        # Convert to BGR
                        if isinstance(dataset, MnistDataSet) and not isinstance(dataset, CifarDataSet):
                            assert len(img.shape) == 2
                            img = 255.0 * np.stack([img, img, img], axis=2)
                        else:
                            assert isinstance(dataset, CifarDataSet)
                        b = np.copy(img[:, :, 2])
                        r = np.copy(img[:, :, 0])
                        img[:, :, 0] = b
                        img[:, :, 2] = r
                        canvas[top: top + _h, left: left + _w, :] = img
                    # Draw distribution info
                    else:
                        cv2.putText(canvas, "%{0:.2f}".format(probability_mass * 100.0), (left, int(top + _h * 0.35)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 0), thickness=0, lineType=cv2.LINE_AA)
                        top_left = (left, int(top + _h * 0.75))
                        relative_prob = probability_mass / max_probability_mass
                        bottom_right = (left + int(relative_prob * _w + 0.5), int(top + _h * 0.95))
                        cv2.rectangle(canvas, top_left, bottom_right, (255, 0, 0), -1)
        cv2.imwrite("Leaf_{0}_None_67.png".format(node.index), canvas)

    def plot_mode_images_v2(self, dataset, node, mode_labels, sample_count_per_class, column_count=2):
        extent_size = 4
        column_margin = 32
        _w = dataset.testSamples.shape[2]
        _h = dataset.testSamples.shape[1]
        class_nums_per_column = [int(len(mode_labels) / column_count)] * column_count
        left_over = len(mode_labels) % column_count
        for i in range(left_over):
            class_nums_per_column[i] += 1
        assert sum(class_nums_per_column) == len(mode_labels)
        column_width = (sample_count_per_class + 3) * extent_size + (sample_count_per_class + 2) * _w
        column_height = (max(class_nums_per_column) + 1) * extent_size + max(class_nums_per_column) * _h
        img_width = column_count * column_width + column_count * column_margin
        img_height = column_height
        canvas = np.ones(shape=(img_height, img_width, 3), dtype=np.uint8)
        canvas[:] = 255
        cummulative_probability = 0.0
        curr_class_idx = 0
        for col_idx, col_class_count in enumerate(class_nums_per_column):
            for idx in range(col_class_count):
                label = mode_labels[curr_class_idx][0]
                probability_mass = mode_labels[curr_class_idx][1]
                cummulative_probability += probability_mass
                top = (idx + 1) * extent_size + idx * _h
                col_left = col_idx * (column_width + column_margin) + extent_size
                curr_class_idx += 1
                cv2.putText(canvas, "#{0}".format(label), (col_left, top + _h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color=(0, 0, 0), thickness=0, lineType=cv2.LINE_AA)
                sample_indices_with_correct_label = np.nonzero(dataset.testLabels == label)[0]
                indices = np.random.choice(sample_indices_with_correct_label, sample_count_per_class, replace=False)
                for img_col in range(1, sample_count_per_class + 2, 1):
                    left = col_left + (img_col + 1) * extent_size + img_col * _w
                    # Draw sample images
                    if img_col < sample_count_per_class + 1:
                        img = dataset.testSamples[indices[img_col - 1]]
                        # Convert to BGR
                        b = np.copy(img[:, :, 2])
                        r = np.copy(img[:, :, 0])
                        img[:, :, 0] = b
                        img[:, :, 2] = r
                        canvas[top: top + _h, left: left + _w, :] = img
                    # Draw distribution info
                    else:
                        cv2.putText(canvas, "+%{0:.3f}".format(probability_mass), (left, int(top + _h * 0.35)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 0), thickness=0, lineType=cv2.LINE_AA)
                        cv2.putText(canvas, "%{0:.3f}".format(cummulative_probability), (left, int(top + _h * 0.7)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 0), thickness=0, lineType=cv2.LINE_AA)
                        top_left = (left, int(top + _h * 0.75))
                        bottom_right = (left + int(cummulative_probability * _w + 0.5), int(top + _h * 0.95))
                        cv2.rectangle(canvas, top_left, bottom_right, (255, 0, 0), -1)
        cv2.imwrite("Leaf_{0}_principal_labels.png".format(node.index), canvas)

    def plot_mode_images(self, dataset, node, mode_labels, sample_count_per_class):
        extent_size = 32
        _w = dataset.testSamples.shape[2]
        _h = dataset.testSamples.shape[1]
        img_width = (sample_count_per_class + 1) * extent_size + sample_count_per_class * _w
        img_height = (len(mode_labels) + 1) * extent_size + len(mode_labels) * _h
        canvas = np.ones(shape=(img_height, img_width, 3), dtype=np.uint8)
        canvas[:] = 255

        img_list = []
        cummulative_probability = 0.0
        for row, tpl in enumerate(mode_labels):
            label = tpl[0]
            probability_mass = tpl[1]
            cummulative_probability += probability_mass
            top = (row + 1) * extent_size + row * _h
            sample_indices_with_correct_label = np.nonzero(dataset.testLabels == label)[0]
            # Pick random indices
            indices = np.random.choice(sample_indices_with_correct_label, sample_count_per_class, replace=False)
            cv2.putText(canvas, "Label:{0} Probability Mass:{1:.6f} Cummulative Probability:{2:.6f}"
                        .format(label, probability_mass, cummulative_probability),
                        (img_width // 25, top - _h // 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        color=(0, 0, 0), thickness=0, lineType=cv2.LINE_AA)
            for col in range(sample_count_per_class):
                img = dataset.testSamples[indices[col]]
                # Convert to BGR
                b = np.copy(img[:, :, 2])
                r = np.copy(img[:, :, 0])
                img[:, :, 0] = b
                img[:, :, 2] = r
                left = (col + 1) * extent_size + col * _w
                canvas[top: top + _h, left: left + _w, :] = img
        cv2.imwrite("Leaf_{0}_principal_labels.png".format(node.index), canvas)
        # for ax, im in zip(grid, img_list):
        #     # Iterating over the grid returns the Axes.
        #     ax.imshow(im)
        #
        # plt.show()

        # counter = 0
        # for ax in grid:
        #     row
        #     # Iterating over the grid returns the Axes.
        #     ax.imshow(im)
        #     ax.set_title("XXX")

        # im1 = np.arange(100).reshape((10, 10))
        # im2 = 200 * np.ones_like(im1)
        # im3 = np.ones_like(im1)
        # im4 = im1.T
        #
        # # fig = plt.figure(figsize=(4., 4.))
        # grid = ImageGrid(fig, 111,  # similar to subplot(111)
        #                  nrows_ncols=(2, 2),  # creates 2x2 grid of axes
        #                  axes_pad=0.15,  # pad between axes in inch.
        #                  )
        #
        # for ax, im in zip(grid, [im1, im2, im3, im4]):
        #     # Iterating over the grid returns the Axes.
        #     ax.imshow(im)
        #     ax.set_title("XXX")
        #
        # plt.show()

        # grid = ImageGrid(fig, 111, nrows_ncols=(len(mode_labels), sample_count_per_class), axes_pad=0.15,
        #                  label_mode="1")
        # for ax in grid:
        #     ax.imshow(dataset.testSamples[0])
        # print("X")
        # plt.show()
        # fig, axs = plt.subplots(nrows=len(mode_labels), ncols=sample_count_per_class,
        #                         sharex=False, sharey=False, figsize=(20, 10))
        #
        # plt.axis('off')
        # fig.suptitle('Mode Labels for Leaf{0}'.format(node.index))
        # for row, tpl in enumerate(mode_labels):
        #     label = tpl[0]
        #     probability_mass = tpl[1]
        #     sample_indices_with_correct_label = np.nonzero(dataset.testLabels == label)[0]
        #     # Pick random indices
        #     indices = np.random.choice(sample_indices_with_correct_label, sample_count_per_class, replace=False)
        #     for col in range(sample_count_per_class):
        #         img = dataset.testSamples[indices[col]]
        #         axs[row, col].imshow(img)
        #         # axs[row, col].set_title("Label:{0} Prob Mass:{1}".format(label, probability_mass))
        # plt.subplots_adjust(hspace=0.01, wspace=0.01)
        # plt.show()

    @staticmethod
    def cifar_100_coarse_class_heat_map():
        class_leaf_distribution = [
            [0, 99, 1, 0],
            [6, 88, 1, 5],
            [0, 97, 3, 0],
            [95, 1, 1, 3],
            [87, 7, 2, 4],
            [3, 0, 97, 0],
            [10, 85, 3, 2],
            [1, 95, 3, 1],
            [1, 82, 15, 2],
            [1, 6, 91, 2],
            [5, 68, 24, 3],
            [6, 88, 5, 1],
            [0, 0, 87, 13],
            [1, 1, 97, 1],
            [13, 84, 2, 1],
            [85, 3, 2, 10],
            [1, 18, 81, 0],
            [3, 0, 88, 9],
            [14, 80, 0, 6],
            [79, 4, 6, 11],
            [1, 3, 95, 1],
            [100, 0, 0, 0],
            [1, 67, 29, 3],
            [9, 3, 1, 87],
            [3, 89, 4, 4], #OK
            [3, 7, 88, 2],
            [20, 79, 0, 1],
            [76, 8, 2, 14],
            [6, 27, 65, 2],
            [76, 7, 4, 13],
            [16, 0, 0, 84],
            [84, 2, 2, 12],
            [65, 22, 6, 7],
            [13, 3, 7, 77],
            [94, 2, 1, 3],
            [5, 92, 2, 1],
            [94, 4, 2, 0],
            [2, 0, 93, 5],
            [95, 1, 3, 1],
            [3, 9, 84, 4],
            [6, 11, 74, 9],
            [4, 29, 67, 0],
            [98, 1, 1, 0],
            [98, 2, 0, 0],
            [60, 32, 5, 3],
            [11, 81, 3, 5],
            [7, 80, 10, 3],
            [4, 3, 2, 91],
            [1, 88, 11, 0],
            [9, 0, 2, 89], # OK
            [92, 6, 1, 1],
            [81, 11, 3, 5],
            [1, 0, 1, 98],
            [1, 99, 0, 0],
            [5, 92, 1, 2],
            [79, 5, 3, 13],
            [8, 8, 2, 82],
            [6, 91, 3, 0],
            [0, 0, 99, 1],
            [7, 1, 1, 91],
            [0, 0, 1, 99],
            [3, 93, 4, 0],
            [1, 96, 0, 3],
            [92, 4, 0, 4],
            [93, 5, 2, 0],
            [86, 4, 7, 3],
            [94, 1, 4, 1],
            [35, 9, 0, 56],
            [2, 0, 5, 93],
            [2, 3, 9, 86],
            [1, 96, 2, 1],
            [1, 0, 2, 97],
            [83, 1, 4, 12],
            [15, 6, 1, 78],
            [92, 7, 0, 1], # OK
            [91, 3, 2, 4],
            [0, 1, 7, 92],
            [71, 27, 1, 1],
            [9, 85, 4, 2],
            [6, 82, 4, 8],
            [92, 1, 1, 6],
            [0, 0, 99, 1],
            [1, 94, 0, 5],
            [0, 93, 6, 1],
            [0, 15, 83, 2],
            [8, 6, 81, 5],
            [2, 5, 91, 2],
            [0, 4, 95, 1],
            [97, 1, 2, 0],
            [3, 12, 83, 2],
            [3, 3, 88, 6],
            [70, 7, 1, 22],
            [1, 96, 0, 3],
            [61, 17, 0, 22],
            [0, 1, 99, 0],
            [18, 1, 2, 79],
            [3, 1, 2, 94],
            [98, 1, 0, 1],
            [8, 83, 6, 3],
            [5, 81, 2, 12]
        ]
        sess = tf.Session()
        dataset = CifarDataSet(session=None, validation_sample_count=0, load_validation_from=None)
        dataset.set_curr_session(sess)
        dataset.set_current_data_set_type(dataset_type=DatasetTypes.test, batch_size=1000)
        fine_labels = []
        coarse_labels = []
        while True:
            batch = dataset.get_next_batch(batch_size=1000)
            if batch is None:
                break
            fine_labels.append(batch.labels)
            coarse_labels.append(batch.coarse_labels)
        fine_labels = np.concatenate(fine_labels)
        coarse_labels = np.concatenate(coarse_labels)
        fine_to_coarse_dict = {}
        coarse_to_fine_dict = {}
        labels_matrix = np.stack([fine_labels, coarse_labels], axis=1)
        for i in range(labels_matrix.shape[0]):
            if labels_matrix[i, 0] not in fine_to_coarse_dict:
                fine_to_coarse_dict[labels_matrix[i, 0]] = set()
            fine_to_coarse_dict[labels_matrix[i, 0]].add(labels_matrix[i, 1])
            if labels_matrix[i, 1] not in coarse_to_fine_dict:
                coarse_to_fine_dict[labels_matrix[i, 1]] = set()
            coarse_to_fine_dict[labels_matrix[i, 1]].add(labels_matrix[i, 0])
        assert all([len(v) == 1 for v in fine_to_coarse_dict.values()])
        print("X")


def main():
    ModeVisualizer.cifar_100_coarse_class_heat_map()
    run_id = 67
    network_name = "None"
    # network_name = "FashionNet_Lite"
    iterations = [119100]
    # node_costs = {0: 67391424.0, 2: 16754176.0, 6: 3735040.0, 5: 3735040.0, 1: 16754176.0, 4: 3735040.0, 3: 3735040.0}
    # pickle.dump(node_costs, open("nodeCosts.sav", "wb"))
    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs"]
    tree = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
    dataset = CifarDataSet(session=None, validation_sample_count=0, load_validation_from=None)
    # dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
    mode_visualizer = ModeVisualizer(network=tree, dataset=dataset, run_id=run_id, iteration=iterations[0],
                                     data_type="",
                                     output_names=output_names)
    mode_visualizer.get_sample_distribution_visual(dataset=dataset, sample_count_per_class=10, mode_threshold=0.85)
    print("X")


if __name__ == "__main__":
    main()
