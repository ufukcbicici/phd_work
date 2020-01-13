import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from data_handling.cifar_dataset import CifarDataSet
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
                if new_cumul_prob >= mode_threshold:
                    break
                cut_off_idx += 1
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
            label_ids = ["Class #{0}".format(idx + jdx) for jdx in range(mat.shape[0])]
            node_ids = ["Node #{0}".format(l_n.index) for l_n in self.leafNodes]

            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            im = ax.imshow(mat)
            ax.set_xticks(np.arange(len(node_ids)))
            ax.set_yticks(np.arange(len(label_ids)))
            ax.set_xticklabels(node_ids)
            ax.set_yticklabels(label_ids)
            ax.tick_params(labelsize=7)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            for i in range(len(label_ids)):
                for j in range(len(node_ids)):
                    text = ax.text(j, i, int(mat[i, j]),
                                   ha="center", va="center", color="w", fontsize=10)

            ax.set_title("Class distribution on leaf nodes")
            fig.tight_layout()
            plt.show()
            fig.savefig("Leaf_distribution_Ids_({0}-{1}).png".format(idx, idx + max_class_count))
            # plt.savefig("Leaf_distribution_Ids_({0}-{1}).png".format(idx, idx + max_class_count))
            print("X")

    def plot_mode_images_v3(self, dataset, node, mode_labels, sample_count_per_class, column_count=2):
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
        cv2.imwrite("Leaf_{0}_principal_labels.png".format(node.index), canvas)

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


def main():
    run_id = 67
    # network_name = "Cifar100_CIGN_Sampling"
    network_name = "None"
    iterations = [119100]
    # node_costs = {0: 67391424.0, 2: 16754176.0, 6: 3735040.0, 5: 3735040.0, 1: 16754176.0, 4: 3735040.0, 3: 3735040.0}
    # pickle.dump(node_costs, open("nodeCosts.sav", "wb"))
    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs"]
    tree = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
    dataset = CifarDataSet(session=None, validation_sample_count=0, load_validation_from=None)
    mode_visualizer = ModeVisualizer(network=tree, dataset=dataset, run_id=67, iteration=119100, data_type="",
                                     output_names=output_names)
    mode_visualizer.get_sample_distribution_visual(dataset=dataset, sample_count_per_class=5, mode_threshold=0.85)
    print("X")


if __name__ == "__main__":
    main()
