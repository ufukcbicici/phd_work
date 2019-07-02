import threading
import numpy as np
from collections import deque


class MultipathCalculator(threading.Thread):
    def __init__(self, thread_id, run_id, iteration, threshold_list,
                 network, sample_count, label_list, branch_probs, posterior_probs):
        threading.Thread.__init__(self)
        self.threadId = thread_id
        self.runId = run_id
        self.iteration = iteration
        self.thresholdList = sorted(threshold_list, reverse=True)
        self.network = network
        self.sampleCount = sample_count
        self.labelList = label_list
        self.branchProbs = branch_probs
        self.posteriorProbs = posterior_probs
        self.kvRows = []

    def run(self):
        root_node = self.network.nodes[0]
        leaf_count = len([node for node in self.network.topologicalSortedNodes if node.isLeaf])
        max_num_of_samples = leaf_count * self.sampleCount
        for path_threshold in self.thresholdList:
            total_correct_simple_avg = 0
            total_correct_weighted_avg = 0
            total_leaves_evaluated = 0
            for sample_index in range(self.sampleCount):
                true_label = self.labelList[sample_index]
                queue = deque([(root_node, 1.0)])
                leaf_path_probs = {}
                leaf_posteriors = {}
                while len(queue) > 0:
                    pair = queue.popleft()
                    curr_node = pair[0]
                    path_probability = pair[1]
                    if not curr_node.isLeaf:
                        p_n_given_sample = self.branchProbs[curr_node.index][sample_index, :]
                        child_nodes = self.network.dagObject.children(node=curr_node)
                        child_nodes_sorted = sorted(child_nodes, key=lambda c_node: c_node.index)
                        for index in range(len(child_nodes_sorted)):
                            if p_n_given_sample[index] >= path_threshold:
                                queue.append((child_nodes_sorted[index], path_probability * p_n_given_sample[index]))
                    else:
                        sample_posterior = self.posteriorProbs[curr_node.index][sample_index, :]
                        assert curr_node.index not in leaf_path_probs and curr_node.index not in leaf_posteriors
                        leaf_path_probs[curr_node.index] = path_probability
                        leaf_posteriors[curr_node.index] = sample_posterior
                assert len(leaf_path_probs) > 0 and \
                       len(leaf_posteriors) > 0 and \
                       len(leaf_path_probs) == len(leaf_posteriors)
                total_leaves_evaluated += len(leaf_posteriors)
                # Method 1: Simply take the average of all posteriors
                final_posterior = None
                # posterior_matrix = np.concatenate(list(leaf_posteriors.values()), axis=0)
                # final_posterior = np.mean(posterior_matrix, axis=0)
                # prediction_simple_avg = np.argmax(final_posterior)
                for posterior in leaf_posteriors.values():
                    if final_posterior is None:
                        final_posterior = np.copy(posterior)
                    else:
                        final_posterior += posterior
                final_posterior = (1.0 / len(leaf_posteriors)) * final_posterior
                prediction_simple_avg = np.argmax(final_posterior)
                # Method 2: Weighted average of all posteriors
                final_posterior_weighted = np.copy(final_posterior)
                final_posterior_weighted.fill(0.0)
                normalizing_constant = 0
                for k, v in leaf_posteriors.items():
                    normalizing_constant += leaf_path_probs[k]
                    final_posterior_weighted += leaf_path_probs[k] * v
                final_posterior_weighted = (1.0 / normalizing_constant) * final_posterior_weighted
                prediction_weighted_avg = np.argmax(final_posterior_weighted)
                if prediction_simple_avg == true_label:
                    total_correct_simple_avg += 1
                if prediction_weighted_avg == true_label:
                    total_correct_weighted_avg += 1
            accuracy_simple_avg = float(total_correct_simple_avg) / float(self.sampleCount)
            accuracy_weighted_avg = float(total_correct_weighted_avg) / float(self.sampleCount)
            print(
                "******* Multipath Threshold:{0} Simple Accuracy:{1} "
                "Weighted Accuracy:{2} Total Leaves Evaluated:{3}*******"
                    .format(path_threshold, accuracy_simple_avg, accuracy_weighted_avg, total_leaves_evaluated))
            self.kvRows.append((self.runId, self.iteration, 0, path_threshold, accuracy_simple_avg,
                                total_leaves_evaluated))
            self.kvRows.append((self.runId, self.iteration, 1, path_threshold, accuracy_weighted_avg,
                                total_leaves_evaluated))
            if total_leaves_evaluated == max_num_of_samples:
                break