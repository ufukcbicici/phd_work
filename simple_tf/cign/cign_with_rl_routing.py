import numpy as np
import tensorflow as tf
import time

from auxillary.constants import DatasetTypes
from algorithms.cign_activation_cost_calculator import CignActivationCostCalculator
from algorithms.cign_reachbility_matrices_calculation import CignReachabilityMatricesCalculation
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants


class CignWithRlRouting(FastTreeNetwork):
    def __init__(self, degree_list, dataset, network_name, rl_fine_tuning_schedule):
        super().__init__(None, None, None, None, None, degree_list, dataset, network_name)
        self.rlFineTuningSchedule = set(rl_fine_tuning_schedule)
        self.actionSpaces = None
        self.networkActivationCosts = None
        self.networkActivationCostsDict = None
        self.reachabilityMatrices = None
        self.posteriorTensors = None
        self.rewardTensors = {}
        self.optimalQTables = {}
        self.dictOfIgMaskVectors = {}
        self.dictOfRlMaskVectors = {}


    def build_network(self):
        # Regular CIGN stuff here
        super().build_network()
        root_node = self.topologicalSortedNodes[0].index
        # self.dictOfIgMaskVectors[root_node.index] = tf.ones(shape=[self.batchSizeTf], dtype=tf.int32)
        # self.dictOfRlMaskVectors[root_node.index] = tf.ones(shape=[self.batchSizeTf], dtype=tf.int32)
        # Reinforcement Learning things here
        self.build_action_spaces()
        self.networkActivationCosts, self.networkActivationCostsDict = \
            CignActivationCostCalculator.calculate_mac_cost(
                network=self,
                node_costs=self.nodeCosts)
        self.reachabilityMatrices = CignReachabilityMatricesCalculation.calculate_reachibility_matrices(
            network=self,
            action_spaces=self.actionSpaces)
        print("X")

    def get_max_trajectory_length(self) -> int:
        return int(self.depth - 1)

    def build_action_spaces(self):
        max_trajectory_length = self.get_max_trajectory_length()
        self.actionSpaces = []
        for t in range(max_trajectory_length):
            next_level_node_count = len(self.orderedNodesPerLevel[t + 1])
            action_count = (2 ** next_level_node_count) - 1
            action_space = []
            for action_id in range(action_count):
                action_code = action_id + 1
                l = [int(x) for x in list('{0:0b}'.format(action_code))]
                k = [0] * (next_level_node_count - len(l))
                k.extend(l)
                binary_node_selection = np.array(k)
                action_space.append(binary_node_selection)
            action_space = np.stack(action_space, axis=0)
            self.actionSpaces.append(action_space)

    def prepare_reward_tensors(self, sess, dataset):
        for tpl in [("validation", DatasetTypes.validation), ("test", DatasetTypes.test)]:
            dataset_name = tpl[0]
            dataset_type = tpl[1]
            leaf_node_collections, inner_node_collections = \
                self.collect_eval_results_from_network(
                    sess=sess, dataset=dataset, dataset_type=dataset_type,
                    use_masking=False,
                    leaf_node_collection_names=["branch_probs", "activations"],
                    inner_node_collections_names=["posterior_probs", "label_tensor"])

    def get_node_sibling_index(self, node):
        parent_nodes = self.dagObject.parents(node=node)
        if len(parent_nodes) == 0:
            return 0
        parent_node = parent_nodes[0]
        siblings_dict = {sibling_node.index: order_index for order_index, sibling_node in
                         enumerate(
                             sorted(self.dagObject.children(node=parent_node),
                                    key=lambda c_node: c_node.index))}
        sibling_index = siblings_dict[node.index]
        return sibling_index

    def mask_input_nodes(self, node):
        if node.isRoot:
            node.batchIndicesTensor = tf.range(0, self.batchSizeTf, 1)
            node.labelTensor = self.labelTensor
            node.indicesTensor = self.indicesTensor
            node.oneHotLabelTensor = self.oneHotLabelTensor
            # node.filteredMask = tf.constant(value=True, dtype=tf.bool, shape=(GlobalConstants.BATCH_SIZE, ))
            node.filteredMask = self.filteredMask
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = tf.size(node.labelTensor)
            node.isOpenIndicatorTensor = tf.constant(value=1.0, dtype=tf.float32)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            return None, None
        else:
            # Obtain the mask vector, sample counts and determine if this node receives samples.
            parent_node = self.dagObject.parents(node=node)[0]
            node_child_index = self.get_node_sibling_index(node=node)
            parent_ig_mask = self.dictOfIgMaskVectors[parent_node.index][:, node_child_index]
            parent_rl_mask = self.dictOfRlMaskVectors[parent_node.index][:, node_child_index]




            # mask_tensor = parent_node.maskTensors[node.index]
            # if GlobalConstants.USE_UNIFIED_BATCH_NORM:
            #     mask_without_threshold = parent_node.masksWithoutThreshold[node.index]
            # if GlobalConstants.USE_MULTI_GPU:
            #     with tf.device("/device:CPU:0"):
            #         mask_tensor = tf.where(self.useMasking > 0, mask_tensor,
            #                                tf.logical_or(x=tf.constant(value=True, dtype=tf.bool), y=mask_tensor))
            # else:
            #     mask_tensor = tf.where(self.useMasking > 0, mask_tensor,
            #                            tf.logical_or(x=tf.constant(value=True, dtype=tf.bool), y=mask_tensor))
            # sample_count_tensor = tf.reduce_sum(tf.cast(mask_tensor, tf.float32))
            # node.evalDict[self.get_variable_name(name="sample_count", node=node)] = sample_count_tensor
            # if GlobalConstants.USE_MULTI_GPU:
            #     with tf.device("/device:CPU:0"):
            #         node.isOpenIndicatorTensor = tf.where(sample_count_tensor > 0.0, 1.0, 0.0)
            # else:
            #     node.isOpenIndicatorTensor = tf.where(sample_count_tensor > 0.0, 1.0, 0.0)
            # node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            # # Mask all inputs: F channel, H  channel, activations from ancestors, labels
            # parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
            # parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
            # for k, v in parent_node.activationsDict.items():
            #     node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
            # if GlobalConstants.USE_MULTI_GPU:
            #     with tf.device("/device:CPU:0"):
            #         node.batchIndicesTensor = tf.boolean_mask(parent_node.batchIndicesTensor, mask_tensor)
            #         node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
            #         node.indicesTensor = tf.boolean_mask(parent_node.indicesTensor, mask_tensor)
            #         node.oneHotLabelTensor = tf.boolean_mask(parent_node.oneHotLabelTensor, mask_tensor)
            #         if GlobalConstants.USE_UNIFIED_BATCH_NORM:
            #             node.filteredMask = tf.boolean_mask(mask_without_threshold, mask_tensor)
            # else:
            #     node.batchIndicesTensor = tf.boolean_mask(parent_node.batchIndicesTensor, mask_tensor)
            #     node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
            #     node.indicesTensor = tf.boolean_mask(parent_node.indicesTensor, mask_tensor)
            #     node.oneHotLabelTensor = tf.boolean_mask(parent_node.oneHotLabelTensor, mask_tensor)
            #     if GlobalConstants.USE_UNIFIED_BATCH_NORM:
            #         node.filteredMask = tf.boolean_mask(mask_without_threshold, mask_tensor)
            # if GlobalConstants.USE_SCALED_GRADIENTS:
            #     parent_child_count = len(self.dagObject.children(node=parent_node))
            #     scale = 1.0 / parent_child_count
            #     parent_F = scale * parent_F + (1 - scale) * tf.stop_gradient(parent_F)
            #     parent_H = scale * parent_H + (1 - scale) * tf.stop_gradient(parent_H)
            # return parent_F, parent_H

    def get_rl_masking(self, depth):
        # Get all nodes in the current layer
        nodes = self.orderedNodesPerLevel[depth]


    def cign_train_step(self, sess, dataset, epoch_id, iteration_counter):
        update_results = self.update_params(sess=sess,
                                            dataset=dataset,
                                            epoch=epoch_id,
                                            iteration=iteration_counter)

    def train(self, sess, dataset, run_id):
        iteration_counter = 0
        train_accuracies = []
        validation_accuracies = []
        print("X")
        self.saver = tf.train.Saver()
        for epoch_id in range(GlobalConstants.TOTAL_EPOCH_COUNT):
            # An epoch is a complete pass on the whole dataset.
            dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
            print("*************Epoch {0}*************".format(epoch_id))
            if epoch_id not in self.rlFineTuningSchedule:
                self.cign_train_step(sess=sess, dataset=dataset, epoch_id=epoch_id, iteration_counter=iteration_counter)

            # total_time = 0.0
            # while True:
            #     print("Iteration:{0}".format(iteration_counter))
            #     start_time = time.time()
            #     update_results = self.update_params(sess=sess,
            #                                         dataset=dataset,
            #                                         epoch=epoch_id,
            #                                         iteration=iteration_counter)
            #     print("Update_results type new:{0}".format(update_results.__class__))
            #     if all([update_results.lr, update_results.sampleCounts, update_results.isOpenIndicators]):
            #         elapsed_time = time.time() - start_time
            #         total_time += elapsed_time
            #         self.print_iteration_info(iteration_counter=iteration_counter, update_results=update_results)
            #         iteration_counter += 1
            #     if dataset.isNewEpoch:
            #         print("Epoch Time={0}".format(total_time))
            #         performance_result = self.calculate_model_performance(sess=sess, dataset=dataset, run_id=run_id,
            #                                                               epoch_id=epoch_id,
            #                                                               iteration=iteration_counter)
            #         if performance_result is not None:
            #             training_accuracy, validation_accuracy = performance_result
            #             train_accuracies.append(training_accuracy)
            #             validation_accuracies.append(validation_accuracy)
            #         break
        return train_accuracies, validation_accuracies
