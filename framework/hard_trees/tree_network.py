import tensorflow as tf
import numpy as np
import time
import sys

from auxillary.constants import ChannelTypes, TreeType, GlobalInputNames, parameterTypes, LossType, DatasetTypes, \
    ProblemType
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DecayingParameter, FixedParameter
from framework.hard_trees.hard_tree_node import HardTreeNode
from framework.network import Network
from framework.network_channel import NetworkChannel
from framework.network_node import NetworkNode
from framework.node_input_outputs import NetworkIOObject
from initializers.xavier import Xavier
from losses.cross_entropy_loss import CrossEntropyLoss
from losses.sample_index_counter import SampleIndexCounter


class TreeNetwork(Network):
    def __init__(self, dataset, parameter_file, problem_type, train_program, explanation,
                 tree_degree, tree_type, list_of_node_builder_functions, ancestor_count=sys.maxsize,
                 eval_sample_distribution=True):
        super().__init__(dataset, parameter_file, problem_type, train_program, explanation)
        self.treeDegree = tree_degree
        self.treeDepth = len(list_of_node_builder_functions)
        self.depthsToNodesDict = {}
        self.nodesToDepthsDict = {}
        self.nodeBuilderFunctions = list_of_node_builder_functions
        self.treeType = tree_type
        self.ancestorCount = ancestor_count
        self.indicatorText = "TreeNetwork"
        self.evalSampleDistribution = eval_sample_distribution

    # Tensorflow specific code (This should be isolated at some point in future)
    # A node can take from an input from any ancestor node. If the ancestor is its parent, it is directly connected
    # to the parent: The parent has already applied the necessary decision mechanism to the output. If the input is
    # from a non-parent ancestor, than the output is taken from that ancestor, propagated through the ancestor laying
    # in between the path (ancestor, current_node). Each intermediate node, inductively takes the input from its parent,
    # applies decision to it and propagate to the its child. In that case the node producing the output and the interme-
    # diate node are different and this is the only case that happens.
    def add_nodewise_input(self, producer_channel, dest_node, producer_node=None, producer_channel_index=0):
        invalid_channels = {ChannelTypes.objective_loss, ChannelTypes.pre_loss, ChannelTypes.gradient}
        if producer_channel in invalid_channels:
            raise Exception("{0} type of channels cannot be input to other nodes.".format(producer_channel.value))
        # Data or label or index input
        if producer_node is None:
            producer_triple = (dest_node, producer_channel, 0)
            if producer_channel == ChannelTypes.data_input \
                    or producer_channel == ChannelTypes.label_input or producer_channel == ChannelTypes.indices_input:
                if producer_channel == ChannelTypes.data_input:
                    tensor = self.add_networkwise_input(name=ChannelTypes.data_input.value, tensor_type=tf.float32)
                elif producer_channel == ChannelTypes.label_input:
                    tensor = self.add_networkwise_input(name=ChannelTypes.label_input.value, tensor_type=tf.int64)
                elif producer_channel == ChannelTypes.indices_input:
                    tensor = self.add_networkwise_input(name=ChannelTypes.indices_input.value, tensor_type=tf.int64)
                else:
                    raise NotImplementedError()
                network_io_object = NetworkIOObject(tensor=tensor, producer_node=None,
                                                    producer_channel=producer_channel,
                                                    producer_channel_index=producer_channel_index)
                dest_node.add_input(producer_triple=(None, producer_channel, 0), input_object=network_io_object)
                dest_node.add_output(producer_triple=producer_triple, output_object=network_io_object)
                return tensor
            else:
                raise Exception("Only data or label inputs can have no source node.")
        # An input either from a 1)directly from a parent node 2)from an ancestor node.
        # In the second case apply the propagation algorithm.
        else:
            producer_triple = (producer_node, producer_channel, producer_channel_index)
            path = self.dag.get_shortest_path(source=producer_node, dest=dest_node)
            last_output = None
            # Assume the path starts from the ancestor to the node
            # Since it is a tree, only a single path exists between the ancestor and the node.
            if len(path) <= 1:
                raise Exception("No path has been found between source node {0} and destination node {1}".
                                format(producer_node.index, dest_node.index))
            for path_node in path:
                # WE ASSUME THAT path_node IS TOPOLOGICALLY ORDERED!!!
                # The source node: The output must already exist before in the ancestor.
                # Output has gone through necessary decision processes.
                if path_node == producer_node:
                    if producer_triple not in path_node.outputs:
                        raise Exception("The output must already exist before.")
                    last_output = path_node.get_output(producer_triple=producer_triple)
                # An ancestor node which on the path [source,dest]. We need to make the desired output of source node
                # pass through the intermediate nodes on the path, such that decisions are applied to them correctly.
                elif path_node != producer_node and path_node != dest_node:
                    # This output is not used by path_node at all.
                    if producer_triple not in path_node.inputs and producer_triple not in path_node.outputs:
                        branched_tensor = path_node.apply_decision(tensor=last_output.tensor)
                        network_io_object = NetworkIOObject(tensor=branched_tensor, producer_node=producer_node,
                                                            producer_channel=producer_channel,
                                                            producer_channel_index=producer_channel_index)
                        path_node.add_input(producer_triple=producer_triple, input_object=network_io_object)
                        path_node.add_output(producer_triple=producer_triple, output_object=network_io_object)
                        last_output = path_node.get_output(producer_triple=producer_triple)
                    # This output is used by the path but for its internal calculations.
                    elif producer_triple in path_node.inputs and producer_triple not in path_node.outputs:
                        network_io_object = path_node.get_input(producer_triple=producer_triple)
                        path_node.add_output(producer_triple=producer_triple, output_object=network_io_object)
                        last_output = path_node.get_output(producer_triple=producer_triple)
                    # There cannot be an operation which is not input but in output. This is invalid.
                    elif producer_triple not in path_node.inputs and producer_triple in path_node.outputs:
                        raise Exception("Operation {0} is not in the input of the node {1} but in its outputs!".format(
                            producer_triple, path_node.index))
                    # Both in input and output.
                    # Then this output from the ancestor is already being broadcast from this node.
                    else:
                        last_output = path_node.get_output(producer_triple=producer_triple)
                # The node which will finally use the output.
                else:  # path_node == dest_node
                    # It shouldn't be in the input dict.
                    if producer_triple in path_node.inputs:
                        network_io_object = path_node.get_input(producer_triple=producer_triple)
                        return network_io_object.tensor
                    else:
                        branched_tensor = path_node.apply_decision(tensor=last_output.tensor)
                        network_io_object = NetworkIOObject(tensor=branched_tensor, producer_node=producer_node,
                                                            producer_channel=producer_channel,
                                                            producer_channel_index=producer_channel_index)
                        path_node.add_input(producer_triple=producer_triple, input_object=network_io_object)
                        return network_io_object.tensor

    def create_global_inputs(self):
        self.add_networkwise_input(name=GlobalInputNames.batch_size.value, tensor_type=tf.float32)
        self.add_networkwise_input(name=GlobalInputNames.branching_prob_threshold.value, tensor_type=tf.float32)

    def create_global_input_drivers(self):
        # Branching probability
        self.globalInputDrivers[GlobalInputNames.branching_prob_threshold.value] = \
            UtilityFuncs.create_parameter_from_train_program(
                parameter_name=GlobalInputNames.branching_prob_threshold.value, train_program=self.trainProgram)
        # Batch Size
        self.globalInputDrivers[GlobalInputNames.batch_size.value] = \
            UtilityFuncs.create_parameter_from_train_program(
                parameter_name=GlobalInputNames.batch_size.value, train_program=self.trainProgram)
        # Evaluation Batch Size
        self.globalInputDrivers[GlobalInputNames.evaluation_batch_size.value] = \
            UtilityFuncs.create_parameter_from_train_program(
                parameter_name=GlobalInputNames.evaluation_batch_size.value, train_program=self.trainProgram)
        # Epoch Count
        self.globalInputDrivers[GlobalInputNames.epoch_count.value] = \
            UtilityFuncs.create_parameter_from_train_program(
                parameter_name=GlobalInputNames.epoch_count.value, train_program=self.trainProgram)
        # All hyper parameters regarding to parameter training
        hyper_parameter_set = {GlobalInputNames.wd.value, GlobalInputNames.lr_initial.value,
                               GlobalInputNames.momentum.value, GlobalInputNames.weight_update_interval.value,
                               GlobalInputNames.lr_update_interval.value, GlobalInputNames.lr_decay_ratio.value}
        for node in self.nodes.values():
            for parameter in node.parametersDict.values():
                if parameter.parameterType == parameterTypes.learnable_parameter:
                    hyper_parameter_dict = {}
                    for hyper_parameter in hyper_parameter_set:
                        parameter_hyper_parameter_name = parameter.get_property_name(property_=hyper_parameter)
                        value_dict = self.trainProgram.load_settings_for_property(property_name=hyper_parameter)
                        hyper_parameter_value = self.trainProgram.decode_json_element_for_parameter(
                            parameter_name=parameter_hyper_parameter_name, json_element=value_dict)
                        hyper_parameter_dict[hyper_parameter] = hyper_parameter_value
                    # Decaying Parameter for the learning rate
                    lr_hyper_param_name = parameter.get_property_name(property_=GlobalInputNames.lr.value)
                    self.globalInputDrivers[lr_hyper_param_name] = \
                        DecayingParameter(name=lr_hyper_param_name,
                                          value=hyper_parameter_dict[GlobalInputNames.lr_initial.value],
                                          decay=hyper_parameter_dict[GlobalInputNames.lr_decay_ratio.value],
                                          decay_period=hyper_parameter_dict[GlobalInputNames.lr_update_interval.value])
                    # Fixed Parameter for the wd
                    wd_hyper_parameter_name = parameter.get_property_name(property_=GlobalInputNames.wd.value)
                    self.globalInputDrivers[wd_hyper_parameter_name] = \
                        FixedParameter(name=wd_hyper_parameter_name,
                                       value=hyper_parameter_dict[GlobalInputNames.wd.value])
                    # Fixed Parameter for the momentum value
                    momentum_hyper_parameter_name = \
                        parameter.get_property_name(property_=GlobalInputNames.momentum.value)
                    self.globalInputDrivers[momentum_hyper_parameter_name] = \
                        FixedParameter(name=momentum_hyper_parameter_name,
                                       value=hyper_parameter_dict[GlobalInputNames.momentum.value])
                    # Weight update parameter
                    weight_update_interval_parameter_name = \
                        parameter.get_property_name(property_=GlobalInputNames.weight_update_interval.value)
                    self.globalInputDrivers[weight_update_interval_parameter_name] = \
                        FixedParameter(name=weight_update_interval_parameter_name,
                                       value=hyper_parameter_dict[GlobalInputNames.weight_update_interval.value])
        print("X")

    def build_network(self):
        curr_index = 0
        # Step 1:
        # Topologically build the node ordering
        for depth in range(0, self.treeDepth):
            node_count_in_depth = pow(self.treeDegree, depth)
            for i in range(0, node_count_in_depth):
                is_root = depth == 0
                is_leaf = depth == (self.treeDepth - 1)
                if self.treeType == TreeType.hard:
                    node = HardTreeNode(index=curr_index, containing_network=self, is_root=is_root,
                                        is_leaf=is_leaf, is_accumulation=False)
                else:
                    raise NotImplementedError()
                self.add_node_to_depth(depth=depth, node=node)
                if is_leaf:
                    self.leafNodes.append(node)
                if not is_root:
                    parent_index = self.get_parent_index(node_index=curr_index)
                    self.dag.add_edge(parent=self.nodes[parent_index], child=node)
                else:
                    self.dag.add_node(node=node)
                curr_index += 1
        # Add a final, accumulation node, which combines all the losses from all parent nodes.
        accumulation_node = HardTreeNode(index=curr_index, containing_network=self, is_root=False, is_leaf=False,
                                         is_accumulation=True)
        self.add_node_to_depth(depth=self.treeDepth, node=accumulation_node)
        for node in self.nodes.values():
            if node.isLeaf:
                self.dag.add_edge(parent=node, child=accumulation_node)
        self.topologicalSortedNodes = self.dag.get_topological_sort()
        # Step 2:
        # Add network wise constant inputs (probability threshold, etc.)
        with tf.variable_scope(GlobalInputNames.global_scope.value):
            self.create_global_inputs()
        # Step 3:
        # Build the complete symbolic graph by building and connecting the symbolic graphs of the nodes
        for node in self.topologicalSortedNodes:
            node_depth = self.nodesToDepthsDict[node]
            # Add customized operations on the top.
            with tf.variable_scope(node.indicatorText):
                # Build the node-wise ops
                print("Node:{0}".format(node.indicatorText))
                # Build non accumulation node networks.
                if not node.isAccumulation:
                    # Build user defined local node network
                    self.nodeBuilderFunctions[node_depth](network=self, node=node)
                # If node is leaf, apply the actual objective_loss function. If accumulation,
                #  fetch all losses from all nodes
                # in the graph, apply gradient calculations, etc.
                if node.isLeaf or node.isAccumulation:
                    node.attach_loss_eval_channels()
                # If node is not leaf or accumulation (inner node), then apply branching mechanism.
                else:
                    node.attach_decision()
                if not node.isAccumulation:
                    # Evaluate sample distribution, if we want to.
                    if self.evalSampleDistribution:
                        sample_counter = SampleIndexCounter(parent_node=node)
                        NetworkNode.apply_loss(loss=sample_counter)
                # Build weight shrinkage losses.
                node.attach_shrinkage_losses()
        # Step 4:
        # Set the drivers for the global constants
        self.create_global_input_drivers()

    def check_grads(self):
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            epoch_count = self.get_networkwise_input_value(name=GlobalInputNames.epoch_count.value)
            batch_size = self.get_networkwise_input_value(name=GlobalInputNames.batch_size.value)
            global_training_hyperparameters_set = set(
                [key for key in
                 self.globalInputs.keys() if key not in {ChannelTypes.data_input.value,
                                                         ChannelTypes.label_input.value,
                                                         ChannelTypes.indices_input.value} and not key.endswith(
                    GlobalInputNames.parameter_update.value)])
            samples, labels, indices_list = self.dataset.get_next_batch()
            # Prepare the feed dict: Samples, labels and indices first.
            feed_dict = {self.get_networkwise_input(name=ChannelTypes.data_input.value): samples,
                         self.get_networkwise_input(name=ChannelTypes.label_input.value): labels,
                         self.get_networkwise_input(name=ChannelTypes.indices_input.value): indices_list}
            # Feed other inputs; except parameter update inputs. (The optimizer will use them)
            for input_name in global_training_hyperparameters_set:
                # Skip parameter update inputs
                if input_name == GlobalInputNames.branching_prob_threshold.value:
                    threshold = (1.0 / self.treeDegree) - self.get_networkwise_input_value(name=input_name)
                    feed_dict[self.globalInputs[input_name]] = threshold
                else:
                    feed_dict[self.globalInputs[input_name]] = self.get_networkwise_input_value(name=input_name)
            param_shapes = [param.get_shape().as_list() for param in self.allLearnableParameterTensorsList]
            x = [tensor for tensor in self.allLearnableParameterTensorsList if "Node_0_Convolution_Bias" in tensor.name][0]
            x_shape = x.shape.as_list()
            print("AAA")
            grad_check = tf.test.compute_gradient(x, x_shape,
                                                  self.totalObjectiveLossTensor, [], extra_feed_dict=feed_dict)
            print("XXX")

    def init_session(self):
        # Init parameters
        self.assignmentOpsList = []
        initial_values_dict = {}
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)
        for node in self.nodes.values():
            for parameter in node.parametersDict.values():
                self.assignmentOpsList.append(parameter.assignOp)

        # initial_values = self.session.run(self.allLearnableParameterTensorsList)
        # # Obtain initial values for each parameter and create the assign operator list for parameter update
        # # np.random.seed(int(time.time()))
        # np_seed = 88
        # np.random.seed(np_seed)
        # xavier_init = Xavier(factor_type="in", magnitude=2.34)

        # for node in self.nodes.values():
        #     for parameter in node.parametersDict.values():
        #         initial_values_dict[parameter.inputTensor] = \
        #             xavier_init.init_weight(arr=initial_values[parameter.globalParameterIndex])
        #         self.assignmentOpsList.append(parameter.assignOp)
        # self.session.run(self.assignmentOpsList, initial_values_dict)
        # initial_values = self.session.run(self.allLearnableParameterTensorsList)
        # for node in self.nodes.values():
        #     for parameter in node.parametersDict.values():
        #         parameter.valueArray = np.array(initial_values[parameter.globalParameterIndex])
        # self.save_parameters()

        # When loading parameters from file
        self.load_parameters(file_name="parameters_runId17")
        for node in self.nodes.values():
            for parameter in node.parametersDict.values():
                initial_values_dict[parameter.inputTensor] = parameter.valueArray
        self.session.run(self.assignmentOpsList, initial_values_dict)

    def train(self):
        # Create the fetch list, get the epoch count and the batch size
        iteration = 0
        epoch_count = self.get_networkwise_input_value(name=GlobalInputNames.epoch_count.value)
        batch_size = self.get_networkwise_input_value(name=GlobalInputNames.batch_size.value)
        global_training_hyperparameters_set = set(
            [key for key in
             self.globalInputs.keys() if key not in {ChannelTypes.data_input.value,
                                                     ChannelTypes.label_input.value,
                                                     ChannelTypes.indices_input.value} and not key.endswith(
                GlobalInputNames.parameter_update.value)])
        # Enter the training loop
        self.evaluate(dataset_type=DatasetTypes.training, iteration=iteration)
        self.evaluate(dataset_type=DatasetTypes.validation, iteration=iteration)
        for epoch_id in range(epoch_count):
            # An epoch is a complete pass on the whole dataset.
            self.dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
            print("*************Epoch {0}*************".format(epoch_id))
            while True:
                kv_store_rows = []
                # Get the next batch
                samples, labels, indices_list = self.dataset.get_next_batch)
                # Prepare the feed dict: Samples, labels and indices first.
                feed_dict = {self.get_networkwise_input(name=ChannelTypes.data_input.value): samples,
                             self.get_networkwise_input(name=ChannelTypes.label_input.value): labels,
                             self.get_networkwise_input(name=ChannelTypes.indices_input.value): indices_list}
                # Feed other inputs; except parameter update inputs. (The optimizer will use them)
                for input_name in global_training_hyperparameters_set:
                    # Skip parameter update inputs
                    if input_name == GlobalInputNames.branching_prob_threshold.value:
                        threshold = (1.0 / self.treeDegree) - self.get_networkwise_input_value(name=input_name)
                        feed_dict[self.globalInputs[input_name]] = threshold
                    else:
                        feed_dict[self.globalInputs[input_name]] = self.get_networkwise_input_value(name=input_name)
                # Run training pass, get individual loss values, total objective and regularization loss values and
                # gradients
                self.trainingResults = self.session.run(self.trainingTensorsDict, feed_dict)
                param_shapes = [param.get_shape().as_list() for param in self.allLearnableParameterTensorsList]
                grad_check = tf.test.compute_gradient(self.allLearnableParameterTensorsList, param_shapes,
                                                      self.totalObjectiveLossTensor, [])
                # self.parentNetwork.objectiveGradientTensors = tf.gradients(self.parentNetwork.totalObjectiveLossTensor,
                #                                                            self.parentNetwork.allLearnableParameterTensorsList)
                # test_grad = tf.test.compute_gradient(self.pa)
                # Print sample distribution
                sample_dist_str = "Iteration {0} ".format(iteration)
                for node in self.leafNodes:
                    sample_index_counter_loss = node.losses[SampleIndexCounter.get_loss_name(node=node)]
                    num_of_samples = self.get_outputs_for_single_loss(loss_object=sample_index_counter_loss)[0]
                    sample_dist_str += "Node{0}:{1} ".format(node.index, num_of_samples)
                print(sample_dist_str)
                # Run the optimizer
                new_values_dict, self.assignmentOpsList = self.optimizer.update()
                # Run the update phase; update trainable parameters with their new values
                # Throw aways ops for the nodes, which did not receive any sample.
                self.session.run(self.assignmentOpsList, new_values_dict)
                # Update the global parameters
                iteration += 1
                for global_param in self.globalInputDrivers.values():
                    global_param.update(iteration=iteration)
                # Check if an epoch has been completed.
                if self.dataset.isNewEpoch:
                    break
            print("*************Epoch {0}*************".format(epoch_id))
            # Evaluate on the training set and validation set
            training_accuracy = self.evaluate(dataset_type=DatasetTypes.training, iteration=iteration)
            validation_accuracy = self.evaluate(dataset_type=DatasetTypes.validation, iteration=iteration)
            log_table_row = (self.runId, iteration, epoch_id, training_accuracy, validation_accuracy, 0.0, 0.0,
                             self.__class__.__name__)
            DbLogger.log_into_log_table([log_table_row])
        final_accuracy = self.evaluate(dataset_type=DatasetTypes.test, iteration=iteration)
        DbLogger.write_into_table(rows=[(self.runId, self.explanation, final_accuracy)], table=DbLogger.runResultsTable,
                                  col_count=3)

    def evaluate(self, dataset_type, iteration):
        self.dataset.set_current_data_set_type(dataset_type=dataset_type)
        eval_batch_size = self.get_networkwise_input_value(name=GlobalInputNames.evaluation_batch_size.value)
        if self.problemType == ProblemType.classification:
            total_correct_count = 0.0
            total_sample_count = 0.0
            corrects_per_leaf = {}
            total_per_leaf = {}
            for node in self.leafNodes:
                corrects_per_leaf[node.index] = 0.0
                total_per_leaf[node.index] = 0.0
            kv_store_rows = []
            while True:
                # Get the next batch
                samples, labels, indices_list = self.dataset.get_next_batch()
                # Prepare the feed dict: Samples, labels and indices first, set the probability thresholds to 0.
                feed_dict = {self.get_networkwise_input(name=ChannelTypes.data_input.value): samples,
                             self.get_networkwise_input(name=ChannelTypes.label_input.value): labels,
                             self.get_networkwise_input(name=ChannelTypes.indices_input.value): indices_list,
                             self.globalInputs[GlobalInputNames.branching_prob_threshold.value]: (
                                 1.0 / self.treeDegree)}
                # Run evaluation pass, get individual loss values, total objective and regularization loss values and
                # gradients
                self.evaluationResults = self.session.run(self.evaluationTensorsDict, feed_dict)
                for node in self.leafNodes:
                    cross_entropy_loss_name = CrossEntropyLoss.get_loss_name(node=node)
                    results = self.evaluationResults[cross_entropy_loss_name]
                    leaf_correct_count = results[0]
                    leaf_sample_count = results[1]
                    corrects_per_leaf[node.index] += leaf_correct_count
                    total_per_leaf[node.index] += leaf_sample_count
                    total_correct_count += leaf_correct_count
                    total_sample_count += leaf_sample_count
                if self.dataset.isNewEpoch:
                    break
            if total_sample_count != self.dataset.get_current_sample_count():
                print("Expected sample count is {0} but {1} samples have been processed.".format(
                    self.dataset.get_current_sample_count(), total_sample_count))
            if dataset_type == DatasetTypes.training:
                dataset_type_str = "training"
            elif dataset_type == DatasetTypes.validation:
                dataset_type_str = "validation"
            elif dataset_type == DatasetTypes.test:
                dataset_type_str = "test"
            else:
                raise NotImplementedError()
            final_accuracy = total_correct_count / total_sample_count
            print("{0} set accuracy:{1}".format(dataset_type_str, final_accuracy))
            for node in self.leafNodes:
                if total_per_leaf[node.index] == 0:
                    leaf_accuracy = -1.0
                else:
                    leaf_accuracy = corrects_per_leaf[node.index] / total_per_leaf[node.index]
                print("Node{0} correct:{1} total:{2} accuracy:{3}".format(node.index, corrects_per_leaf[node.index],
                                                                          total_per_leaf[node.index], leaf_accuracy))
                kv_store_rows.append(
                    (self.runId, iteration, "{0} set, Node{1} Correct Count".
                     format(dataset_type_str, node.index), corrects_per_leaf[node.index]))
                kv_store_rows.append(
                    (self.runId, iteration, "{0} set, Node{1} Total Count".
                     format(dataset_type_str, node.index), total_per_leaf[node.index]))
                kv_store_rows.append(
                    (self.runId, iteration, "{0} set, Node{1} Accuracy".
                     format(dataset_type_str, node.index), leaf_accuracy))
            DbLogger.write_into_table(rows=kv_store_rows, table=DbLogger.runKvStore, col_count=4)
            return final_accuracy
        else:
            raise NotImplementedError()

    def get_outputs_for_single_loss(self, loss_object):
        if loss_object.lossOutputs is None:
            raise Exception("Loss output list is None.")
        return self.trainingResults[loss_object.get_name()]

    def get_total_loss_output(self, loss_type):
        if loss_type == LossType.objective:
            loss = self.trainingResults[ChannelTypes.total_objective_loss.value]
        elif loss_type == LossType.regularization:
            loss = self.trainingResults[ChannelTypes.total_regularization_loss.value]
        else:
            raise Exception("{0} is an unrecognized loss type.".format(loss_type.value))
        return loss

    def get_gradient_for_parameter(self, parameter_object, loss_type):
        if loss_type == LossType.objective:
            grad_tensors_list = self.trainingResults[ChannelTypes.objective_gradients.value]
        elif loss_type == LossType.regularization:
            grad_tensors_list = self.trainingResults[ChannelTypes.regularization_gradients.value]
        else:
            raise Exception("{0} is an unrecognized loss type.".format(loss_type.value))
        grad_for_parameter = grad_tensors_list[parameter_object.globalParameterIndex]
        return grad_for_parameter

    # Private methods
    def get_parent_index(self, node_index):
        parent_index = int((node_index - 1) / self.treeDegree)
        return parent_index

    def add_node_to_depth(self, depth, node):
        if not (depth in self.depthsToNodesDict):
            self.depthsToNodesDict[depth] = []
        self.depthsToNodesDict[depth].append(node)
        self.nodesToDepthsDict[node] = depth
        self.nodes[node.index] = node

    def get_root_node(self):
        return self.nodes[0]
