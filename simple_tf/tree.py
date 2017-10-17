import tensorflow as tf
import numpy as np
from auxillary.dag_utilities import Dag
from simple_tf.global_params import GlobalConstants, GradientType
from simple_tf.node import Node


class TreeNetwork:
    def __init__(self, tree_degree, node_build_funcs, create_new_variables, data, label):
        self.treeDegree = tree_degree
        self.dagObject = Dag()
        self.nodeBuildFuncs = node_build_funcs
        self.depth = len(self.nodeBuildFuncs)
        self.nodes = {}
        self.topologicalSortedNodes = []
        self.createNewVariables = create_new_variables
        self.dataTensor = data
        self.labelTensor = label
        self.evalDict = {}
        self.finalLoss = None
        self.classificationGradients = None
        self.regularizationGradients = None
        self.decisionLossGradients = None
        self.sample_count_tensors = None
        self.isOpenTensors = None
        self.momentumStatesDict = {}
        self.newValuesDict = {}
        self.assignOpsList = []
        self.learningRate = None
        self.globalCounter = None
        self.varToNodesDict = {}

    def get_parent_index(self, node_index):
        parent_index = int((node_index - 1) / self.treeDegree)
        return parent_index

    def build_network(self, network_to_copy_from):
        curr_index = 0
        for depth in range(0, self.depth):
            node_count_in_depth = pow(self.treeDegree, depth)
            for i in range(0, node_count_in_depth):
                is_root = depth == 0
                is_leaf = depth == (self.depth - 1)
                node = Node(index=curr_index, depth=depth, is_root=is_root, is_leaf=is_leaf)
                self.nodes[curr_index] = node
                if not is_root:
                    parent_index = self.get_parent_index(node_index=curr_index)
                    self.dagObject.add_edge(parent=self.nodes[parent_index], child=node)
                else:
                    self.dagObject.add_node(node=node)
                curr_index += 1
        # Build symbolic networks
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        for node in self.topologicalSortedNodes:
            if self.createNewVariables:
                self.nodeBuildFuncs[node.depth](node=node, network=self)
            else:
                self.nodeBuildFuncs[node.depth](node=node, network=self,
                                                variables=network_to_copy_from.nodes[node.index].variablesList)
        # Prepare tensors to evaluate
        for node in self.topologicalSortedNodes:
            # if node.isLeaf:
            #     continue
            # F
            f_output = node.fOpsList[-1]
            self.evalDict["Node{0}_F".format(node.index)] = f_output
            # H
            if len(node.hOpsList) > 0:
                h_output = node.hOpsList[-1]
                self.evalDict["Node{0}_H".format(node.index)] = h_output
            # Activations
            for k, v in node.activationsDict.items():
                self.evalDict["Node{0}_activation_from_{1}".format(node.index, k)] = v
            # Decision masks
            for k, v in node.maskTensorsDict.items():
                self.evalDict["Node{0}_{1}".format(node.index, v.name)] = v
            # Evaluation outputs
            for k, v in node.evalDict.items():
                self.evalDict[k] = v
            # Label outputs
            if node.labelTensor is not None:
                self.evalDict["Node{0}_label_tensor".format(node.index)] = node.labelTensor
        # Prepare losses
        all_network_losses = []
        for node in self.topologicalSortedNodes:
            all_network_losses.extend(node.lossList)
        # Weight decays
        vars = tf.trainable_variables()
        # l2_loss_list = [0.0 * tf.nn.l2_loss(v) if "bias" in v.name else GlobalConstants.WEIGHT_DECAY_COEFFICIENT * tf.nn.l2_loss(v) for v in vars]
        l2_loss_list = []
        for v in vars:
            if "bias" in v.name:
                l2_loss_list.append(0.0 * tf.nn.l2_loss(v))
            else:
                l2_loss_list.append(GlobalConstants.WEIGHT_DECAY_COEFFICIENT * tf.nn.l2_loss(v))
        # weights_and_filters = [v for v in vars if "bias" not in v.name]
        regularizer_loss = tf.add_n(l2_loss_list)
        actual_loss = tf.add_n(all_network_losses)
        self.finalLoss = actual_loss + regularizer_loss
        self.evalDict["RegularizerLoss"] = regularizer_loss
        self.evalDict["ActualLoss"] = actual_loss
        self.evalDict["NetworkLoss"] = self.finalLoss
        self.sample_count_tensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "sample_count" in k}
        self.isOpenTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "is_open" in k}
        self.classificationGradients = tf.gradients(ys=actual_loss, xs=vars)
        self.regularizationGradients = tf.gradients(ys=regularizer_loss, xs=vars)
        # Assign ops for variables
        for var in vars:
            op_name = self.get_assign_op_name(variable=var)
            new_value = tf.placeholder(name=op_name, dtype=GlobalConstants.DATA_TYPE)
            assign_op = tf.assign(ref=var, value=new_value)
            self.newValuesDict[op_name] = new_value
            self.assignOpsList.append(assign_op)
            self.momentumStatesDict[var.name] = np.zeros(shape=var.shape)
            for node in self.topologicalSortedNodes:
                if var in node.variablesSet:
                    if var.name in self.varToNodesDict:
                        raise Exception("{0} is in the parameters already.".format(var.name))
                    self.varToNodesDict[var.name] = node
            if var.name not in self.varToNodesDict:
                raise Exception("{0} is not in the parameters!".format(var.name))
        # Learning rate, counter
        self.globalCounter = tf.Variable(0, dtype=GlobalConstants.DATA_TYPE, trainable=False)
        self.learningRate = tf.train.exponential_decay(
            GlobalConstants.INITIAL_LR,  # Base learning rate.
            self.globalCounter,  # Current index into the dataset.
            GlobalConstants.DECAY_STEP,  # Decay step.
            GlobalConstants.DECAY_RATE,  # Decay rate.
            staircase=True)

    def calculate_accuracy(self, sess, dataset, dataset_type):
        dataset.set_current_data_set_type(dataset_type=dataset_type)
        leaf_predicted_labels_dict = {}
        leaf_true_labels_dict = {}
        while True:
            results = self.eval_network(sess=sess, dataset=dataset)
            for node in self.topologicalSortedNodes:
                if not node.isLeaf:
                    continue
                posterior_probs = results[self.get_variable_name(name="posterior_probs", node=node)]
                true_labels = results[self.get_variable_name(name="labels", node=node)]
                predicted_labels = np.argmax(posterior_probs, axis=1)
                if node.index not in leaf_predicted_labels_dict:
                    leaf_predicted_labels_dict[node.index] = predicted_labels
                else:
                    leaf_predicted_labels_dict[node.index] = np.concatenate((leaf_predicted_labels_dict[node.index],
                                                                             predicted_labels))
                if node.index not in leaf_true_labels_dict:
                    leaf_true_labels_dict[node.index] = true_labels
                else:
                    leaf_true_labels_dict[node.index] = np.concatenate((leaf_true_labels_dict[node.index], true_labels))
            if dataset.isNewEpoch:
                break
        print("****************Dataset:{0}****************".format(dataset_type))
        # Measure Accuracy
        overall_count = 0.0
        overall_correct = 0.0
        for node in self.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            predicted = leaf_predicted_labels_dict[node.index]
            true_labels = leaf_true_labels_dict[node.index]
            if predicted.shape != true_labels.shape:
                raise Exception("Predicted and true labels counts do not hold.")
            correct_count = np.sum(predicted == true_labels)
            total_count = true_labels.shape[0]
            overall_correct += correct_count
            overall_count += total_count
            if total_count > 0:
                print("Leaf {0}: Sample Count:{1} Accuracy:{2}".format(node.index, total_count,
                                                                       float(correct_count) / float(total_count)))
            else:
                print("Leaf {0} is empty.".format(node.index))
        print("*************Overall {0} samples. Overall Accuracy:{1}*************"
              .format(overall_count, overall_correct / overall_count))
        total_accuracy = overall_correct / overall_count
        # Measure overall label distribution in leaves
        for node in self.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            true_labels = leaf_true_labels_dict[node.index]
            frequencies = {}
            distribution_str = ""
            total_sample_count = true_labels.shape[0]
            for label in range(dataset.get_label_count()):
                frequencies[label] = np.sum(true_labels == label)
                distribution_str += "{0}:{1:.3f} ".format(label, frequencies[label] / float(total_sample_count))
            print("Node{0} Label Distribution: {1}".format(node.index, distribution_str))
        # Measure overall information gain
        return overall_correct / overall_count

    def update_params_with_momentum(self, sess, dataset, iteration):
        samples, labels, indices_list = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
        samples = np.expand_dims(samples, axis=3)
        vars = tf.trainable_variables()
        feed_dict = {GlobalConstants.TRAIN_DATA_TENSOR: samples, GlobalConstants.TRAIN_LABEL_TENSOR: labels,
                     self.globalCounter: iteration}
        results = sess.run([self.classificationGradients, self.regularizationGradients,
                            self.sample_count_tensors, vars, self.learningRate, self.isOpenTensors], feed_dict=feed_dict)
        classification_grads = results[0]
        regularization_grads = results[1]
        sample_counts = results[2]
        vars_current_values = results[3]
        lr = results[4]
        is_open_indicators = results[5]
        if (GlobalConstants.GRADIENT_TYPE == GradientType.mixture_of_experts_unbiased) or (
            GlobalConstants.GRADIENT_TYPE == GradientType.parallel_dnns_unbiased):
            update_dict = {}
            for v, g, r, curr_value in zip(vars, classification_grads, regularization_grads, vars_current_values):
                self.momentumStatesDict[v.name][:] *= GlobalConstants.MOMENTUM_DECAY
                self.momentumStatesDict[v.name][:] += -lr * (g + r)
                new_value = curr_value + self.momentumStatesDict[v.name]
                update_dict[self.newValuesDict[self.get_assign_op_name(variable=v)]] = new_value
            sess.run(self.assignOpsList, feed_dict=update_dict)
        elif GlobalConstants.GRADIENT_TYPE == GradientType.mixture_of_experts_biased:
            update_dict = {}
            for v, g, r, curr_value in zip(vars, classification_grads, regularization_grads, vars_current_values):
                self.momentumStatesDict[v.name][:] *= GlobalConstants.MOMENTUM_DECAY
                node = self.varToNodesDict[v.name]
                sample_count_entry_name = self.get_variable_name(name="sample_count", node=node)
                sample_count = sample_counts[sample_count_entry_name]
                gradient_modifier = float(GlobalConstants.BATCH_SIZE) / float(sample_count)
                modified_g = gradient_modifier * g
                self.momentumStatesDict[v.name][:] += -lr * (modified_g + r)
                new_value = curr_value + self.momentumStatesDict[v.name]
                update_dict[self.newValuesDict[self.get_assign_op_name(variable=v)]] = new_value
            sess.run(self.assignOpsList, feed_dict=update_dict)
        else:
            raise NotImplementedError()
        return sample_counts, lr, is_open_indicators

    def get_variable_name(self, name, node):
        return "Node{0}_{1}".format(node.index, name)

    def get_assign_op_name(self, variable):
        return "Assign_{0}".format(variable.name[0:len(variable.name) - 2])

    def mask_input_nodes(self, node):
        if node.isRoot:
            node.labelTensor = self.labelTensor
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = tf.size(node.labelTensor)
            node.isOpenIndicatorTensor = tf.constant(value=1.0, dtype=tf.float32)
        else:
            # Obtain the mask vector, sample counts and determine if this node receives samples.
            parent_node = self.dagObject.parents(node=node)[0]
            # print_op = tf.Print(input_=parent_node.fOpsList[-1], data=[parent_node.fOpsList[-1]], message="Print at Node:{0}".format(node.index))
            # node.evalDict[network.get_variable_name(name="Print", node=node)] = print_op
            mask_tensor = parent_node.maskTensorsDict[node.index]
            sample_count_tensor = tf.reduce_sum(tf.cast(mask_tensor, tf.float32))
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = sample_count_tensor
            node.isOpenIndicatorTensor = tf.where(sample_count_tensor > 0.0, 1.0, 0.0)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            # Mask all inputs: F channel, H channel, activations from ancestors, labels
            if GlobalConstants.USE_CPU_MASKING:
                with tf.device("/cpu:0"):
                    parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
                    parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
                    for k, v in parent_node.activationsDict.items():
                        node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
                    node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
            else:
                parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
                parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
                for k, v in parent_node.activationsDict.items():
                    node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
                node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
            return parent_F, parent_H

    def apply_decision(self, node):
        # child_nodes = sorted(network.dagObject.children(node=node), key=lambda child: child.index)
        arg_max_indices = tf.argmax(input=node.activationsDict[node.index], axis=1)
        node.maskTensorsDict = {}
        for index in range(self.treeDegree):
            child_index = node.index * self.treeDegree + 1 + index
            mask_vector = tf.equal(x=arg_max_indices, y=tf.constant(index, tf.int64),
                                   name="Mask_{0}".format(child_index))
            mask_vector = tf.reshape(mask_vector, [-1])
            # Zero-out the mask if this node is not open
            node.maskTensorsDict[node.index * self.treeDegree + 1 + index] = \
                tf.cond(node.isOpenIndicatorTensor > 0.0, lambda: mask_vector,
                        lambda: tf.logical_and(x=tf.constant(value=False, dtype=tf.bool), y=mask_vector))

    def eval_network(self, sess, dataset):
        # if is_train:
        samples, labels, indices_list = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
        samples = np.expand_dims(samples, axis=3)
        feed_dict = {GlobalConstants.TRAIN_DATA_TENSOR: samples, GlobalConstants.TRAIN_LABEL_TENSOR: labels}
        results = sess.run(self.evalDict, feed_dict)
        # else:
        #     samples, labels, indices_list = dataset.get_next_batch(batch_size=EVAL_BATCH_SIZE)
        #     samples = np.expand_dims(samples, axis=3)
        #     feed_dict = {TEST_DATA_TENSOR: samples, TEST_LABEL_TENSOR: labels}
        #     results = sess.run(network.evalDict, feed_dict)
        return results
