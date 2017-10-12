import tensorflow as tf
import numpy as np
from auxillary.dag_utilities import Dag
from simple_tf.global_params import GlobalConstants
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
        self.gradients = None
        self.sample_count_tensors = None
        self.momentumStatesDict = {}
        self.newValuesDict = {}
        self.assignOpsList = []
        self.learningRate = None
        self.globalCounter = None

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
        weights_and_filters = [v for v in vars if "bias" not in v.name]
        regularizer_loss = GlobalConstants.WEIGHT_DECAY_COEFFICIENT * tf.add_n(
            [tf.nn.l2_loss(v) for v in weights_and_filters])
        actual_loss = tf.add_n(all_network_losses)
        self.finalLoss = actual_loss + regularizer_loss
        self.evalDict["RegularizerLoss"] = regularizer_loss
        self.evalDict["ActualLoss"] = actual_loss
        self.evalDict["NetworkLoss"] = self.finalLoss
        self.sample_count_tensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "sample_count" in k}
        self.gradients = tf.gradients(ys=self.finalLoss, xs=vars)
        # Assign ops for variables
        for var in vars:
            op_name = self.get_assign_op_name(variable=var)
            new_value = tf.placeholder(name=op_name, dtype=GlobalConstants.DATA_TYPE)
            assign_op = tf.assign(ref=var, value=new_value)
            self.newValuesDict[op_name] = new_value
            self.assignOpsList.append(assign_op)
            self.momentumStatesDict[var.name] = np.zeros(shape=var.shape)
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
        results = sess.run([self.gradients, self.sample_count_tensors, vars, self.learningRate], feed_dict=feed_dict)
        grads = results[0]
        sample_counts = results[1]
        vars_current_values = results[2]
        lr = results[3]
        update_dict = {}
        for v, g, curr_value in zip(vars, grads, vars_current_values):
            self.momentumStatesDict[v.name][:] *= GlobalConstants.MOMENTUM_DECAY
            self.momentumStatesDict[v.name][:] += -lr * g
            new_value = curr_value + self.momentumStatesDict[v.name]
            update_dict[self.newValuesDict[self.get_assign_op_name(variable=v)]] = new_value
        sess.run(self.assignOpsList, feed_dict=update_dict)
        return sample_counts, lr

    def get_variable_name(self, name, node):
        return "Node{0}_{1}".format(node.index, name)

    def get_assign_op_name(self, variable):
        return "Assign_{0}".format(variable.name[0:len(variable.name) - 2])

    def apply_decision(self, node):
        # child_nodes = sorted(network.dagObject.children(node=node), key=lambda child: child.index)
        arg_max_indices = tf.argmax(input=node.activationsDict[node.index], axis=1)
        node.maskTensorsDict = {}
        for index in range(self.treeDegree):
            child_index = node.index * self.treeDegree + 1 + index
            mask_vector = tf.equal(x=arg_max_indices, y=tf.constant(index, tf.int64),
                                   name="Mask_{0}".format(child_index))
            mask_vector = tf.reshape(mask_vector, [-1])
            node.maskTensorsDict[node.index * self.treeDegree + 1 + index] = mask_vector

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