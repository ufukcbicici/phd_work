import tensorflow as tf
import numpy as np
from auxillary.dag_utilities import Dag
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants, GradientType
from simple_tf.info_gain import InfoGainLoss
from simple_tf.node import Node


class TreeNetwork:
    def __init__(self, tree_degree, node_build_funcs, create_new_variables):
        self.treeDegree = tree_degree
        self.dagObject = Dag()
        self.nodeBuildFuncs = node_build_funcs
        self.depth = len(self.nodeBuildFuncs)
        self.nodes = {}
        self.topologicalSortedNodes = []
        self.createNewVariables = create_new_variables
        self.dataTensor = GlobalConstants.TRAIN_DATA_TENSOR
        self.labelTensor = GlobalConstants.TRAIN_LABEL_TENSOR
        self.oneHotLabelTensor = GlobalConstants.TRAIN_ONE_HOT_LABELS
        # self.indicesTensor = indices
        self.evalDict = {}
        self.finalLoss = None
        self.classificationGradients = None
        self.regularizationGradients = None
        self.decisionLossGradients = None
        self.sample_count_tensors = None
        self.isOpenTensors = None
        self.momentumStatesDict = {}
        self.newValuesDict = {}
        self.assignOpsDict = {}
        self.learningRate = None
        self.globalCounter = None
        self.weightDecayCoeff = None
        self.probabilityThreshold = None
        self.useThresholding = None
        self.varToNodesDict = {}
        self.paramsDict = {}
        self.mainLossParamsDict = {}
        self.decisionParamsDict = {}
        self.initOp = None

    def get_parent_index(self, node_index):
        parent_index = int((node_index - 1) / self.treeDegree)
        return parent_index

    def get_fixed_variable(self, name, node):
        complete_name = self.get_variable_name(name=name, node=node)
        cnst = tf.constant(value=self.paramsDict["{0}:0".format(complete_name)], dtype=GlobalConstants.DATA_TYPE)
        return tf.Variable(cnst, name=complete_name)

    def get_decision_parameters(self):
        vars = tf.trainable_variables()
        H_vars = [v for v in vars if "hyperplane" in v.name]
        for i in range(len(H_vars)):
            self.decisionParamsDict[H_vars[i]] = i
        return H_vars

    def build_network(self, sess, dataset):
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
        # Probability threshold
        self.useThresholding = tf.placeholder(name="threshold_flag", dtype=tf.int64)
        self.probabilityThreshold = tf.placeholder(name="probability_threshold", dtype=tf.float32)
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        if not GlobalConstants.USE_RANDOM_PARAMETERS:
            self.paramsDict = UtilityFuncs.load_npz(file_name="parameters")
        for node in self.topologicalSortedNodes:
            self.nodeBuildFuncs[node.depth](node=node, network=self)
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
            for k, v in node.maskTensors.items():
                self.evalDict["Node{0}_{1}".format(node.index, v.name)] = v
            # Evaluation outputs
            for k, v in node.evalDict.items():
                self.evalDict[k] = v
            # Label outputs
            if node.labelTensor is not None:
                self.evalDict["Node{0}_label_tensor".format(node.index)] = node.labelTensor
            # One Hot Label outputs
            if node.oneHotLabelTensor is not None:
                self.evalDict["Node{0}_one_hot_label_tensor".format(node.index)] = node.oneHotLabelTensor
        # Learning rate, counter
        self.globalCounter = tf.Variable(0, dtype=GlobalConstants.DATA_TYPE, trainable=False)
        self.learningRate = tf.train.exponential_decay(
            GlobalConstants.INITIAL_LR,  # Base learning rate.
            self.globalCounter,  # Current index into the dataset.
            GlobalConstants.DECAY_STEP,  # Decay step.
            GlobalConstants.DECAY_RATE,  # Decay rate.
            staircase=True)
        # Prepare the cost function
        # Main losses
        primary_losses = []
        for node in self.topologicalSortedNodes:
            primary_losses.extend(node.lossList)
        # Weight decays
        self.weightDecayCoeff = tf.placeholder(name="weight_decay_coefficient", dtype=tf.float32)
        vars = tf.trainable_variables()
        l2_loss_list = []
        for v in vars:
            if "bias" in v.name:
                l2_loss_list.append(0.0 * tf.nn.l2_loss(v))
            else:
                l2_loss_list.append(self.weightDecayCoeff * tf.nn.l2_loss(v))
        # weights_and_filters = [v for v in vars if "bias" not in v.name]
        # Proxy, decision losses
        decision_losses = []
        for node in self.topologicalSortedNodes:
            if node.isLeaf:
                continue
            decision_losses.append(node.infoGainLoss)
        regularizer_loss = tf.add_n(l2_loss_list)
        primary_loss = tf.add_n(primary_losses)
        decision_loss = GlobalConstants.DECISION_LOSS_COEFFICIENT * tf.add_n(decision_losses)
        self.finalLoss = primary_loss + regularizer_loss + decision_loss
        self.evalDict["RegularizerLoss"] = regularizer_loss
        self.evalDict["PrimaryLoss"] = primary_loss
        self.evalDict["DecisionLoss"] = decision_loss
        self.evalDict["NetworkLoss"] = self.finalLoss
        self.sample_count_tensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "sample_count" in k}
        self.isOpenTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "is_open" in k}
        self.initOp = tf.global_variables_initializer()
        sess.run(self.initOp)
        self.get_trainable_vars(loss=primary_loss, sess=sess, dataset=dataset)
        self.classificationGradients = tf.gradients(ys=primary_loss, xs=vars)
        self.regularizationGradients = tf.gradients(ys=regularizer_loss, xs=vars)
        H_vars = self.get_decision_parameters()
        self.decisionLossGradients = tf.gradients(ys=decision_loss, xs=H_vars)
        # Assign ops for variables
        for var in vars:
            op_name = self.get_assign_op_name(variable=var)
            new_value = tf.placeholder(name=op_name, dtype=GlobalConstants.DATA_TYPE)
            assign_op = tf.assign(ref=var, value=new_value)
            self.newValuesDict[op_name] = new_value
            self.assignOpsDict[op_name] = assign_op
            self.momentumStatesDict[var.name] = np.zeros(shape=var.shape)
            for node in self.topologicalSortedNodes:
                if var in node.variablesSet:
                    if var.name in self.varToNodesDict:
                        raise Exception("{0} is in the parameters already.".format(var.name))
                    self.varToNodesDict[var.name] = node
            if var.name not in self.varToNodesDict:
                raise Exception("{0} is not in the parameters!".format(var.name))

    def get_trainable_vars(self, loss, sess, dataset):
        vars = tf.trainable_variables()
        accepted_vars = set()
        rejected_vars = set()
        grad_obj = None
        grads_index_dict = {}
        samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
        samples = np.expand_dims(samples, axis=3)
        feed_dict = {GlobalConstants.TRAIN_DATA_TENSOR: samples,
                     GlobalConstants.TRAIN_LABEL_TENSOR: labels,
                     GlobalConstants.TRAIN_ONE_HOT_LABELS: one_hot_labels,
                     self.globalCounter: 0,
                     self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
                     self.probabilityThreshold: 0.0,
                     self.useThresholding: 0}
        for candidate_v in vars:
            accepted_vars.add(candidate_v)
            print("Trying variable:{0}".format(candidate_v.name))
            grad_obj = tf.gradients(ys=loss, xs=list(accepted_vars))
            try:
                grad_results = sess.run([grad_obj], feed_dict=feed_dict)
            except TypeError as e:
                print(e)
                accepted_vars.remove(candidate_v)
                rejected_vars.add(candidate_v)
            else:
                grads_index_dict[candidate_v] = len(accepted_vars) - 1
        return accepted_vars, grads_index_dict

    def calculate_accuracy(self, sess, dataset, dataset_type, run_id):
        dataset.set_current_data_set_type(dataset_type=dataset_type)
        leaf_predicted_labels_dict = {}
        leaf_true_labels_dict = {}
        info_gain_dict = {}
        while True:
            results = self.eval_network(sess=sess, dataset=dataset)
            batch_sample_count = 0.0
            for node in self.topologicalSortedNodes:
                if not node.isLeaf:
                    info_gain = results[self.get_variable_name(name="info_gain", node=node)]
                    if node.index not in info_gain_dict:
                        info_gain_dict[node.index] = []
                    info_gain_dict[node.index].append(np.asscalar(info_gain))
                    continue
                if results[self.get_variable_name(name="is_open", node=node)] == 0.0:
                    continue
                posterior_probs = results[self.get_variable_name(name="posterior_probs", node=node)]
                true_labels = results[self.get_variable_name(name="labels", node=node)]
                # batch_sample_count += results[self.get_variable_name(name="sample_count", node=node)]
                predicted_labels = np.argmax(posterior_probs, axis=1)
                batch_sample_count += predicted_labels.shape[0]
                if node.index not in leaf_predicted_labels_dict:
                    leaf_predicted_labels_dict[node.index] = predicted_labels
                else:
                    leaf_predicted_labels_dict[node.index] = np.concatenate((leaf_predicted_labels_dict[node.index],
                                                                             predicted_labels))
                if node.index not in leaf_true_labels_dict:
                    leaf_true_labels_dict[node.index] = true_labels
                else:
                    leaf_true_labels_dict[node.index] = np.concatenate((leaf_true_labels_dict[node.index], true_labels))
            if batch_sample_count != GlobalConstants.BATCH_SIZE:
                raise Exception("Incorrect batch size:{0}".format(batch_sample_count))
            if dataset.isNewEpoch:
                break
        print("****************Dataset:{0}****************".format(dataset_type))
        # Measure Information Gain
        for k, v in info_gain_dict.items():
            avg_info_gain = sum(v) / float(len(v))
            print("{0}={1}".format(k, avg_info_gain))
        # Measure Accuracy
        overall_count = 0.0
        overall_correct = 0.0
        confusion_matrix_db_rows = []
        for node in self.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            if node.index not in leaf_predicted_labels_dict:
                continue
            predicted = leaf_predicted_labels_dict[node.index]
            true_labels = leaf_true_labels_dict[node.index]
            if predicted.shape != true_labels.shape:
                raise Exception("Predicted and true labels counts do not hold.")
            correct_count = np.sum(predicted == true_labels)
            # Get the incorrect predictions by preparing a confusion matrix for each leaf
            sparse_confusion_matrix = {}
            for i in range(predicted.shape[0]):
                predicted_label = predicted[i]
                true_label = true_labels[i]
                if (predicted_label, true_label) not in sparse_confusion_matrix:
                    sparse_confusion_matrix[(predicted_label, true_label)] = 0
                sparse_confusion_matrix[(predicted_label, true_label)] += 1
            for k, v in sparse_confusion_matrix.items():
                confusion_matrix_db_rows.append((run_id, dataset_type.value, node.index, k[0], k[1], v))
            # Overall accuracy
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
            if node.index not in leaf_true_labels_dict:
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
        return overall_correct / overall_count, confusion_matrix_db_rows

    def update_params_with_momentum(self, sess, dataset, iteration):
        samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
        samples = np.expand_dims(samples, axis=3)
        vars = tf.trainable_variables()
        use_threshold = int(GlobalConstants.USE_PROBABILITY_THRESHOLD)
        prob_threshold = (1.0 / float(GlobalConstants.TREE_DEGREE)) - GlobalConstants.PROBABILITY_THRESHOLD.value
        print("prob_threshold={0}".format(prob_threshold))
        # info_gain_dicts = {k: v for k, v in self.evalDict.items() if "info_gain" in k}
        feed_dict = {GlobalConstants.TRAIN_DATA_TENSOR: samples,
                     GlobalConstants.TRAIN_LABEL_TENSOR: labels,
                     GlobalConstants.TRAIN_ONE_HOT_LABELS: one_hot_labels,
                     self.globalCounter: iteration,
                     self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
                     self.probabilityThreshold: prob_threshold,
                     self.useThresholding: use_threshold}
        results_threshold = sess.run(
            [self.classificationGradients,
             self.regularizationGradients,
             self.sample_count_tensors,
             vars,
             self.learningRate,
             self.isOpenTensors], feed_dict=feed_dict)
        # Only calculate the derivatives for information gain losses
        classification_grads = results_threshold[0]
        regularization_grads = results_threshold[1]
        sample_counts_threshold = results_threshold[2]
        vars_current_values = results_threshold[3]
        lr = results_threshold[4]
        is_open_indicators_threshold = results_threshold[5]
        decision_grads = None
        sample_counts_no_threshold = None
        is_open_indicators_no_threshold = None
        info_gain_results = None
        if GlobalConstants.USE_INFO_GAIN_DECISION:
            info_gain_dicts = {k: v for k, v in self.evalDict.items() if "info_gain" in k}
            feed_dict = {GlobalConstants.TRAIN_DATA_TENSOR: samples,
                         GlobalConstants.TRAIN_LABEL_TENSOR: labels,
                         GlobalConstants.TRAIN_ONE_HOT_LABELS: one_hot_labels,
                         self.globalCounter: iteration,
                         self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
                         self.probabilityThreshold: prob_threshold,
                         self.useThresholding: 0}
            results_no_threshold = sess.run(
                [self.decisionLossGradients,
                 self.sample_count_tensors,
                 self.isOpenTensors,
                 info_gain_dicts], feed_dict=feed_dict)
            decision_grads_temp = results_no_threshold[0]
            sample_counts_no_threshold = results_no_threshold[1]
            is_open_indicators_no_threshold = results_no_threshold[2]
            info_gain_results = results_no_threshold[3]
            decision_grads = []
            for v in vars:
                if v not in self.decisionParamsDict:
                    decision_grads.append(np.zeros(shape=v.shape))
                elif is_open_indicators_no_threshold[self.get_variable_name(name="is_open",
                                                                                       node=self.varToNodesDict[v.name])] == 0.0:
                    decision_grads.append(np.zeros(shape=v.shape))
                else:
                    decision_grads.append(decision_grads_temp[self.decisionParamsDict[v]])
        GlobalConstants.PROBABILITY_THRESHOLD.update(iteration=iteration)
        # info_gain_results = results[6]
        # UtilityFuncs.save_npz(file_name="parameters", arr_dict=params_dict)
        if (GlobalConstants.GRADIENT_TYPE == GradientType.mixture_of_experts_unbiased) or (
                    GlobalConstants.GRADIENT_TYPE == GradientType.parallel_dnns_unbiased):
            update_dict = {}
            assign_dict = {}
            for v, g, r, d, curr_value in zip(vars, classification_grads, regularization_grads, decision_grads,
                                              vars_current_values):
                node = self.varToNodesDict[v.name]
                is_node_open_threshold = is_open_indicators_threshold[self.get_variable_name(name="is_open", node=node)]
                if not is_node_open_threshold:
                    continue
                self.momentumStatesDict[v.name][:] *= GlobalConstants.MOMENTUM_DECAY
                self.momentumStatesDict[v.name][:] += -lr * (g + r)
                new_value = curr_value + self.momentumStatesDict[v.name]
                op_name = self.get_assign_op_name(variable=v)
                update_dict[self.newValuesDict[op_name]] = new_value
                assign_dict[op_name] = self.assignOpsDict[op_name]
            sess.run(assign_dict, feed_dict=update_dict)
        elif GlobalConstants.GRADIENT_TYPE == GradientType.mixture_of_experts_biased:
            update_dict = {}
            assign_dict = {}
            for v, g, r, d, curr_value in zip(vars, classification_grads, regularization_grads, decision_grads,
                                              vars_current_values):
                node = self.varToNodesDict[v.name]
                is_node_open = is_open_indicators_threshold[self.get_variable_name(name="is_open", node=node)]
                if not is_node_open:
                    # print("Skipping Node{0} Parameter:{1}".format(node.index, v.name))
                    continue
                self.momentumStatesDict[v.name][:] *= GlobalConstants.MOMENTUM_DECAY
                sample_count_entry_name = self.get_variable_name(name="sample_count", node=node)
                sample_count = sample_counts_threshold[sample_count_entry_name]
                gradient_modifier = float(GlobalConstants.BATCH_SIZE) / float(sample_count)
                modified_g = gradient_modifier * g
                self.momentumStatesDict[v.name][:] += -lr * (modified_g + r)
                new_value = curr_value + self.momentumStatesDict[v.name]
                op_name = self.get_assign_op_name(variable=v)
                update_dict[self.newValuesDict[op_name]] = new_value
                assign_dict[op_name] = self.assignOpsDict[op_name]
            sess.run(assign_dict, feed_dict=update_dict)
        else:
            raise NotImplementedError()
        return sample_counts_threshold, lr, is_open_indicators_threshold

    def get_variable_name(self, name, node):
        return "Node{0}_{1}".format(node.index, name)

    def get_node_from_variable_name(self, name):
        node_index_str = name[4:name.find("_")]
        node_index = int(node_index_str)
        return self.nodes[node_index]

    def get_assign_op_name(self, variable):
        return "Assign_{0}".format(variable.name[0:len(variable.name) - 2])

    def mask_input_nodes(self, node):
        if node.isRoot:
            node.labelTensor = self.labelTensor
            node.oneHotLabelTensor = self.oneHotLabelTensor
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = tf.size(node.labelTensor)
            node.isOpenIndicatorTensor = tf.constant(value=1.0, dtype=tf.float32)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
        else:
            # Obtain the mask vector, sample counts and determine if this node receives samples.
            parent_node = self.dagObject.parents(node=node)[0]
            mask_tensor = parent_node.maskTensors[node.index]
            sample_count_tensor = tf.reduce_sum(tf.cast(mask_tensor, tf.float32))
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = sample_count_tensor
            node.isOpenIndicatorTensor = tf.where(sample_count_tensor > 0.0, 1.0, 0.0)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            # TO PREVENT TENSORFLOW FROM CRASHING WHEN THERE ARE NO SAMPLES:
            # If the mask from the parent is completely false, convert it to true; artifically. Note this
            # is only for crash prevention. "node.isOpenIndicatorTensor" will be 0 anyways, so no parameter update will
            # be made for that node. Moreover, when applying decision, we will look control "node.isOpenIndicatorTensor"
            # and set the produced mask vectors to competely false, to propagate the emptyness.
            if GlobalConstants.USE_EMPTY_NODE_CRASH_PREVENTION:
                mask_tensor = tf.where(node.isOpenIndicatorTensor > 0.0, x=mask_tensor,
                                       y=tf.logical_or(x=tf.constant(value=True, dtype=tf.bool), y=mask_tensor))
            # Mask all inputs: F channel, H channel, activations from ancestors, labels
            parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
            parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
            for k, v in parent_node.activationsDict.items():
                node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
            node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
            node.oneHotLabelTensor = tf.boolean_mask(parent_node.oneHotLabelTensor, mask_tensor)
            return parent_F, parent_H

    def apply_decision(self, node):
        p_n_given_x = tf.nn.softmax(node.activationsDict[node.index])
        p_c_given_x = node.oneHotLabelTensor
        node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_n_given_x, p_c_given_x_2d=p_c_given_x)
        node.evalDict[self.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
        node.evalDict[self.get_variable_name(name="p(n|x)", node=node)] = p_n_given_x
        arg_max_indices = tf.argmax(p_n_given_x, axis=1)
        for index in range(self.treeDegree):
            child_index = node.index * self.treeDegree + 1 + index
            branch_prob = p_n_given_x[:, index]
            mask_with_threshold = tf.reshape(tf.greater_equal(x=branch_prob, y=self.probabilityThreshold,
                                                              name="Mask_with_threshold_{0}".format(child_index)), [-1])
            mask_without_threshold = tf.reshape(tf.equal(x=arg_max_indices, y=tf.constant(index, tf.int64),
                                                         name="Mask_without_threshold_{0}".format(child_index)), [-1])
            mask_tensor = tf.where(self.useThresholding > 0, x=mask_with_threshold, y=mask_without_threshold)
            if GlobalConstants.USE_EMPTY_NODE_CRASH_PREVENTION:
                # Zero-out the mask if this node is not open;
                # since we only use the mask vector to avoid Tensorflow crash in this case.
                node.maskTensors[child_index] = tf.where(node.isOpenIndicatorTensor > 0.0, x=mask_tensor,
                                                         y=tf.logical_and(
                                                             x=tf.constant(value=False, dtype=tf.bool), y=mask_tensor))
            else:
                node.maskTensors[child_index] = mask_tensor

    def apply_loss(self, node, logits):
        cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node.labelTensor,
                                                                                   logits=logits)
        parallel_dnn_updates = {GradientType.parallel_dnns_unbiased, GradientType.parallel_dnns_biased}
        mixture_of_expert_updates = {GradientType.mixture_of_experts_biased, GradientType.mixture_of_experts_unbiased}
        if GlobalConstants.GRADIENT_TYPE in parallel_dnn_updates:
            pre_loss = tf.reduce_mean(cross_entropy_loss_tensor)
            loss = tf.where(tf.is_nan(pre_loss), 0.0, pre_loss)
        elif GlobalConstants.GRADIENT_TYPE in mixture_of_expert_updates:
            pre_loss = tf.reduce_sum(cross_entropy_loss_tensor)
            loss = (1.0 / float(GlobalConstants.BATCH_SIZE)) * pre_loss
        else:
            raise NotImplementedError()
        node.fOpsList.extend([cross_entropy_loss_tensor, pre_loss, loss])
        node.lossList.append(loss)

    def eval_network(self, sess, dataset):
        # if is_train:
        samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
        samples = np.expand_dims(samples, axis=3)
        feed_dict = {
            GlobalConstants.TRAIN_DATA_TENSOR: samples,
            GlobalConstants.TRAIN_LABEL_TENSOR: labels,
            GlobalConstants.TRAIN_ONE_HOT_LABELS: one_hot_labels,
            self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
            self.probabilityThreshold: 0.0,
            self.useThresholding: 0}
        results = sess.run(self.evalDict, feed_dict)
        # else:
        #     samples, labels, indices_list = dataset.get_next_batch(batch_size=EVAL_BATCH_SIZE)
        #     samples = np.expand_dims(samples, axis=3)
        #     feed_dict = {TEST_DATA_TENSOR: samples, TEST_LABEL_TENSOR: labels}
        #     results = sess.run(network.evalDict, feed_dict)
        return results
