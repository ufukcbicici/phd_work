import tensorflow as tf
import numpy as np

from auxillary.constants import DatasetTypes
from auxillary.dag_utilities import Dag
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants, GradientType
from simple_tf.info_gain import InfoGainLoss
from simple_tf.node import Node
from collections import deque


class TreeNetwork:
    def __init__(self, node_build_funcs, grad_func, threshold_func, summary_func, degree_list):
        self.dagObject = Dag()
        self.nodeBuildFuncs = node_build_funcs
        self.depth = len(self.nodeBuildFuncs)
        self.nodes = {}
        self.topologicalSortedNodes = []
        self.gradFunc = grad_func
        self.thresholdFunc = threshold_func
        self.summaryFunc = summary_func
        self.degreeList = degree_list
        self.dataTensor = GlobalConstants.TRAIN_DATA_TENSOR
        self.labelTensor = GlobalConstants.TRAIN_LABEL_TENSOR
        self.oneHotLabelTensor = GlobalConstants.TRAIN_ONE_HOT_LABELS
        # self.indicesTensor = indices
        self.evalDict = {}
        self.mainLoss = None
        self.decisionLoss = None
        self.regularizationLoss = None
        self.finalLoss = None
        self.classificationGradients = None
        self.regularizationGradients = None
        self.decisionGradients = None
        self.sample_count_tensors = None
        self.isOpenTensors = None
        self.momentumStatesDict = {}
        self.newValuesDict = {}
        self.assignOpsDict = {}
        self.learningRate = None
        self.globalCounter = None
        self.weightDecayCoeff = None
        self.useThresholding = None
        self.iterationHolder = None
        self.decisionDropoutKeepProb = None
        self.decisionDropoutKeepProbCalculator = GlobalConstants.DROPOUT_CALCULATOR
        self.isTrain = None
        self.useMasking = None
        self.isDecisionPhase = None
        self.varToNodesDict = {}
        self.mainLossParamsDict = {}
        self.regularizationParamsDict = {}
        self.decisionParamsDict = {}
        self.initOp = None
        self.classificationPathSummaries = []
        self.decisionPathSummaries = []
        self.summaryWriter = None
        self.branchingBatchNormAssignOps = []
        self.modesPerLeaves = {}

    # def get_parent_index(self, node_index):
    #     parent_index = int((node_index - 1) / self.treeDegree)
    #     return parent_index

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

    def build_network(self):
        # curr_index = 0
        # prev_depth_node_count = 1
        # for depth in range(0, self.depth):
        #     degree = self.degreeList[depth]
        #     if depth == 0:
        #         node_count_in_depth = 1
        #     else:
        #         node_count_in_depth =
        #     for i in range(0, node_count_in_depth):
        #         is_root = depth == 0
        #         is_leaf = depth == (self.depth - 1)
        #         node = Node(index=curr_index, depth=depth, is_root=is_root, is_leaf=is_leaf)
        #         self.nodes[curr_index] = node
        #         if not is_root:
        #             parent_index = self.get_parent_index(node_index=curr_index)
        #             self.dagObject.add_edge(parent=self.nodes[parent_index], child=node)
        #         else:
        #             self.dagObject.add_node(node=node)
        #         curr_index += 1
        # Create itself
        curr_index = 0
        is_leaf = 0 == (self.depth - 1)
        root_node = Node(index=curr_index, depth=0, is_root=True, is_leaf=is_leaf)
        threshold_name = self.get_variable_name(name="threshold", node=root_node)
        root_node.probabilityThreshold = tf.placeholder(name=threshold_name, dtype=tf.float32)
        softmax_decay_name = self.get_variable_name(name="softmax_decay", node=root_node)
        root_node.softmaxDecay = tf.placeholder(name=softmax_decay_name, dtype=tf.float32)
        self.dagObject.add_node(node=root_node)
        self.nodes[curr_index] = root_node
        d = deque()
        d.append(root_node)
        # Create children if not leaf
        while len(d) > 0:
            # Dequeue
            curr_node = d.popleft()
            if not curr_node.isLeaf:
                for i in range(self.degreeList[curr_node.depth]):
                    new_depth = curr_node.depth + 1
                    is_leaf = new_depth == (self.depth - 1)
                    curr_index += 1
                    child_node = Node(index=curr_index, depth=new_depth, is_root=False, is_leaf=is_leaf)
                    if not child_node.isLeaf:
                        threshold_name = self.get_variable_name(name="threshold", node=child_node)
                        child_node.probabilityThreshold = tf.placeholder(name=threshold_name, dtype=tf.float32)
                        softmax_decay_name = self.get_variable_name(name="softmax_decay", node=child_node)
                        child_node.softmaxDecay = tf.placeholder(name=softmax_decay_name, dtype=tf.float32)
                    self.nodes[curr_index] = child_node
                    self.dagObject.add_edge(parent=curr_node, child=child_node)
                    d.append(child_node)
        # Probability thresholding
        self.useThresholding = tf.placeholder(name="threshold_flag", dtype=tf.int64)
        # Flags
        self.iterationHolder = tf.placeholder(name="iteration", dtype=tf.int64)
        self.isTrain = tf.placeholder(name="is_train_flag", dtype=tf.int64)
        self.useMasking = tf.placeholder(name="use_masking_flag", dtype=tf.int64)
        self.isDecisionPhase = tf.placeholder(name="is_decision_phase", dtype=tf.int64)
        self.decisionDropoutKeepProb = tf.placeholder(name="decision_dropout_keep_prob", dtype=tf.float32)
        # Build symbolic networks
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        if not GlobalConstants.USE_RANDOM_PARAMETERS:
            self.paramsDict = UtilityFuncs.load_npz(file_name="parameters")
        for node in self.topologicalSortedNodes:
            self.nodeBuildFuncs[node.depth](node=node, network=self)
        if len(self.topologicalSortedNodes) == 1:
            GlobalConstants.USE_INFO_GAIN_DECISION = False
            GlobalConstants.USE_CONCAT_TRICK = False
            GlobalConstants.USE_PROBABILITY_THRESHOLD = False
        # Set up mechanism for probability thresholding
        self.thresholdFunc(network=self)
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
            loss_tensor = tf.nn.l2_loss(v)
            self.evalDict["l2_loss_{0}".format(v.name)] = loss_tensor
            if "bias" in v.name:
                l2_loss_list.append(0.0 * loss_tensor)
            else:
                l2_loss_list.append(self.weightDecayCoeff * loss_tensor)
        # weights_and_filters = [v for v in vars if "bias" not in v.name]
        # Proxy, decision losses
        decision_losses = []
        for node in self.topologicalSortedNodes:
            if node.isLeaf:
                continue
            decision_losses.append(node.infoGainLoss)
        self.regularizationLoss = tf.add_n(l2_loss_list)
        self.mainLoss = tf.add_n(primary_losses)
        if len(decision_losses) > 0:
            self.decisionLoss = GlobalConstants.DECISION_LOSS_COEFFICIENT * tf.add_n(decision_losses)
        else:
            self.decisionLoss = tf.constant(value=0.0)
        self.finalLoss = self.mainLoss + self.regularizationLoss + self.decisionLoss
        self.evalDict["RegularizerLoss"] = self.regularizationLoss
        self.evalDict["PrimaryLoss"] = self.mainLoss
        self.evalDict["DecisionLoss"] = self.decisionLoss
        self.evalDict["NetworkLoss"] = self.finalLoss
        self.sample_count_tensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "sample_count" in k}
        self.isOpenTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "is_open" in k}
        self.gradFunc(network=self)
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
                # Add tensorboard ops
                # self.summaryFunc(network=self)
                # self.summaryWriter = tf.summary.FileWriter(GlobalConstants.SUMMARY_DIR + "//train")

    def calculate_accuracy(self, sess, dataset, dataset_type, run_id, iteration):
        dataset.set_current_data_set_type(dataset_type=dataset_type)
        leaf_predicted_labels_dict = {}
        leaf_true_labels_dict = {}
        info_gain_dict = {}
        branch_probs = {}
        while True:
            results = self.eval_network(sess=sess, dataset=dataset, use_masking=True)
            batch_sample_count = 0.0
            for node in self.topologicalSortedNodes:
                if not node.isLeaf:
                    info_gain = results[self.get_variable_name(name="info_gain", node=node)]
                    branch_prob = results[self.get_variable_name(name="p(n|x)", node=node)]
                    UtilityFuncs.concat_to_np_array_dict(dct=branch_probs, key=node.index, array=branch_prob)
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
                UtilityFuncs.concat_to_np_array_dict(dct=leaf_predicted_labels_dict, key=node.index,
                                                     array=predicted_labels)
                UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=node.index,
                                                     array=true_labels)
            if batch_sample_count != GlobalConstants.EVAL_BATCH_SIZE:
                raise Exception("Incorrect batch size:{0}".format(batch_sample_count))
            if dataset.isNewEpoch:
                break
        print("****************Dataset:{0}****************".format(dataset_type))
        # Measure Information Gain
        for k, v in info_gain_dict.items():
            avg_info_gain = sum(v) / float(len(v))
            print("IG_{0}={1}".format(k, -avg_info_gain))
        # Measure Branching Probabilities
        for k, v in branch_probs.items():
            p_n = np.mean(v, axis=0)
            print("p_{0}(n)={1}".format(k, p_n))
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
            for i in range(true_labels.shape[0]):
                true_label = true_labels[i]
                predicted_label = predicted[i]
                label_pair = (np.asscalar(true_label), np.asscalar(predicted_label))
                if label_pair not in sparse_confusion_matrix:
                    sparse_confusion_matrix[label_pair] = 0
                sparse_confusion_matrix[label_pair] += 1
            for k, v in sparse_confusion_matrix.items():
                confusion_matrix_db_rows.append((run_id, dataset_type.value, node.index, iteration, k[0], k[1], v))
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
        # Measure overall label distribution in leaves, get modes
        total_mode_count = 0
        for node in self.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            if node.index not in leaf_true_labels_dict:
                continue
            true_labels = leaf_true_labels_dict[node.index]
            frequencies = {}
            label_distribution = {}
            distribution_str = ""
            total_sample_count = true_labels.shape[0]
            for label in range(dataset.get_label_count()):
                frequencies[label] = np.sum(true_labels == label)
                label_distribution[label] = frequencies[label] / float(total_sample_count)
                distribution_str += "{0}:{1:.3f} ".format(label, label_distribution[label])
            # Get modes
            # if dataset_type == DatasetTypes.training:
            #     cumulative_prob = 0.0
            #     sorted_distribution = sorted(label_distribution.items(), key=lambda tpl: tpl[1], reverse=True)
            #     self.modesPerLeaves[node.index] = set()
            #     for tpl in sorted_distribution:
            #         if cumulative_prob < GlobalConstants.PERCENTILE_THRESHOLD:
            #             self.modesPerLeaves[node.index].add(tpl[0])
            #             cumulative_prob += tpl[1]
            #     total_mode_count += len(self.modesPerLeaves[node.index])
            print("Node{0} Label Distribution: {1}".format(node.index, distribution_str))
        # if dataset_type == DatasetTypes.training and total_mode_count != GlobalConstants.NUM_LABELS:
        #     raise Exception("total_mode_count != GlobalConstants.NUM_LABELS")
        # Measure overall information gain
        return overall_correct / overall_count, confusion_matrix_db_rows

    # def calculate_accuracy_with_route_correction(self, sess, dataset, dataset_type, run_id, iteration):
    #     dataset.set_current_data_set_type(dataset_type=dataset_type)
    #     leaf_predicted_labels_dict = {}
    #     leaf_true_labels_dict = {}
    #     info_gain_dict = {}
    #     branch_probs = {}
    #     one_hot_branch_probs = {}
    #     posterior_probs = {}
    #     while True:
    #         results = self.eval_network(sess=sess, dataset=dataset, use_masking=False)
    #         for node in self.topologicalSortedNodes:
    #             if not node.isLeaf:
    #                 branch_prob = results[self.get_variable_name(name="p(n|x)", node=node)]
    #                 UtilityFuncs.concat_to_np_array_dict(dct=branch_probs, key=node.index,
    #                                                      array=branch_prob)
    #             else:
    #                 posterior_prob = results[self.get_variable_name(name="posterior_probs", node=node)]
    #                 UtilityFuncs.concat_to_np_array_dict(dct=posterior_probs, key=node.index,
    #                                                      array=posterior_prob)
    #         if dataset.isNewEpoch:
    #             break




        # for k, v in branch_probs.items():
        #     zeros_arr = np.zeros(shape=v.shape)
        #     arg_max_indices = np.argmax(v, axis=1)
        #     print("X")
        # # At this stage we have the brancing probabilities p(l|x) and posterior probabilities p(y|l,x) at hand.
        # # for sample_id in range(dataset.get_current_sample_count()):
        # #     curr_node = self.nodes[0]
        # #     while not curr_node.isLeaf:

    def get_probability_thresholds(self, feed_dict, iteration, update):
        for node in self.topologicalSortedNodes:
            if node.isLeaf:
                continue
            if update:
                # Probability Threshold
                node_degree = self.degreeList[node.depth]
                uniform_prob = 1.0 / float(node_degree)
                threshold = uniform_prob - node.probThresholdCalculator.value
                feed_dict[node.probabilityThreshold] = threshold
                print("{0} value={1}".format(node.probThresholdCalculator.name, threshold))
                # Update the threshold calculator
                node.probThresholdCalculator.update(iteration=iteration + 1)
            else:
                feed_dict[node.probabilityThreshold] = 0.0

    def get_softmax_decays(self, feed_dict, iteration, update):
        for node in self.topologicalSortedNodes:
            if node.isLeaf:
                continue
            # Decay for Softmax
            decay = node.softmaxDecayCalculator.value
            feed_dict[node.softmaxDecay] = decay
            if update:
                print("{0} value={1}".format(node.softmaxDecayCalculator.name, decay))
                # Update the Softmax Decay
                node.softmaxDecayCalculator.update(iteration=iteration + 1)

    def get_decision_dropout_prob(self, feed_dict, iteration, update):
        if update:
            prob = self.decisionDropoutKeepProbCalculator.value
            feed_dict[self.decisionDropoutKeepProb] = prob
            print("{0} value={1}".format(self.decisionDropoutKeepProbCalculator.name, prob))
            self.decisionDropoutKeepProbCalculator.update(iteration=iteration + 1)
        else:
            feed_dict[self.decisionDropoutKeepProb] = 1.0

    def get_main_and_regularization_grads(self, sess, samples, labels, one_hot_labels, iteration):
        vars = tf.trainable_variables()
        use_threshold = int(GlobalConstants.USE_PROBABILITY_THRESHOLD)
        if GlobalConstants.USE_INFO_GAIN_DECISION:
            is_decision_phase = 0
        else:
            is_decision_phase = 1
        feed_dict = {GlobalConstants.TRAIN_DATA_TENSOR: samples,
                     GlobalConstants.TRAIN_LABEL_TENSOR: labels,
                     GlobalConstants.TRAIN_ONE_HOT_LABELS: one_hot_labels,
                     self.globalCounter: iteration,
                     self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
                     self.useThresholding: use_threshold,
                     self.isDecisionPhase: is_decision_phase,
                     self.isTrain: 1,
                     self.useMasking: 1,
                     self.iterationHolder: iteration}
        # Add probability thresholds into the feed dict
        self.get_probability_thresholds(feed_dict=feed_dict, iteration=iteration, update=True)
        self.get_softmax_decays(feed_dict=feed_dict, iteration=iteration, update=True)
        self.get_decision_dropout_prob(feed_dict=feed_dict, iteration=iteration,
                                       update=GlobalConstants.USE_DROPOUT_FOR_DECISION)
        run_ops = [self.classificationGradients,
                   self.regularizationGradients,
                   self.sample_count_tensors,
                   vars,
                   self.learningRate,
                   self.isOpenTensors]
        if iteration % GlobalConstants.SUMMARY_PERIOD == 0:
            run_ops.append(self.classificationPathSummaries)
        if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING and is_decision_phase:
            run_ops.extend(self.branchingBatchNormAssignOps)
        results = sess.run(run_ops, feed_dict=feed_dict)
        # Only calculate the derivatives for information gain losses
        classification_grads = results[0]
        regularization_grads = results[1]
        sample_counts = results[2]
        vars_current_values = results[3]
        lr = results[4]
        is_open_indicators = results[5]
        # if iteration % GlobalConstants.SUMMARY_PERIOD == 0:
        #     summary_list = results[6]
        #     for summary in summary_list:
        #         self.summaryWriter.add_summary(summary, iteration)
        # ******************* Calculate grads *******************
        # Main loss
        main_grads = {}
        for k, v in self.mainLossParamsDict.items():
            node = self.varToNodesDict[k.name]
            is_node_open = is_open_indicators[self.get_variable_name(name="is_open", node=node)]
            if not is_node_open:
                continue
            g = classification_grads[v]
            # print("Param:{0} Classification Grad Norm:{1}".format(k.name, np.linalg.norm(g)))
            if (GlobalConstants.GRADIENT_TYPE == GradientType.mixture_of_experts_unbiased) or (
                        GlobalConstants.GRADIENT_TYPE == GradientType.parallel_dnns_unbiased):
                main_grads[k] = g
            elif GlobalConstants.GRADIENT_TYPE == GradientType.mixture_of_experts_biased:
                sample_count_entry_name = self.get_variable_name(name="sample_count", node=node)
                sample_count = sample_counts[sample_count_entry_name]
                gradient_modifier = float(GlobalConstants.BATCH_SIZE) / float(sample_count)
                modified_g = gradient_modifier * g
                main_grads[k] = modified_g
        # Regularization loss
        reg_grads = {}
        for k, v in self.regularizationParamsDict.items():
            node = self.varToNodesDict[k.name]
            is_node_open = is_open_indicators[self.get_variable_name(name="is_open", node=node)]
            if not is_node_open:
                continue
            r = regularization_grads[v]
            reg_grads[k] = r
        return main_grads, reg_grads, lr, vars_current_values, sample_counts, is_open_indicators

    def get_decision_grads(self, sess, samples, labels, one_hot_labels, iteration):
        info_gain_dicts = {k: v for k, v in self.evalDict.items() if "info_gain" in k}
        feed_dict = {GlobalConstants.TRAIN_DATA_TENSOR: samples,
                     GlobalConstants.TRAIN_LABEL_TENSOR: labels,
                     GlobalConstants.TRAIN_ONE_HOT_LABELS: one_hot_labels,
                     self.globalCounter: iteration,
                     self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
                     self.useThresholding: 0,
                     self.isDecisionPhase: 1,
                     self.isTrain: 1,
                     self.useMasking: 1,
                     self.iterationHolder: iteration}
        # Add probability thresholds into the feed dict: They are disabled for decision phase, but still needed for
        # the network to operate.
        self.get_probability_thresholds(feed_dict=feed_dict, iteration=iteration, update=False)
        self.get_softmax_decays(feed_dict=feed_dict, iteration=iteration, update=False)
        self.get_decision_dropout_prob(feed_dict=feed_dict, iteration=iteration, update=False)
        run_ops = [self.decisionGradients, self.sample_count_tensors, self.isOpenTensors, info_gain_dicts]
        if iteration % GlobalConstants.SUMMARY_PERIOD == 0:
            run_ops.append(self.decisionPathSummaries)
        if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
            run_ops.extend(self.branchingBatchNormAssignOps)
        results = sess.run(run_ops, feed_dict=feed_dict)
        decision_grads = results[0]
        sample_counts = results[1]
        is_open_indicators = results[2]
        info_gain_results = results[3]
        # print(info_gain_results)
        # if iteration % GlobalConstants.SUMMARY_PERIOD == 0:
        #     summary_list = results[4]
        #     for summary in summary_list:
        #         self.summaryWriter.add_summary(summary, iteration)
        d_grads = {}
        for k, v in self.decisionParamsDict.items():
            node = self.varToNodesDict[k.name]
            is_node_open = is_open_indicators[self.get_variable_name(name="is_open", node=node)]
            if not is_node_open:
                continue
            d = decision_grads[v]
            # print("Param:{0} Decision Grad Norm:{1}".format(k.name, np.linalg.norm(d)))
            if np.any(np.isnan(d)):
                raise Exception("NAN Gradient!!!")
            d_grads[k] = d
        return d_grads, info_gain_results

    def update_params_with_momentum(self, sess, dataset, iteration):
        vars = tf.trainable_variables()
        samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
        samples = np.expand_dims(samples, axis=3)
        # Decision network
        decision_grads = {}
        if GlobalConstants.USE_INFO_GAIN_DECISION:
            decision_grads, info_gain_results = self.get_decision_grads(sess=sess, samples=samples, labels=labels,
                                                                        one_hot_labels=one_hot_labels,
                                                                        iteration=iteration)
        # Classification network
        main_grads, reg_grads, lr, vars_current_values, sample_counts, is_open_indicators = \
            self.get_main_and_regularization_grads(sess=sess, samples=samples, labels=labels,
                                                   one_hot_labels=one_hot_labels, iteration=iteration)
        update_dict = {}
        assign_dict = {}
        for v, curr_value in zip(vars, vars_current_values):
            total_grad = np.zeros(shape=v.shape)
            is_decision_pipeline_variable = "hyperplane" in v.name or "_decision_" in v.name
            if v in main_grads:
                total_grad += main_grads[v]
            if v in reg_grads:
                if not is_decision_pipeline_variable:
                    total_grad += reg_grads[v]
                elif is_decision_pipeline_variable and GlobalConstants.USE_DECISION_REGULARIZER:
                    total_grad += reg_grads[v]
                else:
                    print("Skipping {0} update".format(v.name))
            if GlobalConstants.USE_INFO_GAIN_DECISION and v in decision_grads:
                total_grad += decision_grads[v]
            self.momentumStatesDict[v.name][:] *= GlobalConstants.MOMENTUM_DECAY
            self.momentumStatesDict[v.name][:] += -lr * total_grad
            new_value = curr_value + self.momentumStatesDict[v.name]
            op_name = self.get_assign_op_name(variable=v)
            update_dict[self.newValuesDict[op_name]] = new_value
            assign_dict[op_name] = self.assignOpsDict[op_name]
        sess.run(assign_dict, feed_dict=update_dict),
        return sample_counts, lr, is_open_indicators

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
            mask_tensor = tf.where(self.useMasking > 0, mask_tensor,
                                   tf.logical_or(x=tf.constant(value=True, dtype=tf.bool), y=mask_tensor))
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
        # node_degree = self.degreeList[node.depth]
        decayed_activation = node.activationsDict[node.index] / node.softmaxDecay
        p_n_given_x = tf.nn.softmax(decayed_activation)
        p_c_given_x = node.oneHotLabelTensor
        node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_n_given_x, p_c_given_x_2d=p_c_given_x)
        node.evalDict[self.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
        node.evalDict[self.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
        node.evalDict[self.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
        node.evalDict[self.get_variable_name(name="p(n|x)", node=node)] = p_n_given_x
        arg_max_indices = tf.argmax(p_n_given_x, axis=1)
        child_nodes = self.dagObject.children(node=node)
        for index in range(len(child_nodes)):
            child_node = child_nodes[index]
            child_index = child_node.index
            branch_prob = p_n_given_x[:, index]
            mask_with_threshold = tf.reshape(tf.greater_equal(x=branch_prob, y=node.probabilityThreshold,
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

    def eval_network(self, sess, dataset, use_masking):
        # if is_train:
        samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(
            batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        samples = np.expand_dims(samples, axis=3)
        feed_dict = {
            GlobalConstants.TRAIN_DATA_TENSOR: samples,
            GlobalConstants.TRAIN_LABEL_TENSOR: labels,
            GlobalConstants.TRAIN_ONE_HOT_LABELS: one_hot_labels,
            self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
            self.useThresholding: 0,
            self.isDecisionPhase: 0,
            self.isTrain: 0,
            self.useMasking: int(use_masking),
            self.iterationHolder: 1000000}
        # Add probability thresholds into the feed dict: They are disabled for decision phase, but still needed for
        # the network to operate.
        self.get_probability_thresholds(feed_dict=feed_dict, iteration=1000000, update=False)
        self.get_softmax_decays(feed_dict=feed_dict, iteration=1000000, update=False)
        self.get_decision_dropout_prob(feed_dict=feed_dict, iteration=1000000, update=False)
        # self.get_probability_hyperparams(feed_dict=feed_dict, iteration=1000000, update_thresholds=False)
        results = sess.run(self.evalDict, feed_dict)
        # else:
        #     samples, labels, indices_list = dataset.get_next_batch(batch_size=EVAL_BATCH_SIZE)
        #     samples = np.expand_dims(samples, axis=3)
        #     feed_dict = {TEST_DATA_TENSOR: samples, TEST_LABEL_TENSOR: labels}
        #     results = sess.run(network.evalDict, feed_dict)
        return results
