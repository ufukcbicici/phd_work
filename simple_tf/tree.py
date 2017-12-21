import tensorflow as tf
import numpy as np

from auxillary.constants import DatasetTypes
from auxillary.dag_utilities import Dag
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants, GradientType
from simple_tf.info_gain import InfoGainLoss
from simple_tf.node import Node
from collections import deque
from simple_tf import batch_norm


class TreeNetwork:
    def __init__(self, node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list):
        self.dagObject = Dag()
        self.nodeBuildFuncs = node_build_funcs
        self.depth = len(self.nodeBuildFuncs)
        self.nodes = {}
        self.topologicalSortedNodes = []
        self.gradFunc = grad_func
        self.thresholdFunc = threshold_func
        self.residueFunc = residue_func
        self.summaryFunc = summary_func
        self.degreeList = degree_list
        self.dataTensor = GlobalConstants.TRAIN_DATA_TENSOR
        self.labelTensor = GlobalConstants.TRAIN_LABEL_TENSOR
        self.oneHotLabelTensor = GlobalConstants.TRAIN_ONE_HOT_LABELS
        self.indicesTensor = GlobalConstants.TRAIN_INDEX_TENSOR
        self.evalDict = {}
        self.mainLoss = None
        self.residueLoss = None
        self.decisionLoss = None
        self.regularizationLoss = None
        self.finalLoss = None
        self.classificationGradients = None
        self.residueGradients = None
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
        self.decisionWeightDecayCoeff = None
        self.useThresholding = None
        self.iterationHolder = None
        self.decisionDropoutKeepProb = None
        self.decisionDropoutKeepProbCalculator = None
        self.classificationDropoutKeepProb = None
        self.informationGainBalancingCoefficient = None
        self.noiseCoefficient = None
        self.noiseCoefficientCalculator = None
        self.isTrain = None
        self.useMasking = None
        self.isDecisionPhase = None
        self.varToNodesDict = {}
        self.mainLossParamsDict = {}
        self.residueParamsDict = {}
        self.regularizationParamsDict = {}
        self.decisionParamsDict = {}
        self.initOp = None
        self.classificationPathSummaries = []
        self.decisionPathSummaries = []
        self.summaryWriter = None
        self.branchingBatchNormAssignOps = []
        self.modesPerLeaves = {}
        self.learningRateCalculator = GlobalConstants.LEARNING_RATE_CALCULATOR
        self.isBaseline = None

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

    def is_decision_variable(self, variable):
        if "scale" in variable.name or "shift" in variable.name or "hyperplane" in variable.name or \
                        "gamma" in variable.name or "beta" in variable.name or "_decision_" in variable.name:
            return True
        else:
            return False

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
        # Flags and hyperparameters
        self.useThresholding = tf.placeholder(name="threshold_flag", dtype=tf.int64)
        self.iterationHolder = tf.placeholder(name="iteration", dtype=tf.int64)
        self.isTrain = tf.placeholder(name="is_train_flag", dtype=tf.int64)
        self.useMasking = tf.placeholder(name="use_masking_flag", dtype=tf.int64)
        self.isDecisionPhase = tf.placeholder(name="is_decision_phase", dtype=tf.int64)
        self.decisionDropoutKeepProb = tf.placeholder(name="decision_dropout_keep_prob", dtype=tf.float32)
        self.classificationDropoutKeepProb = tf.placeholder(name="classification_dropout_keep_prob", dtype=tf.float32)
        self.noiseCoefficient = tf.placeholder(name="noise_coefficient", dtype=tf.float32)
        self.informationGainBalancingCoefficient = tf.placeholder(name="info_gain_balance_coefficient",
                                                                  dtype=tf.float32)
        # Build symbolic networks
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        self.isBaseline = len(self.topologicalSortedNodes) == 1
        if not GlobalConstants.USE_RANDOM_PARAMETERS:
            self.paramsDict = UtilityFuncs.load_npz(file_name="parameters")
        # Set up mechanism for probability thresholding
        if not self.isBaseline:
            self.thresholdFunc(network=self)
        for node in self.topologicalSortedNodes:
            self.nodeBuildFuncs[node.depth](node=node, network=self)
        if self.isBaseline:
            GlobalConstants.USE_INFO_GAIN_DECISION = False
            GlobalConstants.USE_CONCAT_TRICK = False
            GlobalConstants.USE_PROBABILITY_THRESHOLD = False
            self.residueLoss = tf.constant(value=0.0)
        else:
            self.residueLoss = GlobalConstants.RESIDUE_LOSS_COEFFICIENT * self.residueFunc(network=self)
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
                # Sample indices
                self.evalDict["Node{0}_indices_tensor".format(node.index)] = node.indicesTensor
            # One Hot Label outputs
            if node.oneHotLabelTensor is not None:
                self.evalDict["Node{0}_one_hot_label_tensor".format(node.index)] = node.oneHotLabelTensor
        # Learning rate, counter
        self.globalCounter = tf.Variable(0, dtype=GlobalConstants.DATA_TYPE, trainable=False)
        # self.learningRate = tf.train.exponential_decay(
        #     GlobalConstants.INITIAL_LR,  # Base learning rate.
        #     self.globalCounter,  # Current index into the dataset.
        #     GlobalConstants.DECAY_STEP,  # Decay step.
        #     GlobalConstants.DECAY_RATE,  # Decay rate.
        #     staircase=True)
        # Prepare the cost function

        # ******************** Main losses ********************
        primary_losses = []
        for node in self.topologicalSortedNodes:
            primary_losses.extend(node.lossList)
        # ******************** Main losses ********************

        # ******************** Regularization losses ********************
        self.weightDecayCoeff = tf.placeholder(name="weight_decay_coefficient", dtype=tf.float32)
        self.decisionWeightDecayCoeff = tf.placeholder(name="decision_weight_decay_coefficient", dtype=tf.float32)
        vars = tf.trainable_variables()
        l2_loss_list = []
        for v in vars:
            is_decision_pipeline_variable = self.is_decision_variable(variable=v)
            # assert (not is_decision_pipeline_variable)
            loss_tensor = tf.nn.l2_loss(v)
            self.evalDict["l2_loss_{0}".format(v.name)] = loss_tensor
            if "bias" in v.name or "shift" in v.name or "scale" in v.name:
                l2_loss_list.append(0.0 * loss_tensor)
            else:
                if is_decision_pipeline_variable:
                    l2_loss_list.append(self.decisionWeightDecayCoeff * loss_tensor)
                else:
                    l2_loss_list.append(self.weightDecayCoeff * loss_tensor)
        # ******************** Regularization losses ********************

        # ******************** Decision losses ********************
        decision_losses = []
        for node in self.topologicalSortedNodes:
            if node.isLeaf:
                continue
            decision_losses.append(node.infoGainLoss)
        self.regularizationLoss = tf.add_n(l2_loss_list)
        self.mainLoss = tf.add_n(primary_losses)
        if len(decision_losses) > 0 and not self.isBaseline:
            self.decisionLoss = GlobalConstants.DECISION_LOSS_COEFFICIENT * tf.add_n(decision_losses)
        else:
            self.decisionLoss = tf.constant(value=0.0)
        # ******************** Decision losses ********************
        self.finalLoss = self.mainLoss + self.regularizationLoss + self.decisionLoss
        self.evalDict["RegularizerLoss"] = self.regularizationLoss
        self.evalDict["PrimaryLoss"] = self.mainLoss
        self.evalDict["ResidueLoss"] = self.residueLoss
        self.evalDict["DecisionLoss"] = self.decisionLoss
        self.evalDict["NetworkLoss"] = self.finalLoss
        self.sample_count_tensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "sample_count" in k}
        self.isOpenTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "is_open" in k}
        self.gradFunc(network=self)
        # Assign ops for variables
        for var in vars:
            is_residue_variable = "_residue_" in var.name
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
            if not is_residue_variable and var.name not in self.varToNodesDict:
                raise Exception("{0} is not in the parameters!".format(var.name))
                # Add tensorboard ops
                # self.summaryFunc(network=self)
                # self.summaryWriter = tf.summary.FileWriter(GlobalConstants.SUMMARY_DIR + "//train")

    def calculate_branch_probability_histograms(self, branch_probs):
        for k, v in branch_probs.items():
            # Interval analysis
            print("Node:{0}".format(k))
            bin_size = 0.1
            for j in range(v.shape[1]):
                histogram = {}
                for i in range(v.shape[0]):
                    prob = v[i, j]
                    bin_id = int(prob / bin_size)
                    if bin_id not in histogram:
                        histogram[bin_id] = 0
                    histogram[bin_id] += 1
                sorted_histogram = sorted(list(histogram.items()), key=lambda e: e[0], reverse=False)
                print(histogram)

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
                true_labels = results["Node{0}_label_tensor".format(node.index)]
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
        # Measure The Histogram of Branching Probabilities
        self.calculate_branch_probability_histograms(branch_probs=branch_probs)
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
            if dataset_type == DatasetTypes.training:
                cumulative_prob = 0.0
                sorted_distribution = sorted(label_distribution.items(), key=lambda lbl: lbl[1], reverse=True)
                self.modesPerLeaves[node.index] = set()
                for tpl in sorted_distribution:
                    if cumulative_prob < GlobalConstants.PERCENTILE_THRESHOLD:
                        self.modesPerLeaves[node.index].add(tpl[0])
                        cumulative_prob += tpl[1]
                total_mode_count += len(self.modesPerLeaves[node.index])
            print("Node{0} Label Distribution: {1}".format(node.index, distribution_str))
        # if dataset_type == DatasetTypes.training and total_mode_count != GlobalConstants.NUM_LABELS:
        #     raise Exception("total_mode_count != GlobalConstants.NUM_LABELS")
        # Measure overall information gain
        return overall_correct / overall_count, confusion_matrix_db_rows

    def calculate_accuracy_with_route_correction(self, sess, dataset, dataset_type, run_id, iteration):
        dataset.set_current_data_set_type(dataset_type=dataset_type)
        leaf_predicted_labels_dict = {}
        leaf_true_labels_dict = {}
        info_gain_dict = {}
        branch_probs = {}
        one_hot_branch_probs = {}
        posterior_probs = {}
        while True:
            results = self.eval_network(sess=sess, dataset=dataset, use_masking=False)
            for node in self.topologicalSortedNodes:
                if not node.isLeaf:
                    branch_prob = results[self.get_variable_name(name="p(n|x)", node=node)]
                    UtilityFuncs.concat_to_np_array_dict(dct=branch_probs, key=node.index,
                                                         array=branch_prob)
                else:
                    posterior_prob = results[self.get_variable_name(name="posterior_probs", node=node)]
                    true_labels = results["Node{0}_label_tensor".format(node.index)]
                    UtilityFuncs.concat_to_np_array_dict(dct=posterior_probs, key=node.index,
                                                         array=posterior_prob)
                    # UtilityFuncs.concat_to_np_array_dict(dct=leaf_predicted_labels_dict, key=node.index,
                    #                                      array=predicted_labels)
                    UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=node.index,
                                                         array=true_labels)
            if dataset.isNewEpoch:
                break
        label_dict = list(leaf_true_labels_dict.values())[0]
        for v in leaf_true_labels_dict.values():
            assert np.allclose(v, label_dict)
        root_node = self.nodes[0]
        # for node in self.topologicalSortedNodes:
        #     if not node.isLeaf:
        #         continue
        #     path = self.dagObject.get_shortest_path(source=root_node, dest=node)
        #     print("X")
        sample_count = list(leaf_true_labels_dict.values())[0].shape[0]
        total_correct = 0
        total_mode_prediction_count = 0
        total_correct_of_mode_predictions = 0
        samples_with_non_mode_predictions = set()
        wrong_samples_with_non_mode_predictions = set()
        true_labels_dict = {}
        # leaf_histograms = {}
        for sample_index in range(sample_count):
            curr_node = root_node
            probabilities_on_path = []
            while True:
                if not curr_node.isLeaf:
                    p_n_given_sample = branch_probs[curr_node.index][sample_index, :]
                    child_nodes = self.dagObject.children(node=curr_node)
                    child_nodes_sorted = sorted(child_nodes, key=lambda c_node: c_node.index)
                    arg_max_index = np.asscalar(np.argmax(p_n_given_sample))
                    probabilities_on_path.append(p_n_given_sample[arg_max_index])
                    curr_node = child_nodes_sorted[arg_max_index]
                else:
                    sample_posterior = posterior_probs[curr_node.index][sample_index, :]
                    predicted_label = np.asscalar(np.argmax(sample_posterior))
                    true_label = leaf_true_labels_dict[curr_node.index][sample_index]
                    # if curr_node.index not in leaf_histograms:
                    #     leaf_histograms[curr_node.index] = {}
                    # if true_label not in leaf_histograms[curr_node.index]:
                    #     leaf_histograms[curr_node.index][true_label] = 0
                    # leaf_histograms[curr_node.index][true_label] += 1
                    true_labels_dict[sample_index] = true_label
                    if predicted_label not in self.modesPerLeaves[curr_node.index]:
                        samples_with_non_mode_predictions.add(sample_index)
                        # This is just for analysis.
                        if true_label != predicted_label:
                            wrong_samples_with_non_mode_predictions.add(sample_index)
                    else:
                        total_mode_prediction_count += 1.0
                        if true_label == predicted_label:
                            total_correct_of_mode_predictions += 1.0
                            total_correct += 1.0
                    # if true_label == predicted_label:
                    #     total_correct += 1.0
                    # else:
                    #     if sample_index in samples_with_non_mode_predictions:
                    #         wrong_samples_with_non_mode_predictions.add(sample_index)
                    # else:
                    #     print("Wrong!")
                    break
        # Try to correct non mode estimations with a simple heuristics:
        # 1) Check all leaves. Among the leaves which predicts the sample having a label within its modes, choose the
        # prediction with the highest confidence.
        # 2) If all leaves predict the sample as a non mode, pick the estimate with the highest confidence.
        # First Method
        for sample_index in samples_with_non_mode_predictions:
            curr_predicted_label = None
            curr_prediction_confidence = 0.0
            for node in self.topologicalSortedNodes:
                if not node.isLeaf:
                    continue
                sample_posterior = posterior_probs[node.index][sample_index, :]
                leaf_modes = self.modesPerLeaves[node.index]
                predicted_label = np.asscalar(np.argmax(sample_posterior))
                prediction_confidence = sample_posterior[predicted_label] + float(predicted_label in leaf_modes)
                if prediction_confidence > curr_prediction_confidence:
                    curr_predicted_label = predicted_label
                    curr_prediction_confidence = prediction_confidence
            if curr_predicted_label == true_labels_dict[sample_index]:
                total_correct += 1
        corrected_accuracy = total_correct / sample_count
        print("Dataset:{0} Modified Accuracy={1}".format(dataset_type, corrected_accuracy))
        print("Total count of mode predictions={0}".format(total_mode_prediction_count))
        mode_prediction_accuracy = total_correct_of_mode_predictions / total_mode_prediction_count
        print("Mode prediction accuracy={0}".format(mode_prediction_accuracy))
        # Second Method
        avg_total_correct = 0
        for sample_index in samples_with_non_mode_predictions:
            selection_prob_dict = {root_node.index: 1.0}
            avg_posterior = None
            for node in self.topologicalSortedNodes:
                if not node.isLeaf:
                    p_n_given_sample = branch_probs[node.index][sample_index, :]
                    child_nodes = self.dagObject.children(node=node)
                    child_nodes_sorted = sorted(child_nodes, key=lambda c_node: c_node.index)
                    for child_index in range(len(child_nodes_sorted)):
                        selection_prob_dict[child_nodes_sorted[child_index].index] = \
                            selection_prob_dict[node.index] * p_n_given_sample[child_index]
                else:
                    sample_posterior = posterior_probs[node.index][sample_index, :]
                    if avg_posterior is None:
                        avg_posterior = selection_prob_dict[node.index] * sample_posterior
                    else:
                        avg_posterior = avg_posterior + (selection_prob_dict[node.index] * sample_posterior)
            avg_predicted_label = np.asscalar(np.argmax(avg_posterior))
            if avg_predicted_label == true_labels_dict[sample_index]:
                avg_total_correct += 1
        marginalized_corrected_accuracy = (total_correct_of_mode_predictions + avg_total_correct) / sample_count
        print("Marginalized prediction accuracy={0}".format(marginalized_corrected_accuracy))
        return corrected_accuracy, marginalized_corrected_accuracy

    def calculate_accuracy_with_residue_network(self, sess, dataset, dataset_type):
        dataset.set_current_data_set_type(dataset_type=dataset_type)
        leaf_true_labels_dict = {}
        leaf_sample_indices_dict = {}
        leaf_posterior_probs_dict = {}
        residue_true_labels_dict = {}
        residue_sample_indices_dict = {}
        residue_posterior_probs_dict = {}
        while True:
            results = self.eval_network(sess=sess, dataset=dataset, use_masking=False)
            for node in self.topologicalSortedNodes:
                if not node.isLeaf:
                    branch_prob = results[self.get_variable_name(name="p(n|x)", node=node)]
                    UtilityFuncs.concat_to_np_array_dict(dct=leaf_posterior_probs_dict, key=node.index,
                                                         array=branch_prob)
                else:
                    posterior_prob = results[self.get_variable_name(name="posterior_probs", node=node)]
                    true_labels = results["Node{0}_label_tensor".format(node.index)]
                    UtilityFuncs.concat_to_np_array_dict(dct=leaf_posterior_probs_dict, key=node.index,
                                                         array=posterior_prob)
                    # UtilityFuncs.concat_to_np_array_dict(dct=leaf_predicted_labels_dict, key=node.index,
                    #                                      array=predicted_labels)
                    UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=node.index,
                                                         array=true_labels)
            residue_posterior_prob = results["residue_probabilities"]
            residue_true_labels = results["residue_labels"]
            residue_sample_indices = results["residue_indices"]
            UtilityFuncs.concat_to_np_array_dict(dct=residue_posterior_probs_dict, key=-1,
                                                 array=residue_posterior_prob)
            UtilityFuncs.concat_to_np_array_dict(dct=residue_true_labels_dict, key=-1,
                                                 array=residue_true_labels)
            UtilityFuncs.concat_to_np_array_dict(dct=residue_sample_indices_dict, key=-1,
                                                 array=residue_sample_indices)
            if dataset.isNewEpoch:
                break
        label_dict = list(leaf_true_labels_dict.values())[0]
        residue_true_labels = residue_true_labels_dict[-1]
        residue_posterior_prob = residue_posterior_probs_dict[-1]
        residue_sample_indices = residue_sample_indices_dict[-1]
        for v in leaf_true_labels_dict.values():
            assert np.allclose(v, label_dict)
        assert np.allclose(label_dict, residue_true_labels)
        sample_count = list(leaf_true_labels_dict.values())[0].shape[0]
        root_node = self.nodes[0]
        # Accuracy measurements
        mode_samples_count = 0
        non_mode_samples_count = 0
        true_labels_dict = {}
        samples_with_non_mode_predictions = set()
        wrong_samples_with_non_mode_predictions = set()
        for sample_index in range(sample_count):
            curr_node = root_node
            probabilities_on_path = []
            for node in self.topologicalSortedNodes:
                while True:
                    if not curr_node.isLeaf:
                        p_n_given_sample = leaf_posterior_probs_dict[curr_node.index][sample_index, :]
                        child_nodes = self.dagObject.children(node=curr_node)
                        child_nodes_sorted = sorted(child_nodes, key=lambda c_node: c_node.index)
                        arg_max_index = np.asscalar(np.argmax(p_n_given_sample))
                        probabilities_on_path.append(p_n_given_sample[arg_max_index])
                        curr_node = child_nodes_sorted[arg_max_index]
                    else:
                        sample_posterior = leaf_posterior_probs_dict[curr_node.index][sample_index, :]
                        predicted_label = np.asscalar(np.argmax(sample_posterior))
                        true_label = leaf_true_labels_dict[curr_node.index][sample_index]
                        true_labels_dict[sample_index] = true_label
                        if predicted_label not in self.modesPerLeaves[curr_node.index]:
                            samples_with_non_mode_predictions.add(sample_index)
                        #     if true_label != predicted_label:
                        #         wrong_samples_with_non_mode_predictions.add(sample_index)
                        # else:
                        #     total_mode_prediction_count += 1.0
                        #     if true_label == predicted_label:
                        #         total_correct_of_mode_predictions += 1.0
                        #         total_correct += 1.0
                        # if true_label == predicted_label:
                        #     total_correct += 1.0
                        # else:
                        #     if sample_index in samples_with_non_mode_predictions:
                        #         wrong_samples_with_non_mode_predictions.add(sample_index)
                        # else:
                        #     print("Wrong!")
                        break

        # while True:
        #     results = self.eval_network(sess=sess, dataset=dataset, use_masking=True)
        #     # Collect data from leaf nodes
        #     for node in self.topologicalSortedNodes:
        #         if not node.isLeaf:
        #             continue
        #         else:
        #             posterior_prob = results[self.get_variable_name(name="posterior_probs", node=node)]
        #             true_labels = results["Node{0}_label_tensor".format(node.index)]
        #             sample_indices = results["Node{0}_indices_tensor".format(node.index)]
        #             UtilityFuncs.concat_to_np_array_dict(dct=leaf_posterior_probs_dict, key=node.index,
        #                                                  array=posterior_prob)
        #             UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=node.index,
        #                                                  array=true_labels)
        #             UtilityFuncs.concat_to_np_array_dict(dct=leaf_sample_indices_dict, key=node.index,
        #                                                  array=sample_indices)
        #     # Collect data from the residue network
        #     residue_posterior_prob = results["residue_probabilities"]
        #     residue_true_labels = results["residue_labels"]
        #     residue_sample_indices = results["residue_indices"]
        #     UtilityFuncs.concat_to_np_array_dict(dct=residue_posterior_probs_dict, key=-1,
        #                                          array=residue_posterior_prob)
        #     UtilityFuncs.concat_to_np_array_dict(dct=residue_true_labels_dict, key=-1,
        #                                          array=residue_true_labels)
        #     UtilityFuncs.concat_to_np_array_dict(dct=residue_sample_indices_dict, key=-1,
        #                                          array=residue_sample_indices)
        #     if dataset.isNewEpoch:
        #         break
        # Build an index for lookup into residue entries
        residue_network_index = {}
        residue_true_labels = residue_true_labels_dict[-1]
        residue_posterior_prob = residue_posterior_probs_dict[-1]
        residue_sample_indices = residue_sample_indices_dict[-1]








        # for i in range(residue_sample_indices.shape[0]):
        #     residue_network_index[residue_sample_indices[i]] = i
        # # Measure Accuracy
        # mode_prediction_count = 0.0
        # mode_prediction_correct_count = 0.0
        # residue_prediction_count = 0.0
        # residue_prediction_correct_count = 0.0
        # for node in self.topologicalSortedNodes:
        #     if not node.isLeaf:
        #         continue
        #     leaf_sample_indices = leaf_sample_indices_dict[node.index]
        #     leaf_true_labels = leaf_true_labels_dict[node.index]
        #     leaf_posterior_prob = leaf_posterior_probs_dict[node.index]
        #     for i in range(leaf_sample_indices.shape[0]):
        #         sample_index = leaf_sample_indices[i]
        #         leaf_posterior = leaf_posterior_prob[i]
        #         leaf_true_label = leaf_true_labels[i]
        #         leaf_predicted_label = np.argmax(leaf_posterior)
        #         if leaf_predicted_label in self.modesPerLeaves[node.index]:
        #             if leaf_predicted_label == leaf_true_label:
        #                 mode_prediction_correct_count += 1.0
        #             mode_prediction_count += 1.0
        #         else:
        #             residue_index = residue_network_index[sample_index]
        #             residue_posterior = residue_posterior_prob[residue_index]
        #             residue_true_label = residue_true_labels[residue_index]
        #             assert residue_true_label == leaf_true_label
        #             residue_predicted_label = np.argmax(residue_posterior)
        #             if residue_predicted_label == residue_true_label:
        #                 residue_prediction_correct_count += 1.0
        #             residue_prediction_count += 1.0
        # print("Mode Predictions Accuracy:{0}".format(mode_prediction_correct_count / mode_prediction_count))
        # print("Residue Predictions Accuracy:{0}".format(residue_prediction_correct_count / residue_prediction_count))
        # print("Total Accuracy:{0}".format((mode_prediction_correct_count + residue_prediction_correct_count)
        #                                   / (mode_prediction_count + residue_prediction_count)))

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

    def get_noise_coefficient(self, feed_dict, iteration, update):
        noise_coeff = self.noiseCoefficientCalculator.value
        feed_dict[self.noiseCoefficient] = noise_coeff
        print("{0} value={1}".format(self.noiseCoefficientCalculator.name, noise_coeff))
        if update:
            self.noiseCoefficientCalculator.update(iteration=iteration + 1)

    def get_decision_dropout_prob(self, feed_dict, iteration, update):
        if update:
            prob = self.decisionDropoutKeepProbCalculator.value
            feed_dict[self.decisionDropoutKeepProb] = prob
            print("{0} value={1}".format(self.decisionDropoutKeepProbCalculator.name, prob))
            self.decisionDropoutKeepProbCalculator.update(iteration=iteration + 1)
        else:
            feed_dict[self.decisionDropoutKeepProb] = 1.0

    def get_main_and_regularization_grads(self, sess, samples, labels, indices, one_hot_labels, iteration):
        vars = tf.trainable_variables()
        use_threshold = int(GlobalConstants.USE_PROBABILITY_THRESHOLD)
        if GlobalConstants.USE_INFO_GAIN_DECISION:
            is_decision_phase = 0
        else:
            is_decision_phase = 1
        feed_dict = {GlobalConstants.TRAIN_DATA_TENSOR: samples,
                     GlobalConstants.TRAIN_LABEL_TENSOR: labels,
                     GlobalConstants.TRAIN_INDEX_TENSOR: indices,
                     GlobalConstants.TRAIN_ONE_HOT_LABELS: one_hot_labels,
                     self.globalCounter: iteration,
                     self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
                     self.decisionWeightDecayCoeff: GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT,
                     self.useThresholding: use_threshold,
                     self.isDecisionPhase: is_decision_phase,
                     self.isTrain: 1,
                     self.useMasking: 1,
                     self.classificationDropoutKeepProb: GlobalConstants.CLASSIFICATION_DROPOUT_PROB,
                     self.informationGainBalancingCoefficient: GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT,
                     self.iterationHolder: iteration}
        # Add probability thresholds into the feed dict
        if not self.isBaseline:
            self.get_probability_thresholds(feed_dict=feed_dict, iteration=iteration, update=True)
            self.get_softmax_decays(feed_dict=feed_dict, iteration=iteration, update=True)
            self.get_decision_dropout_prob(feed_dict=feed_dict, iteration=iteration,
                                           update=True)
            self.get_noise_coefficient(feed_dict=feed_dict, iteration=iteration, update=True)
        run_ops = [self.classificationGradients,
                   self.regularizationGradients,
                   self.residueGradients,
                   self.sample_count_tensors,
                   vars,
                   self.isOpenTensors]
        if iteration % GlobalConstants.SUMMARY_PERIOD == 0:
            run_ops.append(self.classificationPathSummaries)
        if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING and is_decision_phase:
            run_ops.extend(self.branchingBatchNormAssignOps)
        if GlobalConstants.USE_VERBOSE:
            run_ops.append(self.evalDict)
        results = sess.run(run_ops, feed_dict=feed_dict)
        # *********************For debug purposes*********************
        # cursor = 0
        # eval_results_dict = results[7]
        # residue_labels = eval_results_dict["residue_labels"]
        # residue_indices = eval_results_dict["residue_indices"]
        # residue_features = eval_results_dict["residue_features"]
        # for node in self.topologicalSortedNodes:
        #     if not node.isLeaf:
        #         continue
        #     node_sample_count = eval_results_dict[self.get_variable_name(name="sample_count", node=node)]
        #     node_label_tensor = eval_results_dict["Node{0}_label_tensor".format(node.index)]
        #     node_index_tensor = eval_results_dict["Node{0}_indices_tensor".format(node.index)]
        #     node_final_feature_tensor = eval_results_dict[self.get_variable_name(name="final_eval_feature", node=node)]
        #     assert np.allclose(node_label_tensor, residue_labels[cursor:cursor + int(node_sample_count)])
        #     assert np.allclose(node_index_tensor, residue_indices[cursor:cursor + int(node_sample_count)])
        #     assert np.allclose(node_final_feature_tensor, residue_features[cursor:cursor + int(node_sample_count)])
        #     cursor += int(node_sample_count)
        # *********************For debug purposes*********************
        # Only calculate the derivatives for information gain losses
        classification_grads = results[0]
        regularization_grads = results[1]
        residue_grads = results[2]
        sample_counts = results[3]
        vars_current_values = results[4]
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
        # Residue Loss
        res_grads = {}
        for k, v in self.residueParamsDict.items():
            g = residue_grads[v]
            res_grads[k] = g
        # Regularization loss
        reg_grads = {}
        # if GlobalConstants.USE_ADAPTIVE_WEIGHT_DECAY:
        #     for node in self.topologicalSortedNodes:
        #         sample_count_entry_name = self.get_variable_name(name="sample_count", node=node)
        #         sample_count = sample_counts[sample_count_entry_name]
        #         decay_boost_rate = GlobalConstants.BATCH_SIZE / float(sample_count)
        #         node.weightDecayModifier = \
        #             GlobalConstants.ADAPTIVE_WEIGHT_DECAY_MIXING_RATE * node.weightDecayModifier + \
        #             (1.0 - GlobalConstants.ADAPTIVE_WEIGHT_DECAY_MIXING_RATE) * decay_boost_rate
        for k, v in self.regularizationParamsDict.items():
            is_residue_var = "_residue_" in k.name
            # coeff = 1.0
            if not is_residue_var:
                node = self.varToNodesDict[k.name]
                is_node_open = is_open_indicators[self.get_variable_name(name="is_open", node=node)]
                if not is_node_open:
                    continue
                # if GlobalConstants.USE_ADAPTIVE_WEIGHT_DECAY and not self.is_decision_variable(variable=k):
                #     coeff = node.weightDecayModifier
            r = regularization_grads[v]
            reg_grads[k] = r
        return main_grads, res_grads, reg_grads, vars_current_values, sample_counts, is_open_indicators

    def get_decision_grads(self, sess, samples, labels, indices, one_hot_labels, iteration):
        info_gain_dicts = {k: v for k, v in self.evalDict.items() if "info_gain" in k}
        feed_dict = {GlobalConstants.TRAIN_DATA_TENSOR: samples,
                     GlobalConstants.TRAIN_LABEL_TENSOR: labels,
                     GlobalConstants.TRAIN_INDEX_TENSOR: indices,
                     GlobalConstants.TRAIN_ONE_HOT_LABELS: one_hot_labels,
                     self.globalCounter: iteration,
                     self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
                     self.decisionWeightDecayCoeff: GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT,
                     self.useThresholding: 0,
                     self.isDecisionPhase: 1,
                     self.isTrain: 1,
                     self.useMasking: 1,
                     self.classificationDropoutKeepProb: 1.0,
                     self.informationGainBalancingCoefficient: GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT,
                     self.iterationHolder: iteration}
        # Add probability thresholds into the feed dict: They are disabled for decision phase, but still needed for
        # the network to operate.
        if not self.isBaseline:
            self.get_probability_thresholds(feed_dict=feed_dict, iteration=iteration, update=False)
            self.get_softmax_decays(feed_dict=feed_dict, iteration=iteration, update=False)
            self.get_decision_dropout_prob(feed_dict=feed_dict, iteration=iteration, update=True)
            self.get_noise_coefficient(feed_dict=feed_dict, iteration=iteration, update=False)
        run_ops = [self.decisionGradients, self.sample_count_tensors, self.isOpenTensors, info_gain_dicts]
        if iteration % GlobalConstants.SUMMARY_PERIOD == 0:
            run_ops.append(self.decisionPathSummaries)
        if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
            run_ops.extend(self.branchingBatchNormAssignOps)
        if GlobalConstants.USE_VERBOSE:
            run_ops.append(self.evalDict)
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
        return d_grads, info_gain_results, sample_counts

    def update_params_with_momentum(self, sess, dataset, epoch, iteration):
        vars = tf.trainable_variables()
        samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
        samples = np.expand_dims(samples, axis=3)
        # Decision network
        decision_grads = {}
        decision_sample_counts = None
        if GlobalConstants.USE_INFO_GAIN_DECISION:
            decision_grads, info_gain_results, decision_sample_counts \
                = self.get_decision_grads(sess=sess, samples=samples, labels=labels,
                                          indices=indices_list,
                                          one_hot_labels=one_hot_labels,
                                          iteration=iteration)
        # Classification network
        main_grads, res_grads, reg_grads, vars_current_values, sample_counts, is_open_indicators = \
            self.get_main_and_regularization_grads(sess=sess, samples=samples, labels=labels,
                                                   indices=indices_list,
                                                   one_hot_labels=one_hot_labels, iteration=iteration)
        update_dict = {}
        assign_dict = {}
        self.learningRateCalculator.update(iteration=iteration + 1.0)
        lr = self.learningRateCalculator.value
        for v, curr_value in zip(vars, vars_current_values):
            is_residue_var = "_residue_" in v.name
            if not is_residue_var and epoch >= GlobalConstants.EPOCH_COUNT:
                continue
            total_grad = np.zeros(shape=v.shape)
            # is_decision_pipeline_variable = "hyperplane" in v.name or "_decision_" in v.name
            if v in main_grads:
                total_grad += main_grads[v]
            if v in res_grads:
                total_grad += res_grads[v]
            if v in reg_grads:
                total_grad += reg_grads[v]
            if GlobalConstants.USE_INFO_GAIN_DECISION and v in decision_grads:
                total_grad += decision_grads[v]
            self.momentumStatesDict[v.name][:] *= GlobalConstants.MOMENTUM_DECAY
            self.momentumStatesDict[v.name][:] += -lr * total_grad
            new_value = curr_value + self.momentumStatesDict[v.name]
            if ("scale" in v.name or "shift" in v.name) and iteration % 10 == 0:
                # print("Magnitude of {0}= Changed from {1} to {2}".format(v.name, np.linalg.norm(curr_value),
                #                                                          np.linalg.norm(new_value)))
                print("{0}={1}".format(v.name, new_value))
            op_name = self.get_assign_op_name(variable=v)
            update_dict[self.newValuesDict[op_name]] = new_value
            assign_dict[op_name] = self.assignOpsDict[op_name]
        sess.run(assign_dict, feed_dict=update_dict)
        return sample_counts, decision_sample_counts, lr, is_open_indicators

    # if v in res_grads:
    #     total_grad += res_grads[v]

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
            node.indicesTensor = self.indicesTensor
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
            node.indicesTensor = tf.boolean_mask(parent_node.indicesTensor, mask_tensor)
            node.oneHotLabelTensor = tf.boolean_mask(parent_node.oneHotLabelTensor, mask_tensor)
            return parent_F, parent_H

    def apply_batch_norm_prior_to_decision(self, feature, node):
        normed_data, assign_ops = batch_norm.batch_norm(x=feature, iteration=self.iterationHolder,
                                                        is_decision_phase=self.isDecisionPhase,
                                                        is_training_phase=self.isTrain,
                                                        decay=GlobalConstants.BATCH_NORM_DECAY,
                                                        node=node, network=self)
        self.branchingBatchNormAssignOps.extend(assign_ops)
        return normed_data

    def add_learnable_gaussian_noise(self, node, feature):
        sample_count = tf.shape(feature)[0]
        feature_dim = feature.get_shape().as_list()[-1]
        gaussian = tf.contrib.distributions.MultivariateNormalDiag(loc=np.zeros(shape=(feature_dim,)),
                                                                   scale_diag=np.ones(shape=(feature_dim,)))
        noise_shift = tf.Variable(
            tf.constant(0.0, shape=(feature_dim,), dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="noise_shift", node=node))
        noise_scale = tf.Variable(
            tf.constant(1.0, shape=(feature_dim,), dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="noise_scale", node=node))
        noise_scale_sqrt = tf.square(noise_scale)
        node.variablesSet.add(noise_shift)
        node.variablesSet.add(noise_scale)
        noise = tf.cast(gaussian.sample(sample_shape=sample_count), tf.float32)
        z_noise = noise_scale_sqrt * noise + noise_shift
        # final_feature = tf.where(self.isDecisionPhase > 0, feature, feature + z_noise)
        final_feature = feature + (self.noiseCoefficient * z_noise)
        return final_feature

    def apply_decision(self, node, branching_feature, hyperplane_weights, hyperplane_biases):
        # Apply necessary transformations before decision phase
        if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
            branching_feature = self.apply_batch_norm_prior_to_decision(feature=branching_feature, node=node)
        if GlobalConstants.USE_REPARAMETRIZATION_TRICK:
            self.evalDict[self.get_variable_name(name="branching_feature", node=node)] = branching_feature
            noisy_feature = self.add_learnable_gaussian_noise(node=node, feature=branching_feature)
            self.evalDict[self.get_variable_name(name="noisy_branching_feature", node=node)] = noisy_feature
            branching_feature = tf.where(self.isTrain > 0, noisy_feature, branching_feature)
            self.evalDict[self.get_variable_name(name="final_branching_feature", node=node)] = branching_feature
            # branching_feature = noisy_feature
        if GlobalConstants.USE_DROPOUT_FOR_DECISION:
            branching_feature = tf.nn.dropout(branching_feature, self.decisionDropoutKeepProb)
        activations = tf.matmul(branching_feature, hyperplane_weights) + hyperplane_biases
        node.activationsDict[node.index] = activations
        decayed_activation = node.activationsDict[node.index] / node.softmaxDecay
        p_n_given_x = tf.nn.softmax(decayed_activation)
        p_c_given_x = node.oneHotLabelTensor
        info_gain_balance_coeff = node.infoGainBalanceCoefficient
        node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_n_given_x, p_c_given_x_2d=p_c_given_x,
                                                  balance_coefficient=self.informationGainBalancingCoefficient)
        node.evalDict[self.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
        node.evalDict[self.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
        node.evalDict[self.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
        node.evalDict[self.get_variable_name(name="p(n|x)", node=node)] = p_n_given_x
        self.p_n_given_x = p_n_given_x
        arg_max_indices = tf.argmax(p_n_given_x, axis=1)
        child_nodes = self.dagObject.children(node=node)
        child_nodes = sorted(child_nodes, key=lambda c_node: c_node.index)
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

    def apply_loss(self, node, final_feature, softmax_weights, softmax_biases):
        final_feature_final = final_feature
        if GlobalConstants.USE_DROPOUT_FOR_CLASSIFICATION:
            final_feature_final = tf.nn.dropout(final_feature, self.classificationDropoutKeepProb)
        if GlobalConstants.USE_DECISION_AUGMENTATION:
            concat_list = [final_feature_final]
            concat_list.extend(node.activationsDict.values())
            final_feature_final = tf.concat(values=concat_list, axis=1)
        node.residueOutputTensor = final_feature_final
        logits = tf.matmul(final_feature_final, softmax_weights) + softmax_biases
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
        return final_feature_final, logits

    def eval_network(self, sess, dataset, use_masking):
        # if is_train:
        samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(
            batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        samples = np.expand_dims(samples, axis=3)
        feed_dict = {
            GlobalConstants.TRAIN_DATA_TENSOR: samples,
            GlobalConstants.TRAIN_LABEL_TENSOR: labels,
            GlobalConstants.TRAIN_INDEX_TENSOR: indices_list,
            GlobalConstants.TRAIN_ONE_HOT_LABELS: one_hot_labels,
            self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
            self.decisionWeightDecayCoeff: GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT,
            self.useThresholding: 0,
            self.isDecisionPhase: 0,
            self.isTrain: 0,
            self.useMasking: int(use_masking),
            self.classificationDropoutKeepProb: 1.0,
            self.informationGainBalancingCoefficient: GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT,
            self.noiseCoefficient: 0.0,
            self.iterationHolder: 1000000}
        # Add probability thresholds into the feed dict: They are disabled for decision phase, but still needed for
        # the network to operate.
        if not self.isBaseline:
            self.get_probability_thresholds(feed_dict=feed_dict, iteration=1000000, update=False)
            self.get_softmax_decays(feed_dict=feed_dict, iteration=1000000, update=False)
            self.get_decision_dropout_prob(feed_dict=feed_dict, iteration=1000000, update=False)
        # self.get_probability_hyperparams(feed_dict=feed_dict, iteration=1000000, update_thresholds=False)
        results = sess.run(self.evalDict, feed_dict)
        return results

    def get_transformed_data(self, sess, dataset, dataset_type):
        dataset.set_current_data_set_type(dataset_type=dataset_type)
        leaf_true_labels_dict = {}
        leaf_final_features_dict = {}
        # network.get_variable_name(name="final_eval_feature", node=node)
        while True:
            results = self.eval_network(sess=sess, dataset=dataset, use_masking=True)
            for node in self.topologicalSortedNodes:
                if not node.isLeaf:
                    continue
                final_features = results[self.get_variable_name(name="final_eval_feature", node=node)]
                true_labels = results[self.get_variable_name(name="labels", node=node)]
                UtilityFuncs.concat_to_np_array_dict(dct=leaf_final_features_dict, key=node.index, array=final_features)
                UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=node.index, array=true_labels)
            if dataset.isNewEpoch:
                break
        # Concatenate all data
        transformed_samples = None
        labels = None
        for k, v in leaf_final_features_dict.items():
            if transformed_samples is None:
                transformed_samples = np.array(v)
            else:
                transformed_samples = np.concatenate((transformed_samples, v))
            if labels is None:
                labels = np.array(leaf_true_labels_dict[k])
            else:
                labels = np.concatenate((labels, leaf_true_labels_dict[k]))
        return transformed_samples, labels

    def prepare_residue_input_tensors(self):
        # Get all residue features and labels from leaf nodes
        residue_features = []
        labels = []
        indices = []
        for node in self.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            residue_features.append(node.residueOutputTensor)
            labels.append(node.labelTensor)
            indices.append(node.indicesTensor)
        # Concatenate residue features and labels into a batch
        all_residue_features = tf.concat(values=residue_features, axis=0)
        all_labels = tf.concat(values=labels, axis=0)
        all_indices = tf.concat(values=indices, axis=0)
        return all_residue_features, all_labels, all_indices
