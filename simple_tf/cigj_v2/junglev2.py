import numpy as np
import tensorflow as tf
import time

from auxillary.constants import DatasetTypes
from algorithms.info_gain import InfoGainLoss
from auxillary.dag_utilities import Dag
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cigj.jungle_node import JungleNode, NodeType
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants


class JungleV2(FastTreeNetwork):
    def __init__(self, node_build_funcs, h_dimensions, dataset, network_name):
        self.degreeList = [1] * len(node_build_funcs)
        super().__init__(node_build_funcs, None, None, None, None, self.degreeList, dataset, network_name)
        self.depth = len(self.degreeList)
        curr_index = 0
        self.batchSize = tf.placeholder(name="batch_size", dtype=tf.int64)
        self.evalMultipath = tf.placeholder(name="eval_multipath", dtype=tf.int64)
        self.hDimensions = h_dimensions
        self.depthToNodesDict = {}
        self.currentGraph = tf.get_default_graph()
        self.batchIndices = tf.cast(tf.range(self.batchSize), dtype=tf.int32)
        self.nodeBuildFuncs = node_build_funcs
        self.nodes = {}

    def build_network(self):
        self.dagObject = Dag()
        self.nodes = {}
        curr_index = 0
        net = None
        for depth, build_func in enumerate(self.nodeBuildFuncs):
            if depth == 0:
                node_type = NodeType.root_node
                net = self.dataTensor
            elif depth == len(self.nodeBuildFuncs) - 1:
                node_type = NodeType.leaf_node
            else:
                node_type = NodeType.f_node
            curr_node = JungleNode(index=curr_index, depth=depth, node_type=node_type)
            self.nodes[curr_index] = curr_node
            net = build_func(self, node=curr_node, input_x=net, depth=depth)
            curr_node.fOpsList.append(net)
            if node_type != NodeType.leaf_node:
                self.information_gain_output(node=curr_node, node_output=net, h_dimension=self.hDimensions[depth])
            else:
                self.loss_output(node=curr_node, node_output=net)
            if curr_index - 1 in self.nodes:
                self.dagObject.add_edge(parent=self.nodes[curr_index - 1], child=curr_node)
            curr_index += 1
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        self.isBaseline = False
        self.nodeCosts = {node.index: node.macCost for node in self.topologicalSortedNodes}
        # Build main classification loss
        self.build_main_loss()
        # Build information gain loss
        self.build_decision_loss()
        # Build regularization loss
        self.build_regularization_loss()
        # Final Loss
        self.finalLoss = self.mainLoss + self.regularizationLoss + self.decisionLoss
        if not GlobalConstants.USE_MULTI_GPU:
            self.build_optimizer()
        self.prepare_evaluation_dictionary()

    def information_gain_output(self, node, node_output, h_dimension):
        assert len(node_output.get_shape().as_list()) == 2 or len(node_output.get_shape().as_list()) == 4
        if len(node_output.get_shape().as_list()) == 4:
            net_shape = node_output.get_shape().as_list()
            # Global Average Pooling
            h_net = tf.nn.avg_pool(node_output, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1],
                                   padding='VALID')
            net_shape = h_net.get_shape().as_list()
            h_net = tf.reshape(h_net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
        else:
            h_net = node_output
        # Step 1: Create Hyperplanes
        ig_feature_size = node_output.get_shape().as_list()[-1]
        hyperplane_weights = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node),
            shape=[ig_feature_size, h_dimension],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.truncated_normal([ig_feature_size, h_dimension], stddev=0.1,
                                            seed=GlobalConstants.SEED,
                                            dtype=GlobalConstants.DATA_TYPE))
        hyperplane_biases = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node),
            shape=[h_dimension],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.constant(0.0, shape=[h_dimension], dtype=GlobalConstants.DATA_TYPE))
        if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
            h_net = tf.layers.batch_normalization(inputs=h_net,
                                                  momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                  training=tf.cast(self.isTrain, tf.bool))
        ig_feature = h_net
        node.hOpsList.append(ig_feature)
        activations = FastTreeNetwork.fc_layer(x=ig_feature, W=hyperplane_weights, b=hyperplane_biases, node=node)
        node.activationsDict[node.index] = activations
        decayed_activation = node.activationsDict[node.index] / tf.reshape(node.softmaxDecay, (1,))
        p_F_given_x = tf.nn.softmax(decayed_activation)
        p_c_given_x = self.oneHotLabelTensor
        node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_F_given_x, p_c_given_x_2d=p_c_given_x,
                                                  balance_coefficient=self.informationGainBalancingCoefficient)
        node.evalDict[self.get_variable_name(name="branching_feature", node=node)] = ig_feature
        node.evalDict[self.get_variable_name(name="activations", node=node)] = activations
        node.evalDict[self.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
        node.evalDict[self.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
        node.evalDict[self.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
        node.evalDict[self.get_variable_name(name="branch_probs", node=node)] = p_F_given_x

    def loss_output(self, node, node_output):
        output_feature_dim = node_output.get_shape().as_list()[-1]
        softmax_weights = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="softmax_weights", node=node),
            shape=[output_feature_dim, self.labelCount],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.truncated_normal([output_feature_dim, self.labelCount], stddev=0.1,
                                            seed=GlobalConstants.SEED,
                                            dtype=GlobalConstants.DATA_TYPE))
        softmax_biases = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="softmax_biases", node=node),
            shape=[self.labelCount],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.constant(0.0, shape=[self.labelCount], dtype=GlobalConstants.DATA_TYPE))
        node.labelTensor = self.labelTensor
        final_feature, logits = self.apply_loss(node=node, final_feature=node_output, softmax_weights=softmax_weights,
                                                softmax_biases=softmax_biases)
        node.evalDict[self.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking):
        feed_dict = {self.dataTensor: minibatch.samples,
                     self.labelTensor: minibatch.labels,
                     self.indicesTensor: minibatch.indices,
                     self.oneHotLabelTensor: minibatch.one_hot_labels,
                     # self.globalCounter: iteration,
                     self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
                     self.decisionWeightDecayCoeff: GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT,
                     self.useThresholding: int(use_threshold),
                     # self.isDecisionPhase: int(is_decision_phase),
                     self.isTrain: int(is_train),
                     self.useMasking: int(use_masking),
                     self.informationGainBalancingCoefficient: GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT,
                     self.iterationHolder: iteration,
                     self.filteredMask: np.ones((GlobalConstants.CURR_BATCH_SIZE,), dtype=bool),
                     self.batchSizeTf: GlobalConstants.CURR_BATCH_SIZE}
        if is_train:
            feed_dict[self.classificationDropoutKeepProb] = GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB
            if not self.isBaseline:
                self.get_softmax_decays(feed_dict=feed_dict, iteration=iteration, update=True)
                self.get_decision_dropout_prob(feed_dict=feed_dict, iteration=iteration, update=True)
                self.get_decision_weight(feed_dict=feed_dict, iteration=iteration, update=True)
        else:
            feed_dict[self.classificationDropoutKeepProb] = 1.0
            if not self.isBaseline:
                self.get_softmax_decays(feed_dict=feed_dict, iteration=1000000, update=False)
                self.get_decision_weight(feed_dict=feed_dict, iteration=iteration, update=False)
                feed_dict[self.decisionDropoutKeepProb] = 1.0
        return feed_dict

    def calculate_model_performance(self, sess, dataset, run_id, epoch_id, iteration):
        # moving_results_1 = sess.run(moving_stat_vars)
        is_evaluation_epoch_at_report_period = \
            epoch_id < GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING \
            and (epoch_id + 1) % GlobalConstants.EPOCH_REPORT_PERIOD == 0
        is_evaluation_epoch_before_ending = \
            epoch_id >= GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING
        if is_evaluation_epoch_at_report_period or is_evaluation_epoch_before_ending:
            training_accuracy, training_confusion = \
                self.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training,
                                        run_id=run_id,
                                        iteration=iteration)
            validation_accuracy, validation_confusion = \
                self.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test,
                                        run_id=run_id,
                                        iteration=iteration)
            validation_accuracy_corrected = 0.0
            DbLogger.write_into_table(
                rows=[(run_id, iteration, epoch_id, training_accuracy,
                       validation_accuracy, validation_accuracy_corrected,
                       0.0, 0.0, "XXX")], table=DbLogger.logsTable, col_count=9)

    def train(self, sess, dataset, run_id):
        iteration_counter = 0
        self.saver = tf.train.Saver()
        for epoch_id in range(GlobalConstants.TOTAL_EPOCH_COUNT):
            # An epoch is a complete pass on the whole dataset.
            dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
            print("*************Epoch {0}*************".format(epoch_id))
            total_time = 0.0
            while True:
                print("Iteration:{0}".format(iteration_counter))
                start_time = time.time()
                update_results = self.update_params(sess=sess,
                                                    dataset=dataset,
                                                    epoch=epoch_id,
                                                    iteration=iteration_counter)
                print("Update_results type new:{0}".format(update_results.__class__))
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                self.print_iteration_info(iteration_counter=iteration_counter, update_results=update_results)
                iteration_counter += 1
                if dataset.isNewEpoch:
                    print("Epoch Time={0}".format(total_time))
                    self.calculate_model_performance(sess=sess, dataset=dataset, run_id=run_id, epoch_id=epoch_id,
                                                     iteration=iteration_counter)
                    break
