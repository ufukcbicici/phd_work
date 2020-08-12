import time

import tensorflow as tf

from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants


class CignEarlyExitTree(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset, network_name):

        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                         dataset, network_name)
        self.earlyExitFeatures = {}
        self.earlyExitLogits = {}
        self.earlyExitLosses = {}
        self.earlyExitWeight = tf.placeholder(name="early_exit_loss_weight", dtype=tf.float32)

        self.lateExitFeatures = {}
        self.lateExitLogits = {}
        self.lateExitLosses = {}
        self.lateExitWeight = tf.placeholder(name="late_exit_loss_weight", dtype=tf.float32)

        self.sumEarlyExits = None
        self.sumLateExits = None

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking):
        feed_dict = super().prepare_feed_dict(minibatch=minibatch, iteration=iteration,
                                              use_threshold=use_threshold, is_train=is_train,
                                              use_masking=use_masking)
        feed_dict[self.earlyExitWeight] = GlobalConstants.EARLY_EXIT_WEIGHT
        feed_dict[self.lateExitWeight] = GlobalConstants.LATE_EXIT_WEIGHT
        return feed_dict

    def apply_loss(self, node, final_feature, softmax_weights, softmax_biases):
        node.residueOutputTensor = final_feature
        node.finalFeatures = final_feature
        self.earlyExitFeatures[node.index] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_final", node=node)] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_mag", node=node)] = tf.nn.l2_loss(final_feature)
        logits = FastTreeNetwork.fc_layer(x=final_feature, W=softmax_weights, b=softmax_biases, node=node,
                                          name="early_exit_fc_op")
        node.evalDict[self.get_variable_name(name="logits", node=node)] = logits
        self.earlyExitLogits[node.index] = logits
        loss = self.make_loss(node=node, logits=logits)
        self.earlyExitLosses[node.index] = loss
        return final_feature, logits

    def apply_late_loss(self, node, final_feature, softmax_weights, softmax_biases):
        self.lateExitFeatures[node.index] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_late_exit", node=node)] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_late_exit_mag", node=node)] = \
            tf.nn.l2_loss(final_feature)
        logits = FastTreeNetwork.fc_layer(x=final_feature, W=softmax_weights, b=softmax_biases, node=node,
                                          name="late_exit_fc_op")
        node.evalDict[self.get_variable_name(name="logits_late_exit", node=node)] = logits
        self.lateExitLogits[node.index] = logits
        loss = self.make_loss(node=node, logits=logits)
        self.lateExitLosses[node.index] = loss
        return final_feature, logits

    def build_main_loss(self):
        assert self.earlyExitWeight is not None and self.lateExitWeight is not None
        self.sumEarlyExits = tf.add_n(list(self.earlyExitLosses.values()))
        self.sumLateExits = tf.add_n(list(self.lateExitLosses.values()))
        self.mainLoss = (self.earlyExitWeight * self.sumEarlyExits) + (self.lateExitWeight * self.sumLateExits)

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
            validation_accuracy_late = 0.0
            if not self.isBaseline:
                validation_accuracy_late, validation_confusion_late = \
                    self.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test,
                                            run_id=run_id,
                                            iteration=iteration, posterior_entry_name="posterior_probs_late")
                if is_evaluation_epoch_before_ending:
                    # self.save_model(sess=sess, run_id=run_id, iteration=iteration)
                    t0 = time.time()
                    # self.save_routing_info(sess=sess, run_id=run_id, iteration=iteration,
                    #                        dataset=dataset, dataset_type=DatasetTypes.training)
                    t1 = time.time()
                    self.save_routing_info(sess=sess, run_id=run_id, iteration=iteration,
                                           dataset=dataset, dataset_type=DatasetTypes.test)
                    t2 = time.time()
                    print("t1-t0={0}".format(t1 - t0))
                    print("t2-t1={0}".format(t2 - t1))
            DbLogger.write_into_table(
                rows=[(run_id, iteration, epoch_id, training_accuracy,
                       validation_accuracy, validation_accuracy_late,
                       0.0, 0.0, "XXX")], table=DbLogger.logsTable, col_count=9)