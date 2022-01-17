import numpy as np
import tensorflow as tf
from auxillary.db_logger import DbLogger
from tf_2_cign.cign import Cign
from tf_2_cign.custom_layers.cign_secondary_routing_preparation_layer import CignScRoutingPrepLayer
from tf_2_cign.custom_layers.cign_vanilla_sc_routing_layer import CignVanillaScRoutingLayer
from tf_2_cign.utilities.utilities import Utilities
from tf_2_cign.custom_layers.cign_masking_layer import CignMaskingLayer
from tf_2_cign.custom_layers.cign_decision_layer import CignDecisionLayer
from tf_2_cign.custom_layers.cign_classification_layer import CignClassificationLayer
from collections import Counter
import time


# tf.autograph.set_verbosity(10, True)


class CignNoMask(Cign):
    def __init__(self,
                 batch_size,
                 input_dims,
                 class_count,
                 node_degrees,
                 decision_drop_probability,
                 classification_drop_probability,
                 decision_wd,
                 classification_wd,
                 information_gain_balance_coeff,
                 softmax_decay_controller,
                 learning_rate_schedule,
                 decision_loss_coeff,
                 bn_momentum=0.9):
        super().__init__(batch_size,
                         input_dims,
                         class_count,
                         node_degrees,
                         decision_drop_probability,
                         classification_drop_probability,
                         decision_wd,
                         classification_wd,
                         information_gain_balance_coeff,
                         softmax_decay_controller,
                         learning_rate_schedule,
                         decision_loss_coeff,
                         bn_momentum)
        self.weightedBatchNormOps = {}
        self.maskingLayers = {}
        self.decisionLayers = {}
        self.leafLossLayers = {}
        self.nodeFunctions = {}
        self.isRootDict = {}
        self.igMaskInputsDict = {}
        self.scMaskInputsDict = {}
        self.igActivationsDict = {}
        self.posteriorsDict = {}
        self.secondaryMatricesPerLevelDict = {}
        self.igRoutingMatricesDict = {}
        self.scRoutingMatricesDict = {}
        self.scRoutingPreparationLayers = []
        self.scRoutingCalculationLayers = []
        self.optimizer = None
        self.totalLossTracker = None
        self.classificationLossTrackers = None
        self.igLossTrackers = None
        self.warmUpPeriodInput = tf.keras.Input(shape=(), name="warm_up_period", dtype=tf.bool)
        self.feedDict["warm_up_period"] = self.warmUpPeriodInput

    def node_build_func(self, node):
        pass

    # We don't apply actual masking anymore. We can remove this layer later.
    # Record corresponding ig and sc masks here.
    def mask_inputs(self, node):
        self.maskingLayers[node.index] = CignMaskingLayer(network=self, node=node)

        if node.isRoot:
            parent_ig_matrix = tf.expand_dims(tf.ones_like(self.labels), axis=1)
            parent_sc_matrix = tf.expand_dims(tf.ones_like(self.labels), axis=1)
            parent_F = self.inputs
            parent_H = self.inputs
        else:
            parent_node = self.dagObject.parents(node=node)[0]
            parent_ig_matrix = self.nodeOutputsDict[parent_node.index]["ig_mask_matrix"]
            parent_sc_matrix = self.nodeOutputsDict[parent_node.index]["secondary_mask_matrix"]
            parent_F = self.nodeOutputsDict[parent_node.index]["F"]
            parent_H = self.nodeOutputsDict[parent_node.index]["H"]

        f_net, h_net, ig_mask, sc_mask, sample_count, is_node_open = \
            self.maskingLayers[node.index]([parent_F, parent_H, parent_ig_matrix, parent_sc_matrix])

        self.igMaskInputsDict[node.index] = ig_mask
        self.scMaskInputsDict[node.index] = sc_mask

        self.evalDict[Utilities.get_variable_name(name="f_net", node=node)] = f_net
        self.evalDict[Utilities.get_variable_name(name="h_net", node=node)] = h_net
        self.evalDict[Utilities.get_variable_name(name="ig_mask", node=node)] = ig_mask
        self.evalDict[Utilities.get_variable_name(name="sc_mask", node=node)] = sc_mask
        self.evalDict[Utilities.get_variable_name(name="sample_count", node=node)] = sample_count
        self.evalDict[Utilities.get_variable_name(name="is_node_open", node=node)] = is_node_open
        return f_net, h_net, ig_mask, sc_mask

    # Provide F and H outputs for a given node.
    def apply_node_funcs(self, node, f_net, h_net, ig_mask, sc_mask):
        node_func = self.node_build_func(node=node)
        self.nodeFunctions[node.index] = node_func
        self.nodeOutputsDict[node.index]["H"] = None
        if not node.isLeaf:
            f_net, h_net, pre_branch_feature = self.nodeFunctions[node.index]([f_net, h_net, ig_mask, sc_mask])
            self.evalDict[Utilities.get_variable_name(name="h_net", node=node)] = h_net
            self.evalDict[Utilities.get_variable_name(name="pre_branch_feature", node=node)] = pre_branch_feature
            self.nodeOutputsDict[node.index]["H"] = h_net
        else:
            f_net = self.nodeFunctions[node.index]([f_net, h_net, ig_mask, sc_mask])
        self.evalDict[Utilities.get_variable_name(name="f_net", node=node)] = f_net
        self.nodeOutputsDict[node.index]["F"] = f_net
        return f_net, h_net

    # Provide ig_mask_matrix output for a given inner node and calculate the information gain loss.
    def apply_decision(self, node, ig_mask, h_net=None):
        assert h_net is not None
        self.decisionLayers[node.index] = CignDecisionLayer(network=self, node=node,
                                                            decision_bn_momentum=self.bnMomentum)
        labels = self.labels
        temperature = self.routingTemperatures[node.index]

        h_net_normed, ig_value, output_ig_routing_matrix, ig_activations = self.decisionLayers[node.index](
            [h_net,
             ig_mask,
             labels,
             temperature])
        self.informationGainRoutingLosses[node.index] = ig_value
        self.igActivationsDict[node.index] = ig_activations

        self.evalDict[Utilities.get_variable_name(name="h_net_normed", node=node)] = h_net_normed
        self.evalDict[Utilities.get_variable_name(name="ig_value", node=node)] = ig_value
        self.evalDict[Utilities.get_variable_name(name="output_ig_routing_matrix",
                                                  node=node)] = output_ig_routing_matrix
        self.nodeOutputsDict[node.index]["ig_mask_matrix"] = output_ig_routing_matrix

    # Calculate the cross entropy loss for this leaf node; by paying attention to the secondary mask vector.
    def apply_classification_loss(self, node, f_net=None, sc_mask=None):
        assert f_net is not None and sc_mask is not None
        self.leafLossLayers[node.index] = CignClassificationLayer(network=self, node=node, class_count=self.classCount)
        labels = self.labels

        cross_entropy_loss_tensor, probability_vector, weighted_losses, loss, posteriors = \
            self.leafLossLayers[node.index](
                [f_net, sc_mask, labels])
        self.classificationLosses[node.index] = loss
        self.posteriorsDict[node.index] = posteriors

        self.evalDict[Utilities.get_variable_name(name="cross_entropy_loss_tensor", node=node)] = \
            cross_entropy_loss_tensor
        self.evalDict[Utilities.get_variable_name(name="probability_vector", node=node)] = probability_vector
        self.evalDict[Utilities.get_variable_name(name="weighted_losses", node=node)] = weighted_losses
        self.evalDict[Utilities.get_variable_name(name="loss", node=node)] = loss

    def calculate_secondary_routing_matrix(self, level, input_f_tensor, input_ig_routing_matrix):
        sc_routing_calculation_layer = CignVanillaScRoutingLayer(network=self)
        self.scRoutingCalculationLayers.append(sc_routing_calculation_layer)
        secondary_routing_matrix = sc_routing_calculation_layer([input_f_tensor, input_ig_routing_matrix])
        return secondary_routing_matrix

    def build_secondary_routing_matrices(self, level):
        level_nodes = self.orderedNodesPerLevel[level]
        f_outputs = [self.nodeOutputsDict[node.index]["F"] for node in level_nodes]
        ig_matrices = [self.nodeOutputsDict[node.index]["ig_mask_matrix"] for node in level_nodes]
        sc_masks = [self.scMaskInputsDict[node.index] for node in level_nodes]
        input_prep_layer = CignScRoutingPrepLayer(network=self, level=level)
        self.scRoutingPreparationLayers.append(input_prep_layer)
        input_f_tensor, input_ig_routing_matrix = input_prep_layer([f_outputs, ig_matrices, sc_masks])
        self.igRoutingMatricesDict[level] = input_ig_routing_matrix
        secondary_routing_matrix = self.calculate_secondary_routing_matrix(
            level=level,
            input_f_tensor=input_f_tensor,
            input_ig_routing_matrix=input_ig_routing_matrix)
        # Apply warm-up (all samples to every path), if enabled.
        secondary_routing_matrix_warm_up = tf.ones_like(secondary_routing_matrix)
        secondary_routing_matrix_final = tf.where(tf.cast(self.warmUpPeriodInput, tf.int32) > 0,
                                                  secondary_routing_matrix_warm_up,
                                                  secondary_routing_matrix)
        self.scRoutingMatricesDict[level] = secondary_routing_matrix_final

        # Distribute the results of the secondary routing matrix into the corresponding nodes
        curr_column = 0
        for node in level_nodes:
            node_child_count = len(self.dagObject.children(node=node))
            sc_routing_matrix_for_node = secondary_routing_matrix_final[:, curr_column: curr_column + node_child_count]
            self.evalDict["sc_routing_matrix_for_node_{0}_level_{1}".format(node.index, level)] = \
                sc_routing_matrix_for_node
            self.nodeOutputsDict[node.index]["secondary_mask_matrix"] = sc_routing_matrix_for_node
            curr_column += node_child_count

    def build_network(self):
        self.isBaseline = len(self.topologicalSortedNodes) == 1
        # Build all operations in each node -> Level by level
        for level in range(len(self.orderedNodesPerLevel)):
            for node in self.orderedNodesPerLevel[level]:
                print("Building Node:{0}".format(node.index))
                self.isRootDict[node.index] = tf.constant(node.isRoot)
                # Masking Layer
                f_net, h_net, ig_mask, sc_mask = self.mask_inputs(node=node)
                # Actions to be done in the node
                f_net, h_net = \
                    self.apply_node_funcs(node=node, f_net=f_net, h_net=h_net, ig_mask=ig_mask, sc_mask=sc_mask)
                # Decision Layer for inner nodes
                if level < len(self.orderedNodesPerLevel) - 1:
                    self.routingTemperatures[node.index] = \
                        tf.keras.Input(shape=(), name="routingTemperature_{0}".format(node.index))
                    self.feedDict["routingTemperature_{0}".format(node.index)] = self.routingTemperatures[node.index]
                    self.apply_decision(node=node, ig_mask=ig_mask, h_net=h_net)
                else:
                    self.apply_classification_loss(node=node,
                                                   f_net=f_net,
                                                   sc_mask=sc_mask)
            # Build secondary routing matrices after a level's nodes being built (if not the final layer)
            if level < len(self.orderedNodesPerLevel) - 1:
                self.build_secondary_routing_matrices(level=level)

        # Register node outputs to the eval dict
        for node in self.topologicalSortedNodes:
            for output_name in self.nodeOutputsDict[node.index].keys():
                if self.nodeOutputsDict[node.index][output_name] is None:
                    continue
                self.evalDict[Utilities.get_variable_name(name="node_output_{0}".format(output_name), node=node)] = \
                    self.nodeOutputsDict[node.index][output_name]
                self.evalDict[Utilities.get_variable_name(name="ig_mask_vector", node=node)] = self.igMaskInputsDict[
                    node.index]
                self.evalDict[Utilities.get_variable_name(name="sc_mask_vector", node=node)] = self.scMaskInputsDict[
                    node.index]

        self.evalDict["batch_size"] = self.batchSize

    def get_model_outputs_array(self):
        model_output_arr = [self.evalDict,
                            self.classificationLosses,
                            self.informationGainRoutingLosses,
                            self.posteriorsDict,
                            self.scMaskInputsDict,
                            self.igMaskInputsDict]
        return model_output_arr

    def build_tf_model(self):
        model_outputs_arr = self.get_model_outputs_array()
        self.model = tf.keras.Model(inputs=self.feedDict, outputs=model_outputs_arr)
        variables = self.model.trainable_variables
        self.calculate_regularization_coefficients(trainable_variables=variables)

    # def build_tf_model(self):
    #     # Build the final loss
    #     # Temporary model for getting the list of trainable variables
    #     self.model = tf.keras.Model(inputs=self.feedDict,
    #                                 outputs=[self.evalDict,
    #                                          self.classificationLosses,
    #                                          self.informationGainRoutingLosses,
    #                                          self.posteriorsDict,
    #                                          self.scMaskInputsDict,
    #                                          self.igMaskInputsDict])
    #     variables = self.model.trainable_variables
    #     self.calculate_regularization_coefficients(trainable_variables=variables)

    def init(self):
        self.build_tree()
        self.build_network()
        self.build_tf_model()

    def calculate_total_loss(self, classification_losses, info_gain_losses):
        # Weight decaying
        variables = self.model.trainable_variables
        regularization_losses = []
        for var in variables:
            if var.ref() in self.regularizationCoefficients:
                lambda_coeff = self.regularizationCoefficients[var.ref()]
                regularization_losses.append(lambda_coeff * tf.nn.l2_loss(var))
        total_regularization_loss = tf.add_n(regularization_losses)
        # Classification losses
        classification_loss = tf.add_n([loss for loss in classification_losses.values()])
        # Information Gain losses
        info_gain_loss = self.decisionLossCoeff * tf.add_n([loss for loss in info_gain_losses.values()])
        # Total loss
        total_loss = total_regularization_loss + info_gain_loss + classification_loss
        return total_loss, total_regularization_loss, info_gain_loss, classification_loss

    def print_losses(self, **kwargs):
        eval_dict = kwargs["eval_dict"]
        print("Total Loss:{0}".format(self.totalLossTracker.result().numpy()))
        classification_str = "Classification Losses: "
        for node_id in self.classificationLossTrackers.keys():
            classification_str += "Node_{0}:{1} ".format(
                node_id,
                self.classificationLossTrackers[node_id].result().numpy())
        ig_str = "IG Losses: "
        for node_id in self.igLossTrackers.keys():
            ig_str += "Node_{0}:{1} ".format(
                node_id,
                self.igLossTrackers[node_id].result().numpy())
        sample_counts_str = ""
        for node in self.topologicalSortedNodes:
            sample_counts_str += "Node {0}:{1} ".format(
                node.index, eval_dict[Utilities.get_variable_name(name="sample_count", node=node)])
        print(sample_counts_str)
        print(classification_str)
        print(ig_str)

    def print_train_step_info(self, **kwargs):
        iteration = kwargs["iteration"]
        time_intervals = kwargs["time_intervals"]
        eval_dict = kwargs["eval_dict"]

        # Print outputs
        print("************************************")
        print("Iteration {0}".format(iteration))
        for k, v in time_intervals.items():
            print("{0}={1}".format(k, v))
        self.print_losses(eval_dict=eval_dict)
        print("Temperature:{0}".format(self.softmaxDecayController.get_value()))
        print("Lr:{0}".format(self.optimizer._decayed_lr(tf.float32).numpy()))
        print("************************************")

    def save_log_data(self, **kwargs):
        run_id = kwargs["run_id"]
        iteration = kwargs["iteration"]
        eval_dict = kwargs["eval_dict"]
        # Record ig and classification losses and sample counts into the kv store.
        kv_rows = []
        for node_id in self.igLossTrackers:
            key_ = "IG Node_{0}".format(node_id)
            val_ = np.asscalar(self.igLossTrackers[node_id].result().numpy())
            kv_rows.append((run_id, iteration, key_, val_))

        for node in self.topologicalSortedNodes:
            key_ = "Sample Count Node_{0}".format(node.index)
            val_ = np.asscalar(eval_dict[Utilities.get_variable_name(name="sample_count", node=node)].numpy())
            kv_rows.append((run_id, iteration, key_, val_))

        for node_id in self.classificationLossTrackers:
            key_ = "Classification Node_{0}".format(node_id)
            val_ = np.asscalar(self.classificationLossTrackers[node_id].result().numpy())
            kv_rows.append((run_id, iteration, key_, val_))
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)

    def build_trackers(self):
        self.totalLossTracker = tf.keras.metrics.Mean(name="totalLossTracker")
        # self.classificationLossTracker = tf.keras.metrics.Mean(name="classificationLossTracker")
        self.classificationLossTrackers = \
            {node.index: tf.keras.metrics.Mean(name="classificationLossTracker_Node{0}".format(node.index))
             for node in self.topologicalSortedNodes if node.isLeaf is True}
        self.igLossTrackers = {node.index: tf.keras.metrics.Mean(name="igLossTracker_Node{0}".format(node.index))
                               for node in self.topologicalSortedNodes if node.isLeaf is False}

    def reset_trackers(self):
        self.totalLossTracker.reset_states()
        for node_id in self.classificationLossTrackers:
            self.classificationLossTrackers[node_id].reset_states()
        for node_id in self.igLossTrackers:
            self.igLossTrackers[node_id].reset_states()

    def track_losses(self, **kwargs):
        #  total_loss, classification_losses, info_gain_losses
        total_loss = kwargs["total_loss"]
        classification_losses = kwargs["classification_losses"]
        info_gain_losses = kwargs["info_gain_losses"]
        self.totalLossTracker.update_state(total_loss)
        for node_id in classification_losses:
            self.classificationLossTrackers[node_id].update_state(classification_losses[node_id])
        for node_id in info_gain_losses:
            self.igLossTrackers[node_id].update_state(self.decisionLossCoeff * info_gain_losses[node_id])

    def get_sgd_optimizer(self):
        boundaries = [tpl[0] for tpl in self.learningRateSchedule.schedule]
        values = [self.learningRateSchedule.initialValue]
        values.extend([tpl[1] for tpl in self.learningRateSchedule.schedule])
        learning_rate_scheduler_tf = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries, values=values)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_scheduler_tf, momentum=0.9)
        return optimizer

    def get_adam_optimizer(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        return optimizer

    def measure_performance(self, dataset, run_id, iteration, epoch_id, times_list):
        training_accuracy = self.eval(run_id=run_id, iteration=iteration,
                                      dataset=dataset.trainDataTf, dataset_type="train")
        validation_accuracy = self.eval(run_id=run_id, iteration=iteration,
                                        dataset=dataset.validationDataTf, dataset_type="validation")
        test_accuracy = self.eval(run_id=run_id, iteration=iteration,
                                  dataset=dataset.testDataTf, dataset_type="test")
        print("Train Accuracy:{0}".format(training_accuracy))
        print("Validation Accuracy:{0}".format(validation_accuracy))
        print("Test Accuracy:{0}".format(test_accuracy))
        mean_time_passed = np.mean(np.array(times_list))
        DbLogger.write_into_table(
            rows=[(run_id, iteration, epoch_id, training_accuracy,
                   validation_accuracy, test_accuracy,
                   mean_time_passed, 0.0, "XXX")], table=DbLogger.logsTable)

    def run_model(self, **kwargs):
        X = kwargs["X"]
        y = kwargs["y"]
        iteration = kwargs["iteration"]
        is_training = kwargs["is_training"]
        feed_dict = self.get_feed_dict(x=X, y=y, iteration=iteration, is_training=is_training)

        eval_dict, classification_losses, info_gain_losses, posteriors_dict, \
        sc_masks_dict, ig_masks_dict = self.model(inputs=feed_dict, training=is_training)

        model_output = {
            "eval_dict": eval_dict,
            "classification_losses": classification_losses,
            "info_gain_losses": info_gain_losses,
            "posteriors_dict": posteriors_dict,
            "sc_masks_dict": sc_masks_dict,
            "ig_masks_dict": ig_masks_dict
        }

        return model_output

    def train(self, run_id, dataset, epoch_count):
        self.optimizer = self.get_sgd_optimizer()
        self.build_trackers()

        iteration = 0
        for epoch_id in range(epoch_count):
            self.reset_trackers()
            times_list = []

            # Train for one loop
            for train_X, train_y in dataset.trainDataTf:
                with tf.GradientTape() as tape:
                    t0 = time.time()
                    t1 = time.time()
                    model_output = self.run_model(
                        X=train_X,
                        y=train_y,
                        iteration=iteration,
                        is_training=True)

                    t2 = time.time()
                    total_loss, total_regularization_loss, info_gain_loss, classification_loss = \
                        self.calculate_total_loss(
                            classification_losses=model_output["classification_losses"],
                            info_gain_losses=model_output["info_gain_losses"])
                t3 = time.time()
                # self.unit_test_cign_routing_mechanism(
                #     eval_dict=model_output["eval_dict"],
                #     tape=tape,
                #     classification_loss=classification_loss,
                #     info_gain_loss=info_gain_loss)
                t4 = time.time()
                # Apply grads
                grads = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                t5 = time.time()
                # Track losses
                self.track_losses(total_loss=total_loss, classification_losses=model_output["classification_losses"],
                                  info_gain_losses=model_output["info_gain_losses"])
                times_list.append(t5 - t0)

                self.print_train_step_info(
                    iteration=iteration,
                    time_intervals=[t0, t1, t2, t3, t4, t5],
                    eval_dict=model_output["eval_dict"])
                iteration += 1
                # Print outputs

            self.save_log_data(run_id=run_id,
                               iteration=iteration,
                               eval_dict=model_output["eval_dict"])

            # Eval on training, validation and test sets
            if epoch_id >= epoch_count - 10 or epoch_id % self.trainEvalPeriod == 0:
                self.measure_performance(dataset=dataset,
                                         run_id=run_id,
                                         iteration=iteration,
                                         epoch_id=epoch_id,
                                         times_list=times_list)

    # TODO: Control that.
    def calculate_predictions_of_batch(self, model_output, y):
        leaf_weights = []
        posteriors = []
        leaf_distributions_batch = {}
        for leaf_node in self.leafNodes:
            sc_mask = model_output["sc_masks_dict"][leaf_node.index]
            posterior = model_output["posteriors_dict"][leaf_node.index]
            leaf_weights.append(np.expand_dims(sc_mask, axis=-1))
            posteriors.append(posterior)
            y_leaf = y.numpy()[sc_mask.numpy().astype(np.bool)]
            leaf_distributions_batch[leaf_node.index] = y_leaf
        leaf_weights = np.stack(leaf_weights, axis=-1)
        posteriors = np.stack(posteriors, axis=-1)

        weighted_posteriors = leaf_weights * posteriors
        posteriors_mixture = np.sum(weighted_posteriors, axis=-1)
        y_pred_batch = np.argmax(posteriors_mixture, axis=-1)
        return y_pred_batch, leaf_distributions_batch

    def eval(self, run_id, iteration, dataset, dataset_type):
        if dataset is None:
            return 0.0
        y_true = []
        y_pred = []

        leaf_distributions = {node.index: [] for node in self.leafNodes}
        for X, y in dataset:
            model_output = self.run_model(
                X=X,
                y=y,
                iteration=-1,
                is_training=False)
            y_pred_batch, leaf_distributions_batch = self.calculate_predictions_of_batch(
                model_output=model_output, y=y)
            for leaf_node in self.leafNodes:
                leaf_distributions[leaf_node.index].extend(leaf_distributions_batch[leaf_node.index])
            y_pred.append(y_pred_batch)
            y_true.append(y.numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        truth_vector = y_true == y_pred
        accuracy = np.mean(truth_vector.astype(np.float))

        # Print sample distribution
        kv_rows = []
        for leaf_node in self.leafNodes:
            c = Counter(leaf_distributions[leaf_node.index])
            str_ = "{0} Node {1} Sample Distribution:{2}".format(dataset_type, leaf_node.index, c)
            print(str_)
            kv_rows.append((run_id, iteration,
                            "{0} Node {1} Sample Distribution".format(dataset_type, leaf_node.index),
                            "{0}".format(c)))
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)
        return accuracy

    def unit_test_cign_routing_mechanism(self, eval_dict, **kwargs):
        tape = kwargs["tape"]
        classification_loss = kwargs["classification_loss"]
        info_gain_loss = kwargs["info_gain_loss"]
        # Statistics of sample distribution
        for node in self.topologicalSortedNodes:
            key_name = Utilities.get_variable_name(name="sample_count", node=node)
            if key_name in eval_dict:
                print("Node{0} Sample count:{1}".format(node.index, eval_dict[key_name].numpy()))

        # Assert that all mask vectors sum up to unity at each row in the leaf layer.
        ig_masks = []
        for node in self.orderedNodesPerLevel[-1]:
            ig_masks.append(eval_dict[Utilities.get_variable_name(name="ig_mask_vector", node=node)])
        ig_matrix = np.stack(ig_masks, axis=1)
        row_sums = np.sum(ig_matrix, axis=1)
        assert np.array_equal(row_sums, np.ones_like(row_sums))

        # Assert that all ig routing masks of the child nodes; when concatenated are equal to the parent's ig routing
        # matrix.
        for node in self.topologicalSortedNodes:
            if node.isLeaf:
                continue
            ig_matrix = eval_dict[Utilities.get_variable_name(name="node_output_{0}".format("ig_mask_matrix"),
                                                              node=node)]
            child_nodes = self.dagObject.children(node=node)
            ig_masks_of_children = [None] * len(child_nodes)
            for child_node in child_nodes:
                sibling_index = self.get_node_sibling_index(node=child_node)
                ig_masks_of_children[sibling_index] = eval_dict[Utilities.get_variable_name(name="ig_mask_vector",
                                                                                            node=child_node)]
            ig_matrix_constructed = np.stack(ig_masks_of_children, axis=1)
            assert np.array_equal(ig_matrix, ig_matrix_constructed)

        # Assert that decision variables do not receive gradients from classification loss.
        classification_grads = tape.gradient(classification_loss, self.model.trainable_variables)
        for var_idx, var in enumerate(self.model.trainable_variables):
            if self.is_decision_variable(variable=var):
                assert classification_grads[var_idx] is None
            else:
                assert classification_grads[var_idx] is not None
        print("All assertions are correct")

    def get_explanation_string(self):
        explanation = ""
        total_param_count = 0
        for v in self.model.trainable_variables:
            total_param_count += np.prod(v.get_shape().as_list())

        explanation += "Batch Size:{0}\n".format(self.batchSizeNonTensor)
        explanation += "Tree Degree:{0}\n".format(self.degreeList)
        explanation += "********Lr Settings********\n"
        explanation += self.learningRateSchedule.get_explanation()
        explanation += "********Lr Settings********\n"
        explanation += "Decision Loss Coeff:{0}\n".format(self.decisionLossCoeff)
        explanation += "Batch Norm Decay:{0}\n".format(self.bnMomentum)
        explanation += "Param Count:{0}\n".format(total_param_count)
        explanation += "Classification Wd:{0}\n".format(self.classificationWd)
        explanation += "Decision Wd:{0}\n".format(self.decisionWd)
        explanation += "Information Gain Balance Coefficient:{0}\n".format(self.informationGainBalanceCoeff)
        explanation += "Decision Dropout:{0}\n".format(self.decisionDropProbability)
        explanation += "Classification Dropout:{0}\n".format(self.classificationDropProbability)
        return explanation
