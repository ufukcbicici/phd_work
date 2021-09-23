from collections import Counter

import numpy as np
import tensorflow as tf
import time
import pickle
import os
import shutil

from auxillary.db_logger import DbLogger
from tf_2_cign.cign_no_mask import CignNoMask
from tf_2_cign.cign_rl_routing import CignRlRouting
from tf_2_cign.custom_layers.cign_rl_routing_layer import CignRlRoutingLayer
from tf_2_cign.utilities import Utilities


class CignRlRoutingWithIterativeTraining(CignRlRouting):
    infeasible_action_penalty = -1000000.0

    def __init__(self, valid_prediction_reward, invalid_prediction_penalty, include_ig_in_reward_calculations,
                 lambda_mac_cost, warm_up_period, cign_rl_train_period, batch_size, input_dims, class_count,
                 node_degrees, decision_drop_probability, classification_drop_probability, decision_wd,
                 classification_wd, information_gain_balance_coeff, softmax_decay_controller, learning_rate_schedule,
                 decision_loss_coeff, bn_momentum=0.9):
        super().__init__(valid_prediction_reward, invalid_prediction_penalty, include_ig_in_reward_calculations,
                         lambda_mac_cost, warm_up_period, cign_rl_train_period, batch_size, input_dims, class_count,
                         node_degrees, decision_drop_probability, classification_drop_probability, decision_wd,
                         classification_wd, information_gain_balance_coeff, softmax_decay_controller,
                         learning_rate_schedule, decision_loss_coeff, bn_momentum)

    def get_regularization_loss(self, is_for_q_nets):
        variables = self.model.trainable_variables
        regularization_losses = []
        for var in variables:
            if var.ref() in self.regularizationCoefficients:
                if is_for_q_nets and "q_net" in var.name:
                    lambda_coeff = self.regularizationCoefficients[var.ref()]
                elif is_for_q_nets and "q_net" not in var.name:
                    lambda_coeff = 0.0
                elif not is_for_q_nets and "q_net" in var.name:
                    lambda_coeff = 0.0
                elif not is_for_q_nets and "q_net" not in var.name:
                    lambda_coeff = self.regularizationCoefficients[var.ref()]
                else:
                    raise NotImplementedError()
                regularization_losses.append(lambda_coeff * tf.nn.l2_loss(var))
        total_regularization_loss = tf.add_n(regularization_losses)
        return total_regularization_loss

    def calculate_total_loss(self, classification_losses, info_gain_losses):
        # Weight decaying
        total_regularization_loss = self.get_regularization_loss(is_for_q_nets=False)
        # Classification losses
        classification_loss = tf.add_n([loss for loss in classification_losses.values()])
        # Information Gain losses
        info_gain_loss = self.decisionLossCoeff * tf.add_n([loss for loss in info_gain_losses.values()])
        # Total loss
        total_loss = total_regularization_loss + info_gain_loss + classification_loss
        return total_loss, total_regularization_loss, info_gain_loss, classification_loss

    def train_one_epoch(self,
                        dataset,
                        run_id,
                        iteration,
                        is_in_warm_up_period):
        self.reset_trackers()
        times_list = []
        # Create a validation set; from which can sample batch_size number of samples; for every iteration
        # on the training set. We are going to use these sample for training the Q-Nets.
        assert dataset.valX is not None and dataset.valY is not None
        dataset_size_ratio = int(dataset.trainX.shape[0] / dataset.valX.shape[0]) + 1
        val_dataset = tf.data.Dataset.from_tensor_slices((dataset.valX, dataset.valY)).repeat(
            count=dataset_size_ratio).shuffle(buffer_size=5000).batch(batch_size=self.batchSizeNonTensor)
        val_dataset_iter = iter(val_dataset)
        # mse_losses = [tf.keras.losses.MeanSquaredError() for _ in range(self.get_max_trajectory_length())]
        mse_loss = tf.keras.losses.MeanSquaredError()
        # count = 0
        # while True:
        #     vx, vy = val_dataset_iter.__next__()
        #     count += 1
        # print("X")

        # Draw one minibatch from the training set. Calculate gradients with respect to the
        # main loss and information gain loss and regularization loss.
        for train_X, train_y in dataset.trainDataTf:
            # ********** Loss calculation and gradients for the main loss functions. **********
            with tf.GradientTape() as main_tape:
                t0 = time.time()
                t1 = time.time()
                model_output = self.run_model(
                    X=train_X,
                    y=train_y,
                    iteration=iteration,
                    is_training=True,
                    warm_up_period=is_in_warm_up_period)
                t2 = time.time()
                classification_losses = model_output["classification_losses"]
                info_gain_losses = model_output["info_gain_losses"]
                # Weight decaying; Q-Net variables excluded
                total_regularization_loss = self.get_regularization_loss(is_for_q_nets=False)
                # Classification losses
                classification_loss = tf.add_n([loss for loss in classification_losses.values()])
                # Information Gain losses
                info_gain_loss = self.decisionLossCoeff * tf.add_n([loss for loss in info_gain_losses.values()])
                # Total loss
                total_loss = total_regularization_loss + info_gain_loss + classification_loss
            t3 = time.time()
            # Calculate grads with respect to the main loss
            main_grads = main_tape.gradient(total_loss, self.model.trainable_variables)
            # Check that q_net variables receive zero gradients if the L2 coefficient is zero.
            for idx, v in enumerate(self.model.trainable_variables):
                if "q_net" not in v.name:
                    continue
                grad_arr = main_grads[idx].numpy()
                assert np.array_equal(grad_arr, np.zeros_like(grad_arr))
            # ********** Loss calculation and gradients for the main loss functions. **********

            # ********** Loss calculation and gradients for the Q-Nets. **********
            # Draw samples from the validation set
            val_X, val_y = val_dataset_iter.__next__()
            with tf.GradientTape() as q_tape:
                model_output_val = self.run_model(
                    X=val_X,
                    y=val_y,
                    iteration=iteration,
                    is_training=True,
                    warm_up_period=is_in_warm_up_period)
                # Calculate target values for the Q-Nets
                posteriors_val = {k: v.numpy() for k, v in model_output_val["posteriors_dict"].items()}
                ig_masks_val = {k: v.numpy() for k, v in model_output_val["ig_masks_dict"].items()}
                regs, q_s = self.calculate_q_tables_from_network_outputs(true_labels=val_y.numpy(),
                                                                         posteriors_dict=posteriors_val,
                                                                         ig_masks_dict=ig_masks_val)
                # Q-Net Losses
                q_net_predicted = model_output_val["q_tables_predicted"]
                q_net_losses = []
                for idx, tpl in enumerate(zip(regs, q_net_predicted)):
                    q_truth = tpl[0]
                    q_predicted = tpl[1]
                    q_truth_tensor = tf.convert_to_tensor(q_truth, dtype=q_predicted.dtype)
                    mse = mse_loss(q_truth_tensor, q_predicted)
                    q_net_losses.append(mse)
                full_q_loss = tf.add_n(q_net_losses)
            q_grads = q_tape.gradient(full_q_loss, self.model.trainable_variables)
            # Check that q_net variables do not receive zero gradients and leaf node variables
            # receive no gradients.
            leaf_node_names = ["Node{0}".format(node.index) for node in self.leafNodes]
            for idx, v in enumerate(self.model.trainable_variables):
                print(v.name)
                leaf_names_check_arr = [node_name in v.name for node_name in leaf_node_names]
                grad = q_grads[idx]
                if "decision" in v.name or any(leaf_names_check_arr):
                    assert (grad is None) or np.array_equal(grad.numpy(), np.zeros_like(grad.numpy()))
                else:
                    assert grad is not None
                    assert not np.allclose(grad.numpy(), np.zeros_like(grad.numpy()))

            print("X")

    def train(self, run_id, dataset, epoch_count, **kwargs):
        # q_net_epoch_count = kwargs["q_net_epoch_count"]
        is_in_warm_up_period = True
        self.optimizer = self.get_sgd_optimizer()
        self.build_trackers()

        cign_main_body_variables = self.model.trainable_variables
        epochs_after_warm_up = 0
        iteration = 0
        for epoch_id in range(epoch_count):
            self.train_one_epoch(
                dataset=dataset,
                run_id=run_id,
                iteration=iteration,
                is_in_warm_up_period=is_in_warm_up_period)
            # iteration, times_list = self.train_cign_body_one_epoch(
            #     dataset=dataset,
            #     cign_main_body_variables=cign_main_body_variables,
            #     run_id=run_id,
            #     iteration=iteration,
            #     is_in_warm_up_period=is_in_warm_up_period)
