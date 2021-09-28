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

    def track_losses(self, **kwargs):
        super(CignRlRoutingWithIterativeTraining, self).track_losses(**kwargs)
        q_net_losses = kwargs["q_net_losses"]
        for idx, q_net_loss in enumerate(q_net_losses):
            self.qNetTrackers[idx].update_state(q_net_loss)

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

        q_str = "Q-Net Losses: "
        for idx, q_net_loss in enumerate(self.qNetTrackers):
            q_str += "Q-Net_{0}:{1} ".format(idx, self.qNetTrackers[idx].result().numpy())

        print(q_str)
        print("Temperature:{0}".format(self.softmaxDecayController.get_value()))
        print("Lr:{0}".format(self.optimizer._decayed_lr(tf.float32).numpy()))
        print("************************************")

    def train(self, run_id, dataset, epoch_count, **kwargs):
        # q_net_epoch_count = kwargs["q_net_epoch_count"]
        is_in_warm_up_period = True
        self.optimizer = self.get_sgd_optimizer()
        # OK
        self.build_trackers()
        mse_loss = tf.keras.losses.MeanSquaredError()

        epochs_after_warm_up = 0
        iteration = 0
        for epoch_id in range(epoch_count):
            # OK
            self.reset_trackers()
            times_list = []
            # Create a validation set; from which can sample batch_size number of samples; for every iteration
            # on the training set. We are going to use these sample for training the Q-Nets.
            assert dataset.valX is not None and dataset.valY is not None
            dataset_size_ratio = int(dataset.trainX.shape[0] / dataset.valX.shape[0]) + 1
            val_dataset = tf.data.Dataset.from_tensor_slices((dataset.valX, dataset.valY)).repeat(
                count=dataset_size_ratio).shuffle(buffer_size=5000).batch(batch_size=self.batchSizeNonTensor)
            val_dataset_iter = iter(val_dataset)

            # Draw one minibatch from the training set. Calculate gradients with respect to the
            # main loss and information gain loss and regularization loss.
            time_measurements = {}
            for train_X, train_y in dataset.trainDataTf:
                # ********** Loss calculation and gradients for the main loss functions. **********
                with tf.GradientTape() as main_tape:
                    t0 = time.time()
                    model_output = self.run_model(
                        X=train_X,
                        y=train_y,
                        iteration=iteration,
                        is_training=True,
                        warm_up_period=is_in_warm_up_period)
                    t1 = time.time()
                    time_measurements["main_loss_run_model"] = t1 - t0
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
                t2 = time.time()
                time_measurements["main_loss_calculations"] = t2 - t1
                # Calculate grads with respect to the main loss
                main_grads = main_tape.gradient(total_loss, self.model.trainable_variables)
                # Check that q_net variables receive zero gradients if the L2 coefficient is zero.
                t3 = time.time()
                time_measurements["main_tape.gradient"] = t3 - t2
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
                    t4 = time.time()
                    model_output_val = self.run_model(
                        X=val_X,
                        y=val_y,
                        iteration=iteration,
                        is_training=True,
                        warm_up_period=is_in_warm_up_period)
                    t5 = time.time()
                    time_measurements["q_net_run_model"] = t5 - t4
                    # Calculate target values for the Q-Nets
                    posteriors_val = {k: v.numpy() for k, v in model_output_val["posteriors_dict"].items()}
                    ig_masks_val = {k: v.numpy() for k, v in model_output_val["ig_masks_dict"].items()}
                    regs, q_s = self.calculate_q_tables_from_network_outputs(true_labels=val_y.numpy(),
                                                                             posteriors_dict=posteriors_val,
                                                                             ig_masks_dict=ig_masks_val)
                    t6 = time.time()
                    time_measurements["calculate_q_tables_from_network_outputs"] = t6 - t5
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
                    q_regularization_loss = self.get_regularization_loss(is_for_q_nets=True)
                    total_q_loss = full_q_loss + q_regularization_loss
                    # total_q_loss = 10.0 * total_q_loss
                    t7 = time.time()
                    time_measurements["calculate_q_tables_from_network_outputs"] = t7 - t6
                q_grads = q_tape.gradient(total_q_loss, self.model.trainable_variables)
                t8 = time.time()
                time_measurements["q_tape.gradient"] = t8 - t7
                # Check that q_net variables do not receive zero gradients and leaf node variables
                # receive no gradients. Insert zero gradients for every None variable.
                leaf_node_names = ["Node{0}".format(node.index) for node in self.leafNodes]
                # q_grads_fixed = []
                for idx, v in enumerate(self.model.trainable_variables):
                    # print(v.name)
                    leaf_names_check_arr = [node_name in v.name for node_name in leaf_node_names]
                    q_grad = q_grads[idx]
                    main_grad = main_grads[idx]
                    assert q_grad is not None
                    if "decision" in v.name or any(leaf_names_check_arr):
                        assert np.array_equal(q_grad.numpy(), np.zeros_like(q_grad.numpy()))
                        # assert (q_grad is None) or np.array_equal(q_grad.numpy(), np.zeros_like(q_grad.numpy()))
                        # if q_grad is None:
                        #     q_grad = np.zeros_like(main_grad)
                    else:
                        assert not np.allclose(q_grad.numpy(), np.zeros_like(q_grad.numpy()))
                    assert q_grad.shape == main_grad.shape
                    # q_grads_fixed.append(q_grad)
                # Sum the main CIGN grads and Q-Net grads; apply Gradient Descent step.
                t9 = time.time()
                all_grads = [m_g + q_g for m_g, q_g in zip(main_grads, q_grads)]
                assert all([np.allclose(main_grads[i].numpy() + q_grads[i].numpy(), all_grads[i])
                            for i in range(len(all_grads))])
                self.optimizer.apply_gradients(zip(all_grads, self.model.trainable_variables))
                t10 = time.time()
                time_measurements["self.optimizer.apply_gradients"] = t10 - t9
                # Track losses
                self.track_losses(total_loss=total_loss,
                                  classification_losses=model_output["classification_losses"],
                                  info_gain_losses=model_output["info_gain_losses"],
                                  q_net_losses=q_net_losses)
                self.print_train_step_info(
                    iteration=iteration,
                    time_intervals=time_measurements,
                    eval_dict=model_output["eval_dict"])
                iteration += 1
                # Print outputs

            self.save_log_data(run_id=run_id,
                               iteration=iteration,
                               info_gain_losses=model_output["info_gain_losses"],
                               classification_losses=model_output["classification_losses"],
                               eval_dict=model_output["eval_dict"])
            kv_rows = []
            for idx, q_net_loss in enumerate(self.qNetTrackers):
                key_ = "Q-Net_{0}".format(idx)
                val_ = np.asscalar(self.qNetTrackers[idx].result().numpy())
                kv_rows.append((run_id, iteration, key_, val_))
            DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)

            # Evaluation
            self.measure_performance(dataset=dataset,
                                     run_id=run_id,
                                     iteration=iteration,
                                     epoch_id=epoch_id,
                                     times_list=times_list)
            if epoch_id >= self.warmUpPeriod:
                is_in_warm_up_period = False

            # if epoch_id >= self.warmUpPeriod:
            #     # Run Q-Net learning for the first time.
            #     if is_in_warm_up_period:
            #         self.save_model(run_id=run_id)
            #         is_in_warm_up_period = False
            #         self.train_q_nets(dataset=dataset, q_net_epoch_count=q_net_epoch_count)
            #         self.measure_performance(dataset=dataset,
            #                                  run_id=run_id,
            #                                  iteration=iteration,
            #                                  epoch_id=epoch_id,
            #                                  times_list=times_list)
            #     else:
            #         is_performance_measured = False
            #         if (epochs_after_warm_up + 1) % self.cignRlTrainPeriod == 0:
            #             self.train_q_nets(dataset=dataset, q_net_epoch_count=q_net_epoch_count)
            #             self.measure_performance(dataset=dataset,
            #                                      run_id=run_id,
            #                                      iteration=iteration,
            #                                      epoch_id=epoch_id,
            #                                      times_list=times_list)
            #             is_performance_measured = True
            #         if (epoch_id >= epoch_count - 10 or epoch_id % self.trainEvalPeriod == 0) \
            #                 and is_performance_measured is False:
            #             self.measure_performance(dataset=dataset,
            #                                      run_id=run_id,
            #                                      iteration=iteration,
            #                                      epoch_id=epoch_id,
            #                                      times_list=times_list)
            #     epochs_after_warm_up += 1

