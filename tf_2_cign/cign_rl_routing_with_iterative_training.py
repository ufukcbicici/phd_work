import numpy as np
import tensorflow as tf
import time

from auxillary.db_logger import DbLogger
from tf_2_cign.cign_rl_routing import CignRlRouting


class CignRlRoutingWithIterativeTraining(CignRlRouting):
    infeasible_action_penalty = -1000000.0

    def __init__(self, valid_prediction_reward, invalid_prediction_penalty, include_ig_in_reward_calculations,
                 lambda_mac_cost, warm_up_period, cign_rl_train_period, batch_size, input_dims, class_count,
                 node_degrees, decision_drop_probability, classification_drop_probability, decision_wd,
                 classification_wd, information_gain_balance_coeff, softmax_decay_controller, learning_rate_schedule,
                 decision_loss_coeff, q_net_coeff, bn_momentum=0.9):
        super().__init__(valid_prediction_reward, invalid_prediction_penalty, include_ig_in_reward_calculations,
                         lambda_mac_cost, warm_up_period, cign_rl_train_period, batch_size, input_dims, class_count,
                         node_degrees, decision_drop_probability, classification_drop_probability, decision_wd,
                         classification_wd, information_gain_balance_coeff, softmax_decay_controller,
                         learning_rate_schedule, decision_loss_coeff, bn_momentum)
        self.mseLoss = tf.keras.losses.MeanSquaredError()
        self.qNetCoeff = q_net_coeff

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

    def calculate_total_q_net_loss(self, model_output, y, add_regularization):
        posteriors_val = {k: v.numpy() for k, v in model_output["posteriors_dict"].items()}
        ig_masks_val = {k: v.numpy() for k, v in model_output["ig_masks_dict"].items()}
        regs, q_s = self.calculate_q_tables_from_network_outputs(true_labels=y.numpy(),
                                                                 posteriors_dict=posteriors_val,
                                                                 ig_masks_dict=ig_masks_val)
        # Q-Net Losses
        q_net_predicted = model_output["q_tables_predicted"]
        q_net_losses = []
        for idx, tpl in enumerate(zip(regs, q_net_predicted)):
            q_truth = tpl[0]
            q_predicted = tpl[1]
            q_truth_tensor = tf.convert_to_tensor(q_truth, dtype=q_predicted.dtype)
            mse = self.mseLoss(q_truth_tensor, q_predicted)
            q_net_losses.append(mse)
        full_q_loss = self.qNetCoeff * tf.add_n(q_net_losses)
        if add_regularization:
            q_regularization_loss = self.get_regularization_loss(is_for_q_nets=True)
            total_q_loss = full_q_loss + q_regularization_loss
        else:
            total_q_loss = full_q_loss
        return total_q_loss

    def track_losses(self, **kwargs):
        super(CignRlRoutingWithIterativeTraining, self).track_losses(**kwargs)
        q_net_losses = kwargs["q_net_losses"]
        if q_net_losses is not None:
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

    def run_main_model(self, X, y, iteration, is_in_warm_up_period):
        with tf.GradientTape() as main_tape:
            model_output = self.run_model(
                X=X,
                y=y,
                iteration=iteration,
                is_training=True,
                warm_up_period=is_in_warm_up_period)
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
        # Calculate grads with respect to the main loss
        main_grads = main_tape.gradient(total_loss, self.model.trainable_variables)

        # for idx, v in enumerate(self.model.trainable_variables):
        #     if "q_net" not in v.name:
        #         continue
        #     grad_arr = main_grads[idx].numpy()
        #     assert np.array_equal(grad_arr, np.zeros_like(grad_arr))

        return model_output, main_grads, total_loss

    def run_q_net_model(self, X, y, iteration, is_in_warm_up_period):
        with tf.GradientTape() as q_tape:
            t4 = time.time()
            model_output_val = self.run_model(
                X=X,
                y=y,
                iteration=iteration,
                is_training=True,
                warm_up_period=is_in_warm_up_period)
            # Calculate target values for the Q-Nets
            posteriors_val = {k: v.numpy() for k, v in model_output_val["posteriors_dict"].items()}
            ig_masks_val = {k: v.numpy() for k, v in model_output_val["ig_masks_dict"].items()}
            regs, q_s = self.calculate_q_tables_from_network_outputs(true_labels=y.numpy(),
                                                                     posteriors_dict=posteriors_val,
                                                                     ig_masks_dict=ig_masks_val)
            # Q-Net Losses
            q_net_predicted = model_output_val["q_tables_predicted"]
            q_net_losses = []
            for idx, tpl in enumerate(zip(regs, q_net_predicted)):
                q_truth = tpl[0]
                q_predicted = tpl[1]
                q_truth_tensor = tf.convert_to_tensor(q_truth, dtype=q_predicted.dtype)
                mse = self.mseLoss(q_truth_tensor, q_predicted)
                q_net_losses.append(mse)
            full_q_loss = self.qNetCoeff * tf.add_n(q_net_losses)
            q_regularization_loss = self.get_regularization_loss(is_for_q_nets=True)
            total_q_loss = full_q_loss + q_regularization_loss
        q_grads = q_tape.gradient(total_q_loss, self.model.trainable_variables)
        return model_output_val, q_grads, q_net_losses

    def assert_gradient_validity(self, main_grads, q_grads):
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

    def iterate_rl_cign(self, train_X, train_y, val_X, val_y, iteration,
                        is_in_warm_up_period, is_fine_tune_epoch):
        time_measurements = {}
        t0 = time.time()
        # ********** Loss calculation and gradients for the main loss functions. **********
        model_output, main_grads, total_loss = \
            self.run_main_model(X=train_X,
                                y=train_y,
                                iteration=iteration,
                                is_in_warm_up_period=is_in_warm_up_period)
        # ********** Loss calculation and gradients for the main loss functions. **********
        model_output_val, q_grads, q_net_losses = \
            self.run_q_net_model(X=val_X,
                                 y=val_y,
                                 iteration=iteration,
                                 is_in_warm_up_period=is_in_warm_up_period)
        if not is_fine_tune_epoch:
            # ********** Loss calculation and gradients for the Q-Nets. **********
            # Check that q_net variables do not receive zero gradients and leaf node variables
            # receive no gradients. Insert zero gradients for every None variable.
            # self.assert_gradient_validity(q_grads=q_grads, main_grads=main_grads)
            # Sum the main CIGN grads and Q-Net grads; apply Gradient Descent step.
            t9 = time.time()
            all_grads = [m_g + q_g for m_g, q_g in zip(main_grads, q_grads)]
            # assert all([np.allclose(main_grads[i].numpy() + q_grads[i].numpy(), all_grads[i])
            #             for i in range(len(all_grads))])
            self.optimizer.apply_gradients(zip(all_grads, self.model.trainable_variables))
        else:
            print("Fine Tuning!")
            self.optimizer.apply_gradients(zip(main_grads, self.model.trainable_variables))
        t1 = time.time()
        # Track losses
        self.track_losses(total_loss=total_loss,
                          classification_losses=model_output["classification_losses"],
                          info_gain_losses=model_output["info_gain_losses"],
                          q_net_losses=q_net_losses)
        self.print_train_step_info(
            iteration=iteration,
            time_intervals=time_measurements,
            eval_dict=model_output["eval_dict"])
        return model_output, model_output_val

    def save_log_data(self, **kwargs):
        run_id = kwargs["run_id"]
        iteration = kwargs["iteration"]
        super().save_log_data(**kwargs)
        kv_rows = []
        for idx, q_net_loss in enumerate(self.qNetTrackers):
            key_ = "Q-Net_{0}".format(idx)
            val_ = np.asscalar(self.qNetTrackers[idx].result().numpy())
            kv_rows.append((run_id, iteration, key_, val_))
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)

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


        # if dataset is None:
        #     return 0.0
        # y_true = []
        # y_pred = []
        #
        # leaf_distributions = {node.index: [] for node in self.leafNodes}
        # for X, y in dataset:
        #     model_output = self.run_model(
        #         X=X,
        #         y=y,
        #         iteration=-1,
        #         is_training=False)
        #     y_pred_batch, leaf_distributions_batch = self.calculate_predictions_of_batch(
        #         model_output=model_output, y=y)
        #     for leaf_node in self.leafNodes:
        #         leaf_distributions[leaf_node.index].extend(leaf_distributions_batch[leaf_node.index])
        #     y_pred.append(y_pred_batch)
        #     y_true.append(y.numpy())
        # y_true = np.concatenate(y_true)
        # y_pred = np.concatenate(y_pred)
        # truth_vector = y_true == y_pred
        # accuracy = np.mean(truth_vector.astype(np.float))
        #
        # # Print sample distribution
        # kv_rows = []
        # for leaf_node in self.leafNodes:
        #     c = Counter(leaf_distributions[leaf_node.index])
        #     str_ = "{0} Node {1} Sample Distribution:{2}".format(dataset_type, leaf_node.index, c)
        #     print(str_)
        #     kv_rows.append((run_id, iteration,
        #                     "{0} Node {1} Sample Distribution".format(dataset_type, leaf_node.index),
        #                     "{0}".format(c)))
        # DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)
        # return accuracy

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

    def train(self, run_id, dataset, epoch_count, **kwargs):
        # q_net_epoch_count = kwargs["q_net_epoch_count"]
        fine_tune_epoch_count = kwargs["fine_tune_epoch_count"]
        is_in_warm_up_period = True
        self.optimizer = self.get_adam_optimizer()
        # OK
        self.build_trackers()

        iteration = 0
        for epoch_id in range(epoch_count + fine_tune_epoch_count):
            is_fine_tune_epoch = epoch_id >= epoch_count
            # OK
            self.reset_trackers()
            times_list = []
            if is_fine_tune_epoch:
                # Else, merge training set and validation set
                full_train_X = np.concatenate([dataset.trainX, dataset.valX], axis=0)
                full_train_y = np.concatenate([dataset.trainY, dataset.valY], axis=0)
            else:
                full_train_X = dataset.trainX
                full_train_y = dataset.trainY

            train_tf = tf.data.Dataset.from_tensor_slices((full_train_X, full_train_y)). \
                shuffle(5000).batch(self.batchSizeNonTensor)
            # Create a validation set; from which can sample batch_size number of samples; for every iteration
            # on the training set. We are going to use these sample for training the Q-Nets.
            assert dataset.valX is not None and dataset.valY is not None
            dataset_size_ratio = int(full_train_X.shape[0] / dataset.valX.shape[0]) + 1
            val_dataset = tf.data.Dataset.from_tensor_slices((dataset.valX, dataset.valY)).repeat(
                count=dataset_size_ratio).shuffle(buffer_size=5000).batch(batch_size=self.batchSizeNonTensor)
            val_dataset_iter = iter(val_dataset)
            # Draw one minibatch from the training set. Calculate gradients with respect to the
            # main loss and information gain loss and regularization loss.
            for train_X, train_y in train_tf:
                # Draw samples from the validation set
                val_X, val_y = val_dataset_iter.__next__()
                if is_fine_tune_epoch:
                    print("Fine Tuning!")
                model_output, model_output_val = self.iterate_rl_cign(train_X=train_X, train_y=train_y, val_X=val_X,
                                                                      val_y=val_y, iteration=iteration,
                                                                      is_in_warm_up_period=is_in_warm_up_period,
                                                                      is_fine_tune_epoch=is_fine_tune_epoch)
                iteration += 1

            self.save_log_data(run_id=run_id,
                               iteration=iteration,
                               eval_dict=model_output["eval_dict"])
            # Evaluation
            self.measure_performance(dataset=dataset,
                                     run_id=run_id,
                                     iteration=iteration,
                                     epoch_id=epoch_id,
                                     times_list=times_list)
            if epoch_id >= self.warmUpPeriod:
                is_in_warm_up_period = False

        self.save_model(run_id=run_id)

    def train_without_val_set(self, run_id, dataset, epoch_count, **kwargs):
        is_in_warm_up_period = True
        self.optimizer = self.get_sgd_optimizer()
        # OK
        self.build_trackers()

        full_train_X = np.concatenate([dataset.trainX, dataset.valX], axis=0)
        full_train_y = np.concatenate([dataset.trainY, dataset.valY], axis=0)
        train_tf = tf.data.Dataset.from_tensor_slices((full_train_X, full_train_y)). \
            shuffle(5000).batch(self.batchSizeNonTensor)
        iteration = 0
        for epoch_id in range(epoch_count):
            # OK
            self.reset_trackers()
            times_list = []
            for train_X, train_y in train_tf:
                # ********** Loss calculation and gradients for the main loss functions. **********
                t0 = time.time()
                with tf.GradientTape() as tape:
                    model_output = self.run_model(
                        X=train_X,
                        y=train_y,
                        iteration=iteration,
                        is_training=True,
                        warm_up_period=is_in_warm_up_period)
                    classification_losses = model_output["classification_losses"]
                    info_gain_losses = model_output["info_gain_losses"]
                    # Weight decaying
                    variables = self.model.trainable_variables
                    regularization_losses = []
                    for var in variables:
                        assert var.ref() in self.regularizationCoefficients
                        lambda_coeff = self.regularizationCoefficients[var.ref()]
                        regularization_losses.append(lambda_coeff * tf.nn.l2_loss(var))
                    total_regularization_loss = tf.add_n(regularization_losses)
                    # Classification losses
                    classification_loss = tf.add_n([loss for loss in classification_losses.values()])
                    # Information Gain losses
                    info_gain_loss = self.decisionLossCoeff * tf.add_n([loss for loss in info_gain_losses.values()])
                    # Calculate target values for the Q-Nets
                    posteriors_val = {k: v.numpy() for k, v in model_output["posteriors_dict"].items()}
                    ig_masks_val = {k: v.numpy() for k, v in model_output["ig_masks_dict"].items()}
                    regs, q_s = self.calculate_q_tables_from_network_outputs(true_labels=train_y.numpy(),
                                                                             posteriors_dict=posteriors_val,
                                                                             ig_masks_dict=ig_masks_val)
                    # Q-Net Losses
                    q_net_predicted = model_output["q_tables_predicted"]
                    q_net_losses = []
                    for idx, tpl in enumerate(zip(regs, q_net_predicted)):
                        q_truth = tpl[0]
                        q_predicted = tpl[1]
                        q_truth_tensor = tf.convert_to_tensor(q_truth, dtype=q_predicted.dtype)
                        mse = self.mseLoss(q_truth_tensor, q_predicted)
                        q_net_losses.append(mse)
                    full_q_loss = self.qNetCoeff * tf.add_n(q_net_losses)
                    # Total loss
                    total_loss = total_regularization_loss + info_gain_loss + classification_loss + full_q_loss
                grads = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            self.save_log_data(run_id=run_id,
                               iteration=iteration,
                               eval_dict=model_output["eval_dict"])
            # Evaluation
            self.measure_performance(dataset=dataset,
                                     run_id=run_id,
                                     iteration=iteration,
                                     epoch_id=epoch_id,
                                     times_list=times_list)
            if epoch_id >= self.warmUpPeriod:
                is_in_warm_up_period = False

        self.save_model(run_id=run_id)