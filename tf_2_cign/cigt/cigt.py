import tensorflow as tf
import numpy as np
import time

from auxillary.dag_utilities import Dag
from simple_tf.uncategorized.node import Node
from tf_2_cign.cigt.routing_strategy.approximate_training_strategy import ApproximateTrainingStrategy
from tqdm import tqdm


class Cigt(tf.keras.Model):
    def __init__(self,
                 input_dims, class_count, path_counts, softmax_decay_controller, learning_rate_schedule,
                 decision_loss_coeff, routing_strategy_name, warm_up_period,
                 decision_drop_probability, classification_drop_probability,
                 decision_wd, classification_wd,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathCounts = [1]
        self.pathCounts.extend(path_counts)
        self.classCount = class_count
        self.inputDims = input_dims
        self.learningRateSchedule = learning_rate_schedule
        self.decisionLossCoefficient = decision_loss_coeff
        self.softmaxDecayController = softmax_decay_controller
        self.dagObject = Dag()
        self.cigtBlocks = []
        self.rootNode = None
        self.decisionDropProbability = decision_drop_probability
        self.classificationDropProbability = classification_drop_probability
        self.decisionWd = decision_wd
        self.classificationWd = classification_wd
        self.warmUpPeriod = warm_up_period
        self.numOfTrainingIterations = 0
        self.numOfTrainingEpochs = 0
        self.regularizationCoefficients = {}
        if routing_strategy_name == "Approximate_Training":
            self.routingStrategy = ApproximateTrainingStrategy(warm_up_epoch_count=self.warmUpPeriod)
        else:
            raise NotImplementedError()
        self.metricsDict = {}
        self.model = None
        # self.inputs = tf.keras.Input(shape=input_dims, name="inputs")
        # self.labels = tf.keras.Input(shape=(), name="labels", dtype=tf.int32)

    def get_sgd_optimizer(self):
        boundaries = [tpl[0] for tpl in self.learningRateSchedule.schedule]
        values = [self.learningRateSchedule.initialValue]
        values.extend([tpl[1] for tpl in self.learningRateSchedule.schedule])
        learning_rate_scheduler_tf = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries, values=values)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_scheduler_tf, momentum=0.9)
        return optimizer

    def build_network(self):
        # is_leaf = 0 == self.networkDepth
        last_block = None
        self.metricsDict = {
            "total_loss_metric": tf.keras.metrics.Mean(name="total_loss_metric"),
            "classification_loss_metric": tf.keras.metrics.Mean(name="classification_loss_metric"),
            "regularization_loss_metric": tf.keras.metrics.Mean(name="regularization_loss_metric"),
            "total_information_gain_loss": tf.keras.metrics.Mean(name="total_information_gain_loss"),
            "accuracy_metric": tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy_metric")
        }
        for block_id, path_count in enumerate(self.pathCounts):
            is_root = block_id == 0
            is_leaf = block_id == len(self.pathCounts) - 1
            node = Node(index=block_id, depth=block_id, is_root=is_root, is_leaf=is_leaf)
            if not is_leaf:
                self.metricsDict["info_gain_loss_{0}".format(block_id)] = \
                    tf.keras.metrics.Mean(name="info_gain_loss_{0}".format(block_id))
            self.dagObject.add_node(node=node)
            if not is_root:
                self.dagObject.add_edge(parent=last_block, child=node)
            else:
                self.rootNode = node
            last_block = node

    def call(self, inputs, **kwargs):
        x = inputs[0]
        y = inputs[1]
        batch_size = tf.shape(x)[0]
        if "training" in kwargs:
            is_training = kwargs["training"]
        else:
            is_training = False

        if is_training:
            temperature = self.softmaxDecayController.get_value()
        else:
            temperature = 1.0

        temperature = tf.convert_to_tensor(temperature)
        f_net = x
        # ig_activations = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
        routing_matrix = tf.ones(shape=(batch_size, 1), dtype=tf.int32)
        information_gain_values = []
        logits = None
        posteriors = None
        classification_loss = None
        for block_id, block in enumerate(self.cigtBlocks):
            if block_id < len(self.cigtBlocks) - 1:
                # Run the CIGT block
                f_net, ig_value, ig_activations, routing_probabilities = \
                    block([f_net, routing_matrix, temperature, y], training=is_training)
                # Keep track of the results.
                information_gain_values.append(ig_value)
                # Build the routing matrix for the next block
                routing_matrix = self.routingStrategy(routing_probabilities)
                # Last block
            else:
                logits, posteriors, classification_loss = \
                    block([f_net, routing_matrix, y], training=is_training)

        # Get regularization losses
        regularization_loss = self.get_regularization_loss()
        # Get information gain loss
        total_information_gain_loss = self.routingStrategy.calculate_information_gain_losses(
            ig_losses=information_gain_values, decision_loss_coefficient=self.decisionLossCoefficient)
        # Classification loss is already calculated. Calculate the total loss.
        total_loss = classification_loss + total_information_gain_loss + regularization_loss

        results_dict = {
            "total_loss": total_loss,
            "classification_loss": classification_loss,
            "total_information_gain_loss": total_information_gain_loss,
            "information_gain_loss_list": information_gain_values,
            "regularization_loss": regularization_loss,
            "logits": logits,
            "posteriors": posteriors
        }
        return results_dict

    def update_metrics(self, results_dict, labels):
        self.metricsDict["total_loss_metric"].update_state(values=results_dict["total_loss"])
        self.metricsDict["classification_loss_metric"].update_state(values=results_dict["classification_loss"])
        self.metricsDict["total_information_gain_loss"].update_state(values=results_dict["total_information_gain_loss"])
        self.metricsDict["regularization_loss_metric"].update_state(values=results_dict["regularization_loss"])
        for block_id, path_count in enumerate(self.pathCounts):
            is_leaf = block_id == len(self.pathCounts) - 1
            if is_leaf:
                continue
            self.metricsDict["info_gain_loss_{0}".format(block_id)].update_state(
                values=results_dict["information_gain_loss_list"][block_id])
        self.metricsDict["accuracy_metric"].update_state(y_true=labels, y_pred=results_dict["logits"])

    def report_metrics(self):
        print("Total Loss:{0}".format(self.metricsDict["total_loss_metric"].result().numpy()))
        print("Classification Loss:{0}".format(self.metricsDict["classification_loss_metric"].result().numpy()))
        print("Total Information Gain Loss:{0}".format(self.metricsDict[
                                                           "total_information_gain_loss"].result().numpy()))
        print("Regularization Loss:{0}".format(self.metricsDict["regularization_loss_metric"].result().numpy()))
        for block_id, path_count in enumerate(self.pathCounts):
            is_leaf = block_id == len(self.pathCounts) - 1
            if is_leaf:
                continue
            print("Block {0} Info Gain Loss:{1}".format(
                block_id,
                self.metricsDict["info_gain_loss_{0}".format(block_id)].result().numpy()))
        print("Accuracy:{0}".format(self.metricsDict["accuracy_metric"].result().numpy()))

    def is_decision_variable(self, variable):
        if "scale" in variable.name or "shift" in variable.name or "hyperplane" in variable.name or \
                "gamma" in variable.name or "beta" in variable.name or "decision" in variable.name:
            return True
        else:
            return False

    def calculate_regularization_coefficients(self):
        print("Num of trainable variables:{0}".format(len(self.trainable_variables)))
        l2_loss_list = []
        decayed_variables = []
        non_decayed_variables = []
        decision_variables = []
        for v in self.trainable_variables:
            is_decision_pipeline_variable = self.is_decision_variable(variable=v)
            # assert (not is_decision_pipeline_variable)
            loss_tensor = tf.nn.l2_loss(v)
            # self.evalDict["l2_loss_{0}".format(v.name)] = loss_tensor
            print(v.name)
            if "bias" in v.name or "shift" in v.name or "scale" in v.name or "gamma" in v.name or "beta" in v.name:
                # non_decayed_variables.append(v)
                # l2_loss_list.append(0.0 * loss_tensor)
                self.regularizationCoefficients[v.ref()] = 0.0
            else:
                # decayed_variables.append(v)
                if is_decision_pipeline_variable:
                    # decision_variables.append(v)
                    # l2_loss_list.append(self.decisionWd * loss_tensor)
                    self.regularizationCoefficients[v.ref()] = self.decisionWd
                else:
                    # l2_loss_list.append(self.classificationWd * loss_tensor)
                    self.regularizationCoefficients[v.ref()] = self.classificationWd
        # self.regularizationLoss = tf.add_n(l2_loss_list)

    def get_regularization_loss(self):
        if len(self.trainable_variables) == 0:
            return 0.0
        if len(self.regularizationCoefficients) == 0:
            self.calculate_regularization_coefficients()
        variables = self.trainable_variables
        regularization_losses = []
        for var in variables:
            if var.ref() in self.regularizationCoefficients:
                lambda_coeff = self.regularizationCoefficients[var.ref()]
                regularization_losses.append(lambda_coeff * tf.nn.l2_loss(var))
        total_regularization_loss = tf.add_n(regularization_losses)
        return total_regularization_loss

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose='auto',
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):

        self.numOfTrainingIterations = 0
        train_dataset = x
        val_dataset = validation_data
        for epoch_id in range(epochs):
            print("Start of epoch:{0}".format(epoch_id))

            # Reset all metrics
            for metric in self.metricsDict.values():
                metric.reset_states()

            for train_x, train_y in train_dataset:
                t0 = time.time()
                with tf.GradientTape() as tape:
                    results_dict = self.call(inputs=[train_x, train_y], training=True)
                grads = tape.gradient(results_dict["total_loss"], self.trainable_variables)
                t1 = time.time()
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                print("t1-t0:{0}".format(t1-t0))
                # Update metrics
                self.update_metrics(results_dict=results_dict, labels=train_y)
                print("********** Epoch:{0} Iteration:{1} **********".format(epoch_id, self.numOfTrainingIterations))
                self.report_metrics()
                self.numOfTrainingIterations += 1
            # Train statistics
            print("Epoch {0} Train Statistics".format(epoch_id))
            self.evaluate(x=train_dataset)
            # Validation / Test statistics
            print("Epoch {0} Test Statistics".format(epoch_id))
            self.evaluate(x=val_dataset)

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None,
                 callbacks=None,
                 max_queue_size=10,
                 workers=1,
                 use_multiprocessing=False,
                 return_dict=False,
                 **kwargs):
        dataset = x
        # Reset all metrics
        for metric in self.metricsDict.values():
            metric.reset_states()
        for x_, y_ in tqdm(dataset):
            results_dict = self.call(inputs=[x_, y_], training=False)
            # Update metrics
            self.update_metrics(results_dict=results_dict, labels=y_)
        self.report_metrics()
