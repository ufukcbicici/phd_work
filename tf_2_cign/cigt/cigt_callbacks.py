import tensorflow as tf
import numpy as np


class CigtCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, batch, logs=None):
        self.model.numOfTrainingIterations += 1
        self.model.routingStrategyLayer.set_training_statistics(iteration_count=self.model.numOfTrainingIterations,
                                                                epoch_count=self.model.numOfTrainingEpochs)

    def on_epoch_end(self, epoch, logs=None):
        self.model.numOfTrainingEpochs += 1
        self.model.routingStrategyLayer.set_training_statistics(iteration_count=self.model.numOfTrainingIterations,
                                                                epoch_count=self.model.numOfTrainingEpochs)