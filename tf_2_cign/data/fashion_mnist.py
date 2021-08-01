import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


class FashionMnist:
    def __init__(self, batch_size, validation_size=2000, validation_source="test"):
        self.trainData, self.testData = tf.keras.datasets.fashion_mnist.load_data()
        self.trainX, self.trainY = self.trainData[0], self.trainData[1]
        self.testX, self.testY = self.testData[0], self.testData[1]
        self.trainX = np.expand_dims(self.trainX, axis=-1)
        self.testX = np.expand_dims(self.testX, axis=-1)
        self.valX, self.valY = None, None
        self.batchSize = batch_size
        if validation_size > 0:
            if validation_source == "test":
                self.testX, self.valX, self.testY, self.valY = train_test_split(self.testX, self.testY,
                                                                                test_size=validation_size)
            else:
                self.trainX, self.valX, self.trainY, self.valY = train_test_split(self.trainX, self.trainY,
                                                                                  test_size=validation_size)
        self.trainX = self.trainX / 255.0
        self.testX = self.testX / 255.0
        if self.valX is not None:
            self.valX = self.valX / 255.0

        # tf.data objects
        self.testDataTf = tf.data.Dataset.from_tensor_slices((self.testX, self.testY)).batch(self.batchSize)
        if self.valX is not None:
            self.validationDataTf = tf.data.Dataset.from_tensor_slices((self.valX, self.valY)).batch(self.batchSize)
        else:
            self.validationDataTf = None
        self.trainDataTf = tf.data.Dataset.from_tensor_slices((self.trainX, self.trainY)).\
            shuffle(5000).batch(self.batchSize)
        # dataset_validation = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(config["BATCH_SIZE"])
        # dataset_test = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(config["BATCH_SIZE"])
