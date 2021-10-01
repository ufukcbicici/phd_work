import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


class FashionMnist(object):
    def __init__(self, batch_size, validation_size=2000, validation_source="test"):
        np.random.seed(67)
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

        # with pickle("test_data.sav")
        # Utilities.pickle_save_to_file(path="test_data.sav", file_content=[self.testX, self.testY])
        # Utilities.pickle_save_to_file(path="validation_data.sav", file_content=[self.valX, self.valY])

        # testX_loaded, testY_loaded = Utilities.pickle_load_from_file(path="test_data.sav")
        # valX_loaded, valY_loaded = Utilities.pickle_load_from_file(path="validation_data.sav")
        # assert np.array_equal(self.testX, testX_loaded)
        # assert np.array_equal(self.testY, testY_loaded)
        # assert np.array_equal(self.valX, valX_loaded)
        # assert np.array_equal(self.valY, valY_loaded)


