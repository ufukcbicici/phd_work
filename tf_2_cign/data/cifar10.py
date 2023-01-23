import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


class Cifar10(object):
    CIFAR_SIZE = 32
    TF_RNG = None
    # tf.random.Generator.from_seed(123, alg='philox')

    @staticmethod
    def augment_training_image_fn_with_seed(image, label):
        seed = Cifar10.TF_RNG.make_seeds(2)[0]
        image_normalized = Cifar10.augment_training_image_fn(image=image, seed=seed)
        return image_normalized, label

    @staticmethod
    def augment_training_image_fn(image, seed):
        # assert len(image.shape) == 3
        image = tf.image.resize_with_crop_or_pad(image, Cifar10.CIFAR_SIZE + 8, Cifar10.CIFAR_SIZE + 8)
        image = tf.image.stateless_random_crop(image, [Cifar10.CIFAR_SIZE, Cifar10.CIFAR_SIZE, 3], seed)
        image = tf.image.stateless_random_flip_left_right(image, seed)
        image = (tf.cast(image, dtype=tf.float32) / 255.0)
        mean = tf.convert_to_tensor((0.4914, 0.4822, 0.4465))
        std = tf.convert_to_tensor((0.2023, 0.1994, 0.2010))
        mean = tf.expand_dims(tf.expand_dims(mean, axis=0), axis=0)
        std = tf.expand_dims(tf.expand_dims(std, axis=0), axis=0)
        image_normalized = (image - mean) / std
        return image_normalized

    @staticmethod
    def augment_test_image_fn(image, label):
        image = (tf.cast(image, dtype=tf.float32) / 255.0)
        mean = tf.convert_to_tensor((0.4914, 0.4822, 0.4465))
        std = tf.convert_to_tensor((0.2023, 0.1994, 0.2010))
        mean = tf.expand_dims(tf.expand_dims(mean, axis=0), axis=0)
        std = tf.expand_dims(tf.expand_dims(std, axis=0), axis=0)
        image_normalized = (image - mean) / std
        return image_normalized, label

    def __init__(self, batch_size, validation_size=0, validation_source="test"):
        np.random.seed(67)
        self.trainData, self.testData = tf.keras.datasets.cifar10.load_data()
        self.trainX, self.trainY = self.trainData[0], self.trainData[1]
        self.testX, self.testY = self.testData[0], self.testData[1]
        # self.trainX = np.expand_dims(self.trainX, axis=-1)
        # self.testX = np.expand_dims(self.testX, axis=-1)
        self.valX, self.valY = None, None
        self.batchSize = batch_size
        if validation_size > 0:
            if validation_source == "test":
                self.testX, self.valX, self.testY, self.valY = train_test_split(self.testX, self.testY,
                                                                                test_size=validation_size)
            else:
                self.trainX, self.valX, self.trainY, self.valY = train_test_split(self.trainX, self.trainY,
                                                                                  test_size=validation_size)
        # Create training set
        self.trainDataset = tf.data.Dataset.from_tensor_slices((self.trainX, self.trainY))
        self.trainDataset = self.trainDataset.shuffle(self.trainX.shape[0])
        self.trainDataset = self.trainDataset.map(Cifar10.augment_training_image_fn_with_seed,
                                                  num_parallel_calls=tf.data.AUTOTUNE)
        self.trainDataset = self.trainDataset.batch(batch_size=self.batchSize)
        self.trainDataset = self.trainDataset.prefetch(buffer_size=self.batchSize)

        # Create test set
        self.testDataset = tf.data.Dataset.from_tensor_slices((self.testX, self.testY))
        self.testDataset = self.testDataset.map(Cifar10.augment_test_image_fn,
                                                num_parallel_calls=tf.data.AUTOTUNE)
        self.testDataset = self.testDataset.batch(batch_size=self.batchSize)
        self.testDataset = self.testDataset.prefetch(buffer_size=self.batchSize)

        # Create validation set
        if self.valX is not None and self.valY is not None:
            self.validationDataset = tf.data.Dataset.from_tensor_slices((self.testX, self.testY))
            self.validationDataset = self.validationDataset.map(Cifar10.augment_test_image_fn,
                                                                num_parallel_calls=tf.data.AUTOTUNE)
            self.validationDataset = self.validationDataset.batch(batch_size=self.batchSize)
            self.validationDataset = self.validationDataset.prefetch(buffer_size=self.batchSize)
        else:
            self.validationDataset = None

        # self.trainIter = tf.data.Iterator.from_structure(self.trainDataset.output_types,
        #                                                  self.trainDataset.output_shapes)
