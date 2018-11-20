from auxillary.constants import DatasetTypes
from data_handling.mnist_data_set import MnistDataSet
import os
from auxillary.general_utility_funcs import UtilityFuncs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


class CifarDataSet(MnistDataSet):
    CIFAR_SIZE = 32

    @staticmethod
    def augment_training_image_fn(image, labels):
        # print(image.__class__)
        image = tf.image.resize_image_with_crop_or_pad(image, CifarDataSet.CIFAR_SIZE + 4, CifarDataSet.CIFAR_SIZE + 4)
        image = tf.random_crop(image, [CifarDataSet.CIFAR_SIZE, CifarDataSet.CIFAR_SIZE, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.per_image_standardization(image)
        return image, labels

    def __init__(self,
                 validation_sample_count,
                 save_validation_as=None,
                 load_validation_from=None,
                 is_cifar100=True,
                 augmentation_multiplier=1,
                 test_images_path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + \
                                  os.sep + "data" + os.sep + "cifar100" + os.sep + "test",
                 training_images_path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + \
                                      os.sep + "data" + os.sep + "cifar100" + os.sep + "train",
                 meta_path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + \
                           os.sep + "data" + os.sep + "cifar100" + os.sep + "meta"):
        self.trainDataset = None
        self.validationDataset = None
        self.testDataset = None
        self.trainIter = None
        self.validationIter = None
        self.testIter = None
        self.trainInitOp = None
        self.validationInitOp = None
        self.testInitOp = None
        self.augmentationMultiplier = augmentation_multiplier
        self.batchSize = tf.placeholder(tf.int64)
        super().__init__(validation_sample_count=validation_sample_count, save_validation_as=save_validation_as,
                         load_validation_from=load_validation_from, test_images_path=test_images_path,
                         test_labels_path=None, training_images_path=training_images_path,
                         training_labels_path=None)

    def load_dataset(self):
        training_data = UtilityFuncs.unpickle(file=self.trainImagesPath)
        test_data = UtilityFuncs.unpickle(file=self.testImagesPath)
        self.trainingSamples = training_data[b"data"]
        self.testSamples = test_data[b"data"]
        self.trainingSamples = self.trainingSamples.reshape((self.trainingSamples.shape[0], 3,
                                                             CifarDataSet.CIFAR_SIZE, CifarDataSet.CIFAR_SIZE)) \
            .transpose([0, 2, 3, 1])
        self.testSamples = self.testSamples.reshape((self.testSamples.shape[0], 3,
                                                     CifarDataSet.CIFAR_SIZE, CifarDataSet.CIFAR_SIZE)) \
            .transpose([0, 2, 3, 1])
        # Pack coarse and fine labels into a Nx2 array. Each i.th row corresponds to (coarse,fine) labels.
        training_coarse_labels = np.array(training_data[b"coarse_labels"]).reshape(
            (len(training_data[b"coarse_labels"]), 1))
        training_fine_labels = np.array(training_data[b"fine_labels"]).reshape((len(training_data[b"fine_labels"]), 1))
        test_coarse_labels = np.array(test_data[b"coarse_labels"]).reshape((len(test_data[b"coarse_labels"]), 1))
        test_fine_labels = np.array(test_data[b"fine_labels"]).reshape((len(test_data[b"fine_labels"]), 1))
        self.trainingLabels = np.concatenate([training_coarse_labels, training_fine_labels], axis=1)
        self.testLabels = np.concatenate([test_coarse_labels, test_fine_labels], axis=1)
        if self.validationLoadFile is None:
            indices = np.arange(0, self.validationSampleCount)
            if self.validationSaveFile is not None:
                UtilityFuncs.save_npz(file_name=self.validationSaveFile, arr_dict={"random_indices": indices})
        else:
            indices = UtilityFuncs.load_npz(file_name=self.validationLoadFile)["random_indices"]
        self.validationSamples = self.trainingSamples[indices]
        self.validationLabels = self.trainingLabels[indices]
        self.trainingSamples = np.delete(self.trainingSamples, indices, 0)
        self.trainingLabels = np.delete(self.trainingLabels, indices, 0)
        self.set_current_data_set_type(dataset_type=DatasetTypes.training)
        # Load into Tensorflow Datasets
        # dataset = tf.data.Dataset.from_tensor_slices((self.trainingSamples, self.trainingLabels))
        self.trainDataset = tf.data.Dataset.from_tensor_slices((self.trainingSamples, self.trainingLabels))
        self.validationDataset = tf.data.Dataset.from_tensor_slices((self.validationSamples, self.validationLabels))
        self.testDataset = tf.data.Dataset.from_tensor_slices((self.testSamples, self.testLabels))
        # Create augmented training set
        # self.trainDataset = self.trainDataset.shuffle(buffer_size=self.trainingSamples.shape[0])
        self.trainDataset = self.trainDataset.map(CifarDataSet.augment_training_image_fn)
        self.trainDataset = self.trainDataset.repeat(self.augmentationMultiplier)
        self.trainDataset = self.trainDataset.batch(batch_size=self.batchSize)
        self.trainDataset = self.trainDataset.prefetch(buffer_size=self.batchSize)
        self.trainIter = tf.data.Iterator.from_structure(self.trainDataset.output_types, self.trainDataset.output_shapes)

        features, labels = self.trainIter.get_next()
        self.trainInitOp = self.trainIter.make_initializer(self.trainDataset)
        sess = tf.Session()
        # sess.run(train_init_op, feed_dict={self.batchSize: 250})

        # f, l = sess.run([features, labels])
        # self.visualize_cifar_sample(image=f[5], labels=l[5])

        epoch = 0
        while epoch < 10:
            counter = 0
            sess.run(self.trainInitOp, feed_dict={self.batchSize: 250})
            while True:
                try:
                    f, l = sess.run([features, labels])
                    if counter == 0:
                        self.visualize_cifar_sample(image=f[5], labels=l[5])
                    counter += f.shape[0]
                except tf.errors.OutOfRangeError:
                    break
            print(counter)
            epoch += 1

        print("X")

    def visualize_cifar_sample(self, image, labels):
        # sample_reshaped0 = self.currentSamples[sample_index]
        # sample_reshaped1 = self.currentSamples[sample_index].transpose([1, 2, 0])
        # sample_reshaped2 = self.currentSamples[sample_index].reshape(3, 32, 32).transpose([1, 2, 0])
        plt.title('Label is {label}'.format(label=labels))
        plt.imshow(image)
        plt.show()
