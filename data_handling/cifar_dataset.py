import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from auxillary.constants import DatasetTypes
from auxillary.general_utility_funcs import UtilityFuncs
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from data_handling.data_set import DataSet


class CifarDataSet(DataSet):
    CIFAR_SIZE = 32
    CIFAR100_COARSE_LABEL_COUNT = 20
    CIFAR100_FINE_LABEL_COUNT = 100

    @staticmethod
    def augment_training_image_fn(image, labels, indices, one_hot_labels, coarse_labels, coarse_one_hot_labels):
        # print(image.__class__)
        image = tf.image.resize_image_with_crop_or_pad(image, CifarDataSet.CIFAR_SIZE + 4, CifarDataSet.CIFAR_SIZE + 4)
        image = tf.random_crop(image, [CifarDataSet.CIFAR_SIZE, CifarDataSet.CIFAR_SIZE, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.per_image_standardization(image)
        return image, labels, indices, one_hot_labels, coarse_labels, coarse_one_hot_labels

    @staticmethod
    def augment_test_image_fn(image, labels, indices, one_hot_labels, coarse_labels, coarse_one_hot_labels):
        image = tf.image.per_image_standardization(image)
        return image, labels, indices, one_hot_labels, coarse_labels, coarse_one_hot_labels

    def __init__(self, session, validation_sample_count, save_validation_as=None,
                 load_validation_from=None, is_cifar100=True, augmentation_multiplier=1,
                 test_images_path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + \
                                  os.sep + "data" + os.sep + "cifar100" + os.sep + "test",
                 training_images_path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + \
                                      os.sep + "data" + os.sep + "cifar100" + os.sep + "train"):
        super().__init__()
        # Initial Data
        self.sess = session
        self.trainImagesPath = training_images_path
        self.testImagesPath = test_images_path
        self.validationLoadFile = load_validation_from
        self.validationSaveFile = save_validation_as
        self.validationSampleCount = validation_sample_count
        self.isCifar100 = is_cifar100
        # Dataset Objects
        self.trainDataset = None
        self.validationDataset = None
        self.testDataset = None
        # Iterators
        self.trainIter = None
        self.validationIter = None
        self.testIter = None
        # Init Objects
        self.trainInitOp = None
        self.validationInitOp = None
        self.testInitOp = None
        # Helper objects
        self.outputsDict = {}
        self.augmentationMultiplier = augmentation_multiplier
        self.batchSize = tf.placeholder(tf.int64)
        self.currInitOp = None
        self.currOutputs = None
        self.load_dataset()

    def set_curr_session(self, sess):
        self.sess = sess

    def load_dataset(self):
        training_data = UtilityFuncs.unpickle(file=self.trainImagesPath)
        test_data = UtilityFuncs.unpickle(file=self.testImagesPath)
        training_samples = training_data[b"data"]
        test_samples = test_data[b"data"]
        training_samples = training_samples.reshape((training_samples.shape[0], 3,
                                                     CifarDataSet.CIFAR_SIZE, CifarDataSet.CIFAR_SIZE)) \
            .transpose([0, 2, 3, 1])
        test_samples = test_samples.reshape((test_samples.shape[0], 3,
                                             CifarDataSet.CIFAR_SIZE, CifarDataSet.CIFAR_SIZE)) \
            .transpose([0, 2, 3, 1])
        training_indices = np.arange(training_samples.shape[0])
        test_indices = np.arange(test_samples.shape[0])
        # Unpack fine and coarse labels
        training_coarse_labels = np.array(training_data[b"coarse_labels"]).reshape(
            (len(training_data[b"coarse_labels"]), 1))
        training_fine_labels = np.array(training_data[b"fine_labels"]).reshape((len(training_data[b"fine_labels"]), 1))
        test_coarse_labels = np.array(test_data[b"coarse_labels"]).reshape((len(test_data[b"coarse_labels"]), 1))
        test_fine_labels = np.array(test_data[b"fine_labels"]).reshape((len(test_data[b"fine_labels"]), 1))
        # Pack
        training_labels = training_fine_labels.reshape((training_fine_labels.shape[0],))
        training_coarse_labels = training_coarse_labels.reshape((training_coarse_labels.shape[0],))
        test_labels = test_fine_labels.reshape((test_fine_labels.shape[0],))
        test_coarse_labels = test_coarse_labels.reshape((test_coarse_labels.shape[0],))
        # Convert labels to one hot encoding.
        training_one_hot_labels = UtilityFuncs.convert_labels_to_one_hot(labels=training_labels,
                                                                         max_label=CifarDataSet.CIFAR100_FINE_LABEL_COUNT)
        training_coarse_one_hot_labels = UtilityFuncs.convert_labels_to_one_hot(labels=training_coarse_labels,
                                                                                max_label=CifarDataSet.CIFAR100_COARSE_LABEL_COUNT)
        test_one_hot_labels = UtilityFuncs.convert_labels_to_one_hot(labels=test_labels,
                                                                     max_label=CifarDataSet.CIFAR100_FINE_LABEL_COUNT)
        test_coarse_one_hot_labels = UtilityFuncs.convert_labels_to_one_hot(labels=test_coarse_labels,
                                                                            max_label=CifarDataSet.CIFAR100_COARSE_LABEL_COUNT)
        # Validation Set
        if self.validationLoadFile is None:
            indices = np.arange(0, self.validationSampleCount)
            if self.validationSaveFile is not None:
                UtilityFuncs.save_npz(file_name=self.validationSaveFile, arr_dict={"random_indices": indices})
        else:
            indices = UtilityFuncs.load_npz(file_name=self.validationLoadFile)["random_indices"]
        validation_samples = training_samples[indices]
        validation_labels = training_labels[indices]
        validation_indices = training_indices[indices]
        validation_one_hot_labels = training_one_hot_labels[indices]
        validation_coarse_labels = training_coarse_labels[indices]
        validation_coarse_one_hot_labels = training_coarse_one_hot_labels[indices]
        training_samples = np.delete(training_samples, indices, 0)
        training_labels = np.delete(training_labels, indices, 0)
        training_indices = np.delete(training_indices, indices, 0)
        training_one_hot_labels = np.delete(training_one_hot_labels, indices, 0)
        training_coarse_labels = np.delete(training_coarse_labels, indices, 0)
        training_coarse_one_hot_labels = np.delete(training_coarse_one_hot_labels, indices, 0)
        # Load into Tensorflow Datasets
        # dataset = tf.data.Dataset.from_tensor_slices((training_samples, training_labels))
        self.trainDataset = tf.data.Dataset.from_tensor_slices((training_samples,
                                                                training_labels,
                                                                training_indices,
                                                                training_one_hot_labels,
                                                                training_coarse_labels,
                                                                training_coarse_one_hot_labels))
        self.validationDataset = tf.data.Dataset.from_tensor_slices((validation_samples,
                                                                     validation_labels,
                                                                     validation_indices,
                                                                     validation_one_hot_labels,
                                                                     validation_coarse_labels,
                                                                     validation_coarse_one_hot_labels))
        self.testDataset = tf.data.Dataset.from_tensor_slices((test_samples,
                                                               test_labels,
                                                               test_indices,
                                                               test_one_hot_labels,
                                                               test_coarse_labels,
                                                               test_coarse_one_hot_labels))
        # Create augmented training set
        self.trainDataset = self.trainDataset.shuffle(
            buffer_size=training_samples.shape[0])  # training_samples.shape[0]/10)
        self.trainDataset = self.trainDataset.map(CifarDataSet.augment_training_image_fn)
        self.trainDataset = self.trainDataset.repeat(self.augmentationMultiplier)
        self.trainDataset = self.trainDataset.batch(batch_size=self.batchSize)
        self.trainDataset = self.trainDataset.prefetch(buffer_size=self.batchSize)
        self.trainIter = tf.data.Iterator.from_structure(self.trainDataset.output_types,
                                                         self.trainDataset.output_shapes)
        features, labels, indices, one_hot_labels, coarse_labels, coarse_one_hot_labels = self.trainIter.get_next()
        self.outputsDict[DatasetTypes.training] = [features, labels, indices, one_hot_labels, coarse_labels,
                                                   coarse_one_hot_labels]
        self.trainInitOp = self.trainIter.make_initializer(self.trainDataset)

        # Create validation set
        self.validationDataset = self.validationDataset.map(CifarDataSet.augment_test_image_fn)
        self.validationDataset = self.validationDataset.batch(batch_size=self.batchSize)
        self.validationDataset = self.validationDataset.prefetch(buffer_size=self.batchSize)
        self.validationIter = tf.data.Iterator.from_structure(self.validationDataset.output_types,
                                                              self.validationDataset.output_shapes)
        features, labels, indices, one_hot_labels, coarse_labels, coarse_one_hot_labels = self.validationIter.get_next()
        self.outputsDict[DatasetTypes.validation] = [features, labels, indices, one_hot_labels, coarse_labels,
                                                     coarse_one_hot_labels]
        self.validationInitOp = self.validationIter.make_initializer(self.validationDataset)

        # Create test set
        self.testDataset = self.testDataset.map(CifarDataSet.augment_test_image_fn)
        self.testDataset = self.testDataset.batch(batch_size=self.batchSize)
        self.testDataset = self.testDataset.prefetch(buffer_size=self.batchSize)
        self.testIter = tf.data.Iterator.from_structure(self.testDataset.output_types,
                                                        self.testDataset.output_shapes)
        features, labels, indices, one_hot_labels, coarse_labels, coarse_one_hot_labels = self.testIter.get_next()
        self.outputsDict[DatasetTypes.test] = [features, labels, indices,
                                               one_hot_labels, coarse_labels, coarse_one_hot_labels]
        self.testInitOp = self.testIter.make_initializer(self.testDataset)

        # Experimental
        # # sess = tf.Session()
        # self.sess.run(self.trainInitOp, feed_dict={self.batchSize: 250})
        # labels = tf.placeholder(tf.int64, name="LabelsArr")
        # mask = tf.placeholder(tf.int64, name="MaskArr", shape=(250, ))
        # masked_labels = tf.boolean_mask(labels, mask)
        #
        # nodes_prev = [n.name for n in tf.get_default_graph().as_graph_def().node]
        # print(len(nodes_prev))
        # nodes_prev = None
        # epoch = 0
        # while epoch < 10:
        #     counter = 0
        #     self.sess.run(self.trainInitOp, feed_dict={self.batchSize: 250})
        #     index_set = set()
        #     while True:
        #         try:
        #             x, fine_y, ix, fine_y_one_hot, coarse_y, coarse_y_one_hot = self.sess.run(self.outputsDict[DatasetTypes.training])
        #             index_set = index_set.union(set(ix.tolist()))
        #             np_mask = np.random.binomial(1, 0.5, 250)
        #             masked_results = self.sess.run([masked_labels], {labels: fine_y, mask: np_mask})
        #
        #             # fine_labels = labels[:, 1]
        #             # coarse_labels = labels[:, 0]
        #
        #             # if counter == 0:
        #             #     self.visualize_cifar_sample(image=f[5], labels=l[5])
        #             counter += fine_y.shape[0]
        #             print(counter)
        #             nodes_next = [n.name for n in tf.get_default_graph().as_graph_def().node]
        #             print(len(nodes_next))
        #             nodes_next = None
        #         except tf.errors.OutOfRangeError:
        #             break
        #     print(counter)
        #     epoch += 1
        #
        # #
        # print("X")

    def visualize_cifar_sample(self, image, labels):
        # sample_reshaped0 = self.currentSamples[sample_index]
        # sample_reshaped1 = self.currentSamples[sample_index].transpose([1, 2, 0])
        # sample_reshaped2 = self.currentSamples[sample_index].reshape(3, 32, 32).transpose([1, 2, 0])
        plt.title('Label is {label}'.format(label=labels))
        plt.imshow(image)
        plt.show()

    def reset(self):
        pass

    def set_current_data_set_type(self, dataset_type, batch_size=None):
        assert batch_size is not None
        self.currentDataSetType = dataset_type
        if self.currentDataSetType == DatasetTypes.training:
            self.currInitOp = self.trainInitOp
            self.currOutputs = self.outputsDict[DatasetTypes.training]
        elif self.currentDataSetType == DatasetTypes.test:
            self.currInitOp = self.testInitOp
            self.currOutputs = self.outputsDict[DatasetTypes.test]
        elif self.currentDataSetType == DatasetTypes.validation:
            self.currInitOp = self.validationInitOp
            self.currOutputs = self.outputsDict[DatasetTypes.validation]
        else:
            raise Exception("Unknown dataset type")
        self.sess.run(self.currInitOp, feed_dict={self.batchSize: batch_size})

    def get_label_count(self):
        if self.isCifar100:
            return 100

    def get_next_batch(self, batch_size=None):
        try:
            samples, labels, indices, one_hot_labels, coarse_labels, coarse_one_hot_labels = \
                self.sess.run(self.outputsDict[self.currentDataSetType])
            self.isNewEpoch = False
            return DataSet.MiniBatch(samples, labels, indices, one_hot_labels, None, coarse_labels,
                                     coarse_one_hot_labels)
        except tf.errors.OutOfRangeError:
            self.isNewEpoch = True
            return None

    def get_image_size(self):
        return CifarDataSet.CIFAR_SIZE

    def get_num_of_channels(self):
        return 3
