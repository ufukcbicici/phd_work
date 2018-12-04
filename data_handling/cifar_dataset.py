from auxillary.constants import DatasetTypes
from data_handling.mnist_data_set import MnistDataSet
import os
from auxillary.general_utility_funcs import UtilityFuncs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
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

    def __init__(self,
                 session,
                 batch_sizes,
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
        self.sess = session
        self.isCifar100 = is_cifar100
        self.trainDataset = None
        self.validationDataset = None
        self.testDataset = None
        self.trainingOneHotLabels = None
        self.testOneHotLabels = None
        self.validationOneHotLabels = None
        self.trainIter = None
        self.validationIter = None
        self.testIter = None
        self.trainInitOp = None
        self.validationInitOp = None
        self.testInitOp = None
        self.outputsDict = {}
        self.augmentationMultiplier = augmentation_multiplier
        self.batchSize = tf.placeholder(tf.int64)
        self.currInitOp = None
        self.currOutputs = None
        self.trainingIndices = None
        self.testIndices = None
        self.validationIndices = None
        # Coarse Labels
        self.trainingCoarseLabels = None
        self.trainingCoarseOneHotLabels = None
        self.testCoarseLabels = None
        self.testCoarseOneHotLabels = None
        self.validationCoarseLabels = None
        self.validationCoarseOneHotLabels = None
        super().__init__(batch_sizes=batch_sizes, validation_sample_count=validation_sample_count,
                         save_validation_as=save_validation_as,
                         load_validation_from=load_validation_from, test_images_path=test_images_path,
                         test_labels_path=None, training_images_path=training_images_path,
                         training_labels_path=None)

    def set_curr_session(self, sess):
        self.sess = sess

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
        self.trainingIndices = np.arange(self.trainingSamples.shape[0])
        self.testIndices = np.arange(self.testSamples.shape[0])
        # Unpack fine and coarse labels
        training_coarse_labels = np.array(training_data[b"coarse_labels"]).reshape(
            (len(training_data[b"coarse_labels"]), 1))
        training_fine_labels = np.array(training_data[b"fine_labels"]).reshape((len(training_data[b"fine_labels"]), 1))
        test_coarse_labels = np.array(test_data[b"coarse_labels"]).reshape((len(test_data[b"coarse_labels"]), 1))
        test_fine_labels = np.array(test_data[b"fine_labels"]).reshape((len(test_data[b"fine_labels"]), 1))
        # Pack
        self.trainingLabels = training_fine_labels.reshape((training_fine_labels.shape[0],))
        self.trainingCoarseLabels = training_coarse_labels.reshape((training_coarse_labels.shape[0],))
        self.testLabels = test_fine_labels.reshape((test_fine_labels.shape[0],))
        self.testCoarseLabels = test_coarse_labels.reshape((test_coarse_labels.shape[0],))
        # Convert labels to one hot encoding.
        self.trainingOneHotLabels = UtilityFuncs.convert_labels_to_one_hot(labels=self.trainingLabels,
                                                                           max_label=CifarDataSet.CIFAR100_FINE_LABEL_COUNT)
        self.trainingCoarseOneHotLabels = UtilityFuncs.convert_labels_to_one_hot(labels=self.trainingCoarseLabels,
                                                                                 max_label=CifarDataSet.CIFAR100_COARSE_LABEL_COUNT)
        self.testOneHotLabels = UtilityFuncs.convert_labels_to_one_hot(labels=self.testLabels,
                                                                       max_label=CifarDataSet.CIFAR100_FINE_LABEL_COUNT)
        self.testCoarseOneHotLabels = UtilityFuncs.convert_labels_to_one_hot(labels=self.testCoarseLabels,
                                                                             max_label=CifarDataSet.CIFAR100_COARSE_LABEL_COUNT)
        # Validation Set
        if self.validationLoadFile is None:
            indices = np.arange(0, self.validationSampleCount)
            if self.validationSaveFile is not None:
                UtilityFuncs.save_npz(file_name=self.validationSaveFile, arr_dict={"random_indices": indices})
        else:
            indices = UtilityFuncs.load_npz(file_name=self.validationLoadFile)["random_indices"]
        self.validationSamples = self.trainingSamples[indices]
        self.validationLabels = self.trainingLabels[indices]
        self.validationIndices = self.trainingIndices[indices]
        self.validationOneHotLabels = self.trainingOneHotLabels[indices]
        self.validationCoarseLabels = self.trainingCoarseLabels[indices]
        self.validationCoarseOneHotLabels = self.trainingCoarseOneHotLabels[indices]
        self.trainingSamples = np.delete(self.trainingSamples, indices, 0)
        self.trainingLabels = np.delete(self.trainingLabels, indices, 0)
        self.trainingIndices = np.delete(self.trainingIndices, indices, 0)
        self.trainingOneHotLabels = np.delete(self.trainingOneHotLabels, indices, 0)
        self.trainingCoarseLabels = np.delete(self.trainingCoarseLabels, indices, 0)
        self.trainingCoarseOneHotLabels = np.delete(self.trainingCoarseOneHotLabels, indices, 0)
        # Load into Tensorflow Datasets
        # dataset = tf.data.Dataset.from_tensor_slices((self.trainingSamples, self.trainingLabels))
        self.trainDataset = tf.data.Dataset.from_tensor_slices((self.trainingSamples,
                                                                self.trainingLabels,
                                                                self.trainingIndices,
                                                                self.trainingOneHotLabels,
                                                                self.trainingCoarseLabels,
                                                                self.trainingCoarseOneHotLabels))
        self.validationDataset = tf.data.Dataset.from_tensor_slices((self.validationSamples,
                                                                     self.validationLabels,
                                                                     self.validationIndices,
                                                                     self.validationOneHotLabels,
                                                                     self.validationCoarseLabels,
                                                                     self.validationCoarseOneHotLabels))
        self.testDataset = tf.data.Dataset.from_tensor_slices((self.testSamples,
                                                               self.testLabels,
                                                               self.testIndices,
                                                               self.testOneHotLabels,
                                                               self.testCoarseLabels,
                                                               self.testCoarseOneHotLabels))
        # Create augmented training set
        self.trainDataset = self.trainDataset.shuffle(buffer_size=self.trainingSamples.shape[0]) #self.trainingSamples.shape[0]/10)
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

        self.set_current_data_set_type(dataset_type=DatasetTypes.training,
                                       batch_size=self.batchSizesDict[DatasetTypes.training])

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

    def set_current_data_set_type(self, dataset_type, batch_size):
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

    def get_next_batch(self):
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
