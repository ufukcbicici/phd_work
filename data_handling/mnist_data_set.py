import struct
import os
import platform
import numpy as np
from array import array
from auxillary.constants import DatasetTypes
from auxillary.general_utility_funcs import UtilityFuncs
from data_handling.data_set import DataSet
import matplotlib.pyplot as plt


class MnistDataSet(DataSet):
    MNIST_SIZE = 28

    def __init__(self, validation_sample_count, save_validation_as=None, load_validation_from=None):
        super().__init__()
        # os_name = platform.system()
        self.testImagesPath = os.path.join(os.getcwd(), "..\\data\\mnist\\t10k-images-idx3-ubyte")
        self.testLabelsPath = os.path.join(os.getcwd(), "..\\data\\mnist\\t10k-labels-idx1-ubyte")
        self.trainImagesPath = os.path.join(os.getcwd(), "..\\data\\mnist\\train-images-idx3-ubyte")
        self.trainLabelsPath = os.path.join(os.getcwd(), "..\\data\\mnist\\train-labels-idx1-ubyte")
        self.trainingSamples = None
        self.trainingLabels = None
        self.testSamples = None
        self.testLabels = None
        self.validationSamples = None
        self.validationLabels = None
        self.currentSamples = None
        self.currentLabels = None
        self.currentIndices = None
        self.validationSampleCount = validation_sample_count
        self.validationSaveFile = save_validation_as
        self.validationLoadFile = load_validation_from
        self.load_dataset()
        self.set_current_data_set_type(dataset_type=DatasetTypes.training)
        self.labelCount = None

    # PUBLIC METHODS
    def load_dataset(self):
        self.trainingSamples, self.trainingLabels = self.load(path_img=self.trainImagesPath,
                                                              path_lbl=self.trainLabelsPath)
        self.testSamples, self.testLabels = self.load(path_img=self.testImagesPath, path_lbl=self.testLabelsPath)
        if self.validationLoadFile is None:
            random_indices = np.random.choice(self.trainingSamples.shape[0], size=self.validationSampleCount, replace=False)
            if self.validationSaveFile is not None:
                UtilityFuncs.save_npz(file_name=self.validationSaveFile, arr_dict={"random_indices": random_indices})
        else:
            random_indices = UtilityFuncs.load_npz(file_name=self.validationLoadFile)["random_indices"]
        # print(random_indices[0:5])
        self.validationSamples = self.trainingSamples[random_indices]
        self.validationLabels = self.trainingLabels[random_indices]
        self.trainingSamples = np.delete(self.trainingSamples, random_indices, 0)
        self.trainingLabels = np.delete(self.trainingLabels, random_indices, 0)

    def get_next_batch(self, batch_size):
        num_of_samples = self.get_current_sample_count()
        curr_end_index = self.currentIndex + batch_size - 1
        # Check if the interval [curr_start_index, curr_end_index] is inside data boundaries.
        if 0 <= self.currentIndex and curr_end_index < num_of_samples:
            indices_list = self.currentIndices[self.currentIndex:curr_end_index + 1]
        elif self.currentIndex < num_of_samples <= curr_end_index:
            indices_list = self.currentIndices[self.currentIndex:num_of_samples]
            curr_end_index = curr_end_index % num_of_samples
            indices_list.extend(self.currentIndices[0:curr_end_index + 1])
        else:
            raise Exception("Invalid index positions: self.currentIndex={0} - curr_end_index={1}"
                            .format(self.currentIndex, curr_end_index))
        samples = self.currentSamples[indices_list]
        labels = self.currentLabels[indices_list]
        self.currentIndex = self.currentIndex + batch_size
        if num_of_samples <= self.currentIndex:
            self.currentEpoch += 1
            self.isNewEpoch = True
            np.random.shuffle(self.currentIndices)
            self.currentIndex = self.currentIndex % num_of_samples
        else:
            self.isNewEpoch = False
        return samples, labels, indices_list.astype(np.int64)

    def reset(self):
        self.currentIndex = 0
        indices = np.arange(self.currentSamples.shape[0])
        np.random.shuffle(indices)
        self.currentLabels = self.currentLabels[indices]
        self.currentSamples = self.currentSamples[indices]
        self.currentIndices = np.arange(self.currentSamples.shape[0])
        np.random.shuffle(self.currentIndices)
        self.isNewEpoch = False

    def set_current_data_set_type(self, dataset_type):
        self.currentDataSetType = dataset_type
        if self.currentDataSetType == DatasetTypes.training:
            self.currentSamples = self.trainingSamples
            self.currentLabels = self.trainingLabels
        elif self.currentDataSetType == DatasetTypes.test:
            self.currentSamples = self.testSamples
            self.currentLabels = self.testLabels
        elif self.currentDataSetType == DatasetTypes.validation:
            self.currentSamples = self.validationSamples
            self.currentLabels = self.validationLabels
        else:
            raise Exception("Unknown dataset type")
        self.reset()

    def get_current_sample_count(self):
        return self.currentSamples.shape[0]

    def visualize_sample(self, sample_index):
        plt.title('Label is {label}'.format(label=self.currentLabels[sample_index]))
        plt.imshow(self.currentSamples[sample_index], cmap='gray')
        plt.show()

    def get_label_count(self):
        if self.labelCount is None:
            label_set_count = self.testLabels.shape[0]
            label_dict = {}
            for i in range(0, label_set_count):
                label = self.testLabels[i]
                if not (label in label_dict):
                    label_dict[label] = 0
                label_dict[label] += 1
            self.labelCount = len(label_dict)
        return self.labelCount

    # PRIVATE METHODS
    # Load method taken from https://github.com/sorki/python-mnist/blob/master/mnist/loader.py
    def load(self, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        np_images = np.array(images).reshape((len(images), MnistDataSet.MNIST_SIZE, MnistDataSet.MNIST_SIZE)).astype(
            float)
        np_images /= 255.0
        np_labels = np.array(labels).astype(np.int64)
        return np_images, np_labels
