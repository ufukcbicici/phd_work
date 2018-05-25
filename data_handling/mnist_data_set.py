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

    def __init__(self,
                 validation_sample_count,
                 save_validation_as=None,
                 load_validation_from=None,
                 test_images_path = os.path.join(os.getcwd(), "data\\mnist\\t10k-images-idx3-ubyte"),
                 test_labels_path = os.path.join(os.getcwd(), "data\\mnist\\t10k-labels-idx1-ubyte"),
                 training_images_path = os.path.join(os.getcwd(), "data\\mnist\\train-images-idx3-ubyte"),
                 training_labels_path = os.path.join(os.getcwd(), "data\\mnist\\train-labels-idx1-ubyte")
                 ):
        super().__init__()
        # os_name = platform.system()
        self.testImagesPath = test_images_path
        self.testLabelsPath = test_labels_path
        self.trainImagesPath = training_images_path
        self.trainLabelsPath = training_labels_path
        self.validationSampleCount = validation_sample_count
        self.validationSaveFile = save_validation_as
        self.validationLoadFile = load_validation_from
        self.load_dataset()
        self.set_current_data_set_type(dataset_type=DatasetTypes.training)

    # PUBLIC METHODS
    def load_dataset(self):
        self.trainingSamples, self.trainingLabels = self.load(path_img=self.trainImagesPath,
                                                              path_lbl=self.trainLabelsPath)
        self.testSamples, self.testLabels = self.load(path_img=self.testImagesPath, path_lbl=self.testLabelsPath)
        if self.validationLoadFile is None:
            # random_indices = np.random.choice(self.trainingSamples.shape[0], size=self.validationSampleCount, replace=False)
            indices = np.arange(0, self.validationSampleCount)
            if self.validationSaveFile is not None:
                UtilityFuncs.save_npz(file_name=self.validationSaveFile, arr_dict={"random_indices": indices})
        else:
            indices = UtilityFuncs.load_npz(file_name=self.validationLoadFile)["random_indices"]
        # print(random_indices[0:5])
        self.validationSamples = self.trainingSamples[indices]
        self.validationLabels = self.trainingLabels[indices]
        self.trainingSamples = np.delete(self.trainingSamples, indices, 0)
        self.trainingLabels = np.delete(self.trainingLabels, indices, 0)
        # print("X")

    def visualize_sample(self, sample_index):
        plt.title('Label is {label}'.format(label=self.currentLabels[sample_index]))
        plt.imshow(self.currentSamples[sample_index], cmap='gray')
        plt.show()

    def get_sample_shape(self):
        tpl = (MnistDataSet.MNIST_SIZE, MnistDataSet.MNIST_SIZE, 1)
        return tpl

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
