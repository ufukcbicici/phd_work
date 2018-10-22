from auxillary.constants import DatasetTypes
from data_handling.mnist_data_set import MnistDataSet
import os
from auxillary.general_utility_funcs import UtilityFuncs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


class Cifar100DataSet(MnistDataSet):
    CIFAR_SIZE = 32

    def __init__(self,
                 validation_sample_count,
                 save_validation_as=None,
                 load_validation_from=None,
                 test_images_path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + \
                                  os.sep + "data" + os.sep + "cifar100" + os.sep + "test",
                 training_images_path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + \
                                      os.sep + "data" + os.sep + "cifar100" + os.sep + "train",
                 meta_path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + \
                           os.sep + "data" + os.sep + "cifar100" + os.sep + "meta"):
        super().__init__(validation_sample_count=validation_sample_count, save_validation_as=save_validation_as,
                         load_validation_from=load_validation_from, test_images_path=test_images_path,
                         test_labels_path=None, training_images_path=training_images_path,
                         training_labels_path=None)

    def load_dataset(self):
        training_data = UtilityFuncs.unpickle(file=self.trainImagesPath)
        test_data = UtilityFuncs.unpickle(file=self.testImagesPath)
        # self.trainingSamples = training_data[b"data"].reshape((training_data[b"data"].shape[0],
        #                                                       Cifar100DataSet.CIFAR_SIZE,
        #                                                       Cifar100DataSet.CIFAR_SIZE, 3)).astype(float)
        # self.testSamples = test_data[b"data"].reshape((test_data[b"data"].shape[0],
        #                                               Cifar100DataSet.CIFAR_SIZE, Cifar100DataSet.CIFAR_SIZE, 3)).astype(
        #     float)
        self.trainingSamples = training_data[b"data"]
        self.testSamples = test_data[b"data"]
        self.trainingSamples = self.trainingSamples.reshape((self.trainingSamples.shape[0], 3,
                                                             Cifar100DataSet.CIFAR_SIZE, Cifar100DataSet.CIFAR_SIZE))\
            .transpose([0, 2, 3, 1])
        self.testSamples = self.testSamples.reshape((self.testSamples.shape[0], 3,
                                                     Cifar100DataSet.CIFAR_SIZE, Cifar100DataSet.CIFAR_SIZE))\
            .transpose([0, 2, 3, 1])
        # Pack coarse and fine labels into a Nx2 array. Each i.th row corresponds to (coarse,fine) labels.
        training_coarse_labels = np.array(training_data[b"coarse_labels"]).reshape((len(training_data[b"coarse_labels"]), 1))
        training_fine_labels = np.array(training_data[b"fine_labels"]).reshape((len(training_data[b"fine_labels"]), 1))
        test_coarse_labels =   np.array(test_data[b"coarse_labels"]).reshape((len(test_data[b"coarse_labels"]), 1))
        test_fine_labels = np.array(test_data[b"fine_labels"]).reshape((len(test_data[b"fine_labels"]), 1))
        self.trainingLabels = np.concatenate([training_coarse_labels,  training_fine_labels], axis=1)
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
        # Preprocessing and augmentation
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            zca_whitening=True,
            fill_mode='nearest')





        self.visualize_sample(sample_index=1613)
        print("X")

    def visualize_sample(self, sample_index):
        # sample_reshaped0 = self.currentSamples[sample_index]
        # sample_reshaped1 = self.currentSamples[sample_index].transpose([1, 2, 0])
        # sample_reshaped2 = self.currentSamples[sample_index].reshape(3, 32, 32).transpose([1, 2, 0])
        plt.title('Label is {label}'.format(label=self.currentLabels[sample_index]))
        plt.imshow(self.currentSamples[sample_index])
        plt.show()
