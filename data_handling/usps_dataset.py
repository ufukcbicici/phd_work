from auxillary.constants import DatasetTypes
from data_handling.mnist_data_set import MnistDataSet
from sklearn.model_selection import train_test_split
import h5py
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from auxillary.general_utility_funcs import UtilityFuncs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from data_handling.data_set import DataSet


class UspsDataset(MnistDataSet):
    USPS_SIZE = 16

    def __init__(self, validation_sample_count):
        super().__init__(validation_sample_count)

    def load_dataset(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "usps", "usps.h5")
        with h5py.File(path, 'r') as hf:
            train = hf.get('train')
            self.trainingSamples = train.get('data')[:]
            self.trainingLabels = train.get('target')[:]
            test = hf.get('test')
            self.testSamples = test.get('data')[:]
            self.testLabels = test.get('target')[:]
        if self.validationSampleCount > 0:
            self.trainingSamples, self.trainingLabels, self.validationSamples, self.validationLabels = \
                train_test_split(self.trainingSamples, self.trainingLabels, self.validationSampleCount)
        # Bootstrap sampling
        # training_indices = np.random.choice(np.arange(self.trainingSamples.shape[0]),
        #                                     size=self.trainingSamples.shape[0], replace=True)
        # self.trainingSamples = self.trainingSamples[training_indices]
        # self.trainingLabels = self.trainingLabels[training_indices]

        # Experimental
        # scaler = StandardScaler()
        # scaler.fit(self.trainingSamples)
        # training_samples_scaled = scaler.transform(self.trainingSamples)
        # test_samples_scaled = scaler.transform(self.testSamples)
        # pca = PCA()
        # pca.fit(training_samples_scaled)
        # self.trainingSamples = pca.transform(training_samples_scaled)
        # self.testSamples = pca.transform(test_samples_scaled)
        # print("X")


