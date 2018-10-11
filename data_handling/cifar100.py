from data_handling.mnist_data_set import MnistDataSet
import os
from auxillary.general_utility_funcs import UtilityFuncs
import numpy as np


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
        self.trainingSamples = training_data[b"data"].reshape((training_data[b"data"].shape[0],
                                                              Cifar100DataSet.CIFAR_SIZE,
                                                              Cifar100DataSet.CIFAR_SIZE, 3)).astype(float)
        self.testSamples = test_data[b"data"].reshape((test_data[b"data"].shape[0],
                                                      Cifar100DataSet.CIFAR_SIZE, Cifar100DataSet.CIFAR_SIZE, 3)).astype(
            float)
        # Pack coarse and fine labels into a Nx2 array. Each i.th row corresponds to (coarse,fine) labels.
        training_coarse_labels = np.array(training_data[b"coarse_labels"])
        training_fine_labels = np.array(training_data[b"fine_labels"])
        test_coarse_labels = np.array(test_data[b"coarse_labels"])
        test_fine_labels = np.array(test_data[b"fine_labels"])
        self.trainingLabels = np.concatenate([training_coarse_labels,  training_fine_labels], axis=1)
        self.testLabels = np.concatenate([test_coarse_labels, test_fine_labels], axis=1)
        self.testLabels = np.concatenate([test_data[b"coarse_labels"], test_data[b"fine_labels"]], axis=1)
        if self.validationLoadFile is None:
            # random_indices = np.random.choice(self.trainingSamples.shape[0], size=self.validationSampleCount, replace=False)
            indices = np.arange(0, self.validationSampleCount)
            if self.validationSaveFile is not None:
                UtilityFuncs.save_npz(file_name=self.validationSaveFile, arr_dict={"random_indices": indices})
        else:
            indices = UtilityFuncs.load_npz(file_name=self.validationLoadFile)["random_indices"]
        self.validationSamples = self.trainingSamples[indices]
        self.validationLabels = self.trainingLabels[indices]
        self.trainingSamples = np.delete(self.trainingSamples, indices, 0)
        self.trainingLabels = np.delete(self.trainingLabels, indices, 0)
        print("X")
