from auxillary.constants import DatasetTypes
import numpy as np


class DataSet:
    def __init__(self):
        self.dataShape = None
        self.targetShape = None
        self.currentDataSetType = None
        self.currentIndex = 0
        self.currentEpoch = 0
        self.isNewEpoch = True
        self.trainingSamples = None
        self.trainingLabels = None
        self.testSamples = None
        self.testLabels = None
        self.validationSamples = None
        self.validationLabels = None
        self.currentLabels = None
        self.currentSamples = None
        self.currentIndices = None
        self.validationSampleCount = 0
        self.labelCount = None

    def load_dataset(self):
        pass

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
        one_hot_labels = np.zeros(shape=(batch_size, self.get_label_count()))
        one_hot_labels[np.arange(batch_size), labels.astype(np.int)] = 1.0
        self.currentIndex = self.currentIndex + batch_size
        if num_of_samples <= self.currentIndex:
            self.currentEpoch += 1
            self.isNewEpoch = True
            np.random.shuffle(self.currentIndices)
            self.currentIndex = self.currentIndex % num_of_samples
        else:
            self.isNewEpoch = False
        return samples, labels, indices_list.astype(np.int64), one_hot_labels

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

    def get_label_count(self):
        if self.labelCount is None:
            label_set_count = self.trainingLabels.shape[0]
            label_dict = {}
            for i in range(0, label_set_count):
                label = self.trainingLabels[i]
                if not (label in label_dict):
                    label_dict[label] = 0
                label_dict[label] += 1
            self.labelCount = len(label_dict)
        return self.labelCount

    def get_sample_shape(self):
        pass

    def visualize_sample(self, sample_index):
        pass
