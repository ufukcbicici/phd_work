from auxillary.constants import DatasetTypes
from collections import namedtuple
import numpy as np


class DataSet:
    MiniBatch = namedtuple('MiniBatch', ['samples', 'labels', 'indices', 'one_hot_labels', 'hash_codes'])

    def __init__(self):
        self.dataShape = None
        self.targetShape = None
        self.currentDataSetType = None
        self.currentIndex = 0
        self.currentEpoch = 0
        self.isNewEpoch = True

    def load_dataset(self):
        pass

    def get_next_batch(self, batch_size):
        pass

    def reset(self):
        pass

    def set_current_data_set_type(self, dataset_type):
        pass

    def get_current_sample_count(self):
        pass

    def get_label_count(self):
        pass

    def visualize_sample(self, sample_index):
        pass

    def get_unique_codes(self, samples):
        sample_count = samples.shape[0]
        hash_codes = np.zeros(shape=(sample_count, ), dtype=np.int64)
        for i in range(sample_count):
            sample = samples[i]
            hash_codes[i] = hash(sample.tostring())
        return hash_codes
