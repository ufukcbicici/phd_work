from auxillary.constants import DatasetTypes
from collections import namedtuple


class DataSet:
    MiniBatch = namedtuple('MiniBatch', ['samples', 'labels', 'indices', 'one_hot_labels'])

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
