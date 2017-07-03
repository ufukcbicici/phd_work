from auxillary.constants import DatasetTypes


class DataSet:
    def __init__(self):
        self.dataShape = None
        self.targetShape = None
        self.currentDataSetType = None
        self.currentIndex = 0

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
