from auxillary.constants import DatasetTypes


class DataSet:
    def __init__(self):
        self.trainingSampleCount = None
        self.testSampleCount = None
        self.validationSampleCount = None
        self.dataShape = None
        self.targetShape = None
        self.currentDataSet = DatasetTypes.training
        self.currentIndex = 0

    def load_dataset(self):
        pass

    def get_next_batch(self, batch_size):
        pass

    def reset(self):
        self.currentIndex = 0

    def get_current_sample_count(self):
        if self.currentDataSet == DatasetTypes.training:
            return self.trainingSampleCount
        elif self.currentDataSet == DatasetTypes.test:
            return self.testSampleCount
        elif self.currentDataSet == DatasetTypes.validation:
            return self.validationSampleCount
        else:
            raise Exception("Unknown dataset type.")


