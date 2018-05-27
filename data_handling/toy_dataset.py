import os
import numpy as np
from auxillary.constants import DatasetTypes
from data_handling.data_set import DataSet
import matplotlib.pyplot as plt


class ToyDataset(DataSet):
    def __init__(self,
                 validation_sample_count,
                 # path=os.path.join(os.getcwd(), "data\\toy_data\\spiral.txt")
                 path="C:\\Users\\ufuk.bicici\\Desktop\\tf\\phd_work\\data\\toy_data\\spiral.txt"
                 ):
        super().__init__()
        # os_name = platform.system()
        self.dataPath = path
        self.validationSampleCount = validation_sample_count
        self.load_dataset()
        # self.set_current_data_set_type(dataset_type=DatasetTypes.training)
        self.labelCount = None
        self.load_dataset()

    def load_dataset(self):
        with open(self.dataPath) as f:
            content = f.readlines()
        content = [x.strip().split("\t") for x in content]
        sample_dim = len(content[0]) - 1
        # All samples are of equal dimension
        assert len(set([len(l) for l in content])) == 1
        self.trainingSamples = np.zeros(shape=(len(content), sample_dim))
        self.trainingLabels = np.zeros(shape=(len(content),))
        for i in range(len(content)):
            self.trainingSamples[i, :] = np.array(content[i][0:sample_dim])
            self.trainingLabels[i] = content[i][-1]
        # self.set_current_data_set_type(dataset_type=DatasetTypes.training)
        # self.reset()
        # Change the label mappings, so labels range in [0,self.labelsCount-1]
        label_set_count = self.trainingLabels.shape[0]
        label_dict = {}
        for i in range(0, label_set_count):
            label = self.trainingLabels[i]
            if not (label in label_dict):
                label_dict[label] = 0
            label_dict[label] += 1
        sorted_labels = sorted(label_dict.keys())
        mapping_dict = {}
        for i in range(len(sorted_labels)):
            mapping_dict[sorted_labels[i]] = i
        for i in range(self.trainingLabels.shape[0]):
            self.trainingLabels[i] = mapping_dict[self.trainingLabels[i]]
        indices = np.arange(self.trainingSamples.shape[0])
        np.random.shuffle(indices)
        self.trainingSamples = self.trainingSamples[indices]
        self.trainingLabels = self.trainingLabels[indices]
        indices = np.arange(0, self.validationSampleCount)
        self.validationSamples = self.trainingSamples[indices]
        self.validationLabels = self.trainingLabels[indices]
        self.trainingSamples = np.delete(self.trainingSamples, indices, 0)
        self.trainingLabels = np.delete(self.trainingLabels, indices, 0)

    def visualize_dataset(self, dataset_type):
        self.set_current_data_set_type(dataset_type=dataset_type)
        plt.title(dataset_type)
        for l in range(self.get_label_count()):
            label_mask = self.currentLabels == l
            plt.plot(self.currentSamples[label_mask, 0], self.currentSamples[label_mask, 1], '.',
                     label="class {0}".format(l))
        plt.legend()
        plt.show()
