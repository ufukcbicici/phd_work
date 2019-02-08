import pickle
import numpy as np
from collections import namedtuple
from sklearn import preprocessing
from park_here.constants import Constants
from enum import Enum


class SignalDataSet:
    class DatasetTypes(Enum):
        Training = 0
        Validation = 1
        Test = 2

    MiniBatch = \
        namedtuple('MiniBatch',
                   ['sequences',
                    'labels',
                    'sequence_lengths'])

    ClassOverSampleRatios = \
        {"CAR_IN": 1.0,
         "CAR_OUT": 1.0,
         "PEDESTRIAN": 3.0,
         "MOTORCYCLE": 6.0
         }

    def __init__(self, test_ratio=0.1):
        self.testRatio = test_ratio
        self.currentIndex = 0
        [X, Y1, Y2, _] = pickle.load(open('data\\challenge_data.pkl', 'rb'), encoding='latin1')
        # Set numpy random seed
        np.random.seed(seed=Constants.RANDOM_SEED)
        # Remove "OTHER" Labels
        indices_with_label_other = np.nonzero(Y1 == "OTHER")
        X = np.delete(X, indices_with_label_other, 0)
        Y1 = np.delete(Y1, indices_with_label_other, 0)
        # Encode Labels
        self.labelEncoder = preprocessing.LabelEncoder()
        self.labelEncoder.fit(Y1)
        Y1 = self.labelEncoder.transform(Y1)
        self.maxLength = max([len(l) for l in X])
        # Training - Test Split
        self.testSampleCount = int(test_ratio * len(X))
        random_indices = np.random.choice(len(X), size=self.testSampleCount, replace=False)
        self.testSamples = X[random_indices]
        self.testLabels = Y1[random_indices]
        self.trainingSamples = np.delete(X, random_indices, 0)
        self.trainingLabels = np.delete(Y1, random_indices, 0)
        self.currentLabels = self.trainingLabels
        self.currentSamples = self.trainingSamples
        self.currentIndices = np.arange(self.currentSamples.shape[0])
        self.isNewEpoch = False
        self.currentEpoch = 0
        self.labelCount = None
        self.currentDataSetType = None
        self.oversample()
        # Not used
        self.validationSamples = None
        self.validationLabels = None

    def oversample(self):
        # Oversample to address class imbalance
        oversampled_samples = None
        oversampled_labels = None
        for class_name, ratio in SignalDataSet.ClassOverSampleRatios.items():
            class_id = self.labelEncoder.transform([class_name])[0]
            indices = np.nonzero(self.trainingLabels == class_id)[0]
            if ratio == 1.0:
                samples = self.trainingSamples[indices]
            else:
                class_count = indices.shape[0]
                sampled_class_count = int(class_count * ratio)
                samples = np.random.choice(self.trainingSamples[indices], size=sampled_class_count, replace=True)
            labels = np.zeros(shape=(samples.shape[0],), dtype=np.int32)
            labels[:] = class_id
            if oversampled_samples is None:
                oversampled_samples = samples
            else:
                oversampled_samples = np.concatenate((oversampled_samples, samples), axis=0)
            if oversampled_labels is None:
                oversampled_labels = labels
            else:
                oversampled_labels = np.concatenate((oversampled_labels, labels), axis=0)
        self.trainingSamples = oversampled_samples
        self.trainingLabels = oversampled_labels

    def reset(self):
        self.currentIndex = 0
        indices = np.arange(self.currentSamples.shape[0])
        np.random.shuffle(indices)
        self.currentLabels = self.currentLabels[indices]
        self.currentSamples = self.currentSamples[indices]
        self.currentIndices = np.arange(self.currentSamples.shape[0])
        np.random.shuffle(self.currentIndices)
        self.isNewEpoch = False

    def get_current_sample_count(self):
        return self.currentSamples.shape[0]

    def get_label_count(self):
        if self.labelCount is None:
            label_set_count = self.testLabels.shape[0]
            label_dict = {}
            for i in range(0, label_set_count):
                label = self.testLabels[i]
                if not (label in label_dict):
                    label_dict[label] = 0
                label_dict[label] += 1
            self.labelCount = len(label_dict)
        return self.labelCount

    def get_next_batch(self, batch_size, wrap_around):
        assert batch_size is not None
        num_of_samples = self.get_current_sample_count()
        curr_end_index = self.currentIndex + batch_size - 1
        # Check if the interval [curr_start_index, curr_end_index] is inside data boundaries.
        if 0 <= self.currentIndex and curr_end_index < num_of_samples:
            indices_list = self.currentIndices[self.currentIndex:curr_end_index + 1]
        elif self.currentIndex < num_of_samples <= curr_end_index:
            indices_list = self.currentIndices[self.currentIndex:num_of_samples]
            if wrap_around:
                curr_end_index = curr_end_index % num_of_samples
                indices_list = np.concatenate([indices_list, self.currentIndices[0:curr_end_index + 1]])
        else:
            raise Exception("Invalid index positions: self.currentIndex={0} - curr_end_index={1}"
                            .format(self.currentIndex, curr_end_index))
        # samples = self.currentSamples[indices_list]
        # labels = self.currentLabels[indices_list]
        # one_hot_labels = np.zeros(shape=(batch_size, self.get_label_count()))
        # one_hot_labels[np.arange(batch_size), labels.astype(np.int)] = 1.0
        self.currentIndex = self.currentIndex + batch_size
        # Prepare sequence data
        sequences = np.zeros(shape=(batch_size, self.maxLength, Constants.ORIGINAL_DATA_DIMENSION), dtype=np.float32)
        sequence_lengths = np.zeros(shape=(batch_size,), dtype=np.int32)
        labels = np.zeros(shape=(batch_size,), dtype=np.int32)
        for batch_index, global_index in enumerate(indices_list):
            sequence = self.currentSamples[global_index]
            sequence_lengths[batch_index] = len(sequence)
            labels[batch_index] = self.currentLabels[global_index]
            sequences[batch_index, 0:sequence_lengths[batch_index]] = sequence
        if num_of_samples <= self.currentIndex:
            self.currentEpoch += 1
            self.isNewEpoch = True
            np.random.shuffle(self.currentIndices)
            self.currentIndex = self.currentIndex % num_of_samples
        else:
            self.isNewEpoch = False
        return SignalDataSet.MiniBatch(sequences, labels, sequence_lengths)

    def set_current_dataset_type(self, dataset_type):
        self.currentDataSetType = dataset_type
        if self.currentDataSetType == SignalDataSet.DatasetTypes.Training:
            self.currentSamples = self.trainingSamples
            self.currentLabels = self.trainingLabels
        elif self.currentDataSetType == SignalDataSet.DatasetTypes.Test:
            self.currentSamples = self.testSamples
            self.currentLabels = self.testLabels
        elif self.currentDataSetType == SignalDataSet.DatasetTypes.Validation:
            self.currentSamples = self.validationSamples
            self.currentLabels = self.validationLabels
        else:
            raise Exception("Unknown dataset type")
        self.reset()