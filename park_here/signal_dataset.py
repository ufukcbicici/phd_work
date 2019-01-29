import pickle
import numpy as np


class SignalDataSet:
    def __init__(self, test_ratio=0.1):
        self.testRatio = test_ratio
        self.currentIndex = 0
        [X, Y1, Y2, _] = pickle.load(open('data\\challenge_data.pkl', 'rb'), encoding='latin1')
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

    def reset(self):
        self.currentIndex = 0
        indices = np.arange(self.currentSamples.shape[0])
        np.random.shuffle(indices)
        self.currentLabels = self.currentLabels[indices]
        self.currentSamples = self.currentSamples[indices]
        self.currentIndices = np.arange(self.currentSamples.shape[0])
        np.random.shuffle(self.currentIndices)
        self.isNewEpoch = False
