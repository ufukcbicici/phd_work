import pickle
import numpy as np


class SignalDataSet:
    def __init__(self, test_ratio=0.1):
        self.testRatio = test_ratio
        [X, Y1, Y2, _] = pickle.load(open('data\\challenge_data.pkl', 'rb'), encoding='latin1')
        print("X")