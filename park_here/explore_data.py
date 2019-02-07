import pickle

from park_here.rnn_classifier import RnnClassifier
from park_here.signal_dataset import SignalDataSet

dataset = SignalDataSet()
rnn_classifier = RnnClassifier(dataset=dataset)
rnn_classifier.build_classifier()
print("X")
