import pickle

from park_here.signal_dataset import SignalDataSet

dataset = SignalDataSet()
batch = dataset.get_next_batch(batch_size=125)
print("X")
