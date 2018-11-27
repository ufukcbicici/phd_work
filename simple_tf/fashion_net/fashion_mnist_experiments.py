import numpy as np

from auxillary.constants import DatasetTypes
from data_handling.fashion_mnist import FashionMnistDataSet

dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
dataset.set_current_data_set_type(dataset_type=DatasetTypes.test)
samples, labels, indices_list, one_hot_labels = dataset.get_next_batch()

sums = set()
for sample in samples:
    img_sum_0 = np.sum(sample[0:14])
    img_sum_1 = np.sum(sample[14:28])
    sums.add((img_sum_0, img_sum_1))
print("X")