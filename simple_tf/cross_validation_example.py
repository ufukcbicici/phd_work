import numpy as np

from algorithms.cross_validation import CrossValidation
from simple_tf.global_params import GlobalConstants

sample_count = 17569
cv = CrossValidation(sample_count=sample_count, fold_count=10)

for partition in cv:
    print(partition)
    print(partition.shape)
    training_indices = set(range(sample_count)).difference(set(partition))
    print("X")

print("X")
