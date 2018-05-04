import tensorflow as tf

from data_handling.fashion_mnist import FashionMnistDataSet

dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)