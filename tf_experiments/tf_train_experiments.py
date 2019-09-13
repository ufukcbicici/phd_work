import tensorflow as tf

from auxillary.constants import DatasetTypes
from data_handling.cifar_dataset import CifarDataSet
from simple_tf.global_params import GlobalConstants

GlobalConstants.BATCH_SIZE = 750
sess = tf.Session()
dataset = CifarDataSet(session=sess, validation_sample_count=0, load_validation_from=None)
dataset.set_curr_session(sess=sess)
dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
batches = []

while True:
    minibatch = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
    batches.append(minibatch)
    if dataset.isNewEpoch:
        break
print("X")