import numpy as np
import tensorflow as tf


class RoutingMaskFromNodeIndicesGenerator:

    def __init__(self):
        pass

    @staticmethod
    def generate_routing_matrix(batch_size_, node_count_, node_indices_):
        assert node_count_ == len(node_indices_)
        indices_list = []
        for node_id in range(node_count_):
            node_index_array = node_id * tf.ones(shape=node_indices_[node_id].get_shape(), dtype=tf.int32)
            batch_index_array = node_indices_[node_id]
            batch_index_array = tf.stack([batch_index_array, node_index_array], axis=1)
            indices_list.append(batch_index_array)
        indices_ = tf.concat(indices_list, axis=0)
        updates = tf.ones(shape=tf.shape(indices_)[0], dtype=tf.int32)
        routing_matrix = tf.scatter_nd(indices=indices_, updates=updates, shape=(batch_size_, node_count_))
        return routing_matrix


if __name__ == '__main__':
    node_count = 5
    batch_size = 125
    node_indices = []
    routing_matrix_np = np.zeros(shape=(batch_size, node_count), dtype=np.int32)
    for i in range(node_count):
        sample_count = np.random.randint(low=10, high=125)
        indices = np.random.choice(batch_size, size=(sample_count,), replace=False)
        indices_tf = tf.constant(indices)
        node_indices.append(indices_tf)
        for idx in indices:
            routing_matrix_np[idx, i] = 1
    routing_matrix_tf = \
        RoutingMaskFromNodeIndicesGenerator.generate_routing_matrix(batch_size_=batch_size, node_count_=node_count,
                                                                    node_indices_=node_indices)
    sz = tf.placeholder(dtype=tf.int32)
    mask_vector = tf.ones(shape=[sz], dtype=tf.int32)
    sess = tf.Session()
    results = sess.run([routing_matrix_tf, mask_vector], feed_dict={sz: 125})
    assert np.array_equal(routing_matrix_np, results[0])
    print("X")
