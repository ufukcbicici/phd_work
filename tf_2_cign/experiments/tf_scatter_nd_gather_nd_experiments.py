import numpy as np
import tensorflow as tf
import time

batch_size_param = 128
node_degree = 2
x = np.random.uniform(size=(batch_size_param, 32))
routing_values = np.random.uniform(size=(batch_size_param, node_degree), low=-10.0, high=10.0)

gpus = tf.config.list_physical_devices('GPU')

with tf.device('/GPU:0'):
    # Init
    tf_x = tf.keras.Input(shape=(32, ), name="tf_x")
    tf_router = tf.keras.Input(shape=(node_degree, ), name="tf_router")
    tf_temperature = tf.keras.Input(shape=(), name="tf_temperature")

    tempered_values = tf_router / tf_temperature
    p_n_given_x = tf.nn.softmax(tempered_values)
    tf_batch_size = tf.shape(tf_x)[0]
    batch_indices = tf.range(0, tf_batch_size, 1)
    routing_matrix = tf.one_hot(tf.argmax(p_n_given_x, axis=1), node_degree)


    mask_vectors = [routing_matrix[:, c] for c in range(node_degree)]
    batch_indices_masked = [tf.boolean_mask(batch_indices, mask_vectors[c]) for c in range(node_degree)]
    x_list = [tf.boolean_mask(tf_x, mask_vectors[c]) for c in range(node_degree)]
    # scatter_nd_shapes = [[tf_batch_size] + x_list[c].shape[1:] for c in range(node_degree)]
    scatter_nd_shapes = [tf.concat([tf.expand_dims(tf_batch_size, axis=0), tf.shape(x_list[c])[1:]], axis=0)
                         for c in range(node_degree)]
    x_list_scattered = [tf.scatter_nd(tf.expand_dims(batch_indices_masked[c], axis=-1),
                                      x_list[c], scatter_nd_shapes[c]) for c in range(node_degree)]
    x_concat = tf.concat(x_list_scattered, axis=-1)
    x_list_split = [x_concat[..., c*x_list[c].shape[-1]:(c+1)*x_list[c].shape[-1]] for c in range(node_degree)]
    x_list_split_gathered = [tf.gather_nd(x_list_split[c], tf.expand_dims(batch_indices_masked[c], axis=-1))
                             for c in range(node_degree)]

    model = tf.keras.Model(inputs={"tf_x": tf_x, "tf_router": tf_router, "tf_temperature": tf_temperature},
                           outputs=[routing_matrix,
                                    p_n_given_x,
                                    mask_vectors,
                                    batch_indices_masked,
                                    x_list,
                                    x_list_scattered,
                                    x_concat,
                                    x_list_split,
                                    x_list_split_gathered,
                                    tf_temperature])

    trial_count = 1000
    for trial_id in range(trial_count):
        t0 = time.time()
        print("Trial:{0}".format(trial_id))
        x = np.random.uniform(size=(batch_size_param, 32))
        routing_values = np.random.uniform(size=(batch_size_param, node_degree), low=-10.0, high=10.0)
        results = model(inputs={"tf_x": x, "tf_temperature": 5.0, "tf_router": routing_values}, training=False)

        # Check if tf.scatter_nd works
        x_shape = [batch_size_param] + list(x.shape[1:])
        x_shape[-1] = 2 * x_shape[-1]
        manuel_x_concat = np.zeros(shape=x_shape)
        tf_x_concat = results[6].numpy()
        for c in range(node_degree):
            indices = results[3][c].numpy()
            for sample_index in indices:
                manuel_x_concat[sample_index, ..., c*x.shape[-1]:(c+1)*x.shape[-1]] = x[sample_index]
        assert np.allclose(tf_x_concat, manuel_x_concat)

        # Check if tf.gather_nd works
        tf_x_before_scatter = results[4]
        tf_x_split_back = results[8]
        for c in range(node_degree):
            assert np.array_equal(tf_x_before_scatter[c].numpy(), tf_x_split_back[c].numpy())
        t1 = time.time()
        print("Time Spent:{0}".format(t1 - t0))





