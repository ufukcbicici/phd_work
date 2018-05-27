import tensorflow as tf


class LdaLoss:
    def __init__(self):
        pass

    @staticmethod
    def get_loss(data, labels, dataset, use_verbose=True):
        # weights = []
        # class_counts = []
        sum_cov_list = []
        sum_inner_class_variances = []
        inner_class_cov_matrices = []
        var_list = []
        mean_all, var_all = tf.nn.moments(x=data, axes=0)
        mean_all = tf.expand_dims(mean_all, 1)
        N = tf.cast(tf.size(labels), tf.float32)
        for k in range(dataset.get_label_count()):
            label_mask = tf.equal(x=labels, y=tf.constant(k, tf.int64))
            N_k = tf.reduce_sum(tf.cast(label_mask, tf.float32))
            data_k = tf.boolean_mask(data, label_mask)
            mean_k, var_k = tf.nn.moments(x=data_k, axes=0)
            var_list.append(var_k)
            sum_var_k = tf.reduce_sum(var_k)
            mean_k = tf.expand_dims(mean_k, 1)
            delta_mean_k = mean_k - mean_all
            weight_k = N_k / N
            delta_matrix_k = tf.matmul(delta_mean_k, tf.transpose(delta_mean_k))
            sum_cov_list.append(N_k * delta_matrix_k)
            sum_inner_class_variances.append(sum_var_k)
            data_k_mean_subtracted = data_k - tf.transpose(mean_k)
            data_k_mean_subtracted_T = tf.transpose(data_k_mean_subtracted)
            inner_class_cov_matrix = tf.matmul(data_k_mean_subtracted_T, data_k_mean_subtracted)
            inner_class_cov_matrices.append(inner_class_cov_matrix)
        inner_class_covariance_matrix = tf.add_n(inputs=inner_class_cov_matrices)
        between_class_covariance_matrix = tf.add_n(inputs=sum_cov_list)
        total_between_class_variance = tf.trace(between_class_covariance_matrix)
        # total_inner_class_variance = tf.add_n(inputs=sum_inner_class_variances)
        total_inner_class_variance = tf.trace(inner_class_covariance_matrix)
        objective = total_inner_class_variance / total_between_class_variance
        # objective = -1.0 * total_between_class_variance + total_inner_class_variance
        # -1.0 * (total_between_class_variance / total_inner_class_variance)
        return objective, total_between_class_variance, total_inner_class_variance, \
               between_class_covariance_matrix, inner_class_covariance_matrix, var_list, inner_class_cov_matrices

        # mean_all, var_all = tf.nn.moments(x=tf_data, axes=0)
        # mean_all = tf.expand_dims(mean_all, 1)
        # N = tf.cast(tf.size(tf_labels), tf.float32)
        # run_ops = [mean_all, var_all, N]
        # class_means = []
        # inner_class_variances = []
        # sum_inner_class_variances = []
        # class_counts = []
        # weights = []
        # sum_cov_list = []
        # for k in range(toy_dataset.get_label_count()):
        #     label_mask = tf.equal(x=tf_labels, y=tf.constant(k, tf.int64))
        #     N_k = tf.reduce_sum(tf.cast(label_mask, tf.float32))
        #     data_k = tf.boolean_mask(tf_data, label_mask)
        #     mean_k, var_k = tf.nn.moments(x=data_k, axes=0)
        #     sum_var_k = tf.reduce_sum(var_k)
        #     mean_k = tf.expand_dims(mean_k, 1)
        #     delta_mean_k = mean_k - mean_all
        #     weight_k = N_k / N
        #     weights.append(weight_k)
        #     delta_matrix_k = tf.matmul(delta_mean_k, tf.transpose(delta_mean_k))
        #     sum_cov_list.append(weight_k * delta_matrix_k)
        #     class_means.append(mean_k)
        #     inner_class_variances.append(var_k)
        #     sum_inner_class_variances.append(sum_var_k)
        #     class_counts.append(N_k)
        # between_class_covariance_matrix = tf.add_n(inputs=sum_cov_list)
        # total_inner_class_variance = tf.add_n(inputs=sum_inner_class_variances)
        # total_between_class_variance = tf.trace(between_class_covariance_matrix)
        # objective = -1.0 * (total_between_class_variance / total_inner_class_variance)
