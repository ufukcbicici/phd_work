import tensorflow as tf
import numpy as np

BATCH_SIZE = 32
IMG_W = 256
IMG_H = 256
FEATURE_COUNT = 64
NUM_ROIS = 50

ROI_LIST = [(0.6, 0.7, 0.2, 0.3), (0.2, 0.1, 0.4, 0.4), (0.8, 0.8, 0.15, 0.125)]


class RoIPooling:

    @staticmethod
    def roi_pool(feature_map, roi_list):
        feature_map_width = int(feature_map.shape[2])
        feature_map_height = int(feature_map.shape[1])
        split_features = tf.split(feature_map, BATCH_SIZE, axis=0)
        split_features = [f_map[0, :] for f_map in split_features]
        split_rois = tf.split(roi_list, BATCH_SIZE, axis=0)
        split_rois = [roi_arr[0, :] for roi_arr in split_features]
        return split_features, split_rois

        # roi_feature_maps = []
        # for roi_dims in roi_list:
        #     # assert roi_dims[0] + roi_dims[2] <= 1.0 and roi_dims[1] + roi_dims[3] <= 1.0
        #     h_start = tf.cast(feature_map_height * roi_dims[0], 'int32')
        #     w_start = tf.cast(feature_map_width * roi_dims[1], 'int32')
        #     h_end = tf.cast(feature_map_height * roi_dims[2], 'int32')
        #     w_end = tf.cast(feature_map_width * roi_dims[3], 'int32')
        #     roi_feature_map = x[:, h_start:h_end, w_start:w_end, :]
        #     roi_feature_maps.append(roi_feature_map)
        # print("X")


def main():
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, IMG_W, IMG_H, FEATURE_COUNT])
    rois_tensor = tf.placeholder(tf.float32, shape=(None, NUM_ROIS, 4))
    split_features, split_rois = RoIPooling.roi_pool(feature_map=input_tensor, roi_list=rois_tensor)
    # RoIPooling.roi_pool(x=input_tensor, roi_list=ROI_LIST)

    # Prepare dummy image data
    random_imgs = np.random.uniform(size=(BATCH_SIZE, IMG_W, IMG_H, FEATURE_COUNT))
    # Prepare dummy roi data
    rois_top_coord = np.random.uniform(low=0.0, high=0.75, size=(BATCH_SIZE, NUM_ROIS))
    rois_bottom_coord = (np.random.uniform(low=0.0, high=1.0, size=(BATCH_SIZE, NUM_ROIS)) *
                         (1.0 - rois_top_coord)) + rois_top_coord
    assert np.all(rois_bottom_coord <= 1.0)
    rois_left_coord = np.random.uniform(low=0.0, high=0.75, size=(BATCH_SIZE, NUM_ROIS))
    rois_right_coord = (np.random.uniform(low=0.0, high=1.0, size=(BATCH_SIZE, NUM_ROIS)) *
                        (1.0 - rois_left_coord)) + rois_left_coord
    assert np.all(rois_right_coord <= 1.0)
    roi_arr = np.stack([rois_top_coord, rois_left_coord, rois_bottom_coord, rois_right_coord], axis=2)
    sess = tf.Session()

    results = sess.run([split_features, split_rois, input_tensor],
                       feed_dict={input_tensor: random_imgs, rois_tensor: roi_arr})
    print("X")


if __name__ == "__main__":
    main()
