import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
IMG_W = 256
IMG_H = 256
FEATURE_COUNT = 64
NUM_ROIS = 50

ROI_LIST = [(0.6, 0.7, 0.2, 0.3), (0.2, 0.1, 0.4, 0.4), (0.8, 0.8, 0.15, 0.125)]


# class RoIPooling:
#     @staticmethod
#     def roi_pool(feature_map, roi_list):
#         feature_map_width = int(feature_map.shape[2])
#         feature_map_height = int(feature_map.shape[1])
#         # Split feature maps and roi lists from the first dimension
#         split_features = tf.split(feature_map, BATCH_SIZE, axis=0)
#         split_features = [f_map[0, :] for f_map in split_features]
#         split_rois = tf.split(roi_list, BATCH_SIZE, axis=0)
#         split_rois = [roi_arr[0, :] for roi_arr in split_rois]
#         # Get cropped roi tensors
#         roi_tensors = []
#         for f_map_idx in range(BATCH_SIZE):
#             print("f_map_idx={0}".format(f_map_idx))
#             f_map = split_features[f_map_idx]
#             f_map_roi_tensors = []
#             for roi_idx in range(NUM_ROIS):
#                 roi_dims = split_rois[f_map_idx][roi_idx]
#                 h_start = tf.cast(feature_map_height * roi_dims[0], 'int32')
#                 w_start = tf.cast(feature_map_width * roi_dims[1], 'int32')
#                 h_end = tf.cast(feature_map_height * roi_dims[2], 'int32')
#                 w_end = tf.cast(feature_map_width * roi_dims[3], 'int32')
#                 roi_tensor = f_map[h_start:h_end, w_start:w_end, :]
#                 f_map_roi_tensors.append(roi_tensor)
#             roi_tensors.append(f_map_roi_tensors)
#         return split_features, split_rois, roi_tensors
#
#         # roi_feature_maps = []
#         # for roi_dims in roi_list:
#         #     # assert roi_dims[0] + roi_dims[2] <= 1.0 and roi_dims[1] + roi_dims[3] <= 1.0
#         #     h_start = tf.cast(feature_map_height * roi_dims[0], 'int32')
#         #     w_start = tf.cast(feature_map_width * roi_dims[1], 'int32')
#         #     h_end = tf.cast(feature_map_height * roi_dims[2], 'int32')
#         #     w_end = tf.cast(feature_map_width * roi_dims[3], 'int32')
#         #     roi_feature_map = x[:, h_start:h_end, w_start:w_end, :]
#         #     roi_feature_maps.append(roi_feature_map)
#         # print("X")


class RoIPooling:
    # def __init__(self, pooled_width, pooled_height):
    #     self.pooledWidth = pooled_width
    #     self.pooledHeight = pooled_height
    @staticmethod
    def pool_roi_single_f_map_single_roi(feature_map, roi, pooled_height, pooled_width):
        # Feature map: Single feature map
        # roi: Single region of interest
        feature_map_height = int(feature_map.shape[0])
        feature_map_width = int(feature_map.shape[1])
        h_start = tf.cast(feature_map_height * roi[0], 'int32')
        w_start = tf.cast(feature_map_width * roi[1], 'int32')
        h_end = tf.cast(feature_map_height * roi[2], 'int32')
        w_end = tf.cast(feature_map_width * roi[3], 'int32')
        region = feature_map[h_start:h_end, w_start:w_end, :]
        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width = w_end - w_start
        h_step = tf.cast(region_height / pooled_height, 'int32')
        w_step = tf.cast(region_width / pooled_width, 'int32')
        areas = [[(i * h_step, j * w_step,
                   (i + 1) * h_step if i + 1 < pooled_height else region_height,
                   (j + 1) * w_step if j + 1 < pooled_width else region_width)
                  for j in range(pooled_width)]
                 for i in range(pooled_height)]
        pooled_areas = [[tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0, 1]) for x in row]
                        for row in areas]
        pooled_features = tf.stack(pooled_areas)
        return pooled_features

    @staticmethod
    def pool_roi_single_f_map_many_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single feature map and more than one ROIs
        """

        def fn(roi):
            return RoIPooling.pool_roi_single_f_map_single_roi(feature_map, roi, pooled_height, pooled_width)

        pooled_areas = tf.map_fn(fn, rois, dtype=tf.float32)
        return pooled_areas

    @staticmethod
    def roi_pool(x, pooled_height, pooled_width):

        def fn(_x):
            return RoIPooling.pool_roi_single_f_map_many_rois(feature_map=_x[0], rois=_x[1],
                                                              pooled_height=pooled_height,
                                                              pooled_width=pooled_width)

        pooled_areas = tf.map_fn(fn, x, dtype=tf.float32)
        return pooled_areas


def get_dummy_roi_tensor():
    rois_top_coord = np.random.uniform(low=0.0, high=0.75, size=(BATCH_SIZE, NUM_ROIS))
    rois_bottom_coord = (np.random.uniform(low=0.25, high=1.0, size=(BATCH_SIZE, NUM_ROIS)) *
                         (1.0 - rois_top_coord)) + rois_top_coord
    assert np.all(rois_bottom_coord <= 1.0)
    rois_left_coord = np.random.uniform(low=0.0, high=0.75, size=(BATCH_SIZE, NUM_ROIS))
    rois_right_coord = (np.random.uniform(low=0.25, high=1.0, size=(BATCH_SIZE, NUM_ROIS)) *
                        (1.0 - rois_left_coord)) + rois_left_coord
    assert np.all(rois_right_coord <= 1.0)
    roi_arr = np.stack([rois_top_coord, rois_left_coord, rois_bottom_coord, rois_right_coord], axis=2)
    min_roi_height = np.min(rois_bottom_coord - rois_top_coord) * IMG_H
    min_roi_width = np.min(rois_right_coord - rois_left_coord) * IMG_W
    print("min_roi_height={0}".format(min_roi_height))
    print("min_roi_width={0}".format(min_roi_width))
    return roi_arr


def test_roi_pooling_layer():
    pooled_height = 7
    pooled_width = 7
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, IMG_W, IMG_H, FEATURE_COUNT])
    rois_tensor = tf.placeholder(tf.float32, shape=(None, NUM_ROIS, 4))
    sess = tf.Session()
    for iteration_id in range(1000):
        # Prepare dummy image data
        random_imgs = np.random.uniform(size=(BATCH_SIZE, IMG_W, IMG_H, FEATURE_COUNT))
        # Prepare dummy roi data
        roi_arr = get_dummy_roi_tensor()

        # Network
        roi_output = RoIPooling.roi_pool(x=[input_tensor, rois_tensor],
                                         pooled_height=pooled_height, pooled_width=pooled_width)

        # Get the Tensorflow result.
        results = sess.run([roi_output], feed_dict={input_tensor: random_imgs, rois_tensor: roi_arr})
        tf_result = results[0]

        # Calculate a manual simulation, with very basic for loops
        pooled_imgs = []
        for img_idx in range(random_imgs.shape[0]):
            feature_map = random_imgs[img_idx]
            img_rois = roi_arr[img_idx]
            pooled_maps = []
            for roi_idx in range(img_rois.shape[0]):
                roi = img_rois[roi_idx]
                feature_map_height = int(feature_map.shape[0])
                feature_map_width = int(feature_map.shape[1])
                h_start = int(feature_map_height * roi[0])
                w_start = int(feature_map_width * roi[1])
                h_end = int(feature_map_height * roi[2])
                w_end = int(feature_map_width * roi[3])
                region = feature_map[h_start:h_end, w_start:w_end, :]
                # Divide the region into non overlapping areas
                region_height = h_end - h_start
                region_width = w_end - w_start
                h_step = int(region_height / pooled_height)
                w_step = int(region_width / pooled_width)
                pooled_map = np.zeros(shape=(pooled_height, pooled_width, region.shape[-1]))
                for i in range(pooled_height):
                    delta_h = h_step if i != pooled_height - 1 else region_height - i * h_step
                    for j in range(pooled_width):
                        delta_w = w_step if j != pooled_width - 1 else region_width - j * w_step
                        sub_region = region[i*h_step:i*h_step + delta_h, j*w_step:j*w_step + delta_w, :]
                        max_val = np.max(sub_region, axis=(0, 1))
                        pooled_map[i, j, :] = max_val
                pooled_maps.append(pooled_map)
            pooled_maps = np.stack(pooled_maps, axis=0)
            pooled_imgs.append(pooled_maps)
        np_result = np.stack(pooled_imgs, axis=0)
        assert np.allclose(tf_result, np_result)
        print("Passed iteration {0}".format(iteration_id))


def main():
    test_roi_pooling_layer()
    print("X")


if __name__ == "__main__":
    main()

