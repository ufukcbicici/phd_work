import numpy as np
import cv2


def normalize_depth_array(depth_array):
    # Find minimum and maximum distances in the depth array
    min_distance = np.min(depth_array)
    max_distance = np.max(depth_array)
    # Calculate the interval length
    interval_length = max_distance - min_distance
    # We are going to bin the distances and quantize them into 256 equal length bins, in order to convert the
    # depth map to a grayscale image. Firstly, we find the length of a single bin.
    bin_length = interval_length / 256.0
    # We are going to use numpy's digitize method, which quantizes an array into bins; provided a monotonically
    # increasing or decreasing bin array. We need to first build this bin array. It should be built like that:
    # [min_distance, min_distance + bin_length, min_distance + 2*bin_length, ..., min_distance + 256*bin_length]
    bins = np.arange(257).astype(np.float64)
    bins = bin_length * bins
    bins = bins + min_distance
    assert np.allclose([min_distance, max_distance], [bins[0], bins[-1]])
    # We are going to apply digitize method. It does that by applying a binary search; a O(log N) operation, instead
    # of linear iteration of the whole array. right=False parameter applies binning as: bins[i-1] <= x < bins[i]
    gray_scale = np.digitize(depth_array, bins, right=False)
    # np.digitize() returns bin indices starting from 1. We, therefore, subtract 1 from every element.
    # We clip the bin indices into [0,255] range. Since min_distance = bins[0], this creates an edge case where the
    # corresponding entry gets 0 index and when subtracted by 1, becomes -1. We need to prevent that.
    gray_scale = np.clip(gray_scale - 1, a_min=0, a_max=255)
    # The corresponding grayscale image is returned.

    # Test if the above logic works.
    for r in range(0, depth_array.shape[0]):
        for c in range(0, depth_array.shape[1]):
            gray_val = int((depth_array[r, c] - min_distance) / bin_length)
            # (gray_val == 256 and gray_scale[r, c] == 255) part of the assertion comes from the fact that
            # int((max_distance - min_distance) / bin_length) might give 256 due to floating point inprecision.
            assert gray_val == gray_scale[r, c] or (gray_val == 256 and gray_scale[r, c] == 255)

    return gray_scale


def depth_array_algorithm(width, height):
    da = np.random.uniform(low=0.5, high=10.0, size=(height, width))
    return da


if __name__ == "__main__":
    da = depth_array_algorithm(width=640, height=480)
    gs = normalize_depth_array(da)
