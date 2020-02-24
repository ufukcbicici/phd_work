class Constants:
    def __init__(self):
        pass

    # Model Name
    MODEL_NAME = "ResNet_Detector"

    # Image Scales
    IMG_WIDTHS = [640]

    # Test / Whole Dataset Ratio
    TEST_RATIO = 0.15

    # Bounding Box Clustering:
    MAX_INCLUSIVENESS_BB = 0.975 # Coverage
    MAX_MEDOID_COUNT = 8
    MAX_IOU_DISTANCE = 0.5

    # Object Detection and Testing: Positive and Negative Bounding Box Selection
    POSITIVE_IOU_THRESHOLD = 0.5
    NEGATIVE_IOU_THRESHOLD = 0.4

    # FAST RCNN - RoI Sampling Parameters
    POSITIVE_PROPOSAL_SAMPLING_STD = 20.0
    IMAGE_COUNT_PER_BATCH = 2
    ROI_SAMPLE_COUNT_PER_IMAGE = 128
    POSITIVE_SAMPLE_RATIO_PER_IMAGE = 0.25

    # ROI POOLING PARAMETERS
    POOLED_WIDTH = 7
    POOLED_HEIGHT = 7

    # BackBone ResNet Parameters
    NUM_OF_RESIDUAL_UNITS = 3
    NUM_OF_FEATURES_PER_BLOCK = [4, 8, 16, 32]
    FIRST_CONV_FILTER_SIZE = 3
    RELU_LEAKINESS = 0.1
    FILTER_STRIDES = [1, 2, 2]
    ACTIVATE_BEFORE_RESIDUALS = [True, False, False]
    BATCH_NORM_DECAY = 0.9

    # Detector ResNet Parameters
    DETECTOR_NUM_OF_RESIDUAL_UNITS = 3
    DETECTOR_NUM_OF_FEATURES_PER_BLOCK = [32, 64, 128]
    DETECTOR_FIRST_CONV_FILTER_SIZE = 3
    DETECTOR_RELU_LEAKINESS = 0.1
    DETECTOR_FILTER_STRIDES = [1, 2]
    DETECTOR_ACTIVATE_BEFORE_RESIDUALS = [True, False]
    DETECTOR_BATCH_NORM_DECAY = 0.9

    # Roi Feature Vector Transformations
    CLASSIFIER_HIDDEN_LAYERS = [128]

    # L2 Norm Regularizer Strength
    L2_LAMBDA = 0.0005

    # Parameters for testing
    STRIDE_WIDTH = 10
    STRIDE_HEIGHT = 25
    TEST_BATCH_SIZE = 5000
    NMS_THRESHOLD = 0.5
    RESULT_REPORTING_PERIOD = 10 # In Epochs
    MODEL_SAVING_PERIOD = 100 # In Epochs
