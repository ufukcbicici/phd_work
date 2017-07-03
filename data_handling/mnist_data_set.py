import struct
from array import array
from auxillary.constants import DatasetTypes
from data_handling.data_set import DataSet


class MnistDataSet(DataSet):
    def __init__(self):
        super().__init__()
        self.testImagesPath = "../data/t10k-images-idx3-ubyte"
        self.testLabelsPath = "../data/t10k-labels-idx1-ubyte"
        self.trainImagesPath = "../data/train-images-idx3-ubyte"
        self.trainLabelsPath = "../data/train-labels-idx1-ubyte"

    def load_dataset(self):
        train_images, train_labels = self.load(path_img=self.trainImagesPath, path_lbl=self.trainLabelsPath)
        test_images, test_labels = self.load(path_img=self.testImagesPath, path_lbl=self.testLabelsPath)

    # PRIVATE METHODS
    # Load method taken from https://github.com/sorki/python-mnist/blob/master/mnist/loader.py
    def load(self, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels
