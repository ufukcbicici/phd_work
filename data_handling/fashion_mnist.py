import os

from data_handling.mnist_data_set import MnistDataSet


class FashionMnistDataSet(MnistDataSet):
    def __init__(self,
                 validation_sample_count,
                 save_validation_as=None,
                 load_validation_from=None,
                 test_images_path=os.path.join(os.getcwd(), "..\\data\\fashion_mnist\\t10k-images-idx3-ubyte"),
                 test_labels_path=os.path.join(os.getcwd(), "..\\data\\fashion_mnist\\t10k-labels-idx1-ubyte"),
                 training_images_path=os.path.join(os.getcwd(), "..\\data\\fashion_mnist\\train-images-idx3-ubyte"),
                 training_labels_path=os.path.join(os.getcwd(), "..\\data\\fashion_mnist\\train-labels-idx1-ubyte")):
        super().__init__(validation_sample_count=validation_sample_count, save_validation_as=save_validation_as,
                         load_validation_from=load_validation_from, test_images_path=test_images_path,
                         test_labels_path=test_labels_path, training_images_path=training_images_path,
                         training_labels_path=training_labels_path)

    def get_label_def(self, label):
        if label == 0:
            return "T-shirt/top"
        elif label == 1:
            return "Trouser"
        elif label == 2:
            return "Pullover"
        elif label == 3:
            return "Dress"
        elif label == 4:
            return "Coat"
        elif label == 5:
            return "Sandal"
        elif label == 6:
            return "Shirt"
        elif label == 7:
            return "Sneaker"
        elif label == 8:
            return "Bag"
        elif label == 9:
            return "Ankle boot"
        else:
            raise Exception("Unexpected label.")