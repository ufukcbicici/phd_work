import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL
# import torch
from torchvision import transforms
from tf_2_cign.utilities.utilities import Utilities

CIFAR_SIZE = 32
library = "torch"
tf_rng = tf.random.Generator.from_seed(123, alg='philox')
# seed = tf_rng.make_seeds(2)

train_data, test_data = tf.keras.datasets.cifar10.load_data()
X_ = np.concatenate([train_data[0], test_data[0]], axis=0)
Utilities.pickle_save_to_file(path="cifar10.sav", file_content=X_)

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


# normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

def visualize(original, augmented_tf, augmented_torch):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # plt.subplot(1, 2, 1)
    ax1.set_title('Original image')
    ax1.imshow(original)

    # plt.subplot(1, 2, 2)
    ax2.set_title('Tensorflow Augmented image')
    ax2.imshow(augmented_tf)

    # plt.subplot(1, 2, 3)
    ax3.set_title('Pytorch Augmented image')
    ax3.imshow(augmented_torch)


def augment_training_image_fn_with_seed(image):
    seed = tf_rng.make_seeds(2)[0]
    image_normalized = augment_training_image_fn(image=image, seed=seed)
    return image_normalized


def augment_training_image_fn(image, seed):
    # print(image.__class__)
    image = tf.image.resize_with_crop_or_pad(image, CIFAR_SIZE + 8, CIFAR_SIZE + 8)
    image = tf.image.stateless_random_crop(image, [CIFAR_SIZE, CIFAR_SIZE, 3], seed)
    image = tf.image.stateless_random_flip_left_right(image, seed)
    image = (tf.cast(image, dtype=tf.float32) / 255.0)
    mean = tf.convert_to_tensor((0.4914, 0.4822, 0.4465))
    std = tf.convert_to_tensor((0.2023, 0.1994, 0.2010))
    mean = tf.expand_dims(tf.expand_dims(mean, axis=0), axis=0)
    std = tf.expand_dims(tf.expand_dims(std, axis=0), axis=0)
    image_normalized = (image - mean) / std
    return image_normalized


def augment_test_image_fn(image):
    image = (tf.cast(image, dtype=tf.float32) / 255.0)
    mean = tf.convert_to_tensor((0.4914, 0.4822, 0.4465))
    std = tf.convert_to_tensor((0.2023, 0.1994, 0.2010))
    mean = tf.expand_dims(tf.expand_dims(mean, axis=0), axis=0)
    std = tf.expand_dims(tf.expand_dims(std, axis=0), axis=0)
    image_normalized = (image - mean) / std
    return image_normalized


X_means_tf = []
X_means_torch = []
# tf_sample_count = 0.0
# torch_sample_count = 0.0

train_dataset_tf = tf.data.Dataset.from_tensor_slices((X_,))
train_dataset_tf = train_dataset_tf.map(map_func=augment_training_image_fn_with_seed, num_parallel_calls=None)

test_dataset_tf = tf.data.Dataset.from_tensor_slices((X_,))
test_dataset_tf = test_dataset_tf.map(map_func=augment_test_image_fn, num_parallel_calls=None)

for outer_idx in range(100):
    # Tensorflow augmentations
    tf_samples = []
    print("Tensorflow Augmentations")
    for x_tf in tqdm(test_dataset_tf):
        tf_samples.append(x_tf.numpy())
    tf_samples = np.stack(tf_samples, axis=0)
    tf_mean = np.mean(tf_samples, axis=0)
    X_means_tf.append(tf_mean)

    # Torch augmentations
    torch_samples = []
    print("Torch Augmentations")
    for idx in tqdm(range(X_.shape[0])):
        x_3d = X_[idx]
        img = PIL.Image.fromarray(x_3d).convert('RGB')
        x_torch = transform_test(img).numpy()
        x_torch = np.transpose(x_torch, axes=(1, 2, 0))
        torch_samples.append(x_torch)
    torch_samples = np.stack(torch_samples, axis=0)
    torch_mean = np.mean(torch_samples, axis=0)
    X_means_torch.append(torch_mean)

    tf_mean_image = np.mean(np.stack(X_means_tf, axis=0), axis=0)
    torch_mean_image = np.mean(np.stack(X_means_torch, axis=0), axis=0)
    print("Iteration:{0} Mean Difference:{1}".format(outer_idx, np.linalg.norm(tf_mean_image - torch_mean_image)))
