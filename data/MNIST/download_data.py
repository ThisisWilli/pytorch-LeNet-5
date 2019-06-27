'''
@project : LeNet-5
@author  : Hoodie_Willi
@description: ${}
@time   : 2019-06-26 10:52:05
'''
"""`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
import os
from skimage import io
import torchvision.datasets.mnist as mnist
import struct
root="../"
train_set = (
    # mnist.read_image_file('D:\\PycharmProject\\LeNet-5\\data\\MNIST\\t10k-images-idx3-ubyte.gz'),
    mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte')),
    # mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte.gz')),
    mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
        )
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
        )
print("training set :", train_set[0].size())
print("test set :", test_set[0].size())
