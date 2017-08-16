# -*- coding: utf-8 -*-
"""Top-level package for Tensorflow Datasets."""

__author__ = """Thibaut Mattio"""
__email__ = 'thibaut.mattio@gmail.com'
__version__ = '0.1.0'

from tf_datasets.datasets.caltech_pedestrian import caltech_pedestrian
from tf_datasets.datasets.cifar10 import cifar10
from tf_datasets.datasets.cifar100 import cifar100
from tf_datasets.datasets.fddb import fddb
from tf_datasets.datasets.flowers import flowers
from tf_datasets.datasets.mnist import mnist
from tf_datasets.datasets.svhn import svhn
from tf_datasets.datasets.wider_face import wider_face

_dataset_name_map = {
    'caltech_pedestrian': caltech_pedestrian,
    'cifar10': cifar10,
    'cifar100': cifar100,
    'fddb': fddb,
    'flowers': flowers,
    'mnist': mnist,
    'svhn': svhn,
    'wider_face': wider_face,
}


def get_dataset(dataset_name, dataset_dir):
    return _dataset_name_map[dataset_name](dataset_dir)
