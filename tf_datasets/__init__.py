# -*- coding: utf-8 -*-
"""Top-level package for Tensorflow Datasets."""

__author__ = """Thibaut Mattio"""
__email__ = 'thibaut.mattio@gmail.com'
__version__ = '0.1.0'

from tf_datasets.datasets.mnist import mnist
from tf_datasets.datasets.ciraf10 import ciraf10
from tf_datasets.datasets.ciraf100 import ciraf100
from tf_datasets.datasets.flowers import flowers
from tf_datasets.datasets.fddb import fddb
from tf_datasets.datasets.wider_face import wider_face

_dataset_name_map = {
    'mnist': mnist,
    'ciraf10': ciraf10,
    'ciraf100': ciraf100,
    'flowers': flowers,
    'fddb': fddb,
    'wider_face': wider_face,
}


def get_dataset(dataset_name, dataset_dir):
    return _dataset_name_map[dataset_name](dataset_dir)
