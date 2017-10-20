# -*- coding: utf-8 -*-
"""Top-level package for Tensorflow Datasets."""

__author__ = """Thibaut Mattio"""
__email__ = 'thibaut.mattio@gmail.com'
__version__ = '0.1.0'

from tf_datasets.datasets import mnist
# from tf_datasets.datasets import caltech_pedestrian
# from tf_datasets.datasets import cbsr_webface
# from tf_datasets.datasets import cifar10
# from tf_datasets.datasets import cifar100
# from tf_datasets.datasets import fddb
# from tf_datasets.datasets import flowers
# from tf_datasets.datasets import svhn
# from tf_datasets.datasets import wider_face

_dataset_name_map = {
    'mnist': mnist,
    # 'caltech_pedestrian': caltech_pedestrian,
    # 'cbsr_webface': cbsr_webface,
    # 'cifar10': cifar10,
    # 'cifar100': cifar100,
    # 'fddb': fddb,
    # 'flowers': flowers,
    # 'svhn': svhn,
    # 'wider_face': wider_face,
}


def get_dataset(dataset_name, dataset_dir):
  return _dataset_name_map[dataset_name](dataset_dir)
