# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import shutil
import sys
from six.moves import cPickle
import numpy as np
import tensorflow as tf
from tf_datasets.core.download import download_http, extract_tgz
from tf_datasets.core.base_dataset import BaseDataset
from tf_datasets.core.dataset_utils import create_image_example
from tf_datasets.core.dataset_utils import create_dataset_split
from tf_datasets.core.dataset_utils import ImageCoder


slim = tf.contrib.slim


def _get_data_points_from_cifar_file(filepath):
  with open(filepath, 'rb') as f:
    if sys.version_info < (3,):
      data = cPickle.load(f)
    else:
      data = cPickle.load(f, encoding='bytes')

  images = data[b'data']
  num_images = images.shape[0]

  images = images.reshape((num_images, 3, 32, 32))
  images = [np.squeeze(image).transpose((1, 2, 0)) for image in images]
  labels = data[b'fine_labels']

  return [(images[i], labels[i]) for i in range(num_images)]


class cifar100(BaseDataset):
  image_size = 32
  image_channel = 3
  class_names = ['apples',
                 'aquarium_fish',
                 'baby',
                 'bear',
                 'beaver',
                 'bed',
                 'bee',
                 'beetle',
                 'bicycle',
                 'bottles',
                 'bowls',
                 'boy',
                 'bridge',
                 'bus',
                 'butterfly',
                 'camel',
                 'cans',
                 'castle',
                 'caterpillar',
                 'cattle',
                 'chair',
                 'chimpanzee',
                 'clock',
                 'cloud',
                 'cockroach',
                 'computer_keyboard',
                 'couch',
                 'crab',
                 'crocodile',
                 'cups',
                 'dinosaur',
                 'dolphin',
                 'elephant',
                 'flatfish',
                 'forest',
                 'fox',
                 'girl',
                 'hamster',
                 'house',
                 'kangaroo',
                 'lamp',
                 'lawn_mower',
                 'leopard',
                 'lion',
                 'lizard',
                 'lobster',
                 'man',
                 'maple',
                 'motorcycle',
                 'mountain',
                 'mouse',
                 'mushrooms',
                 'oak',
                 'oranges',
                 'orchids',
                 'otter',
                 'palm',
                 'pears',
                 'pickup_truck',
                 'pine',
                 'plain',
                 'plates',
                 'poppies',
                 'porcupine',
                 'possum',
                 'rabbit',
                 'raccoon',
                 'ray',
                 'road',
                 'rocket',
                 'roses',
                 'sea',
                 'seal',
                 'shark',
                 'shrew',
                 'skunk',
                 'skyscraper',
                 'snail',
                 'snake',
                 'spider',
                 'squirrel',
                 'streetcar',
                 'sunflowers',
                 'sweet_peppers',
                 'table',
                 'tank',
                 'telephone',
                 'television',
                 'tiger',
                 'tractor',
                 'train',
                 'trout',
                 'tulips',
                 'turtle',
                 'wardrobe',
                 'whale',
                 'willow',
                 'wolf',
                 'woman',
                 'worm']
  public_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

  def __init__(self, dataset_dir):
    super().__init__(dataset_dir, self.class_names, zero_based_labels=True)
    self.dataset_name = 'cifar100'
    self.download_dir = os.path.join(self.dataset_dir, 'download')
    self._coder = ImageCoder()

  def download(self):
    try:
      os.makedirs(self.download_dir)
    except FileExistsError:
      pass

    output_path = os.path.join(
        self.download_dir, 'cifar-100-python.tar.gz')
    if not os.path.exists(output_path):
      download_http(self.public_url, output_path)

  def extract(self):
    output_path = os.path.join(self.download_dir, 'cifar-100-python')
    if not os.path.exists(output_path):
      extract_tgz(
          os.path.join(self.download_dir, 'cifar-100-python.tar.gz'),
          self.download_dir
      )

  def _get_data_points(self):
    train_filename = os.path.join(self.download_dir,
                                  'cifar-100-python',
                                  'train')
    train_datapoints = _get_data_points_from_cifar_file(train_filename)

    test_filename = os.path.join(self.download_dir,
                                 'cifar-100-python',
                                 'test')
    val_datapoints = _get_data_points_from_cifar_file(test_filename)

    return train_datapoints, val_datapoints

  def convert(self):
    splits = self._get_data_points()
    split_names = ['train', 'validation']

    for split, split_name in zip(splits, split_names):
      create_dataset_split('cifar100',
                           self.dataset_dir,
                           split_name,
                           split,
                           self._convert_to_example)

    self.write_label_file()

  def cleanup(self):
    shutil.rmtree(self.download_dir)

  def _convert_to_example(self, data_point):
    image, label = data_point
    encoded = self._coder.encode_png(image)
    image_format = 'png'
    height, width, channels = (
        self.image_size,
        self.image_size,
        self.image_channel
    )
    class_name = self.labels_to_class_names[label]
    key = hashlib.sha256(encoded).hexdigest()

    return create_image_example(height,
                                width,
                                channels,
                                key,
                                encoded,
                                image_format,
                                class_name,
                                label)

  def load(self, split_name, reader=None):
    # TODO(tmattio): Implement the load methods
    pass
