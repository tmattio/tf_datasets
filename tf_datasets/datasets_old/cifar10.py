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
  labels = data[b'labels']

  return [(images[i], labels[i]) for i in range(num_images)]


class cifar10(BaseDataset):
  image_size = 32
  image_channel = 3
  num_train_files = 5
  class_names = [
      'airplane',
      'automobile',
      'bird',
      'cat',
      'deer',
      'dog',
      'frog',
      'horse',
      'ship',
      'truck',
  ]
  public_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

  def __init__(self, dataset_dir):
    super().__init__(dataset_dir, self.class_names, zero_based_labels=True)
    self.dataset_name = 'cifar10'
    self.download_dir = os.path.join(self.dataset_dir, 'download')
    self._coder = ImageCoder()

  def download(self):
    try:
      os.makedirs(self.download_dir)
    except FileExistsError:
      pass

    output_path = os.path.join(self.download_dir, 'cifar-10-python.tar.gz')
    if not os.path.exists(output_path):
      download_http(self.public_url, output_path)

  def extract(self):
    output_path = os.path.join(self.download_dir, 'cifar-10-batches-py')
    if not os.path.exists(output_path):
      extract_tgz(
          os.path.join(self.download_dir, 'cifar-10-python.tar.gz'),
          self.download_dir
      )

  def _get_data_points(self):
    train_datapoints = []
    for i in range(self.num_train_files):
      filename = os.path.join(self.download_dir,
                              'cifar-10-batches-py',
                              'data_batch_%d' % (i + 1))
      train_datapoints += _get_data_points_from_cifar_file(filename)

    test_filename = os.path.join(self.download_dir,
                                 'cifar-10-batches-py',
                                 'test_batch')
    val_datapoints = _get_data_points_from_cifar_file(test_filename)

    return np.stack(train_datapoints), val_datapoints

  def convert(self):
    splits = self._get_data_points()
    split_names = ['train', 'validation']

    for split, split_name in zip(splits, split_names):
      create_dataset_split('cifar10',
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
