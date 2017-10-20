# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import shutil
import numpy as np
import tensorflow as tf
from tf_datasets.core.download import download_http, extract_gzip
from tf_datasets.core.base_dataset import BaseDataset
from tf_datasets.core.dataset_utils import create_image_example
from tf_datasets.core.dataset_utils import create_dataset_split
from tf_datasets.core.dataset_utils import ImageCoder


slim = tf.contrib.slim


class pascal_voc_2017(BaseDataset):
  class_names = [
      'zero',
      'one',
      'two',
      'three',
      'four',
      'five',
      'size',
      'seven',
      'eight',
      'nine',
  ]
  filenames = ['VOCtest_06-Nov-2007',
               'VOCtrainval_06-Nov-2007',
               'VOCdevkit_08-Jun-2007']

  def __init__(self, dataset_dir):
    super().__init__(dataset_dir, self.class_names, zero_based_labels=True)
    self.dataset_name = 'mnist'
    self.download_dir = os.path.join(self.dataset_dir, 'download')
    self._coder = ImageCoder()

  def download(self):
    try:
      os.makedirs(self.download_dir)
    except FileExistsError:
      pass

    data_url = 'http://yann.lecun.com/exdb/mnist/'
    for filename in self.filenames:
      output_path = os.path.join(self.download_dir, filename)
      if not os.path.exists(output_path):
        download_http(data_url + filename, output_path)

  def extract(self):
    for filename in self.filenames:
      output_path = os.path.join(self.download_dir, filename[:-3])
      if not os.path.exists(output_path):
        extract_gzip(os.path.join(self.download_dir,
                                  filename), self.download_dir)

  def convert(self):
    splits = self._get_data_points()
    split_names = ['train', 'validation']

    for split, split_name in zip(splits, split_names):
      create_dataset_split('mnist',
                           self.dataset_dir,
                           split_name,
                           split,
                           self._convert_to_example)

    self.write_label_file()

  def cleanup(self):
    shutil.rmtree(self.download_dir)

  def _get_data_points(self):
    def _get_data_points_from_mnist_files(image_file, label_file,
                                          num_data_points):
      with open(os.path.join(self.download_dir, label_file), 'rb') as f:
        f.read(8)
        buf = f.read(1 * num_data_points)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

      with open(os.path.join(self.download_dir, image_file), 'rb') as f:
        f.read(16)
        buf = f.read(self.image_size * self.image_size *
                     num_data_points * self.image_channel)
        images = np.frombuffer(buf, dtype=np.uint8)
        images = images.reshape(
            num_data_points,
            self.image_size,
            self.image_size,
            self.image_channel
        )

      assert labels.shape[0] == images.shape[0]

      data_points = []
      for i in range(images.shape[0]):
        data_points.append((labels[i], images[i]))

      return data_points

    training = _get_data_points_from_mnist_files(
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        self.num_training)

    validation = _get_data_points_from_mnist_files(
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte',
        self.num_validation)

    return training, validation

  def _convert_to_example(self, data_point):
    label, image = data_point
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
