# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import numpy as np
from tf_datasets.core.base_dataset import BaseDataset
from tf_datasets.core.dataset_utils import create_image_example
from tf_datasets.core.dataset_utils import ImageCoder


class MNISTDataset(BaseDataset):

  def __init__(self, dataset_dir):
    config_filepath = os.path.join(os.path.dirname(__file__), 'mnist.config')
    super().__init__(dataset_dir, config_filepath)

    self._image_size = 28
    self._image_channel = 1
    self._num_training = 60000
    self._num_validation = 10000
    self._coder = ImageCoder()

  def _get_data_points(self, download_dir):
    training = self.__get_data_points_from_mnist_files(
        os.path.join(download_dir, 'train-images-idx3-ubyte'),
        os.path.join(download_dir, 'train-labels-idx1-ubyte'),
        self._num_training)

    validation = self.__get_data_points_from_mnist_files(
        os.path.join(download_dir, 't10k-images-idx3-ubyte'),
        os.path.join(download_dir, 't10k-labels-idx1-ubyte'),
        self._num_validation)

    return training, validation

  def _data_point_to_example(self, data_point):
    label, image = data_point
    encoded = self._coder.encode_png(image)
    image_format = 'png'
    height, width, channels = (
        self._image_size,
        self._image_size,
        self._image_channel
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
    pass

  def __get_data_points_from_mnist_files(self, image_file, label_file,
                                         num_data_points):
    with open(label_file, 'rb') as f:
      f.read(8)
      buf = f.read(1 * num_data_points)
      labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    with open(image_file, 'rb') as f:
      f.read(16)
      buf = f.read(self._image_size * self._image_size *
                   num_data_points * self._image_channel)
      images = np.frombuffer(buf, dtype=np.uint8)
      images = images.reshape(
          num_data_points,
          self._image_size,
          self._image_size,
          self._image_channel
      )

    assert labels.shape[0] == images.shape[0]

    data_points = []
    for i in range(images.shape[0]):
      data_points.append((labels[i], images[i]))

    return data_points
