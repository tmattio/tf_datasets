# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import hashlib
import h5py
import tensorflow as tf
from tf_datasets.core import dataset_utils
from tf_datasets.core.base_dataset import BaseDataset
from tf_datasets.core.dataset_utils import create_dataset_split
from tf_datasets.core.dataset_utils import ImageCoder
from tf_datasets.core.download import download_http, extract_tgz


slim = tf.contrib.slim


def read_digit_struct(filepath):
  def _bbox_helper(attr):
    if (len(attr) > 1):
      attr = [mat_f[attr.value[j].item()].value[0][0]
              for j in range(len(attr))]
    else:
      attr = [attr.value[0][0]]
    return attr

  mat_f = h5py.File(filepath, 'r')
  bin_names = [mat_f[name_ref[0]].value for name_ref in mat_f[
      'digitStruct']['name']]
  bin_bboxes = [mat_f[bb.item()] for bb in mat_f['digitStruct']['bbox']]

  names = [''.join([chr(c[0]) for c in bin_name]) for bin_name in bin_names]
  names = [os.path.join(os.path.dirname(filepath), name) for name in names]

  bboxes = [{
      'name': names[i],
      'label': _bbox_helper(bbox['label']),
      'top': _bbox_helper(bbox['top']),
      'left': _bbox_helper(bbox['left']),
      'height': _bbox_helper(bbox['height']),
      'width': _bbox_helper(bbox['width']),
  } for i, bbox in enumerate(bin_bboxes)]

  return bboxes


class svhn(BaseDataset):
  base_url = 'http://ufldl.stanford.edu/housenumbers/'
  filenames = [
      'train.tar.gz',
      'test.tar.gz',
  ]

  num_training = 73257
  num_validation = 26032

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

  def __init__(self, dataset_dir):
    super().__init__(dataset_dir, class_names=self.class_names,
                     zero_based_labels=False)
    self.dataset_name = 'svhn'
    self.download_dir = os.path.join(self.dataset_dir, 'download')
    self._coder = ImageCoder()

  def download(self):
    try:
      os.makedirs(self.download_dir)
    except FileExistsError:
      pass

    for filename in self.filenames:
      output_path = os.path.join(self.download_dir, filename)
      if not os.path.exists(output_path):
        download_http(self.base_url + filename, output_path)

  def extract(self):
    for filename in self.filenames:
      output_path = os.path.join(self.download_dir, filename[:-7])
      if not os.path.exists(output_path):
        extract_tgz(
            os.path.join(self.download_dir, filename),
            self.download_dir
        )

  def convert(self):
    splits = self._get_data_points()
    split_names = ['train', 'validation']

    for split, split_name in zip(splits, split_names):
      create_dataset_split('svhn',
                           self.dataset_dir,
                           split_name,
                           split,
                           self._convert_to_example)

    self.write_label_file()

  def cleanup(self):
    shutil.rmtree(self.download_dir)

  def _get_data_points(self):
    train_filepath = os.path.join(
        self.download_dir, 'train', 'digitStruct.mat')
    train_points = read_digit_struct(train_filepath)

    eval_filepath = os.path.join(
        self.download_dir, 'test', 'digitStruct.mat')
    eval_points = read_digit_struct(eval_filepath)

    return train_points, eval_points

  def _convert_to_example(self, data_point):
    filename = data_point['name']
    with open(filename, 'rb') as f:
      encoded_image = f.read()

    decoded_image = self._coder.decode_jpeg(encoded_image)
    image_format = 'jpg'
    height, width, channels = decoded_image.shape
    key = hashlib.sha256(encoded_image).hexdigest()

    xmin = [p / width for p in data_point['left']]
    ymin = [p / height for p in data_point['top']]
    xmax = [(data_point['left'][i] + data_point['width'][i]) / width
            for i in range(len(data_point['left']))]
    ymax = [(data_point['top'][i] + data_point['height'][i]) / height
            for i in range(len(data_point['left']))]
    classes = [int(cl) for cl in data_point['label']]
    classes_text = [self.labels_to_class_names[cl].encode('utf8')
                    for cl in classes]

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_utils.int64_feature(height),
        'image/width': dataset_utils.int64_feature(width),
        'image/channels': dataset_utils.int64_feature(channels),
        'image/filename': dataset_utils.bytes_feature(
            filename.encode('utf8')),
        'image/key/sha256':
            dataset_utils.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_utils.bytes_feature(encoded_image),
        'image/format':
            dataset_utils.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_utils.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_utils.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_utils.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_utils.float_list_feature(ymax),
        'image/object/class/text':
            dataset_utils.bytes_list_feature(classes_text),
        'image/object/class/label':
            dataset_utils.int64_list_feature(classes),
    }))

  def load(self, split_name, reader=None):
    # TODO(tmattio): Implement the load methods
    pass
