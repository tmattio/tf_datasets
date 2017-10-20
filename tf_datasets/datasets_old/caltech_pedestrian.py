# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tensorflow as tf
from tf_datasets.core.base_dataset import BaseDataset
from tf_datasets.core.dataset_utils import ImageCoder
from tf_datasets.core.download import download_http, extract_tgz, extract_zip


slim = tf.contrib.slim


class caltech_pedestrian(BaseDataset):

  base_url = 'http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/'
  filenames = [
      'set00.tar',
      'set01.tar',
      'set02.tar',
      'set03.tar',
      'set04.tar',
      'set05.tar',
      'set06.tar',
      'set07.tar',
      'set08.tar',
      'set09.tar',
      'set10.tar',
  ]
  annotation_filename = 'annotations.zip'

  def __init__(self, dataset_dir):
    super().__init__(dataset_dir, class_names=[
        'pedistrian', ], zero_based_labels=False)
    self.dataset_name = 'caltech_pedestrian'
    self.download_dir = os.path.join(self.dataset_dir, 'download')
    self._coder = ImageCoder()

  def download(self):
    try:
      os.makedirs(self.download_dir)
    except FileExistsError:
      pass

    for filename in self.filenames + [self.annotation_filename, ]:
      output_path = os.path.join(self.download_dir, filename)
      if not os.path.exists(output_path):
        download_http(self.base_url + filename, output_path)

  def extract(self):
    for filename in self.filenames:
      output_path = os.path.join(self.download_dir, filename[:-4])
      if not os.path.exists(output_path):
        extract_tgz(os.path.join(self.download_dir,
                                 filename), self.download_dir)

    output_path = os.path.join(
        self.download_dir, self.annotation_filename[:-4])
    if not os.path.exists(output_path):
      extract_zip(os.path.join(self.download_dir,
                               self.annotation_filename), self.download_dir)

  def convert(self):
    pass

  def cleanup(self):
    shutil.rmtree(self.download_dir)

  def _get_data_points(self):
    pass

  def _convert_to_example(self, data_point):
    pass

  def load(self, split_name, reader=None):
    # TODO(tmattio): Implement the load methods
    pass
