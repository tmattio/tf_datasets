# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tensorflow as tf
from tf_datasets.core.base_dataset import BaseDataset
from tf_datasets.core.dataset_utils import ImageCoder
from tf_datasets.core.download import download_ftp
from tf_datasets.core.download import extract_zip


slim = tf.contrib.slim

_FTP_HOST = 'cbsr.ia.ac.cn'
_FTP_PORT = 21
_FTP_USER = 'xxxxxxxx'
_FTP_PASSWD = 'xxxxxxxx'
_FTP_BLOCK_SIZE = 1024


class cbsr_webface(BaseDataset):

  filename = 'CASIA-WebFace.zip'

  def __init__(self, dataset_dir):
    super().__init__(dataset_dir,
                     class_names=['face', ],
                     zero_based_labels=False)
    self.dataset_name = 'fddb'
    self.download_dir = os.path.join(self.dataset_dir, 'download')
    self._coder = ImageCoder()

  def download(self):
    output_path = os.path.join(self.download_dir, self.filename)
    if not os.path.exists(output_path):
      download_ftp(_FTP_HOST,
                   self.filename,
                   output_path,
                   _FTP_USER,
                   _FTP_PASSWD,
                   _FTP_PORT)

  def extract(self):
    output_path = os.path.join(self.download_dir, self.filename)
    extract_zip(output_path, self.download_dir)

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
