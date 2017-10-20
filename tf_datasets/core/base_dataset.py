# -*- coding: utf-8 -*-
import os
import shutil
from abc import ABCMeta, abstractmethod
from google.protobuf import text_format

from ..protos import dataset_config_pb2
from ..protos import public_file_pb2

from .dataset_utils import split_dataset
from .dataset_utils import create_dataset_split
from .dataset_utils import write_label_file

from .download_utils import download_http
from .download_utils import download_google_drive
from .download_utils import download_ftp
from .download_utils import extract_tgz
from .download_utils import extract_zip
from .download_utils import extract_gzip

_download_protocol_to_function_map = {
    public_file_pb2.PublicFile.HTTP: download_http,
    public_file_pb2.PublicFile.FTP: download_ftp,
    public_file_pb2.PublicFile.GOOGLE_DRIVE: download_google_drive,
    public_file_pb2.PublicFile.SSH: None,
}

_archive_type_to_function_map = {
    public_file_pb2.PublicFile.ArchiveInfo.AUTO: extract_tgz,
    public_file_pb2.PublicFile.ArchiveInfo.GZIP: extract_gzip,
    public_file_pb2.PublicFile.ArchiveInfo.ZIP: extract_zip,
}


class BaseDataset(metaclass=ABCMeta):
  """TODO (tmattio): Complete docstring
  """

  def __init__(self, dataset_dir, config_filepath):
    """TODO (tmattio): Complete docstring
    """
    if not os.path.exists(config_filepath):
      raise FileNotFoundError(
          "The configuration file for the MNIST dataset does not exist."
      )

    self.dataset_dir = dataset_dir

    with open(config_filepath, 'r') as f:
      config_proto_text = f.read()

    self.config = dataset_config_pb2.DatasetConfig()
    text_format.Merge(config_proto_text, self.config)

    # If the download path is relative in the configuration,
    # make it relative to the directory root directory
    if not os.path.isabs(self.config.download_dir):
      self.config.download_dir = os.path.join(
          self.dataset_dir, self.config.download_dir)

    try:
      os.makedirs(dataset_dir)
    except FileExistsError:
      pass

  @property
  def class_names(self):
    """Get the list of the classes in the dataset.

    Returns:
      A list or tuple of the classes names of this dataset.
    """
    return self.config.class_names

  @class_names.setter
  def set_class_names(self, class_names):
    """Sets the list of the classes in the dataset.
    """
    self.config.class_names = class_names

  @property
  def num_classes(self):
    """Get the total number of classes in this dataset.

    Returns:
      An int of the number of classes
    """
    return len(self.class_names)

  @property
  def class_names_to_labels(self):
    """Build an association of the dataset classes to their ID.

    Returns:
      A dictionnary of the classes names mapped to the labels.
    """
    if self.config.zero_based_labels:
      return dict(zip(self.class_names, range(self.num_classes)))
    else:
      return dict(zip(self.class_names, range(1, self.num_classes + 1)))

  @property
  def labels_to_class_names(self):
    """Build an association of the dataset labels to their names.

    Returns:
      A dictionnary of the labels mapped to the classes names.
    """
    if self.config.zero_based_labels:
      return dict(zip(range(self.num_classes), self.class_names))
    else:
      return dict(zip(range(1, self.num_classes + 1), self.class_names))

  def download(self, force=False):
    """TODO (tmattio): Complete docstring
    """
    try:
      os.makedirs(self.config.download_dir)
    except FileExistsError:
      pass

    for public_file in self.config.public_files:
      output_path = os.path.join(
          self.config.download_dir, public_file.filename)
      if force or not os.path.exists(output_path):
        download_func = _download_protocol_to_function_map[
            public_file.protocol]
        download_func(
            public_file.uri,
            output_path,
            public_file.username,
            public_file.password
        )

  def extract(self, force=False):
    """TODO (tmattio): Complete docstring
    """
    for public_file in self.config.public_files:
      if not public_file.is_archive:
        continue

      output_path = os.path.join(
          self.config.download_dir,
          public_file.archive_info.extracted_filename
      )
      if not os.path.exists(output_path):
        extract_func = _archive_type_to_function_map[
            public_file.archive_info.type]
        archive_path = os.path.join(
            self.config.download_dir, public_file.filename)
        extract_func(
            archive_path,
            self.config.download_dir
        )

  def cleanup(self):
    """TODO (tmattio): Complete docstring
    """
    shutil.rmtree(self.config.download_dir)

  def convert(self):
    """Convert the data points from the source format to TF-Records files.

    If the dataset already exist, this will not generate new tfrecord files.

    This will split the datasets from the data source into a training set and a
    validation set. This will then convert the two set separately.

    This will also generate a label file containing all the classes names and
    labels of this dataset.

    The generated tfrecords and the labels file can be found in
    `self.dataset_dir`.
    """
    if self.config.custom_splits:
      splits = self._get_data_points(self.config.download_dir)

      if len(splits) != len(self.config.splits):
        raise ValueError(
            "The number of splits given is not the same as the config file.")

      split_num = map(lambda s: s.data_point_num, self.config.splits)
      if sum(map(len, splits)) != sum(split_num):
        raise ValueError(
            "The number of total data point is not the same as the config file.")
    else:
      split_ratios = map(lambda s: s.ratios, self.config.splits)
      if sum(split_ratios) != 1.0:
        raise ValueError("The sum of the ratios does not equal 1.")

      data_points = self._get_data_points(self.config.download_dir)
      splits = split_dataset(data_points, split_ratios)

    for split, split_config in zip(splits, self.config.splits):
      create_dataset_split(
          self.config.name,
          self.dataset_dir,
          split_config.name,
          split,
          self._data_point_to_example,
          split_config.num_shards,
          split_config.num_threads
      )

    write_label_file(self.labels_to_class_names, self.dataset_dir)

  @abstractmethod
  def load(self, split_name, reader=None):
    """Gets a dataset tuple with instructions for reading the data points.

    Args:
      split_name: A train/test split name.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """
    pass

  @abstractmethod
  def _get_data_points(self, download_dir):
    """TODO (tmattio): Complete docstring
    """
    pass

  @abstractmethod
  def _data_point_to_example(self, data_point):
    """TODO (tmattio): Complete docstring
    """
    pass
