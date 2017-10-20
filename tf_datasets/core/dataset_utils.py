# -*- coding: utf-8 -*-
"""Contains utilities for downloading and converting datasets.

This module is based on several versions of dataset_utils.py in
https://github.com/tensorflow/models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from datetime import datetime
import random
import threading
import numpy as np
import tensorflow as tf

LABELS_FILENAME = 'labels.txt'


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._png_data, channels=3)

    # Initializes function that decodes RGB JPEG data.
    self._jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._jpeg_data, channels=3)

    # Initializes function that encode RGB JPEG/PNG data.
    self._image = tf.placeholder(dtype=tf.uint8)
    self._encoded_png = tf.image.encode_png(self._image)
    self._encoded_jpeg = tf.image.encode_jpeg(self._image)

  def decode_png(self, image_data):
    image = self._sess.run(self._decode_png,
                           feed_dict={self._png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def encode_png(self, image_data):
    string = self._sess.run(self._encoded_png,
                            feed_dict={self._image: image_data})
    return string

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def encode_jpeg(self, image_data):
    string = self._sess.run(self._encoded_jpeg,
                            feed_dict={self._image: image_data})
    return string


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_image_example(height,
                         width,
                         channels,
                         key,
                         encoded,
                         image_format,
                         class_name,
                         label):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/channels': int64_feature(channels),
      'image/key/sha256': bytes_feature(key.encode('utf8')),
      'image/encoded': bytes_feature(encoded),
      'image/format': bytes_feature(image_format.encode('utf8')),
      'image/class/text': bytes_feature(class_name.encode('utf8')),
      'image/class/label': int64_feature(label),
  }))


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with open(labels_filename, 'r') as f:
    lines = f.read()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index + 1:]
  return labels_to_class_names


def dataset_exists(dataset_name, dataset_dir, split_names_to_shards_num):
  for split_name in split_names_to_shards_num.keys():
    for shard_id in range(split_names_to_shards_num[split_name]):
      output_filename = get_dataset_filepath(dataset_name,
                                             split_name,
                                             shard_id,
                                             split_names_to_shards_num[split_name])
      output_filename = os.path.join(dataset_dir, output_filename)
      if not tf.gfile.Exists(output_filename):
        return False

  return True


def get_dataset_filepath(dataset_name, split_name, shard_id, total_shards):
  output_filename = dataset_name + '_%s_%05d-of-%05d.tfrecord' % (
      split_name,
      shard_id,
      total_shards)
  return output_filename


def _process_data_points_batch(dataset_name,
                               dataset_dir,
                               split_name,
                               thread_index,
                               num_threads,
                               num_shards,
                               ranges,
                               data_points,
                               convert_to_example_fn):
  """Process a batch of data point for a specific thread index.

  Each thread produces N shards where N = int(num_shards / num_threads).
  For instance, if num_shards = 128, and the num_threads is 2, then the first
  thread would produce shards [0, 64).

  Args:
      split_name: string, The name of the split the data belong to.
      thread_index: int, The current running thread.
      ranges: list, The list of ranges indexes for each shards.
      data_points: list, The data_points to create for this split.
  """
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g.
    # 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s

    output_filepath = get_dataset_filepath(
        dataset_name, split_name, shard, num_shards)
    output_filepath = os.path.join(dataset_dir, output_filepath)
    writer = tf.python_io.TFRecordWriter(output_filepath)

    shard_counter = 0
    files_in_shard = np.arange(
        shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      data_point = data_points[i]

      example = convert_to_example_fn(data_point)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if num_threads > 1:
        if not counter % 100:
          print('%s [thread %d]: Processed %d of %d data points in thread batch.' %
                (datetime.now(), thread_index, counter, num_files_in_thread))
          sys.stdout.flush()
      else:
        sys.stdout.write('\r>> Converting %.1f%%' % (
            float(counter) / float(num_files_in_thread) * 100.0))
        sys.stdout.flush()

    writer.close()

    if num_threads > 1:
      print('%s [thread %d]: Wrote %d data points to %s' %
            (datetime.now(), thread_index, shard_counter, output_filepath))
      sys.stdout.flush()
    shard_counter = 0

  if num_threads > 1:
    print('%s [thread %d]: Wrote %d data points to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()
  else:
    print()


def create_dataset_split(dataset_name,
                         dataset_dir,
                         split_name,
                         data_points,
                         convert_to_example_fn,
                         num_shards=1,
                         num_threads=1):
  """Create the TFRecords of a split for this dataset.

  If there is only one thread to run for this split, this will run synchronous.
  Otherwise, this will create the threads to process each data points batches.

  Args:
      split_name: string, The name of this split.
      data_points: list, The list of the data_points in this split.
  """

  assert not num_shards % num_threads, (
      'Please make the number of thread commensurate with the number of shards.')

  spacing = np.linspace(0, len(data_points), num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  print('Launching %d threads for spacings: %s' % (num_threads, ranges))
  sys.stdout.flush()

  if len(ranges) > 1:
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(len(ranges)):
      args = (dataset_name,
              dataset_dir,
              split_name,
              thread_index,
              num_threads,
              num_shards,
              ranges,
              data_points,
              convert_to_example_fn)
      t = threading.Thread(target=_process_data_points_batch, args=args)
      t.start()
      threads.append(t)

    coord.join(threads)
  else:
    args = (dataset_name,
            dataset_dir,
            split_name,
            0,
            num_threads,
            num_shards,
            ranges,
            data_points,
            convert_to_example_fn)
    _process_data_points_batch(*args)

  print('%s: Finished writing all %d data points in data set.' %
        (datetime.now(), len(data_points)))
  sys.stdout.flush()


def split_dataset(data_points, split_factor=[0.8, 0.2], shuffle=True):
  """Split the dataset into different splits.

  If `suffle` is `True`, we shuffle the split before returning it with a given seed for
  reproducibility.

  Args:
      data_points: list, A list of data point to split. Or if `user_defined_splits` is `True`,
      a list of two splits to use.
      shuffle: bool, `True` to shuffle the split before returning it.

  Returns:
      A tuple of two numpy array containing the data_points to use for each split.
  """
  sample_counts = []
  spacing = []
  split_number = len(split_factor)
  for i in range(split_number):
    sample_count = len(data_points) * split_factor[i]
    sample_counts.append(int(sample_count))
    spacing.append([sum(sample_counts[:i]), sum(sample_counts[:i + 1])])
  splits = [data_points[s[0]:s[1]] for s in spacing]

  if shuffle:
    random.seed(12345)
    for split in splits:
      random.shuffle(split)

  return splits
