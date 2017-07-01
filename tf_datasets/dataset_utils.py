# -*- coding: utf-8 -*-
"""Contains utilities for downloading and converting datasets.

This module is based on several versions of dataset_utils.py in
https://github.com/tensorflow/models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    """Writes a file with the list of class names.

    Args:
      labels_to_class_names: A map of (integer) labels to class names.
      dataset_dir: The directory in which the labels file should be written.
      filename: The filename where the class names are written.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
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
    with tf.gfile.Open(labels_filename, 'r') as f:
        lines = f.read()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
    return labels_to_class_names
