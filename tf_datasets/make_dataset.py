#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Downloads and converts a particular dataset.

Usage:
```shell
$ tf_datasets --dataset_name=mnist --dataset_dir=/tmp/mnist
```
"""
import sys

import tensorflow as tf
from tensorflow.python.platform import flags

import tf_datasets as tfd

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    None,
    'The name of the dataset to convert, one of "cifar10", "flowers", "mnist".')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_bool(
    'force',
    False,
    'If true, the dataset will be recreated even if it already exists.')

tf.app.flags.DEFINE_bool(
    'cleanup',
    True,
    'If true, the dataset will be recreated even if it already exists.')


def main(_):
    if not FLAGS.dataset_name:
        raise ValueError('You must supply the dataset name with --dataset_name')

    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset name with --dataset_dir')

    dataset = tfd.get_dataset(FLAGS.dataset_name, FLAGS.dataset_dir)

    print('Downloading the files...')
    dataset.download()
    print('Extracting the files...')
    dataset.extract()
    print('Converting to tfrecords...')
    dataset.convert()
    if FLAGS.cleanup:
        print('Cleaning up temporary files...')
        dataset.cleanup()

    print('\nThe dataset has been generated!')


def run(argv=None):
    """Runs the program with an optional 'main' function and 'argv' list."""
    f = flags.FLAGS
    args = argv[1:] if argv else None
    flags_passthrough = f._parse_flags(args=args)
    sys.exit(main(sys.argv[:1] + flags_passthrough))


if __name__ == '__main__':
    run()
