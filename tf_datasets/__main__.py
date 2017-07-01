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

from tf_datasets import download_and_convert_mnist
from tf_datasets import download_and_convert_flowers

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
    'download_only',
    False,
    'If true, the dataset will not be extracted and converted to tfrecords.')

tf.app.flags.DEFINE_bool(
    'cleanup',
    False,
    'If true, the temporary files downloaded and extracted to create the dataset will be deleted.')

tf.app.flags.DEFINE_bool(
    'force',
    False,
    'If true, the dataset will be recreated even if it already exists.')


dataset_map = {
    'mnist': download_and_convert_mnist.MNISTDataset,
    'flowers': download_and_convert_flowers.FlowersDataset,
}


def main(_):
    if not FLAGS.dataset_name:
        raise ValueError('You must supply the dataset name with --dataset_name')

    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    if FLAGS.dataset_name not in dataset_map.keys():
        raise ValueError('The dataset `{}` does not exists!'.format(FLAGS.dataset_name))

    dataset = dataset_map[FLAGS.dataset_name](
        dataset_dir=FLAGS.dataset_dir
    )

    if not FLAGS.force and dataset.exists():
        print('The dataset already exists, leaving without re-creating it.')
        return

    dataset.maybe_download()

    if not FLAGS.download_only:
        dataset.maybe_extract()
        dataset.convert(FLAGS.force)

        if FLAGS.cleanup:
            dataset.cleanup_temporary_files()


def run(argv=None):
    """Runs the program with an optional 'main' function and 'argv' list."""
    f = flags.FLAGS
    args = argv[1:] if argv else None
    flags_passthrough = f._parse_flags(args=args)
    sys.exit(main(sys.argv[:1] + flags_passthrough))


if __name__ == '__main__':
    run()
