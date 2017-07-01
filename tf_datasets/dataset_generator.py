# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from datetime import datetime
import os
import random
import threading
from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf

from pirnn_class.datasets.dataset_utils import write_label_file


class cached_property(object):
    """A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.

    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class DatasetGenerator:
    """Define a workflow for generating a Tensorflow dataset.

    When creating a dataset, this allows the user to focus on datasets specific tasks:
      * Definition of the classes
      * How to get each data points
      * How to convert the data points from the original data source to an Example proto

    The dataset generator will take care of the workflow to generate the final TFRecords files:
      * Splitting between training and validation.
      * Creating the dataset split into N shards.
      * Assign a shard creation to a child process.

    A BaseDataGenerator sub-class should define the following class methods:
      * _get_data_points: returns the ids of the data points from the data source.
      * _get_class_names: returns the list of the classes of the dataset.
      * _convert_to_example: returns a Example proto from a data point id.

    Once the above class methods have been created, one can easily generate the dataset by
    creating an instance and calling the generate() function.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 dataset_name,
                 dataset_dir,
                 split_names=['train', 'validation'],
                 split_factor=[0.8, 0.2],
                 split_max_samples=None,
                 split_shards_num=[1, 1],
                 split_threads_num=[1, 1],
                 user_defined_splits=False):
        self.split_number = len(split_names)

        error_len = "Please, make sure you provide the arguments for each dataset split."

        if not user_defined_splits:
            assert len(split_factor) == self.split_number, error_len
            assert sum(split_factor) == 1., "Please, make sure the sum of the split factors is 1."
            if split_max_samples is not None:
                assert len(split_max_samples) == self.split_number, error_len

        for l in [split_shards_num, split_threads_num]:
            assert len(l) == self.split_number, error_len

        for i in range(self.split_number):
            assert not split_shards_num[i] % split_threads_num[i], (
                'Please make the number of thread commensurate with the number of shards.')

        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.split_names = split_names
        self.split_shards_num = split_shards_num
        self.split_threads_num = split_threads_num
        self.user_defined_splits = user_defined_splits
        self.split_factor = None
        self.split_max_samples = None

        if not self.user_defined_splits:
            self.split_factor = split_factor

            if split_max_samples:
                self.split_max_samples = [s if s else np.inf for s in split_max_samples]

        if not tf.gfile.Exists(self.dataset_dir):
            tf.gfile.MakeDirs(self.dataset_dir)

    @abstractmethod
    def _get_data_points(self):
        """Get all the data points of the dataset.

        The data point can be any time. Here are some examples:
        * If the data source is the filesystem, this can be the filepath of the data point.
        * If the data source is a database, this can be the id of the field in the database.

        Returns:
          A list or tuple of all the data points in this dataset.
        """
        pass

    @abstractmethod
    def _convert_to_example(self, data_point):
        """Build an Example proto for a data point.

        The data_point should contain enough informations to allow the method to convert it
        to an Example proto.

        Args:
          data_point: a data point to convert, the same type as the items returned by _get_class_names()

        Returns:
          Example proto
        """
        pass

    @abstractmethod
    def _get_class_names(self):
        """Get the list of the classes in the dataset.

        The classes should be strings, and we later be converted to ID.

        Returns:
          A list or tuple of the classes names of this dataset.
        """
        pass

    @cached_property
    def class_names(self):
        """Get the list of the classes in the dataset.

        This is a wrapper around the _get_class_names abstract method
        that will cache the result.

        Returns:
          A list or tuple of the classes names of this dataset.
        """
        return self._get_class_names()

    @cached_property
    def class_names_to_labels(self):
        """Build an association of the dataset classes to their ID.

        Returns:
          A dictionnary of the classes names mapped to the labels.
        """
        return dict(zip(self.class_names, range(len(self.class_names))))

    @cached_property
    def labels_to_class_names(self):
        """Build an association of the dataset labels to their names.

        Returns:
          A dictionnary of the labels mapped to the classes names.
        """
        return dict(zip(range(len(self.class_names)), self.class_names))

    def get_num_shards(self, split_name):
        """Get the number of shards for a given split name.

        The number of shards can be specified when creating the instance,
        by default it's one for every split.

        If there is no split with this name, this will raise a ValueError.

        Args:
          split_name: string, the name of the split to get the number of shards
        Returns:
          A int of the number of shards for this split.
        """
        index = self.split_names.index(split_name)
        return self.split_shards_num[index]

    def get_num_threads(self, split_name):
        """Get the number of threads for a given split name.

        The number of thread can be specified when creating the instance,
        by default it's one for every split.

        If there is no split with this name, this will raise a ValueError.

        Args:
          split_name: string, the name of the split to get the number of threads
        Returns:
          A int of the number of threads for this split.
        """
        index = self.split_names.index(split_name)
        return self.split_threads_num[index]

    def get_dataset_filepath(self, split_name, shard_id, total_shards):
        """Build a dataset filepath given the parameters of the dataset.

        Args:
          dataset_name: string, the name of the dataset.
          split_name: string, the split name of the dataset, can be either 'train' or 'validation'.
          shard_id: int, the number of the shard of the dataset file, this is zero-based.
          total_shards: int, the total number of shards for this split, this is one-based.
        Returns:
          The filename of the dataset file given all the parameters.
        """
        output_filename = self.dataset_name + '_%s_%05d-of-%05d.tfrecord' % (
            split_name,
            shard_id,
            total_shards)
        return output_filename

    def exists(self):
        """Check if this datasets already exists or not.

        This will take the file pattern of the TFRecords, and iterate through the number of
        thread/shards. If all the files exist, the dataset is considered to be complete.
        If there is one or more file missing, the dataset is considered to be incomplete.

        This will not check the data of the files, if all the files exits but the data is not
        valid, this will still return True.

        Returns:
          True if the dataset exists, False if the dataset does not exist.
        """
        for split_name in self.split_names:
            for shard_id in range(self.get_num_shards(split_name)):
                output_filename = self.get_dataset_filepath(split_name, shard_id, self.get_num_shards(split_name))
                output_filename = os.path.join(self.dataset_dir, output_filename)
                if not tf.gfile.Exists(output_filename):
                    return False
        return True

    def convert(self, force=False):
        """Convert the data points from the source format to TF-Records files.

        If the dataset already exist, this will not generate new tfrecord files.

        This will split the datasets from the data source into a training set and a
        validation set. This will then convert the two set separately.

        This will also generate a label file containing all the classes names and labels
        of this dataset.

        The generated tfrecords and the labels file can be found in `self.dataset_dir`.
        """
        if not force and self.dataset_exists():
            print('Dataset files already exist. Exiting without re-creating them.')
            return

        data_points = self._get_data_points()
        splits = self.__split_dataset(data_points, shuffle=True)

        log = 'Creating the dataset with'
        for i, split in enumerate(splits):
            log += ' {} samples in {}'.format(len(split), self.split_names[i])
        print(log)

        for i, split in enumerate(splits):
            self._create_dataset_split(self.split_names[i], split)

        write_label_file(self.labels_to_class_names, self.dataset_dir)

        print('\nFinished converting the {} dataset!'.format(self.dataset_name))

    def _process_data_points_batch(self,
                                   split_name,
                                   thread_index,
                                   ranges,
                                   data_points):
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
        num_threads = self.get_num_threads(split_name)
        num_shards = self.get_num_shards(split_name)
        num_shards_per_batch = int(num_shards / num_threads)

        shard_ranges = np.linspace(ranges[thread_index][0],
                                   ranges[thread_index][1],
                                   num_shards_per_batch + 1).astype(int)
        num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

        counter = 0
        for s in range(num_shards_per_batch):
            # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
            shard = thread_index * num_shards_per_batch + s

            output_filepath = self.get_dataset_filepath(split_name, shard, num_shards)
            output_filepath = os.path.join(self.dataset_dir, output_filepath)
            writer = tf.python_io.TFRecordWriter(output_filepath)

            shard_counter = 0
            files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
            for i in files_in_shard:
                data_point = data_points[i]

                example = self._convert_to_example(data_point)
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

    def _create_dataset_split(self, split_name, data_points):
        """Create the TFRecords of a split for this dataset.

        If there is only one thread to run for this split, this will run synchronous.
        Otherwise, this will create the threads to process each data points batches.

        This will create

        Args:
          split_name: string, The name of this split.
          data_points: list, The list of the data_points in this split.
        """
        num_threads = self.get_num_threads(split_name)

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
                args = (split_name,
                        thread_index,
                        ranges,
                        data_points)
                t = threading.Thread(target=self._process_data_points_batch, args=args)
                t.start()
                threads.append(t)

            coord.join(threads)
        else:
            self._process_data_points_batch(split_name,
                                            0,
                                            ranges,
                                            data_points)

        print('%s: Finished writing all %d data points in data set.' %
              (datetime.now(), len(data_points)))
        sys.stdout.flush()

    def __split_dataset(self, data_points, shuffle=True):
        """Split the dataset into training set and validation set.

        If the `user_definied_splits` is `True`, the dataset should already be splitted into two
        different set.

        If not, we consider the `validation_split_factor` and the maximum sample in each splits
        to split the dataset in two.

        If `suffle` is `True`, we shuffle the split before returning it with a given seed for
        reproducibility.

        Args:
          data_points: list, A list of data point to split. Or if `user_defined_splits` is `True`,
            a list of two splits to use.
          shuffle: bool, `True` to shuffle the split before returning it.
        Returns:
          A tuple of two numpy array containing the data_points to use for each split.
        """
        if self.user_defined_splits:
            assert len(data_points) == len(self.split_names),\
                "Please, return the datasets splits as a list in the _get_data_points function"

            splits = data_points
        else:
            sample_counts = []
            spacing = []
            for i in range(self.split_number):
                if self.split_max_samples is not None:
                    sample_count = min(len(data_points) * self.split_factor[i], self.split_max_samples[i])
                else:
                    sample_count = len(data_points) * self.split_factor[i]
                sample_counts.append(int(sample_count))
                spacing.append([sum(sample_counts[:i]), sum(sample_counts[:i + 1])])
            splits = [data_points[s[0]:s[1]] for s in spacing]
        if shuffle:
            random.seed(12345)
            for split in splits:
                random.shuffle(split)

        return splits
