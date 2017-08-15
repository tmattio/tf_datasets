import os
from abc import ABCMeta, abstractmethod

from tf_datasets.core.dataset_utils import write_label_file


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


class BaseDataset(metaclass=ABCMeta):

    def __init__(self, dataset_dir, class_names, zero_based_labels=False):
        try:
            os.makedirs(dataset_dir)
        except FileExistsError:
            pass

        self.dataset_dir = dataset_dir
        self._class_names = class_names
        self.zero_based_labels = zero_based_labels

    @cached_property
    def class_names(self):
        """Get the list of the classes in the dataset.

        Returns:
          A list or tuple of the classes names of this dataset.
        """
        return list(self._class_names)

    @cached_property
    def num_classes(self):
        """Get the total number of classes in this dataset.

        Returns:
          An int of the number of classes
        """
        return len(self.class_names)

    @cached_property
    def class_names_to_labels(self):
        """Build an association of the dataset classes to their ID.

        Returns:
          A dictionnary of the classes names mapped to the labels.
        """
        if self.zero_based_labels:
            return dict(zip(self.class_names, range(self.num_classes)))
        else:
            return dict(zip(self.class_names, range(1, self.num_classes + 1)))

    @cached_property
    def labels_to_class_names(self):
        """Build an association of the dataset labels to their names.

        Returns:
          A dictionnary of the labels mapped to the classes names.
        """
        if self.zero_based_labels:
            return dict(zip(range(self.num_classes), self.class_names))
        else:
            return dict(zip(range(1, self.num_classes + 1), self.class_names))

    def write_label_file(self):
        write_label_file(self.labels_to_class_names, self.dataset_dir)

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def convert(self):
        """Convert the data points from the source format to TF-Records files.

        If the dataset already exist, this will not generate new tfrecord files.

        This will split the datasets from the data source into a training set and a
        validation set. This will then convert the two set separately.

        This will also generate a label file containing all the classes names and labels
        of this dataset.

        The generated tfrecords and the labels file can be found in `self.dataset_dir`.
        """
        pass

    @abstractmethod
    def cleanup(self):
        pass

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
