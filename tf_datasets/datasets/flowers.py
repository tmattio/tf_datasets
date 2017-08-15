from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import hashlib
import os

import tensorflow as tf

from tf_datasets.core.base_dataset import BaseDataset
from tf_datasets.core.download import download_http, extract_tgz
from tf_datasets.core.dataset_utils import split_dataset, create_dataset_split, ImageCoder, create_image_example


slim = tf.contrib.slim


class flowers(BaseDataset):

    num_training = 3303
    num_validation = 367
    filename = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

    def __init__(self, dataset_dir):
        super().__init__(dataset_dir, self._get_class_names(), zero_based_labels=True)
        self.dataset_name = 'flowers'
        self.download_dir = os.path.join(self.dataset_dir, 'download')
        self._coder = ImageCoder()

    def download(self):
        try:
            os.makedirs(self.download_dir)
        except FileExistsError:
            pass

        output_path = os.path.join(self.download_dir, 'flower_photos.tgz')
        if not os.path.exists(output_path):
            download_http(self.filename, output_path)

    def extract(self):
        output_path = os.path.join(self.download_dir, 'flower_photos')
        if not os.path.exists(output_path):
            extract_tgz(os.path.join(self.download_dir, 'flower_photos.tgz'), self.download_dir)

    def convert(self):
        data_points = list(self._get_data_points())
        splits = split_dataset(data_points, split_factor=[0.9, 0.1], shuffle=True)
        split_names = ['train', 'validation']

        for split, split_name in zip(splits, split_names):
            create_dataset_split('flowers',
                                 self.dataset_dir,
                                 split_name,
                                 split,
                                 self._convert_to_example)

        self.write_label_file()

    def cleanup(self):
        shutil.rmtree(self.download_dir)

    def _get_data_points(self):
        flower_root = os.path.join(self.download_dir, 'flower_photos')
        directories = []
        for filename in os.listdir(flower_root):
            path = os.path.join(flower_root, filename)
            if os.path.isdir(path):
                directories.append(path)

        for directory in directories:
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                yield path

    def _get_class_names(self):
        flower_root = os.path.join(self.download_dir, 'flower_photos')
        for filename in os.listdir(flower_root):
            path = os.path.join(flower_root, filename)
            if os.path.isdir(path):
                yield filename

    def _convert_to_example(self, data_point):
        filepath = data_point
        encoded = tf.gfile.FastGFile(filepath, 'rb').read()
        image_data = self._coder.decode_jpeg(encoded)
        height, width, channels = image_data.shape
        image_format = 'jpg'
        class_name = os.path.basename(os.path.dirname(filepath))
        label = self.class_names_to_labels[class_name]
        key = hashlib.sha256(encoded).hexdigest()

        return create_image_example(height,
                                    width,
                                    channels,
                                    key,
                                    encoded,
                                    image_format,
                                    class_name,
                                    label)

    def load(self, split_name, reader=None):
        # TODO(tmattio): Implement the load methods
        pass
