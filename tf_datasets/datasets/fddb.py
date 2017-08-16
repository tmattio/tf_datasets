from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import shutil
import glob
import math
import tensorflow as tf
from tf_datasets.core.download import download_http, extract_tgz
from tf_datasets.core.base_dataset import BaseDataset
from tf_datasets.core import dataset_utils
from tf_datasets.core.dataset_utils import create_dataset_split
from tf_datasets.core.dataset_utils import split_dataset
from tf_datasets.core.dataset_utils import ImageCoder


slim = tf.contrib.slim


def _load_fddb_label_file(label_file):
    filenames_to_coords = {}
    with open(label_file, 'r') as f:
        while True:
            l = f.readline()

            if not l:
                break

            filename = l.strip() + '.jpg'

            ellipse_count = int(f.readline().strip())
            faces = [f.readline().strip().split()
                     for ellipse in range(ellipse_count)]

            filenames_to_coords[filename] = faces
    return filenames_to_coords


def _ellipse_to_rectangle(coord, img_width, img_height):
    r1 = float(coord[0])
    r2 = float(coord[1])
    angle = float(coord[2])
    cx = float(coord[3])
    cy = float(coord[4])

    height = 2 * r1 * (math.cos(math.radians(abs(angle))))
    width = 2 * r2 * (math.cos(math.radians(abs(angle))))

    lx = int(max(0, cx - width / 2))
    ly = int(max(0, cy - height / 2))
    rx = int(min(img_width - 1, cx + width / 2))
    ry = int(min(img_height - 1, cy + height / 2))

    return lx, ly, rx, ry


class fddb(BaseDataset):

    image_size = 32
    image_channel = 3
    urls = [
        'http://tamaraberg.com/faceDataset/originalPics.tar.gz',
        'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz',
    ]

    def __init__(self, dataset_dir):
        super().__init__(dataset_dir, class_names=[
            'face', ], zero_based_labels=False)
        self.dataset_name = 'fddb'
        self.download_dir = os.path.join(self.dataset_dir, 'download')
        self._coder = ImageCoder()

    def download(self):
        try:
            os.makedirs(self.download_dir)
        except FileExistsError:
            pass

        for url in self.urls:
            output_path = os.path.join(self.download_dir, url.split('/')[-1])
            if not os.path.exists(output_path):
                download_http(url, output_path)

    def extract(self):
        extract_tgz(os.path.join(self.download_dir,
                                 'FDDB-folds.tgz'), self.download_dir)
        extract_tgz(os.path.join(self.download_dir,
                                 'originalPics.tar.gz'), self.download_dir)

    def convert(self):
        data_points = list(self._get_data_points())
        splits = split_dataset(data_points, split_factor=[
                               0.8, 0.2], shuffle=True)
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
        filenames_to_coords = {}
        for f in glob.glob(os.path.join(self.download_dir,
                                        'FDDB-folds',
                                        '*-ellipseList.txt')):
            filenames_to_coords.update(_load_fddb_label_file(f))

        return filenames_to_coords.items()

    def _convert_to_example(self, data_point):
        filename, coords = data_point
        filename = os.path.join(self.download_dir, filename)
        with open(filename, 'rb') as f:
            encoded_image = f.read()
        decoded_image = self._coder.decode_jpeg(encoded_image)
        image_format = 'jpg'
        height, width, channels = decoded_image.shape
        key = hashlib.sha256(encoded_image).hexdigest()
        faces = [_ellipse_to_rectangle(e, height, width) for e in coords]

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        for bbox in faces:
            xmin.append(float(bbox[0]) / width)
            ymin.append(float(bbox[1]) / height)
            xmax.append(float(bbox[2] + bbox[0]) / width)
            ymax.append(float(bbox[3] + bbox[1]) / height)
            classes.append(0)
            classes_text.append('face'.encode('utf8'))

        return tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_utils.int64_feature(height),
            'image/width': dataset_utils.int64_feature(width),
            'image/channels': dataset_utils.int64_feature(channels),
            'image/filename': dataset_utils.bytes_feature(
                filename.encode('utf8')),
            'image/key/sha256':
                dataset_utils.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_utils.bytes_feature(encoded_image),
            'image/format':
                dataset_utils.bytes_feature(image_format.encode('utf8')),
            'image/object/bbox/xmin': dataset_utils.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_utils.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_utils.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_utils.float_list_feature(ymax),
            'image/object/class/text':
                dataset_utils.bytes_list_feature(classes_text),
            'image/object/class/label':
                dataset_utils.int64_list_feature(classes),
        }))

    def load(self, split_name, reader=None):
        # TODO(tmattio): Implement the load methods
        pass
