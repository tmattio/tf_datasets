from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import shutil
import glob
from scipy.io import loadmat
import tensorflow as tf
from tf_datasets.core.download import download_google_drive, extract_zip
from tf_datasets.core.base_dataset import BaseDataset
from tf_datasets.core import dataset_utils
from tf_datasets.core.dataset_utils import create_dataset_split
from tf_datasets.core.dataset_utils import split_dataset
from tf_datasets.core.dataset_utils import ImageCoder


slim = tf.contrib.slim


class wider_face(BaseDataset):

    data_url = 'https://docs.google.com/uc?export=download'
    train_data_id = '0B6eKvaijfFUDQUUwd21EckhUbWs'
    validation_data_id = '0B6eKvaijfFUDd3dIRmpvSk8tLUk'
    test_data_id = '0B6eKvaijfFUDbW4tdGpaYjgzZkU'
    train_data_size = 1465602149
    validation_data_size = 362752168
    test_data_size = 1844140520

    annotation_data_url = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/'
    'support/bbx_annotation/wider_face_split.zip'

    def __init__(self, dataset_dir):
        super().__init__(
            dataset_dir,
            class_names=self._get_class_names(),
            zero_based_labels=False
        )
        self.dataset_name = 'wider_face'
        self.download_dir = os.path.join(self.dataset_dir, 'download')
        self._coder = ImageCoder()

    def download(self):
        for file_id, filename, file_size in [
            (self.train_data_id, 'WIDER_train.zip', self.train_data_size),
            (self.validation_data_id, 'WIDER_val.zip',
             self.validation_data_size),
            (self.test_data_id, 'WIDER_test.zip', self.test_data_size)
        ]:
            output_path = os.path.join(self.download_dir, filename)
            if not os.path.exists(output_path):
                download_google_drive(
                    file_id, output_path, self.base_url, file_id)

    def extract(self):
        for filename in ['WIDER_train.zip',
                         'WIDER_val.zip',
                         'wider_face_split.zip']:
            extract_zip(os.path.join(self.download_dir,
                                     filename), self.download_dir)

    def convert(self):
        data_points = list(self._get_data_points())
        splits = split_dataset(data_points, split_factor=[
                               0.8, 0.2], shuffle=True)
        split_names = ['train', 'validation']

        for split, split_name in zip(splits, split_names):
            create_dataset_split('wider_face',
                                 self.dataset_dir,
                                 split_name,
                                 split,
                                 self._convert_to_example)

        self.write_label_file()

    def cleanup(self):
        shutil.rmtree(self.download_dir)

    def _get_class_names(self):
        root = os.path.join(self.download_dir, 'WIDER_train')
        for filename in os.listdir(root):
            path = os.path.join(root, filename)
            if os.path.isdir(path):
                yield filename

    def _get_data_points(self):
        train = glob.glob(
            os.path.join(self.download_dir, 'WIDER_train', '**.png'),
            recursive=True
        )
        eval = glob.glob(os.path.join(self.download_dir,
                                      'WIDER_eval', '**.png'), recursive=True)

        filenames_to_bboxes = {}
        filenames_to_bboxes.update(self._load_label_file(os.path.join(
            self.download_dir, 'wider_face_split', 'wider_face_train.mat')))
        filenames_to_bboxes.update(self._load_label_file(os.path.join(
            self.download_dir, 'wider_face_split', 'wider_face_val.mat')))

        train_split = [(f, filenames_to_bboxes[f]) for f in train]
        eval_split = [(f, filenames_to_bboxes[f]) for f in eval]

        train_split, eval_split

    def _load_label_file(label_file):
        mat_f = loadmat(label_file)
        file_list = mat_f['file_list']
        face_bbx_list = mat_f['face_bbx_list']

        assert len(file_list) == len(face_bbx_list)

        filename_to_bboxes = {}
        for i in range(len(file_list)):
            for j in range(len(file_list[i][0])):
                filename = file_list[i][0][j][0][0]
                bboxes = face_bbx_list[i][0][j][0]
                filename_to_bboxes[filename] = bboxes
        return filename_to_bboxes

    def _convert_to_example(self, data_point):
        filename, bboxes = data_point
        filename = os.path.join(self.download_dir, filename)
        with open(filename, 'rb') as f:
            encoded_image = f.read()
        decoded_image = self._coder.decode_jpeg(encoded_image)
        image_format = 'jpg'
        height, width, channels = decoded_image.shape
        key = hashlib.sha256(encoded_image).hexdigest()

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        for bbox in bboxes:
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
            'image/filename':
                dataset_utils.bytes_feature(filename.encode('utf8')),
            'image/key/sha256':
                dataset_utils.bytes_feature(key.encode('utf8')),
            'image/encoded':
                dataset_utils.bytes_feature(encoded_image.tobytes()),
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
