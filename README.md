# Tensorflow Datasets

Python scripts to download public datasets and generate tfrecords.

## Features

* Show progress of the download of the dataset
* Show progress of the convertion of the dataset
* Create the TFRecords datasets multithreaded
* Split the TFRecords datasets into several shards

## Usage

To download and create a dataset, you can use the `tf_datasets` command:

    # Create the MNIST dataset
    tf_datasets --dataset_name=mnist --dataset_dir=data/mnist --cleanup

## Supported Dataset

Image Classification:

* [MNIST](http://yann.lecun.com/exdb/mnist/): The MNIST database of handwritten digits
* [Flowers](https://github.com/tensorflow/models/blob/master/slim/datasets/flowers.py): The Tensorflow flowers dataset.

## TODO

* Write example of DatasetProvider
* Create repository of already created dataset
* Create API to download already created dataset and data provider

* Support Ciraf-10 dataset
* Support Ciraf-100 dataset
