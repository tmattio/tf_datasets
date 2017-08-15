# Tensorflow Datasets

Python scripts to download public datasets and generate tfrecords.

## Features

* Show progress of the download of the dataset
* Show progress of the conversion of the dataset
* Create the TFRecords datasets multithreaded
* Split the TFRecords datasets into several shards

## Usage

To install Tensorflow Datasets, you can install directly from sources:

    git clone git@github.com:tmattio/tf_datasets.git
    cd tf_datasets
    make install

To download and create a dataset, you can use the `tf_datasets` command installed with the package:

    # Create the MNIST dataset
    tf_make_dataset --dataset_name=mnist --dataset_dir=data/mnist --cleanup

To use a dataset:

    import tf_datasets as tfd

    mnist = tfd.get_dataset('mnist', './data/mnist')
    mnist.download()
    mnist.extract()
    mnist.convert()
    mnist.cleanup()

    # This will raise an error if the dataset does not exist
    images, labels = mnist.load('train')

## Supported Dataset

### Image Classification

* **mnist** - [MNIST](http://yann.lecun.com/exdb/mnist/): The MNIST database of handwritten digits
* **flowers** - [Flowers](https://github.com/tensorflow/models/blob/master/slim/datasets/flowers.py): The Tensorflow flowers dataset.
* **cifar10** - [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html): The CIFAR-10 is a labeled subset of the 80 million tiny images dataset.
* **cifar100** - [Cifar-100](https://www.cs.toronto.edu/~kriz/cifar.html): The CIFAR-100 is a labeled subset of the 80 million tiny images dataset.

### Object Detection

* **fddb** - [FDDB](http://vis-www.cs.umass.edu/fddb/): Face Detection Data Set and Benchmark
* **wider_face** - [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/): WIDER FACE: A Face Detection Benchmark

## TODO

* Add unit tests
* Add loads method for the datasets
* Create API to download already created dataset
* Support MSCoco dataset
* Support Pascal VOC 2007/2012 dataset
* Support CBSR-Webface dataset
