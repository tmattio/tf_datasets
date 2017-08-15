import tensorflow as tf

import tf_datasets as tfd
from model import lenet, load_batch

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('data_dir', './data/mnist',
                    'Directory with the mnist data.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('num_batches', None,
                     'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', './logs/train',
                    'Directory with the log data.')
FLAGS = flags.FLAGS


def main(args):
    # load the dataset
    mnist = tfd.get_dataset('mnist', FLAGS.data_dir)
    dataset = mnist.load('train')

    # load batch of dataset
    images, labels = load_batch(
        dataset,
        FLAGS.batch_size)

    # run the image through the model
    predictions = lenet(images)

    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)
    slim.losses.softmax_cross_entropy(
        predictions,
        one_hot_labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    # use RMSProp to optimize
    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)

    # create train op
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        summarize_gradients=True)

    # run training
    slim.learning.train(
        train_op,
        FLAGS.log_dir,
        save_summaries_secs=20)


if __name__ == '__main__':
    tf.app.run()
