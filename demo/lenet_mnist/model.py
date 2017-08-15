import tensorflow as tf

slim = tf.contrib.slim


def preprocess_image(image, output_height, output_width):
    image = tf.to_float(image)
    image = tf.image.resize_image_with_crop_or_pad(
        image, output_width, output_height)
    image = tf.subtract(image, 128.0)
    image = tf.div(image, 128.0)
    return image


def lenet(images):
    net = slim.conv2d(images, 20, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.conv2d(net, 50, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.flatten(net, scope='flatten3')
    net = slim.fully_connected(net, 500, scope='fc4')
    net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
    return net


def load_batch(dataset, batch_size=32, height=28, width=28):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

    image, label = data_provider.get(['image', 'label'])

    image = preprocess_image(
        image,
        height,
        width)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        allow_smaller_final_batch=True)

    return images, labels
