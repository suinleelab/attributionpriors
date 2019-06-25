# Code gratuitously stolen from: 
# https://github.com/hwalsuklee/tensorflow-mnist-cnn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
FLAGS = tf.app.flags.FLAGS

def model(inputs, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        x = tf.reshape(inputs, [-1, 28, 28, 1])

        net = slim.conv2d(x, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.flatten(net, scope='flatten3')

        net = slim.fully_connected(net, 1024, scope='fc3')
        net = slim.dropout(net, is_training=is_training, scope='dropout3', keep_prob=FLAGS.keep_prob)  # 0.5 by default
        outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs