# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train and Eval the MNIST network.
This version is like fully_connected_feed.py but uses data converted
to a TFRecords file containing tf.train.Example protocol buffers.
See:
https://www.tensorflow.org/guide/reading_data#reading_from_files
for context.
YOU MUST run convert_to_records before running this (but you only need to
run it once).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import math

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import mnist

# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
# You will need to change this to match where you have MNIST downloaded
TRAIN_FILE = '/data/image_datasets/mnist/train.tfrecords'
VALIDATION_FILE = '/data/image_datasets/mnist/validation.tfrecords'
TEST_FILE = '/data/image_datasets/mnist/test.tfrecords'

def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape((mnist.IMAGE_PIXELS))
    image = tf.reshape(image, [28, 28, 1])

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label

def augment(image, label):
    """Placeholder for data augmentation."""
    with tf.variable_scope('augmentations'):
        num_images_ = tf.shape(image)[0]
        # random rotate
        angle = 15.0
        processed_data = tf.contrib.image.rotate(image,
                                                 tf.random_uniform([num_images_],
                                                                   maxval=math.pi / 180 * angle,
                                                                   minval=math.pi / 180 * -angle))
        # random shift
        rank    = 2
        max_pad = 4
        min_pad = 0
        random_pad = tf.random_uniform(shape=(rank, 2),
                                          minval=min_pad,
                                          maxval=max_pad + 1,
                                          dtype=tf.int32)
        random_pad = tf.pad(random_pad, paddings=[[0, 1], [0, 0]])
        image = tf.pad(image, random_pad)
        slice_begin = random_pad[:, 1]
        slice_end = [28, 28, 1]
        image = tf.slice(image,
                         slice_begin,
                         slice_end,
                         name='translate')
    
    return image, label

def normalize(image, label):
    """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label


def inputs(train, batch_size, num_epochs):
    """Reads input data num_epochs times.
    Args:
      train: Selects between the training ('train'),  validation ('vald') data and 'test' data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
      This function creates a one_shot_iterator, meaning that it will only iterate
      over the dataset once. On the other hand there is no special initialization
      required.
    """
    if not num_epochs:
        num_epochs = None
    
    if train == 'train':
        filename = TRAIN_FILE
    elif train == 'vald':
        filename = VALIDATION_FILE
    elif train == 'test':
        filename = TEST_FILE
    elif train == 'ref':
        filename = TRAIN_FILE
    else:
        raise ValueError('Invalid argument `{}` -  must be one of: train, vald, test'.format(train))

    with tf.name_scope('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset(filename)

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(decode)
        
        if train == 'train':
            dataset = dataset.map(augment)
            
        dataset = dataset.map(normalize)

        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        if train == 'train' or train == 'ref':
            dataset = dataset.shuffle(1000 + 3 * batch_size)
        
        if num_epochs is not None:
            dataset = dataset.repeat(num_epochs)
        elif train == 'ref':
            dataset = dataset.repeat()
        
        dataset = dataset.batch(batch_size, drop_remainder=True) #Drop remainder for consistent batch size
        
        if train == 'ref':
            return dataset.make_one_shot_iterator()
            
        iterator = dataset.make_initializable_iterator()
    return iterator

