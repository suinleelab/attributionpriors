# Code gratuitously stolen from: 
# https://github.com/hwalsuklee/tensorflow-mnist-cnn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

import tensorflow as tf  
import numpy as np
import tensorflow.contrib.slim as slim
import datetime 

import mnist_model
import mnist_reader

from attributionpriors.ops import AttributionPriorExplainer

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 50,
                       """Batch size for training.""")
tf.app.flags.DEFINE_integer('epochs', 60,
                       """Number of epochs to train for.""")
tf.app.flags.DEFINE_float('lamb', 0.0,
                         """Penalty between training loss (0.0) and regularization loss (1.0)""")
tf.app.flags.DEFINE_float('eta', 0.0001,
                         """initial learning rate""")
tf.app.flags.DEFINE_float('keep_prob', 0.5,
                         'Dropout keep probability')

MODEL_DIRECTORY = "models/{}/model.ckpt"
TRAIN_LOGS_DIRECTORY = "logs/{}/train"
EVAL_LOGS_DIRECTORY  = "logs/{}/vald"

# Params for Train
validation_step = 500

# Params for test
TEST_BATCH_SIZE = 50

#Data params
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 50

def batch_standardize(frames):
        return tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), frames)

def inputs(train_batch, train_epochs,
           vald_batch=50,
           test_batch=50):
    training_iterator   = mnist_reader.inputs('train', train_batch, train_epochs)
    validation_iterator = mnist_reader.inputs('vald',  vald_batch, None)
    test_iterator       = mnist_reader.inputs('test',  test_batch, None)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
                handle, (tf.float32, tf.int32), ((None, 28, 28, 1), (None, )))
    x, y_ = iterator.get_next()
    return training_iterator, validation_iterator, test_iterator, handle, x, y_

def get_model(cond_input_op):
    train_pl = tf.placeholder_with_default(False, shape=(), name='train_pl')
    y = mnist_model.model(cond_input_op, train_pl)
    return y, train_pl

def get_learning_rate(init_learning_rate, train_batch):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(init_learning_rate, 
                                                global_step * train_batch, 
                                                60000,
                                                0.95,
                                                staircase=True)
    return learning_rate

def pipeline(use_old_model=False):
    # Data iterators
    training_iterator, validation_iterator, test_iterator, handle, x, y_ = inputs(FLAGS.batch_size, FLAGS.epochs,
                                                                                  VALIDATION_SIZE,
                                                                                  TEST_BATCH_SIZE)
    
    # Get explainer conditional input
    explainer = AttributionPriorExplainer()
    cond_input_op, train_eg = explainer.input_to_samples_delta(x)
    
    # Predict
    y, train_pl = get_model(cond_input_op)
    
    # Get explanations
    expected_gradients_op = explainer.shap_value_op(y, cond_input_op, sparse_labels_op=y_)
    expected_gradients_op = batch_standardize(expected_gradients_op)
    
    learning_rate = get_learning_rate(FLAGS.eta, FLAGS.batch_size)
    global_step = tf.train.get_or_create_global_step()
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize((1.0 - FLAGS.lamb) * loss, global_step=global_step)

    eg_loss  = tf.reduce_mean(tf.image.total_variation(expected_gradients_op, name='variation'))
    eg_train = tf.train.AdamOptimizer(learning_rate).minimize(FLAGS.lamb * eg_loss, global_step=global_step)
    
    y_pred = tf.argmax(y, 1)
    accuracy_op, accuracy_update_op = tf.metrics.accuracy(y_pred, y_)
    reset_metrics_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
    
    return cond_input_op, x, y, y_, train_pl, loss, train_step, eg_loss, eg_train, train_eg, accuracy_op, accuracy_update_op, reset_metrics_op, training_iterator, validation_iterator, test_iterator, handle

def train(save_dir, sess, cond_input_op, y, train_pl, loss, train_step, eg_loss, eg_train, \
          train_eg, accuracy_op, accuracy_update_op, reset_metrics_op, training_iterator, \
          validation_iterator, test_iterator, handle):
    
    
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    training_handle   = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    test_handle       = sess.run(test_iterator.string_handle())

    max_acc = 0.

    sess.run(training_iterator.initializer)
    global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver()
    
    validation_accuracies = []
    validation_total_variances = []
    while True:
        try:
            _, _, i = sess.run([train_step, accuracy_update_op, global_step] , feed_dict={handle: training_handle,
                                                                                          train_pl: True})
            if FLAGS.lamb > 0.0:
                _, train_var_loss = sess.run([eg_train, eg_loss], feed_dict={handle: training_handle, 
                                                                             train_eg: True})
        except tf.errors.OutOfRangeError:
            break

        if i % validation_step == 0:
            sess.run(reset_metrics_op)
            sess.run(validation_iterator.initializer)
            while True:
                try:
                    sess.run(accuracy_update_op, feed_dict={handle: validation_handle})
                except tf.errors.OutOfRangeError:
                    break

            sess.run(validation_iterator.initializer)
            vald_var_loss = sess.run(eg_loss, feed_dict={handle: validation_handle, train_eg: True})
            validation_accuracy = sess.run(accuracy_op, feed_dict={handle: validation_handle})
            
            validation_total_variances.append(vald_var_loss)
            validation_accuracies.append(validation_accuracy)

            print('Iteration: {}, validation accuracy: {:.6f}, e-variation vald (batch): {:.6f}'.format(i, validation_accuracy, vald_var_loss), end='\r')
            sess.run(reset_metrics_op)

        if validation_accuracy > max_acc:
            max_acc = validation_accuracy
            save_path = saver.save(sess, save_dir)
    
    # Restore variables from disk
    saver.restore(sess, save_dir)

    # Loop over all batches
    sess.run(reset_metrics_op)
    sess.run(test_iterator.initializer)
    while True:
        try:
            sess.run(accuracy_update_op, feed_dict={handle: test_handle, train_pl: False})
        except tf.errors.OutOfRangeError:
            break
    test_accuracy = sess.run(accuracy_op)
    print('Test accuracy: {:.6f}'.format(test_accuracy))
    return validation_total_variances, validation_accuracies, test_accuracy