import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
    
def batch_standardize(frames):
    return tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), frames)
    
def get_learning_rate(init_learning_rate, train_batch):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(init_learning_rate, 
                                                global_step * train_batch, 
                                                60000,
                                                0.95,
                                                staircase=True)
    return learning_rate

def simple_model(inputs, is_training=None):
    net = tf.layers.BatchNormalization()(inputs)
    net = tf.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu, use_bias=True, padding='SAME')(net)
    net = tf.layers.BatchNormalization()(net)
    net = tf.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu, use_bias=True, padding='SAME')(net)
    net = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(net)
    
    net = tf.layers.BatchNormalization()(net)
    net = tf.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu, use_bias=True, padding='SAME')(net)
    net = tf.layers.BatchNormalization()(net)
    net = tf.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu, use_bias=True, padding='SAME')(net)
    net = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(net)
    
    net = tf.layers.Flatten()(net)
    net = tf.layers.Dense(128, activation=tf.nn.relu, use_bias=True)(net)
    net = tf.layers.Dense(10, activation=None, use_bias=False)(net)
    return net

def model(inputs, is_training, cnn_filters=[32, 64], fully_connected=[1024]):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        net = tf.reshape(inputs, [-1, 28, 28, 1])

        i = 0
        for num_filters in cnn_filters:
            net = slim.conv2d(net, num_filters, [5, 5], scope='conv{}'.format(i))
            net = slim.max_pool2d(net, [2, 2], scope='pool{}'.format(i))
            i = i + 1
            
        net = slim.flatten(net, scope='flatten')

        i = 0
        for num_hidden in fully_connected:
            net = slim.fully_connected(net, num_hidden, scope='fc{}'.format(i))
            net = slim.dropout(net, is_training=is_training, scope='dropout{}'.format(i), keep_prob=0.5)
            i = i + 1
        outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs

def normalize(dict_op):
    """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
    image_op = dict_op['image']
    image_op = tf.cast(image_op, tf.float32) * (1. / 255) - 0.5
    dict_op['image'] = image_op 
    return dict_op

def augment(dict_op):
    """Placeholder for data augmentation."""
    with tf.variable_scope('augmentations'):
        image = dict_op['image']
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
        dict_op['image'] = image
    return dict_op

def salt_and_pepper(image_pl, minval=0.0, maxval=1.0, percent_flipped=0.2):
    z = 1.0 - (percent_flipped) * 0.5
    if tf.contrib.framework.is_tensor(z):
        t  = tf.distributions.Normal(0.0, 1.0).quantile(z)
    else:
        t = st.norm.ppf(z)
    
    
    random_normal  = tf.random_normal(shape=image_pl.shape)
    clipped_normal = tf.cast(tf.greater(random_normal, t), tf.float32) - \
                     tf.cast(tf.less(random_normal, -t), tf.float32)
    max_range = maxval - minval
    clipped_normal = clipped_normal * max_range
    
    aug_im = image_pl + clipped_normal
    aug_im = tf.clip_by_value(aug_im, 
                              clip_value_min=minval,
                              clip_value_max=maxval)
    return aug_im