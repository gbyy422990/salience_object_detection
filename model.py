# coding:utf-8
# Bin GAO

import os
import tensorflow as tf
import numpy as np
import vgg16

learning_rate = 1e-5
num_class = 2
loss_weight = np.array([1, 1])

g_mean = [142.53, 129.53, 120.20]


def conv2d(x, weight, n_filters, training, name, activation=tf.nn.relu):
    with tf.variable_scope('layer{}'.format(name)):
        for index, filter in enumerate(n_filters):
            conv = tf.layers.conv2d(x, filter, weight, strides=1, padding='same', activation=None,
                                    name='conv_{}'.format(index + 1))

            if training != False:
                conv = tf.layers.batch_normalization(conv, training=training, name='bn_{}'.format(index + 1))

        if activation == None:
            return conv

        conv = activation(conv, name='relu{}_{}'.format(name, index + 1))

        return conv


def pool2d(x, pool_size, pool_stride, name):
    pool = tf.layers.max_pooling2d(x, pool_size, pool_stride, name='pool_{}'.format(name), padding='same')
    return pool


def deconv2d(x, kernel, strides, training, name, output_shape, activation=None):
    kernel_shape = [kernel, kernel, 1, 1]
    strides = [1, strides, strides, 1]
    kernel = tf.get_variable('weight_{}'.format(name), shape=kernel_shape,
                             initializer=tf.random_normal_initializer(mean=0, stddev=1))
    deconv = tf.nn.conv2d_transpose(x, kernel, strides=strides, output_shape=output_shape, padding='SAME',
                                    name='upsample_{}'.format(name))

    # Now output.get_shape() is equal (?,?,?,?) which can become a problem in the
    # next layers. This can be repaired by reshaping the tensor to its shape:
    deconv = tf.reshape(deconv, output_shape)
    # now the shape is back to (?, H, W, C) or (?, C, H, W)

    if training != False:
        deconv = tf.layers.batch_normalization(deconv, training=training, name='bn{}'.format(name))
    if activation is None:
        return deconv

    deconv = activation(deconv, name='sigmoid_{}'.format(name))

    return deconv


def upsampling_2d(tensor, name, size=(2, 2)):
    h_, w_, c_ = tensor.get_shape().as_list()[1:]
    h_multi, w_multi = size
    h = h_multi * h_
    w = w_multi * w_
    target = tf.image.resize_nearest_neighbor(tensor, size=(h, w), name='deconv_{}'.format(name))

    return target


def upsampling_concat(input_A, input_B, name):
    upsampling = upsampling_2d(input_A, name=name, size=(2, 2))
    up_concat = tf.concat([upsampling, input_B], axis=-1, name='up_concat_{}'.format(name))
    return up_concat


def unet(input, training):
    # 归一化[-1,1]
    input = input / 127.5 - 1
    # input = tf.layers.conv2d(input,3,(1,1),name = 'color')   #filters:一个整数，输出空间的维度，也就是卷积核的数量

    vgg = vgg16.Vgg16()
    vgg.build(input)

    default_shape = tf.stack([tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2], 1])

    '''conv1_1 = conv2d(input, (3, 3), [32], training, name='conv1_1')
    conv1_2 = conv2d(conv1_1, (3, 3), [32], training, name='conv1_2')
    pool1 = pool2d(conv1_2,pool_size=(2,2),pool_stride=2,name='pool1')

    conv2_1=conv2d(pool1, (3, 3), [64], training, name='conv2_1')
    conv2_2=conv2d(conv2_1, (3, 3), [64], training, name='conv2_2')
    pool2 = pool2d(conv2_2,pool_size=(2,2),pool_stride=2,name='pool2')

    conv3_1 = conv2d(pool2, (3, 3), [128], training, name='conv3_1')
    conv3_2 = conv2d(conv3_1, (3, 3), [128], training, name='conv3_2')
    conv3_3 = conv2d(conv3_2, (3, 3), [128], training, name='conv3_3')
    pool3 = pool2d(conv3_3, pool_size=(2, 2), pool_stride=2, name='pool3')

    conv4_1 = conv2d(pool3, (3, 3), [256], training, name='conv4_1')
    conv4_2 = conv2d(conv4_1, (3, 3), [256], training, name='conv4_2')
    conv4_3 = conv2d(conv4_2, (3, 3), [256], training, name='conv4_3')
    pool4 = pool2d(conv4_3, pool_size=(2, 2), pool_stride=2, name='pool4')

    conv5_1 = conv2d(pool4, (3, 3), [256], training, name='conv5_1')
    conv5_2 = conv2d(conv5_1, (3, 3), [256], training, name='conv5_2')
    conv5_3 = conv2d(conv5_2, (3, 3), [256], training, name='conv5_3')
    pool5 = pool2d(conv5_3, pool_size=(3, 3), pool_stride=2, name='pool5')'''

    # pool6 = tf.layers.average_pooling2d(vgg.pool5,pool_size=(3,3),strides=1,padding='SAME',name='pool5a')

    pool6 = pool2d(vgg.pool5, pool_size=(3, 3), pool_stride=1, name='pool5a')

    conv1_dsn6 = conv2d(pool6, (7, 7), [256], training, name='conv1-dsn6')
    conv2_dsn6 = conv2d(conv1_dsn6, (7, 7), [256], training, name='conv2-dsn6')
    conv3_dsn6 = conv2d(conv2_dsn6, (1, 1), [1], training=training, name='conv3-dsn6', activation=None)
    score_dsn6_up = deconv2d(conv3_dsn6, 64, 32, training=False, name='upsample32_in_dsn6_sigmoid-dsn6',
                             output_shape=default_shape, activation=None)

    conv1_dsn5 = conv2d(vgg.conv5_3, (5, 5), [256], training, name='conv1_dsn5')
    conv2_dsn5 = conv2d(conv1_dsn5, (5, 5), [256], training, name='conv2-dsn5')
    conv3_dsn5 = conv2d(conv2_dsn5, (1, 1), [1], training=training, name='conv3-dsn5', activation=None)
    score_dsn5_up = deconv2d(conv3_dsn5, 32, 16, training=False, name='upsample16_in_dsn5_sigmoid-dsn5',
                             output_shape=default_shape, activation=None)

    conv1_dsn4 = conv2d(vgg.conv4_3, (5, 5), [128], training, name='conv1-dsn4')
    conv2_dsn4 = conv2d(conv1_dsn4, (5, 5), [128], training, name='conv2-dsn4')
    conv3_dsn4 = conv2d(conv2_dsn4, (1, 1), [1], training, name='conv3-dsn4', activation=None)
    score_dsn6_up_4 = deconv2d(conv3_dsn6, 8, 4, training, name='upsample4_dsn6',
                               output_shape=tf.shape(conv3_dsn4))
    score_dsn5_up_4 = deconv2d(conv3_dsn5, 4, 2, training, name='upsample2_dsn5',
                               output_shape=tf.shape(conv3_dsn4))
    concat_dsn4 = tf.concat([score_dsn6_up_4, score_dsn5_up_4, conv3_dsn4], axis=-1, name='concat_dsn4')
    conv4_dsn4 = conv2d(concat_dsn4, (1, 1), [1], training=training, name='conv4-dsn4', activation=None)
    score_dsn4_up = deconv2d(conv4_dsn4, 16, 8, training=False, name='upsample8_in_dsn4_sigmoid-dsn4',
                             output_shape=default_shape, activation=None)

    conv1_dsn3 = conv2d(vgg.conv3_3, (5, 5), [128], training, name='conv1-dsn3')
    conv2_dsn3 = conv2d(conv1_dsn3, (5, 5), [128], training, name='conv2-dsn3')
    conv3_dsn3 = conv2d(conv2_dsn3, (1, 1), [1], training, name='conv3-dsn3')
    score_dsn6_up_3 = deconv2d(conv3_dsn6, 16, 8, training, name='upsample8_dsn6',
                               output_shape=tf.shape(conv3_dsn3))
    score_dsn5_up_3 = deconv2d(conv3_dsn5, 8, 4, training, name='upsample4_dsn5',
                               output_shape=tf.shape(conv3_dsn3))
    concat_dsn3 = tf.concat([score_dsn6_up_3, score_dsn5_up_3, conv3_dsn3], axis=-1, name='concat_dsn3')
    conv4_dsn3 = conv2d(concat_dsn3, (1, 1), [1], training=training, name='conv4-dsn3', activation=None)
    score_dsn3_up = deconv2d(conv4_dsn3, 8, 4, training=False, name='upsample4_in_dsn3_sigmoid-dsn3',
                             output_shape=default_shape, activation=None)

    conv1_dsn2 = conv2d(vgg.conv2_2, (3, 3), [64], training, name='conv1-dsn2')
    conv2_dsn2 = conv2d(conv1_dsn2, (3, 3), [64], training, name='conv2-dsn2')
    conv3_dsn2 = conv2d(conv2_dsn2, (1, 1), [1], training, name='conv3-dsn2')
    score_dsn6_up_2 = deconv2d(conv3_dsn6, 32, 16, training, name='upsample16_dsn6',
                               output_shape=tf.shape(conv3_dsn2))
    score_dsn5_up_2 = deconv2d(conv3_dsn5, 16, 8, training, name='upsample8_dsn5',
                               output_shape=tf.shape(conv3_dsn2))
    score_dsn4_up_2 = deconv2d(conv3_dsn4, 8, 4, training, name='upsample4_dsn4',
                               output_shape=tf.shape(conv3_dsn2))
    score_dsn3_up_2 = deconv2d(conv3_dsn3, 4, 2, training, name='upsample2_dsn3',
                               output_shape=tf.shape(conv3_dsn2))
    concat_dsn2 = tf.concat([score_dsn6_up_2, score_dsn5_up_2, score_dsn4_up_2, score_dsn3_up_2, conv3_dsn2], axis=-1,
                            name='concat_dsn2')
    conv4_dsn2 = conv2d(concat_dsn2, (1, 1), [1], training, name='conv4-dsn2', activation=None)
    score_dsn2_up = deconv2d(conv4_dsn2, 4, 2, training=False, name='upsample2_in_dsn2_sigmoid-dsn2',
                             output_shape=default_shape, activation=None)

    conv1_dsn1 = conv2d(vgg.conv1_2, (3, 3), [64], training, name='conv1-dsn1')
    conv2_dsn1 = conv2d(conv1_dsn1, (3, 3), [64], training, name='conv2-dsn1')
    conv3_dsn1 = conv2d(conv2_dsn1, (1, 1), [1], training, name='conv3-dsn1', activation=None)
    score_dsn6_up_1 = deconv2d(conv3_dsn6, 64, 32, training, name='upsample32_dsn6',
                               output_shape=tf.shape(conv3_dsn1))
    score_dsn5_up_1 = deconv2d(conv3_dsn5, 32, 16, training, name='upsample16_dsn5',
                               output_shape=tf.shape(conv3_dsn1))
    score_dsn4_up_1 = deconv2d(conv3_dsn4, 16, 8, training, name='upsample8_dsn4',
                               output_shape=tf.shape(conv3_dsn1))
    score_dsn3_up_1 = deconv2d(conv3_dsn3, 8, 4, training, name='upsample4_dsn3',
                               output_shape=tf.shape(conv3_dsn1))
    concat_dsn1 = tf.concat([score_dsn6_up_1, score_dsn5_up_1, score_dsn4_up_1, score_dsn3_up_1, conv3_dsn1], axis=-1,
                            name='concat_dsn1')
    score_dsn1_up = conv2d(concat_dsn1, (1, 1), [1], training=False, name='conv4-dsn1', activation=None)

    concat_upscore = tf.concat(
        [score_dsn6_up, score_dsn5_up, score_dsn4_up, score_dsn3_up, score_dsn2_up, score_dsn1_up],
        axis=-1, name='concat')
    upscore_fuse = conv2d(concat_upscore, (1, 1), [1], training=False, name='new-score-weighting', activation=None)
    # upscore_fuse = tf.layers.conv2d(concat_upscore,filters=1,kernel_size=(1,1),strides=(1,1),padding='SAME',name='output',activation=tf.nn.sigmoid)

    return score_dsn6_up, score_dsn5_up, score_dsn4_up, score_dsn3_up, score_dsn2_up, score_dsn1_up, upscore_fuse


# IOU损失
def loss_IOU(y_pred, y_true):
    H, W, _ = y_pred.get_shape().as_list()[1:]
    flat_logits = tf.reshape(y_pred, [-1, H * W])
    flat_labels = tf.reshape(y_true, [-1, H * W])
    intersection = 2 * tf.reduce_sum(flat_logits * flat_labels, axis=1) + 1e-7
    denominator = tf.reduce_sum(flat_logits, axis=1) + tf.reduce_sum(flat_labels, axis=1) + 1e-7
    iou = 1 - tf.reduce_mean(intersection / denominator)

    return iou


def loss_CE(y_pred, y_true):
    '''flat_logits = tf.reshape(y_pred,[-1,num_class])
    flat_labels = tf.reshape(y_true,[-1,num_class])
    class_weights = tf.constant(loss_weight,dtype=np.float32)
    weight_map = tf.multiply(flat_labels,class_weights)
    weight_map = tf.reduce_sum(weight_map,axis=1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits(labels=flat_labels,logits=flat_logits)

    #weighted_loss = tf.multiply(loss_map,weight_map)

    #cross_entropy_mean = tf.reduce_mean(weighted_loss)
    cross_entropy_mean = -tf.reduce_mean(tf.reduce_sum(y_true*tf.log(y_pred)))'''

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    return cross_entropy_mean


def train_op(loss, learning_rate):
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    return optimizer.minimize(loss, global_step=global_step)
