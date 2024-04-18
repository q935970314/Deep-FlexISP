#/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 22-02-09
# @Author  : zhangyuqian@xiaomi.com

import tensorflow.compat.v1 as tf
import tf_slim as slim


def relu(x):
    return tf.nn.relu(x)


def upsample_and_sum(x1, x2,output_channels,in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.add(deconv,x2)
    return deconv_output

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output





def sc_net_1f(input):
    # scratch capture single frame denoise network
    # unet_2down_res_relu_64c5
    
    with slim.arg_scope([slim.conv2d], weights_initializer=slim.variance_scaling_initializer(),
                        weights_regularizer=slim.l1_regularizer(0.0001),biases_initializer = None):

        conv1 = slim.conv2d(input, 64, [3, 3], rate=1, activation_fn=relu, scope='conv1_1')
        res_conv1 = slim.conv2d(conv1, 64, [3, 3], rate=1, activation_fn=relu, scope='res_conv1_1')
        res_conv1 = slim.conv2d(res_conv1, 64, [3, 3], rate=1, activation_fn=relu, scope='res_conv1_2')
        res_block1 = conv1 + res_conv1


        pool2 = slim.avg_pool2d(res_block1,[2,2],padding='SAME')
        res_conv2 = slim.conv2d(pool2, 64, [3, 3], rate=1, activation_fn=relu, scope='res_conv2_1')
        res_conv2 = slim.conv2d(res_conv2, 64, [3, 3], rate=1, activation_fn=relu, scope='res_conv2_2')
        res_block2 = pool2 + res_conv2

        pool3 = slim.avg_pool2d(res_block2,[2,2],padding='SAME')
        res_conv3 = slim.conv2d(pool3, 64, [3, 3], rate=1, activation_fn=relu, scope='res_conv3_1')
        res_conv3 = slim.conv2d(res_conv3, 64, [3, 3], rate=1, activation_fn=relu, scope='res_conv3_2')
        res_block3 = pool3 + res_conv3

        deconv1 = upsample_and_sum(res_block3, res_block2, 64, 64)

        conv4 = slim.conv2d(deconv1, 64, [3, 3], rate=1, stride=1, activation_fn=relu, scope='conv4_1')
        res_conv4 = slim.conv2d(conv4, 64, [3, 3], rate=1, activation_fn=relu, scope='res_conv4_1')
        res_conv4 = slim.conv2d(res_conv4, 64, [3, 3], rate=1, activation_fn=relu, scope='res_conv4_2')
        res_block4 = conv4 + res_conv4

        deconv2 = upsample_and_sum(res_block4, res_block1, 64, 64)

        conv5 = slim.conv2d(deconv2, 64, [3, 3], rate=1, stride=1, activation_fn=relu, scope='conv5_1')
        res_conv5 = slim.conv2d(conv5, 64, [3, 3], rate=1, activation_fn=relu, scope='res_conv5_1')
        res_conv5 = slim.conv2d(res_conv5, 64, [3, 3], rate=1, activation_fn=relu, scope='res_conv5_2')
        res_block5 = conv5 + res_conv5

        conv6 = slim.conv2d(res_block5, 64, [3, 3], rate=1, stride=1, activation_fn=relu, scope='conv6_1')
        conv7 = slim.conv2d(conv6, 4, [3, 3], rate=1, stride=1, activation_fn=None, scope='conv7_1')

        out = conv7

    return out




def sc_net(input):
    
    with slim.arg_scope([slim.conv2d], weights_initializer=slim.variance_scaling_initializer(),
                        weights_regularizer=slim.l1_regularizer(0.0001),biases_initializer = None):

        conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=relu, scope='g_conv1_1')
        conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=relu, scope='g_conv1_2')
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

        conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=relu, scope='g_conv2_1')
        conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=relu, scope='g_conv2_2')
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=relu, scope='g_conv3_1')
        conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=relu, scope='g_conv3_2')
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=relu, scope='g_conv4_1')
        conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=relu, scope='g_conv4_2')
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=relu, scope='g_conv5_1')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=relu, scope='g_conv5_2')

        up6 = upsample_and_concat(conv5, conv4, 256, 512)
        conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=relu, scope='g_conv6_1')
        conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=relu, scope='g_conv6_2')

        up7 = upsample_and_concat(conv6, conv3, 128, 256)
        conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=relu, scope='g_conv7_1')
        conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=relu, scope='g_conv7_2')

        up8 = upsample_and_concat(conv7, conv2, 64, 128)
        conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=relu, scope='g_conv8_1')
        conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=relu, scope='g_conv8_2')

        up9 = upsample_and_concat(conv8, conv1, 32, 64)
        conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=relu, scope='g_conv9_1')
        conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=relu, scope='g_conv9_2')

        conv10 = slim.conv2d(conv9, 4, [1, 1], rate=1, activation_fn=relu, scope='g_conv10')

        out = conv10

    return out




















