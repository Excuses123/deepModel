# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""


import tensorflow as tf

"""
dice激活函数:
  sigmoid( BN(input) ) * input +  (1 - sigmoid( BN(input) )) * alpha * input
"""
def dice(inputs, name='', training=True):
    alphas = tf.get_variable(name=f'alpha_{name}',
                             shape=inputs.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    inputs_bn = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        epsilon=1e-9,
        center=False,
        scale=False,
        training=training)
    p = tf.sigmoid(inputs_bn)
    return p * inputs + (1.0 - p) * alphas * inputs

