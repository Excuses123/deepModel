# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""


import tensorflow as tf


def mae(logits, labels):
    return tf.reduce_mean(tf.abs(labels - logits))


def mse(logits, labels):
    return tf.reduce_mean(tf.pow(labels - logits, 2))


def rmse(logits, labels):
    return tf.sqrt(mse(logits, labels))


def softmax_cross_entropy(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))


def sigmoid_cross_entropy(logits, labels):
    return tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), axis=-1))


def loss_layer(cost_fun):
    if not isinstance(cost_fun, str):
        raise ValueError(f"{cost_fun} is not str !")
    if cost_fun == 'mae':
        return mse
    elif cost_fun == 'mse':
        return mse
    elif cost_fun == 'rmse':
        return rmse
    elif cost_fun == 'softmax_cross_entropy':
        return softmax_cross_entropy
    elif cost_fun == 'sigmoid_cross_entropy':
        return sigmoid_cross_entropy
    else:
        raise ValueError(f"cost function of {cost_fun} is not supported !")






