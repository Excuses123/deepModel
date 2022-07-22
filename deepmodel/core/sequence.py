# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""

import tensorflow.compat.v1 as tf
from .layers import fc_layer

class SequencePoolingLayer(object):
    """
    序列池化
    seqs_emb:  (batch_size, T, embedding_size)
    click_len: (batch_size, )
    """

    def __init__(self, mode='mean', keep_dims=False):

        if mode not in ['mean', 'sum', 'max', 'min']:
            raise ValueError("mode must in ['mean', 'sum', 'max', 'min']")
        self.mode = mode
        self.keep_dims = keep_dims

    def run(self, seqs_emb, click_len):

        seq_max_len, embedding_size = tf.shape(seqs_emb)[1], tf.shape(seqs_emb)[2]

        if len(click_len.shape) == 2:
            click_len = tf.squeeze(click_len)

        mask = tf.sequence_mask(click_len, seq_max_len, dtype=tf.float32)
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, embedding_size])

        if self.mode == 'max':
            hist = seqs_emb + (mask - 1) * 1e8
            return tf.reduce_max(hist, axis=1, keep_dims=self.keep_dims)

        if self.mode == 'min':
            hist = seqs_emb - (mask - 1) * 1e8
            return tf.reduce_min(hist, axis=1, keep_dims=self.keep_dims)

        hist = tf.reduce_sum(seqs_emb * mask, axis=1, keep_dims=self.keep_dims)

        if self.mode == 'mean':
            hist = tf.math.divide_no_nan(hist, tf.cast(tf.expand_dims(click_len, 1), tf.float32))

        return hist


class WeightedPoolingLayer(object):
    """
    带权序列池化
    seqs_emb:  (batch_size, T, embedding_size)
    click_len: (batch_size, )
    weight:    (batch_size, T)
    """

    def __init__(self, weight_normalization=True, keep_dims=False):

        self.keep_dims = keep_dims
        self.weight_normalization = weight_normalization

    def run(self, seqs_emb, click_len, weight):

        seq_max_len, embedding_size = tf.shape(seqs_emb)[1], tf.shape(seqs_emb)[2]

        if len(click_len.shape) == 2:
            click_len = tf.squeeze(click_len)

        mask = tf.sequence_mask(click_len, seq_max_len)
        mask = tf.expand_dims(mask, -1)
        weight = tf.expand_dims(weight, -1)
        if weight.dtype != tf.float32:
            weight = tf.cast(weight, dtype=tf.float32)

        if self.weight_normalization:
            paddings = tf.ones_like(weight) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(weight)
        weight = tf.where(mask, weight, paddings)

        if self.weight_normalization:
            weight = tf.nn.softmax(weight, dim=1)

        weight = tf.tile(weight, [1, 1, embedding_size])

        hist = tf.reduce_sum(seqs_emb * weight, axis=1, keep_dims=self.keep_dims)

        return hist


class AttentionSequencePoolingLayer(object):
    """
    基于注意力机制的序列池化 for DIN
    gameid_emb(query): (batch, embedding_size)
    hist_emb(key):   (batch, sl, embedding_size)
    click_len: (batch, )
    """

    def __init__(self, att_hidden_units=(80, 40), att_activation='dice', weight_normalization=True,
                 return_score=False,
                 supports_masking=True):

        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        self.supports_masking = supports_masking

    def run(self, hist_emb, gameid_emb, click_len):

        hist_mask = tf.sequence_mask(click_len, tf.shape(hist_emb)[1])  # (batch, sl)
        hist_mask = tf.expand_dims(hist_mask, axis=1)  # (batch, 1, sl)

        gameid_emb = tf.tile(tf.expand_dims(gameid_emb, 1),
                             [1, tf.shape(hist_emb)[1], 1])  # (batch, sl, embedding_size)
        # din中原始代码的实现方法；
        atten_input = tf.concat([gameid_emb, hist_emb, gameid_emb - hist_emb, gameid_emb * hist_emb],
                                axis=-1)  # (batch, sl, embedding_size * 4)

        dnn = fc_layer(atten_input, hidden_units=self.att_hidden_units, activation=self.att_activation, name='att_dnn')
        att_score = tf.layers.dense(dnn, 1, activation=None, name='dnn3')
        outputs = tf.transpose(att_score, (0, 2, 1))  # (batch, 1, sl)

        if self.weight_normalization:
            padding = tf.ones_like(outputs) * (-2 ** 32 + 1)  # (batch, 1, sl)
        else:
            padding = tf.zeros_like(outputs)

        outputs = tf.where(hist_mask, outputs, padding)  # (batch, 1, sl)  mask为0的位置填充上一个极小值(-2 ** 32 + 1) softmax后为0

        # scale
        # outputs = outputs / (tf.shape(hist_emb)[-1] ** 0.5)    # outputs/(embedding_size ** 0.5)
        if self.weight_normalization:
            outputs = tf.nn.softmax(outputs)  # (batch, 1, sl)

        # weighted_sum_pooling  (batch, 1, sl) * (batch, sl, embedding_size)
        outputs = tf.matmul(outputs, hist_emb)  # (batch, 1, embedding_size)
        outputs = tf.squeeze(outputs, axis=1)  # (batch, embedding_size)

        return outputs


class Transformer(object):
    """

    """

    def __init__(self):
        pass

    def run(self):
        pass
