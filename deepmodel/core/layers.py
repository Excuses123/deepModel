# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""


import tensorflow.compat.v1 as tf


"""
全连接
"""
def fc_layer(inputs, hidden_units, activation=tf.nn.relu, use_bn=False,
             training=True, keep_prob=1.0, name='fc', seed=1024):

    outputs = inputs if not use_bn else \
        tf.layers.batch_normalization(inputs=inputs, training=training)

    for i, units in enumerate(hidden_units):
        fc = tf.layers.dense(outputs, units, name=f'{name}_{i}')
        fc = activation(fc, name=f'activation_{i}')
        fc = tf.nn.dropout(fc, keep_prob=keep_prob, seed=seed)
        outputs = fc

    return outputs

"""
池化
"""
def pool_layer(seqs_emb, click_len, weight=None, mode='mean', weight_normalization=True,
               keep_dims=False):

    seq_max_len, embedding_size = tf.shape(seqs_emb)[1], tf.shape(seqs_emb)[2]
    if len(click_len.shape) == 2:
        click_len = tf.squeeze(click_len)

    mask = tf.sequence_mask(click_len, seq_max_len, dtype=tf.float32)
    mask = tf.expand_dims(mask, -1)

    if weight is None:
        mask = tf.tile(mask, [1, 1, embedding_size])
        if mode == 'max':
            seqs_emb = seqs_emb + (mask - 1) * 1e8
            return tf.reduce_max(seqs_emb, axis=1, keep_dims=keep_dims)

        if mode == 'min':
            seqs_emb = seqs_emb - (mask - 1) * 1e8
            return tf.reduce_min(seqs_emb, axis=1, keep_dims=keep_dims)

        seqs_emb = tf.reduce_sum(seqs_emb * mask, axis=1, keep_dims=keep_dims)
        if mode == 'mean':
            seqs_emb = tf.math.divide_no_nan(seqs_emb, tf.cast(tf.expand_dims(click_len, 1), tf.float32))
        return seqs_emb
    else:
        weight = tf.expand_dims(weight, -1)
        if weight.dtype != tf.float32:
            weight = tf.cast(weight, dtype=tf.float32)

        paddings = tf.ones_like(weight) * (-2 ** 32 + 1) if weight_normalization else tf.zeros_like(weight)
        weight = tf.where(tf.cast(mask, tf.bool), weight, paddings)

        if weight_normalization:
            weight = tf.nn.softmax(weight, dim=1)

        weight = tf.tile(weight, [1, 1, embedding_size])
        seqs_emb = tf.reduce_sum(seqs_emb * weight, axis=1, keep_dims=keep_dims)

        if weight_normalization:
            return seqs_emb
        else:
            return tf.math.divide_no_nan(seqs_emb, tf.reduce_sum(weight, axis=1))

"""
注意力机制
"""
def attention_layer(key, query, mask, hidden_units=(80, 40), activation=tf.nn.sigmoid,
                    weight_normalization=True):

    seq_len = tf.shape(key)[1]

    mask = tf.sequence_mask(mask, seq_len)
    mask = tf.expand_dims(mask, axis=1)

    query = tf.tile(tf.expand_dims(query, 1), [1, seq_len, 1])

    # din中原始代码的实现方法
    atten_input = tf.concat([query, key, query - key, query * key], axis=-1)

    atten_output = fc_layer(atten_input, hidden_units=hidden_units, activation=activation, name='att_dnn')
    att_score = tf.layers.dense(atten_output, 1, activation=None, name='fc_att')
    att_score = tf.transpose(att_score, (0, 2, 1))

    if weight_normalization:
        padding = tf.ones_like(att_score) * (-2 ** 32 + 1)
    else:
        padding = tf.zeros_like(att_score)

    att_score = tf.where(mask, att_score, padding)

    if weight_normalization:
        att_score = tf.nn.softmax(att_score)

    outputs = tf.matmul(att_score, key)
    outputs = tf.squeeze(outputs, axis=1)  # (batch, embedding_size)

    return outputs



