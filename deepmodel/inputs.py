# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""

import os
import tensorflow.compat.v1 as tf



class Feature(object):
    """ 特征信息类 """
    def __init__(self, name, dtype='int64', dim=1, dense=False):
        self.name = name
        self.dtype = dtype
        self.dim = dim
        self.dense = dense


class Inputs(object):
    """ 批量加载tfrecord数据 """

    def __init__(self, features, filepath, repeats=1, shuffle_size=1, prefetch_size=1, num_parallel_calls=16):
        self.filepath = filepath
        self.repeats = repeats
        self.shuffle_size = shuffle_size
        self.prefetch_size = prefetch_size
        self.num_parallel_calls = num_parallel_calls

        self.toDense = []
        self.keepDim = []
        self.dicts = {}
        for feature in features:
            name = feature.name
            if feature.dim == 1:
                self.dicts[name] = tf.FixedLenFeature([1], dtype=feature.dtype)
                self.keepDim.append(name)
            else:
                self.dicts[name] = tf.VarLenFeature(dtype=feature.dtype)
                if feature.dense:
                    self.toDense.append(name)

    def parser(self, record):
        return tf.parse_single_example(record, self.dicts)

    def load_batch(self, batch_size):

        dataset = tf.data.TFRecordDataset(self.filenames())

        iterator = dataset \
            .map(self.parser, num_parallel_calls=self.num_parallel_calls) \
            .repeat(self.repeats) \
            .shuffle(buffer_size=self.shuffle_size) \
            .batch(batch_size) \
            .prefetch(buffer_size=self.prefetch_size) \
            .make_one_shot_iterator()

        batch_x = iterator.get_next()

        for feat in self.toDense:
            batch_x[feat] = tf.sparse_tensor_to_dense(batch_x[feat])
        for feat in self.keepDim:
            batch_x[feat] = batch_x[feat][:, 0]

        return batch_x

    def filenames(self):
        if not os.path.isdir(self.filepath):
            return self.filepath
        else:
            return [os.path.join(self.filepath, i) for i in os.listdir(self.filepath)]
