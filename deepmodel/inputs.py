# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""

import os
import tensorflow as tf

dtypes = {'int32': tf.int32, 'int64': tf.int64, 'float32': tf.float32, 'float64': tf.float64, 'str': tf.string}

class Inputs(object):

    def __init__(self, features, filepath, repeats=1, shuffle_size=1, prefetch_size=1, num_parallel_calls=16):
        self.filepath = filepath
        self.repeats = repeats
        self.shuffle_size = shuffle_size
        self.prefetch_size = prefetch_size
        self.num_parallel_calls = num_parallel_calls

        self.toDense = []
        self.keepDim = []
        self.dicts = {}
        for k, v in features.items():
            if v['arrayType'] == 1:
                self.dicts[k] = tf.FixedLenFeature([1], dtype=dtypes[v['dtype']])
                self.keepDim.append(k)
            else:
                self.dicts[k] = tf.VarLenFeature(dtype=dtypes[v['dtype']])
                if v['to_dense']:
                    self.toDense.append(k)

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
        return [os.path.join(self.filepath, i) for i in os.listdir(self.filepath) if i != '_SUCCESS']






