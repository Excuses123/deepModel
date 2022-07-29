# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""


import os
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from .utils.loss import loss_layer


class Feature(object):
    """ 特征信息类
    # Arguments
        name: String, 特征类名.
        name_from: String, 源数据对应的特征名.
        dtype: String, 返回的数据类型.
        dtype_from: String, 源数据类型.
        dim: Int, 特征纬度.
        dense: Boolean, 是否转换为稠密特征.
    """
    def __init__(self, name, name_from=None, dtype='int64',
                 dtype_from=None, dim=1, dense=False, **kwargs):
        self.name = name
        self.name_from = name \
            if name_from is None else name_from
        self.dtype = dtype
        self.dtype_from = dtype \
            if dtype_from is None else dtype_from
        self.dim = dim
        self.dense = dense

        allowed_kwargs = {
            'emb_count',
            'emb_size',
            'emb_share',
            'feat_size',
            'for_train',
            'attention',
            'label',
            'label_type',  # ['single', 'multi']
            'num_class',
            'onehot',
            'loss'   # ['mae', 'mse', 'rmse', 'softmax_cross_entropy', 'sigmoid_cross_entropy']
        }
        # Validate optional keyword arguments.
        generic_utils.validate_kwargs(kwargs, allowed_kwargs)

        self.for_train = kwargs.pop('for_train', True)

        self.emb_count = kwargs.pop('emb_count', None)

        self.emb_size = kwargs.pop('emb_size', 1)

        self.feat_size = kwargs.pop('feat_size', 1)

        self.emb_share = kwargs.pop('emb_share', None)

        self.attention = kwargs.pop('attention', None)

        self.label = kwargs.pop('label', False)
        self.label_type = kwargs.pop('label_type', 'single')

        self.num_class = kwargs.pop('num_class', None)

        self.onehot = kwargs.pop('onehot', False)

        loss = kwargs.pop('loss', 'softmax_cross_entropy')
        self.loss = loss_layer(loss)


class TFRecordLoader(object):
    """ 批量加载tfrecord数据 """

    def __init__(self, features, filepath, repeats=1, shuffle_size=1,
                 prefetch_size=1, num_parallel_calls=16):
        self.features = features
        self.filepath = filepath
        self.repeats = repeats
        self.shuffle_size = shuffle_size
        self.prefetch_size = prefetch_size
        self.num_parallel_calls = num_parallel_calls

        self.toDense = []
        self.keepDim = []
        self.dicts = {}
        for feature in self.features:
            name = feature.name_from
            if feature.dim == 1:
                self.dicts[name] = tf.FixedLenFeature([1], dtype=self.dtype(feature.dtype_from))
                self.keepDim.append(name)
            else:
                self.dicts[name] = tf.VarLenFeature(dtype=self.dtype(feature.dtype_from))
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

        for feat in self.features:
            batch_x[feat.name] = tf.cast(batch_x.pop(feat.name_from), feat.dtype)

        return batch_x

    def dtype(self, dtype):
        if dtype.startswith('int'):
            return 'int64'
        elif dtype.startswith('float'):
            return 'float32'
        else:
            return 'string'

    def filenames(self):
        if not os.path.isdir(self.filepath):
            return self.filepath
        else:
            return [os.path.join(self.filepath, i) for i in os.listdir(self.filepath)]


class DataFrameLoader(object):
    """ 生成dataframe批数据 """

    def __init__(self, features, data, repeats=1, shuffle_size=1, prefetch_size=1):
        self.features = features
        self.data = data
        self.repeats = repeats
        self.shuffle_size = shuffle_size
        self.prefetch_size = prefetch_size
        self.__to_dict()

    def __to_dict(self):
        self.data_dict = {}
        for feature in self.features:
            if feature.dim == 1:
                self.data_dict[feature.name] = self.data[feature.name_from].values
            else:
                self.data_dict[feature.name] = pad_sequences(self.data[feature.name_from].to_list(),
                                                             dtype=feature.dtype, padding='post')

    def load_batch(self, batch_size):

        iterator = tf.data.Dataset.from_tensor_slices(self.data_dict) \
            .repeat(self.repeats) \
            .shuffle(buffer_size=self.shuffle_size) \
            .batch(batch_size) \
            .prefetch(buffer_size=self.prefetch_size) \
            .make_one_shot_iterator()

        batch_x = iterator.get_next()

        for feat in self.features:
            batch_x[feat.name] = tf.cast(batch_x[feat.name], feat.dtype)

        return batch_x


def __Int64List(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def __FloatList(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def __BytesList(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def __FeatureList(value):
    return tf.train.Feature(feature_list=tf.train.FloatList(value=value))


def save2tfrecord(filename, data, features):
    writer = tf.python_io.TFRecordWriter(filename)
    for _, row in data.iterrows():
        feats_dict = {}
        for feature in features:
            if feature.dtype.startswith("int"):
                feats_dict[feature.name] = __Int64List([row[feature.name]] if feature.dim == 1
                                                     else row[feature.name])
            if feature.dtype.startswith("float"):
                feats_dict[feature.name] = __FloatList([row[feature.name]] if feature.dim == 1
                                                     else row[feature.name])
            if feature.dtype.startswith("str"):
                feats_dict[feature.name] = __BytesList([row[feature.name]] if feature.dim == 1
                                                     else row[feature.name])
        tf_example = tf.train.Example(features=tf.train.Features(feature=feats_dict))
        writer.write(tf_example.SerializeToString())
    writer.close()



