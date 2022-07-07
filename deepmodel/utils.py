# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""

import tensorflow.compat.v1 as tf


def save2tfrecord(filename, data, sparse_feats, dense_feats, seq_feats, label_name):
    writer = tf.python_io.TFRecordWriter(filename)
    for ind, line in data.iterrows():
        feats_dict = {feat: tf.train.Feature(int64_list=tf.train.Int64List(value=[int(line[feat])])) for feat in
                      sparse_feats}
        feats_dict.update(
            {feat: tf.train.Feature(float_list=tf.train.FloatList(value=[line[feat]])) for feat in dense_feats})
        feats_dict.update(
            {feat: tf.train.Feature(int64_list=tf.train.Int64List(value=line[feat])) for feat in seq_feats})
        feats_dict.update(
            {f"{feat}_len": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(line[feat])])) for feat in
             seq_feats})
        feats_dict[label_name] = tf.train.Feature(float_list=tf.train.FloatList(value=[line[label_name]]))
        tf_example = tf.train.Example(features=tf.train.Features(feature=feats_dict))
        writer.write(tf_example.SerializeToString())
    writer.close()

