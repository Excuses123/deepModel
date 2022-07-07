# -*- coding:utf-8 -*-

"""
save
"""
import pandas as pd
from deepmodel.utils import save2tfrecord

df = pd.DataFrame(
    {'a': [1, 3, 1, 2, 1], 'b': [1, 3, 1, 2, 1], 'c': [[12, 89], [11, 42, 19, 10], [1, 21, 39], [11, 7], [15, 22, 9]],
     'd': [0.1, 0.22, 0.53, 0.71, 0.4], 'e': [0.13, 0.2, 0.3, 0.1, 0.44], 'label': [1, 0, 0, 1, 0]})

sparse_feats = ['a', 'b']
dense_feats = ['d', 'e']
seq_feats = ['c']
label = 'label'

save2tfrecord("./examples/test.tfrecord", df, sparse_feats, dense_feats, seq_feats, label)


"""
load
"""
import tensorflow as tf
from deepmodel.inputs import Inputs, Feature

features = [
    Feature(name='a', dtype='int64', dim=1),
    Feature(name='b', dtype='int64', dim=1),
    Feature(name='c', dtype='int64', dim=2, dense=True),
    Feature(name='d', dtype='float32', dim=2, dense=True),
    Feature(name='e', dtype='float32', dim=2, dense=True),
    Feature(name='label', dtype='float32', dim=1)
]

inputs = Inputs(features, 'examples/test.tfrecord', repeats=10)
batch_x = inputs.load_batch(3)

sess = tf.Session()
sess.run(batch_x)
