# -*- coding:utf-8 -*-

"""
save
"""
import random
import pandas as pd
from deepmodel.data import Feature, save2tfrecord

num_sample = 1000
featMap = {
    'id': [f'id_{i}'.encode() for i in range(num_sample)],
    'a': [random.randint(0, 10) for _ in range(num_sample)],
    'b': [random.randint(0, 100) for _ in range(num_sample)],
    'c': [[random.randint(0, 100) for _ in range(random.randint(1, 10))] for _ in range(num_sample)],
    'd': [random.random() for _ in range(num_sample)],
    'e': [[random.random() for _ in range(10)] for _ in range(num_sample)],
    'label': [random.randint(0, 1) for _ in range(num_sample)]
}
featMap['c_len'] = list(map(len, featMap['c']))

data = pd.DataFrame(featMap)

features = [
    Feature(name='id', dtype='string', dim=1),
    Feature(name='a', dtype='int32', dim=1),
    Feature(name='b', dtype='int64', dim=1),
    Feature(name='c', dtype='int64', dim=2, dense=True),
    Feature(name='c_len', dtype='int64', dim=1),
    Feature(name='d', dtype='float32', dim=1),
    Feature(name='e', dtype='float64', dim=2, dense=True),
    Feature(name='label', dtype='float32', dim=1)
]

save2tfrecord("./examples/test.tfrecord", data, features)


"""
load
"""
import tensorflow as tf
from deepmodel.data import Inputs, Feature

features = [
    Feature(name='id', dtype='string', dim=1),
    Feature(name='a', dtype='int32', dim=1),
    Feature(name='b', dtype='int64', dim=1),
    Feature(name='c', dtype='int64', dim=2, dense=True),
    Feature(name='c_len', dtype='int64', dim=1),
    Feature(name='d', dtype='float32', dim=1),
    Feature(name='e', dtype='float64', dim=2, dense=True),
    Feature(name='label', dtype='float32', dim=1)
]

inputs = Inputs(features, 'examples/test.tfrecord', repeats=10)
batch_x = inputs.load_batch(3)

sess = tf.Session()
sess.run(batch_x)
