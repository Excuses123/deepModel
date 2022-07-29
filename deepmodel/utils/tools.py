# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""


import os
import time
import datetime
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.platform import gfile


def choice_gpu(memory=1024):
    """选择当前空闲内存最大的gpu
    memory: Int, 选择gpu内存的最小阈值 单位m.
    """
    while True:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > gpu_tmp')
        gpu_memory = [int(x.split()[2]) for x in open('gpu_tmp', 'r').readlines()]
        os.system('rm gpu_tmp')

        gpu = np.argmax(gpu_memory)
        if gpu_memory[gpu] >= memory:
            break
        time.sleep(60)
    return gpu


def get_datekey(datekey, day, format="%Y%m%d"):
    return (datetime.datetime.strptime(datekey, format) + datetime.timedelta(days=day)).strftime(format)


def one_hot(label, num_classes=None, dtype=tf.float32):
    label = tf.cast(label, tf.int64)
    if label.shape.__len__() == 1:
        label = tf.expand_dims(label, axis=-1)

    if not num_classes:
        num_classes = tf.reduce_max(label) + 1

    row = tf.expand_dims(tf.range(tf.shape(label)[0]), axis=-1)
    row = tf.reshape(tf.tile(tf.cast(row, tf.int64), [1, tf.shape(label)[1]]), [-1, 1])

    col = tf.reshape(tf.expand_dims(label, axis=-1), [-1, 1])

    cates = tf.SparseTensor(indices=tf.cast(tf.concat([row, col], axis=1), tf.int64),
                            values=tf.ones(tf.shape(col)[0], dtype),
                            dense_shape=[tf.shape(label)[0], num_classes])

    return tf.sparse_tensor_to_dense(cates)


def save_ckpt(sess, save_path, global_step, name='model'):
    saver = tf.train.Saver()
    saver.save(sess, save_path=os.path.join(save_path, name), global_step=global_step)


def load_ckpt(sess, path):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and ckpt.model_checkpoint_path:
        step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, save_path=ckpt.model_checkpoint_path)
        print("load model of step %s success" % step)
    else:
        print("no checkpoint!")

    return sess


def ckpt2pb(args, features, orgin_model, out_names=None, in_names=None, **kwargs):
    tf.reset_default_graph()
    graph = tf.get_default_graph()

    allowed_kwargs = {
        'pred_feature',
    }
    generic_utils.validate_kwargs(kwargs, allowed_kwargs)
    pred_feature = kwargs.pop('pred_feature', None)

    op_names = []
    with graph.as_default():
        batch_x = {}
        features = features if in_names is None else \
            [feature for feature in features if feature.name in in_names]
        for feature in features:
            op_names.append(feature.name)
            batch_x[feature.name] = tf.placeholder(feature.dtype, [None, None], name=feature.name)

        dense_feats = args.dense_feats if args.contains('dense_feats') else {}

        size = tf.shape(batch_x[args.item_name])[1]

        for feature in features:
            feat = batch_x[feature.name]
            feat_size = tf.shape(feat)[1] if feature.dim == 1 else tf.shape(feat)[0]
            if orgin_model.type == 'Rank':
                feat = tf.cond(feat_size < size, lambda: tf.tile(feat, [size, 1]), lambda: feat)
            batch_x[feature.name] = tf.reshape(feat, [-1]) if feature.dim == 1 else feat

            if feature.name in dense_feats:
                batch_x[feature.name] = tf.gather(dense_feats[feature.name], batch_x[args.item_name])

        if pred_feature is not None:
            batch_x[pred_feature.name] = tf.placeholder(pred_feature.dtype, [1, None], name=pred_feature.name)
            op_names.append(pred_feature.name)

        args.bn_training = False
        model = orgin_model(args, features, batch_x, 1.0)
        model.pred(pred_feature.name) if pred_feature else model.pred()

        output = {}
        model_out = model.output
        out_names = model_out.keys() if out_names is None \
            else out_names
        for out_name in out_names:
            output[out_name] = tf.identity(model_out[out_name], name=out_name)
            op_names.append(out_name)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess = load_ckpt(sess, args.model_path)

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, op_names)

        tf.train.write_graph(output_graph_def, '.', os.path.join(args.model_path, args.model_name), as_text=False)


class load_pb(object):
    def __init__(self, path, features, out_name, in_names=None):
        self.path = path
        self.out_name = out_name
        self.features = features if in_names is None \
            else [feature for feature in features if feature.name in in_names]
        self.__load()

    def __load(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        with gfile.FastGFile(self.path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
        self.sess.run(tf.global_variables_initializer())
        # 输入
        self.inputMap = {}
        for feature in self.features:
            self.inputMap[feature.name] = self.sess.graph.get_tensor_by_name(f'{feature.name}:0')
        # 输出
        self.output = self.sess.graph.get_tensor_by_name(f'{self.out_name}:0')

    def predict(self, feed_dict):
        return self.sess.run(self.output,
                             feed_dict=dict([(v, feed_dict[k]) for k, v in self.inputMap.items() if k in feed_dict]))



