# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""

import os
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile

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


def ckpt2pb(args, features, orgin_model, out_name, in_names=None):
    tf.reset_default_graph()
    graph = tf.get_default_graph()

    op_names = []
    with graph.as_default():
        batch_x = {}
        features = features if in_names is None \
            else [feature for feature in features if feature.name in in_names]
        for feature in features:
            op_names.append(feature.name)
            batch_x[feature.name] = tf.placeholder(feature.dtype, [None] if feature.dim == 1 else [None, None],
                                                   name=feature.name)

        model = orgin_model(args, features, batch_x, 1.0)
        output = tf.identity(model.pred, name=out_name)
        op_names.append(out_name)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess = load_ckpt(sess, args.save_path)

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, op_names)

        tf.train.write_graph(output_graph_def, '.', args.save_path + '/' + args.model_name, as_text=False)


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



