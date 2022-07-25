# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""


import tensorflow.compat.v1 as tf
from ..core.layers import fc_layer


class DeepFM(object):
    """
    deepfm model
    """

    def __init__(self, args, features, batch, keep_prob):
        self.args = args
        self.features = features
        self.batch = batch
        self.keep_prob = keep_prob

        self.train_feat = []
        self.emb_feat = []
        for feature in features:
            if feature.for_train:
                self.train_feat.append(feature)
            if feature.emb_count:
                self.emb_feat.append(feature)
            if feature.label:
                self.label = batch[feature.name]

        self.build_model()

    def build_model(self):
        """FM"""
        # lr参数、交叉项参数
        w1, v = {}, {}
        for feat in self.emb_feat:
            w1[f'{feat.name}_w1'] = tf.get_variable(f'{feat.name}_w1', shape=[feat.emb_count, 1])
            v[f'{feat.name}_emb'] = tf.get_variable(f'{feat.name}_emb', shape=[feat.emb_count, self.args.K])
        # 偏置
        b = tf.get_variable('bias', shape=[1], initializer=tf.constant_initializer(0.01))

        # line = tf.sparse_tensor_dense_matmul(self.X, w1)   shape: (batch, 1)
        # x1 = tf.sparse_tensor_dense_matmul(self.X, v)    shape: (batch, k)
        # x2 = tf.sparse_tensor_dense_matmul(self.X, tf.pow(v, 2):   值都为1，tf.pow(self.X, 2) = self.X
        line, x1, x2 = [], [], []
        for feat in self.train_feat:
            if feat.emb_count and feat.dtype.startswith('int'):
                line.append(tf.nn.embedding_lookup(w1[f'{feat.name}_w1'], self.batch[feat.name]) if feat.dim < 2
                            else SeqsPool(w1[f'{feat.name}_w1'], self.batch[feat.name], self.batch[f'{feat.name}_len']))

                x1.append(tf.nn.embedding_lookup(v[f'{feat.name}_emb'], self.batch[feat.name]) if feat.dim < 2
                          else SeqsPool(v[f'{feat.name}_emb'], self.batch[feat.name], self.batch[f'{feat.name}_len']))

                x2.append(
                    tf.nn.embedding_lookup(tf.pow(v[f'{feat.name}_emb'], 2), self.batch[feat.name]) if feat.dim < 2
                    else SeqsPool(tf.pow(v[f'{feat.name}_emb'], 2), self.batch[feat.name],
                                  self.batch[f'{feat.name}_len']))
            else:
                line.append(tf.matmul(tf.reshape(self.batch[feat.name], [-1, 1]) if feat.dim < 2
                                      else self.batch[feat.name], w1[f'{feat.name}_w1']))

                x1.append(tf.matmul(tf.reshape(self.batch[feat.name], [-1, 1]) if feat.dim < 2
                                    else self.batch[feat.name], v[f'{feat.name}_emb']))

                x2.append(tf.matmul(tf.reshape(tf.pow(self.batch[feat.name], 2), [-1, 1]) if feat.dim < 2
                                    else self.batch[feat.name], tf.pow(v[f'{feat.name}_emb'], 2)))

        self.line = tf.add_n(line) + b
        self.x1 = tf.add_n(x1)
        self.x2 = tf.add_n(x2)

        self.inter = tf.multiply(0.5,
                                 tf.reduce_sum(
                                     tf.subtract(tf.pow(self.x1, 2), self.x2),  # (batch, K)
                                     axis=1, keep_dims=True)
                                 )  # (batch, 1)

        self.y_fm = tf.add(self.line, self.inter)  # (batch, 1)

        """DNN"""
        # tf.nn.embedding_lookup(v, self.feature_inds)  shape: (batch, field_num * k)
        dnn_emb = tf.concat(x1, axis=1)

        dnn_emb = fc_layer(dnn_emb, hidden_units=[512, 256], use_bn=True, training=self.args.bn_training, keep_prob=0.8)
        self.y_dnn = tf.layers.dense(dnn_emb, 1, use_bias=False, activation=None, name="y_dnn")

        # 输出
        self.logits = tf.add(self.y_fm, self.y_dnn)
        self.probability = tf.nn.sigmoid(self.logits)[:, 0]

    def train(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits[:, 0], labels=self.label))
        self.log_loss = tf.losses.log_loss(self.label, self.probability)
        self.mse_loss = tf.reduce_mean(tf.pow(self.label - self.probability, 2))

        tf.summary.scalar("loss", self.loss)

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def test(self, cols):
        self.output = {
            'probability': self.probability,
        }
        for col in cols:
            self.output[col] = self.batch[col]

    def pred(self):
        self.output = {
            'probability': self.probability,
        }

def SeqsPool(embedding, item_seqs, click_len, keep_dims=False):
    seqs_emb = tf.nn.embedding_lookup(embedding, item_seqs)
    mask = tf.sequence_mask(click_len, tf.shape(seqs_emb)[1], dtype=tf.float32)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, tf.shape(seqs_emb)[2]])
    seqs_emb *= mask

    return tf.reduce_sum(seqs_emb, axis=1, keep_dims=keep_dims)
