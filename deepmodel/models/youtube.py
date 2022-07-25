# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""


import tensorflow.compat.v1 as tf
from ..core.layers import fc_layer, pool_layer


class YouTubeRecall(object):
    """
    youtube recall model
    """
    def __init__(self, args, features, batch, keep_prob):

        self.args = args
        self.features = features
        self.batch = batch
        self.keep_prob = keep_prob
        self.item_name = args.item_name

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

        self.embeddings = {}
        for feat in self.emb_feat:
            shape = [feat.emb_count, feat.emb_size]
            self.embeddings[f'{feat.name}_emb'] = (tf.get_variable(f'{feat.name}_emb', shape=shape), shape)

        self.input_b = tf.get_variable("input_b", [self.embeddings[f'{self.item_name}_emb'][1][0]],
                                       initializer=tf.constant_initializer(0.0))

        concat_list, concat_dim = [], 0

        for feat in self.train_feat:
            if feat.dtype.startswith('int'):
                f_emb = tf.nn.embedding_lookup(self.embeddings[f'{feat.emb_share}_emb'][0] if feat.emb_share
                                               else self.embeddings[f'{feat.name}_emb'][0], self.batch[feat.name])
                shape = self.embeddings[f'{feat.emb_share}_emb'][1] if feat.emb_share else self.embeddings[f'{feat.name}_emb'][1]
                if feat.dim == 2:
                    f_emb = pool_layer(f_emb, self.batch[f'{feat.name}_len'])
                concat_list.append(f_emb)
                concat_dim += shape[1]
            else:
                f_emb = tf.reshape(self.batch[feat.name], [-1, 1]) if feat.dim < 2 else self.batch[feat.name]
                concat_list.append(f_emb)
                concat_dim += feat.feat_size

        inputs = tf.reshape(tf.concat(concat_list, axis=-1), [-1, concat_dim])

        outputs = fc_layer(inputs, hidden_units=self.args.hidden_units, use_bn=True, training=self.args.bn_training, keep_prob=self.keep_prob)

        self.user_emb = tf.layers.dense(outputs, self.embeddings[f'{self.item_name}_emb'][1][1], activation=tf.nn.relu)

        self.logits = tf.matmul(self.user_emb, self.embeddings[f'{self.item_name}_emb'][0], transpose_b=True) + self.input_b

        self.probability = tf.nn.softmax(self.logits)

        self.pred_topn = tf.nn.top_k(self.probability, self.args.topK).indices


    def train(self):
        y = tf.sequence_mask(tf.ones(tf.shape(self.label)[0]), tf.shape(self.label)[1], dtype=tf.float32)
        sample_b = tf.nn.embedding_lookup(self.input_b, self.label)
        sample_w = tf.concat([tf.nn.embedding_lookup(self.embeddings[f'{self.item_name}_emb'][0], self.label)], axis=2)

        user_v = tf.expand_dims(self.user_emb, 1)
        sample_w = tf.transpose(sample_w, perm=[0, 2, 1])

        self.logits = tf.squeeze(tf.matmul(user_v, sample_w), axis=1) + sample_b
        self.yhat = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(-y * tf.log(self.yhat + 1e-24))

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        tf.summary.scalar("loss", self.loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


    def test(self, cols):

        self.output = {
            'pred_topn': self.pred_topn,
            'pred_topn_str': tf.reduce_join(tf.as_string(self.pred_topn), separator=",", axis=1)
        }

        if self.args.contains('item_key'):
            self.output['pred_topn_key'] = tf.gather(self.args.item_key, self.pred_topn)

        for col in cols:
            self.output[col] = self.batch[col]


    def pred(self):

        self.output = {
            'user_emb': self.user_emb,
            'probability': self.probability,
            'pred_topn': self.pred_topn
        }

        if self.args.contains('item_key'):
            self.output['pred_topn_key'] = tf.gather(self.args.item_key, self.pred_topn)


class YouTubeRank(object):
    """
    youtube rank model
    """
    def __init__(self, args, features, batch, keep_prob):

        self.args = args
        self.features = features
        self.batch = batch
        self.keep_prob = keep_prob
        self.item_name = args.item_name

        self.train_feat = []
        self.emb_feat = []
        self.label = []
        for feature in features:
            if feature.for_train:
                self.train_feat.append(feature)
            if feature.emb_count:
                self.emb_feat.append(feature)
            if feature.label:
                self.label.append(batch[feature.name])

        self.build_model()

    def build_model(self):

        self.embeddings = {}
        for feat in self.emb_feat:
            shape = [feat.emb_count, feat.emb_size]
            self.embeddings[f'{feat.name}_emb'] = (tf.get_variable(f'{feat.name}_emb', shape=shape), shape)

        concat_list, concat_dim = [], 0
        for feat in self.train_feat:
            if feat.dtype.startswith('int'):
                f_emb = tf.nn.embedding_lookup(self.embeddings[f'{feat.emb_share}_emb'][0] if feat.emb_share
                                               else self.embeddings[f'{feat.name}_emb'][0], self.batch[feat.name])
                shape = self.embeddings[f'{feat.emb_share}_emb'][1] if feat.emb_share else self.embeddings[f'{feat.name}_emb'][1]
                if feat.dim == 2:
                    f_emb = pool_layer(f_emb, self.batch[f'{feat.name}_len'])
                concat_list.append(f_emb)
                concat_dim += shape[1]
            else:
                f_emb = tf.reshape(self.batch[feat.name], [-1, 1]) if feat.dim < 2 else self.batch[feat.name]
                concat_list.append(f_emb)
                concat_dim += feat.feat_size

        inputs = tf.reshape(tf.concat(concat_list, axis=-1), [-1, concat_dim])

        outputs = fc_layer(inputs, hidden_units=self.args.hidden_units, use_bn=True,
                           training=self.args.bn_training, keep_prob=self.keep_prob)

        self.logits = [tf.layers.dense(outputs, 2, activation=None, name=f"logits_label_{i}")
                       for i in range(len(self.label))]

        self.probability_list = [tf.nn.softmax(logit)[:, 1] for logit in self.logits]

        self.probability = tf.reduce_mean(self.probability_list, axis=0)

    def train(self):
        self.loss_list = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits[i], labels=tf.cast(label, tf.int32))) for i, label in enumerate(self.label)]
        self.loss = tf.reduce_mean(self.loss_list)

        tf.summary.scalar("loss", self.loss)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def test(self, cols):
        self.output = {
            'probability': self.probability
        }
        for col in cols:
            self.output[col] = self.batch[col]

    def pred(self):
        self.output = {
            'probability': self.probability
        }

        if self.args.contains('item_key'):
            self.item = tf.cast(self.batch[self.item_name], dtype=tf.int32)
            self.item_key = tf.gather(self.args.item_key, self.item)

            self.output['item'] = self.item
            self.output['item_key'] = self.item_key



