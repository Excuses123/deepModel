# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""


import tensorflow.compat.v1 as tf
from ..core.layers import fc_layer, pool_layer, attention_layer
from ..core.activation import dice
from ..utils.tools import one_hot


class DIN(object):
    type = 'Rank'

    def __init__(self, args, features, batch, keep_prob):

        self.args = args
        self.features = features
        self.batch = batch
        self.keep_prob = keep_prob

        self.train_feat = []
        self.emb_feat = []
        self.din_feat = {}
        self.label = []

        for feature in features:
            if feature.for_train:
                if feature.attention:
                    self.din_feat[feature.attention] = feature
                else:
                    self.train_feat.append(feature)

            if feature.emb_count:
                self.emb_feat.append(feature)

            if feature.label:
                self.label.append(feature)

        self.build_model()


    def build_model(self):

        self.embeddings = {}
        for feat in self.emb_feat:
            self.embeddings[f'{feat.name}_emb'] = tf.get_variable(f'{feat.name}_emb',
                                                                  shape=[feat.emb_count, feat.emb_size])

        concat_list, concat_dim = [], 0
        for feat in self.train_feat:
            if feat.dtype.startswith('int') and (feat.emb_count or feat.emb_share):
                f_emb = tf.nn.embedding_lookup(self.embeddings[f'{feat.emb_share}_emb'] if feat.emb_share
                                               else self.embeddings[f'{feat.name}_emb'], self.batch[feat.name])
                shape = self.embeddings[f'{feat.emb_share}_emb'].shape if feat.emb_share \
                    else self.embeddings[f'{feat.name}_emb'].shape

                if feat.dim == 1 and f'{feat.name}_len' in self.batch:
                    f_emb *= tf.expand_dims(tf.cast(self.batch[f'{feat.name}_len'], f_emb.dtype), 1)
                if feat.dim == 2:
                    weight = None
                    if self.args.contains('use_weight'):
                        weight = self.batch.get(f'{feat.name}_weight', None) if self.args.use_weight else weight
                    f_emb = pool_layer(f_emb, mask_len=self.batch.get(f'{feat.name}_len', None), weight=weight)
                concat_list.append(f_emb)
                concat_dim += shape[1]
            else:
                f_emb = tf.reshape(self.batch[feat.name], [-1, 1]) if feat.dim < 2 else self.batch[feat.name]
                concat_list.append(tf.cast(f_emb, tf.float32))
                concat_dim += feat.feat_size

        self.din_emb = {}
        for k, feat in self.din_feat.items():
            self.din_emb[f'{k}_emb'] = tf.nn.embedding_lookup(
                self.embeddings[f'{feat.emb_share}_emb'] if feat.emb_share else self.embeddings[f'{feat.name}_emb'],
                self.batch[feat.name])

        att_emb = attention_layer(key=self.din_emb['key_emb'],
                                  query=self.din_emb['query_emb'],
                                  mask=self.batch[f"{self.din_feat['key'].name}_len"],
                                  hidden_units=[80, 40], activation=dice)
        concat_list.append(att_emb)

        concat_dim += self.embeddings[f"{self.din_feat['key'].emb_share}_emb"].shape[1] \
            if self.din_feat['key'].emb_share else self.embeddings[f'{self.din_feat["key"].name}_emb'].shape[1]

        inputs = tf.reshape(tf.concat(concat_list, axis=-1), [-1, concat_dim])

        outputs = fc_layer(inputs, hidden_units=self.args.hidden_units, use_bn=True, training=self.args.bn_training,
                           keep_prob=self.keep_prob)

        self.logits = dict([(feat.name, tf.layers.dense(outputs, feat.num_class, name=f"logits_{feat.name}"))
                            for i, feat in enumerate(self.label)])

        self.probas = {}
        for i, feat in enumerate(self.label):
            if feat.label_type == 'multi':
                self.probas[f'proba_{feat.name}'] = tf.nn.sigmoid(self.logits[feat.name])
            else:
                self.probas[f'proba_{feat.name}'] = tf.nn.softmax(self.logits[feat.name])[:, 1] if feat.num_class == 2 \
                    else tf.nn.softmax(self.logits[feat.name])

        if self.args.union_label:
            self.probas['proba_union'] = tf.reduce_mean(list(self.probas.values()), axis=0)

    def train(self):
        self.loss_list = dict([(feat.name, feat.loss(logits=self.logits[feat.name],
                                                     labels=one_hot(self.batch[feat.name], num_classes=feat.num_class)
                                                     if feat.onehot else self.batch[feat.name]))
                               for i, feat in enumerate(self.label)])

        self.loss = tf.reduce_mean(list(self.loss_list.values()))

        tf.summary.scalar("loss", self.loss)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def test(self, cols):
        self.output = self.probas
        for col in cols:
            self.output[col] = self.batch[col]

    def pred(self):
        self.output = self.probas

        if self.args.contains('item_key'):
            self.item = tf.cast(self.batch[self.args.item_name], dtype=tf.int32)
            self.item_key = tf.gather(self.args.item_key, self.item)

            self.proba = self.probas['proba_union']

            self.output['item'] = tf.gather(self.item, tf.nn.top_k(self.proba, k=tf.shape(self.proba)[0]).indices)
            self.output['item_key'] = tf.gather(self.item_key,
                                                tf.nn.top_k(self.proba, k=tf.shape(self.proba)[0]).indices)

