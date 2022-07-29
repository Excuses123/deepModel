# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""


import tensorflow.compat.v1 as tf
from ..core.layers import fc_layer, pool_layer
from ..utils.tools import one_hot


class YouTubeRecall(object):
    """
    youtube recall model
    """
    type = 'Recall'

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
                self.loss_fun = feature.loss

        self.build_model()

    def build_model(self):

        self.embeddings = {}
        for feat in self.emb_feat:
            self.embeddings[f'{feat.name}_emb'] = tf.get_variable(f'{feat.name}_emb',
                                                                  shape=[feat.emb_count, feat.emb_size])

        self.input_b = tf.get_variable("input_b", [self.embeddings[f'{self.item_name}_emb'].shape[0]],
                                       initializer=tf.constant_initializer(0.0))

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

        inputs = tf.reshape(tf.concat(concat_list, axis=-1), [-1, concat_dim])

        outputs = fc_layer(inputs, hidden_units=self.args.hidden_units, use_bn=True,
                           training=self.args.bn_training, keep_prob=self.keep_prob)

        self.user_emb = tf.layers.dense(outputs, self.embeddings[f'{self.item_name}_emb'].shape[1], activation=tf.nn.relu)

        self.logits = tf.matmul(self.user_emb, self.embeddings[f'{self.item_name}_emb'], transpose_b=True) + self.input_b

        self.proba = tf.nn.softmax(self.logits)

        self.pred_topn = tf.nn.top_k(self.proba, self.args.topK).indices

    def train(self):
        y = tf.sequence_mask(tf.ones(tf.shape(self.label)[0]), tf.shape(self.label)[1], dtype=tf.float32)
        sample_b = tf.nn.embedding_lookup(self.input_b, self.label)
        sample_w = tf.nn.embedding_lookup(self.embeddings[f'{self.item_name}_emb'], self.label)

        user_v = tf.expand_dims(self.user_emb, 1)
        sample_w = tf.transpose(sample_w, perm=[0, 2, 1])

        self.logits = tf.squeeze(tf.matmul(user_v, sample_w), axis=1) + sample_b

        self.loss = self.loss_fun(logits=self.logits, labels=y)

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

    def pred(self, name=None):
        if name in self.batch:
            pool = self.batch[name]
            indices = tf.nn.top_k(tf.gather(tf.squeeze(self.proba), pool), self.args.topK).indices
            self.pred_topn = tf.gather(tf.squeeze(pool), indices)

        self.output = {
            'user_emb': self.user_emb,
            'pred_topn': self.pred_topn
        }

        if self.args.contains('item_key'):
            self.output['pred_topn_key'] = tf.gather(self.args.item_key, self.pred_topn)


class YouTubeRank(object):
    """
    youtube rank model
    """
    type = 'Rank'

    def __init__(self, args, features, batch, keep_prob):

        self.args = args
        self.features = features
        self.batch = batch
        self.keep_prob = keep_prob

        self.train_feat = []
        self.emb_feat = []
        self.label = []

        for feature in features:
            if feature.for_train:
                self.train_feat.append(feature)

            if feature.emb_count:
                self.emb_feat.append(feature)

            if feature.label:
                self.label.append(feature)

        self.build_model()

    def build_model(self):

        self.embeddings = {}
        for feat in self.emb_feat:
            # todo 支持指定embedding初始化
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

        inputs = tf.reshape(tf.concat(concat_list, axis=-1), [-1, concat_dim])

        outputs = fc_layer(inputs, hidden_units=self.args.hidden_units, use_bn=True,
                           training=self.args.bn_training, keep_prob=self.keep_prob)

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




