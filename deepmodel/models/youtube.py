
import tensorflow.compat.v1 as tf
from ..core.layers import fc_layer, pool_layer


class YouTubeRecall(object):
    """
    youtube recall model
    """
    def __init__(self, args, features, batch, keep_prob, infer_feat, label='label'):

        self.args = args
        self.features = features
        self.batch = batch
        self.keep_prob = keep_prob
        self.infer_feat = infer_feat

        self.train_feat = []
        self.emb_feat = []
        for feature in features:
            if feature.for_train:
                self.train_feat.append(feature)
            if feature.emb_count:
                self.emb_feat.append(feature)

        self.label = batch[label]

        self.build_model()

    def build_model(self):

        self.embeddings = {}
        for feat in self.emb_feat:
            shape = [feat.emb_count, feat.emb_size]
            self.embeddings[f'{feat.name}_emb'] = (tf.get_variable(f'{feat.name}_emb', shape=shape), shape)

        self.input_b = tf.get_variable("input_b", [1], initializer=tf.constant_initializer(0.0))

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

        self.user_emb = tf.layers.dense(outputs, self.args.embedding_size, activation=tf.nn.relu, name="user_v")

    def train(self):
        y = tf.sequence_mask(tf.ones(tf.shape(self.label)[0]), tf.shape(self.label)[1], dtype=tf.float32)
        sample_b = tf.nn.embedding_lookup(self.input_b, self.batch['label'])
        sample_w = tf.concat([tf.nn.embedding_lookup(self.embeddings[f'{self.infer_feat}_emb'], self.batch['label'])], axis=2)

        user_v = tf.expand_dims(self.user_emb, 1)
        sample_w = tf.transpose(sample_w, perm=[0, 2, 1])

        self.logits = tf.squeeze(tf.matmul(user_v, sample_w), axis=1) + sample_b
        self.yhat = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(-y * tf.log(self.yhat + 1e-24))

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        tf.summary.scalar("loss", self.loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def test(self):
        logits = tf.matmul(self.user_emb, self.embeddings[f'{self.infer_feat}_emb'], transpose_b=True) + self.input_b
        pred = tf.nn.softmax(logits)

        pred_topn = tf.gather(self.args.id_key, tf.nn.top_k(pred, k=self.args.recall_topN).indices)

        self.output = {
            'pred_topn': pred_topn,
            'pred_topn_str': tf.reduce_join(tf.as_string(pred_topn), separator=",", axis=1),
            'label': tf.gather(self.args.id_key, self.label)
        }


class YouTubeRank(object):
    """
    youtube rank model
    """
    def __init__(self, args, features, batch, keep_prob, label='label'):

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

        self.label = batch[label]

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

        outputs = fc_layer(inputs, hidden_units=self.args.hidden_units, use_bn=True, training=self.args.bn_training, keep_prob=self.keep_prob)

        self.logits = tf.layers.dense(outputs, 2, activation=None, name="logits")

        self.pred = tf.nn.softmax(self.logits)[:, 1]

    def train(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                  labels=tf.cast(self.label, tf.int32)))
        tf.summary.scalar("loss", self.loss)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def test(self, cols):
        self.output = {
            'pred_label': self.pred
        }
        for col in cols:
            self.output[col] = self.batch[col]




