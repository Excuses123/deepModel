
import pandas as pd
import tensorflow.compat.v1 as tf
from ..core.layers import DNN
from ..core.sequence import SequencePoolingLayer, WeightedPoolingLayer


class YouTubeRecall(object):

    def __init__(self, args, batch_x, flag):

        self.args = args

        self.hist_click = batch_x['gameids']
        self.weights = batch_x['weights']
        self.click_len = batch_x['click_len']
        self.last_click = batch_x['last_click']

        self.modelid = batch_x['model']
        self.version = batch_x['version']
        self.channel = batch_x['channel']
        self.version = batch_x['version']

        self.channel_game = batch_x['channel_game']
        self.ischannelgame = batch_x['ischannelgame']

        self.create_user_vector()

        if flag == "pred":
            self.game_pool = batch_x['game_pool']
            self.pred()
        else:
            self.udid = batch_x['udid']
            self.label = batch_x['label']
            self.train() if flag == "train" else self.test()

    def create_user_vector(self):

        self.gameid_emb_w = tf.get_variable("gameid_emb_w", [self.args.gameid_count, self.args.embedding_size])
        self.model_emb_w = tf.get_variable("model_emb_w", [self.args.model_count, 32])
        self.version_emb_w = tf.get_variable("version_emb_w", [self.args.version_count, 12])
        self.channel_emb_w = tf.get_variable("channel_emb_w", [self.args.channel_count, 20])

        self.input_b = tf.get_variable("input_b", [self.args.gameid_count], initializer=tf.constant_initializer(0.0))

        h_emb = tf.nn.embedding_lookup(self.gameid_emb_w, self.hist_click)
        if self.args.use_weight:
            hists = WeightedPoolingLayer().run(h_emb, self.click_len, self.weights)
        else:
            hists = SequencePoolingLayer(mode="mean").run(h_emb, self.click_len)

        last_emb = tf.nn.embedding_lookup(self.gameid_emb_w, self.last_click)
        click_len = tf.cast(tf.expand_dims(self.click_len, 1), dtype=tf.float32)
        last_emb = tf.math.divide_no_nan((last_emb * click_len), click_len)

        modelid_emb = tf.nn.embedding_lookup(self.model_emb_w, self.modelid)
        version_emb = tf.nn.embedding_lookup(self.version_emb_w, self.version)
        channel_emb = tf.nn.embedding_lookup(self.channel_emb_w, self.channel)

        channel_game_emb = tf.nn.embedding_lookup(self.gameid_emb_w, self.channel_game)
        channel_game_emb = channel_game_emb * tf.cast(tf.expand_dims(self.ischannelgame, 1), dtype=tf.float32)

        inputs = tf.reshape(
            tf.concat([hists, last_emb, modelid_emb, version_emb, channel_emb, channel_game_emb], axis=1),
            shape=[-1, 256])

        outputs = DNN(hidden_units=[512, 256], keep_prob=self.args.keep_prob).run(inputs)

        self.user_emb = tf.layers.dense(outputs, self.args.embedding_size, activation=tf.nn.relu, name="user_v")

    def train(self):

        y = tf.sequence_mask(tf.ones(tf.shape(self.label)[0]), tf.shape(self.label)[1], dtype=tf.float32)
        sample_b = tf.nn.embedding_lookup(self.input_b, self.label)
        sample_w = tf.concat([tf.nn.embedding_lookup(self.gameid_emb_w, self.label)], axis=2)

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
        size = len(self.args.gameid_key)
        logits = tf.matmul(self.user_emb, self.gameid_emb_w[:size], transpose_b=True) + self.input_b[:size]
        pred = tf.nn.softmax(logits)

        pred_topn = tf.gather(self.args.gameid_key, tf.nn.top_k(pred, k=self.args.recall_topN).indices)

        self.output = {
            'udid': self.udid,
            'model': self.modelid,
            'version': self.version,
            'channel': self.channel,
            'pred_topn': pred_topn,
            'pred_topn_str': tf.reduce_join(tf.as_string(pred_topn), separator=",", axis=1),
            'label': tf.gather(self.args.gameid_key, self.label),
            'last_click': tf.gather(self.args.gameid_key, self.last_click)
        }

    def pred(self):
        logits = tf.matmul(self.user_emb, tf.gather(self.gameid_emb_w, self.game_pool), transpose_b=True) + tf.gather(
            self.input_b, self.game_pool)
        output = tf.nn.softmax(logits)

        recommend_ind = tf.nn.top_k(tf.squeeze(output), k=self.args.recall_topN).indices
        self.pred_mapped = tf.gather(self.game_pool, recommend_ind)

        self.pred_orgin = tf.gather(self.args.gameid_key, self.pred_mapped)

        self.output_topN_str = tf.reduce_join(tf.as_string(self.pred_orgin), separator="|", axis=1)


class YouTubeRank(object):
    def __init__(self, args, batch_x, flag):

        self.args = args

        self.hist_click = batch_x['gameids']
        self.weights = batch_x['weights']
        self.click_len = batch_x['click_len']
        self.last_click = batch_x['last_click']

        self.channel_game = batch_x['channel_game']
        self.ischannelgame = batch_x['ischannelgame']
        self.isnew = batch_x['isnew']

        self.packages = batch_x['packages']
        self.package_num = batch_x['package_num']

        self.gameid = batch_x['gameid']
        self.position = batch_x['position']

        self.modelid = batch_x['model']
        self.version = batch_x['version']
        self.connectiontype = batch_x['connectiontype']
        self.channel = batch_x['channel']

        if flag == "pred":
            game_feat_w = tf.constant(self.args.game_features, dtype=tf.float32, name="game_features")
            self.dense_features = tf.gather(game_feat_w, self.gameid)
            self.build_model()
            self.pred()
        else:
            self.versionv = batch_x['versionv']
            self.udid = batch_x['udid']
            self.isclick = batch_x['isclick']
            self.isdownload = batch_x['isdownload']
            self.isrealshow = batch_x['isrealshow']
            self.is_detail_entry = batch_x['is_detail_entry']
            self.dense_features = batch_x['dense_features']
            self.build_model()
            self.train() if flag == "train" else self.test()

    def __load_w2v(self, path, expectDim):
        print("loading embedding!")
        emb = pd.read_csv(path, header=None)
        emb = emb.iloc[:, 1:].values.astype("float32")
        assert emb.shape[1] == expectDim
        return emb

    def build_model(self):

        #         if self.args.fine_tune:
        #             self.gameid_emb_w = tf.Variable(self.__load_w2v(self.args.emb_path, self.args.embedding_size), dtype=tf.float32, name='gameid_emb_w')
        #         else:
        #             self.gameid_emb_w = tf.Variable(tf.random_uniform([self.args.gameid_count, self.args.embedding_size], seed=42), name='gameid_emb_w')

        self.gameid_emb_w = tf.get_variable("gameid_emb_w", [self.args.gameid_count + 100, self.args.embedding_size])

        self.model_emb_w = tf.get_variable("model_emb_w", [self.args.model_count + 100, 15])

        self.version_emb_w = tf.get_variable("version_emb_w", [self.args.version_count + 100, 10])

        self.connectiontype_emb_w = tf.get_variable("connectiontype_emb_w", [self.args.connectiontype_count + 100, 5])

        self.channel_emb_w = tf.get_variable("channel_emb_w", [self.args.channel_count + 100, 10])

        self.position_emb_w = tf.get_variable("position_emb_w", [200, 5])

        gameid_feats = tf.layers.dense(tf.reshape(self.dense_features, [-1, 18]), 18, activation=None,
                                       name='missing_value_layer')

        h_emb = tf.nn.embedding_lookup(self.gameid_emb_w, self.hist_click)
        if self.args.use_weight:
            hists = WeightedPoolingLayer().run(h_emb, self.click_len, self.weights)
        else:
            hists = SequencePoolingLayer(mode="mean").run(h_emb, self.click_len)

        #         last_emb = tf.nn.embedding_lookup(self.gameid_emb_w, self.last_click)

        channel_game_emb = tf.nn.embedding_lookup(self.gameid_emb_w, self.channel_game)
        channel_game_emb = channel_game_emb * tf.cast(tf.expand_dims(self.ischannelgame, 1), dtype=tf.float32)

        package_emb = tf.nn.embedding_lookup(self.gameid_emb_w, self.packages)
        packages = SequencePoolingLayer(mode="mean").run(package_emb, self.package_num)

        model_emb = tf.nn.embedding_lookup(self.model_emb_w, self.modelid)
        version_emb = tf.nn.embedding_lookup(self.version_emb_w, self.version)
        connectiontype_emb = tf.nn.embedding_lookup(self.connectiontype_emb_w, self.connectiontype)
        channel_emb = tf.nn.embedding_lookup(self.channel_emb_w, self.channel)

        position_emb = tf.nn.embedding_lookup(self.position_emb_w, self.position)
        self.logits_bias = tf.layers.dense(position_emb, 2, activation=None, name="logits_bias")

        gameid_emb = tf.nn.embedding_lookup(self.gameid_emb_w, self.gameid)

        isNew = tf.cast(tf.expand_dims(self.isnew, 1), dtype=tf.float32)

        inputs = tf.reshape(tf.concat(
            [hists, packages, model_emb, version_emb, connectiontype_emb, channel_emb, channel_game_emb, gameid_emb,
             isNew, gameid_feats], axis=-1), [-1, 315])

        outputs = DNN(hidden_units=[512, 256, 128], use_bn=True, keep_prob=self.args.keep_prob).run(inputs)

        self.logits_detail = tf.layers.dense(outputs, 2, activation=None, name="logits_detail") + self.logits_bias
        self.logits_click = tf.layers.dense(outputs, 2, activation=None, name="logits_click") + self.logits_bias

    def train(self):
        self.detail_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_detail,
                                                                                         labels=tf.squeeze(
                                                                                             self.is_detail_entry)))
        self.click_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_click, labels=tf.squeeze(self.isclick)))

        self.joint_loss = self.detail_loss + self.click_loss

        tf.summary.scalar("loss", self.joint_loss)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.train_op = optimizer.minimize(self.joint_loss, global_step=self.global_step)

    def test(self):
        self.output = {
            'isrealshow': self.isrealshow,
            'is_detail_entry': self.is_detail_entry,
            'isclick': self.isclick,
            'isdownload': self.isdownload,
            'isnew': self.isnew,
            'pred_detail': tf.nn.softmax(self.logits_detail)[:, 1],
            'pred_click': tf.nn.softmax(self.logits_click)[:, 1],
            'versionv': self.versionv,
            'udid': self.udid,
            'position': self.position,
            'gameid': tf.gather(self.args.gameid_key, self.gameid),
            'model': tf.gather(self.args.model_key, self.modelid),
            'version': tf.gather(self.args.version_key, self.version),
            'connectiontype': tf.gather(self.args.connectiontype_key, self.connectiontype)
        }

    def pred(self):
        self.gameid_map = tf.cast(self.gameid, dtype=tf.int32)
        self.gameid = tf.gather(self.args.gameid_key, self.gameid)
        self.pred_detail = tf.nn.softmax(self.logits_detail, name="pred_detail")[:, 1]
        self.pred_click = tf.nn.softmax(self.logits_click, name="pred_click")[:, 1]


