
import tensorflow.compat.v1 as tf
from ..core.layers import DNN
from ..core.sequence import AttentionSequencePoolingLayer


class DIN(object):
    def __init__(self, args, batch_x, flag):
        self.args = args
        self.hist_click = batch_x['gameids']
        self.weights = batch_x['weights']
        self.click_len = batch_x['click_len']
        self.last_click = batch_x['last_click']

        self.modelid = batch_x['model']
        self.version = batch_x['version']
        self.connectiontype = batch_x['connectiontype']

        self.gameid = batch_x['gameid']
        if flag == "pred":
            game_feat_w = tf.constant(self.args.game_features, dtype=tf.float32, name="game_features")
            self.dense_features = tf.gather(game_feat_w, self.gameid)
            self.build_model()
            self.pred()
        else:
            self.isclick = batch_x['isclick']
            self.isdownload = batch_x['isdownload']
            self.dense_features = batch_x['dense_features']
            self.build_model()
            self.train() if flag == "train" else self.test()

    def build_model(self):
        # init embedding
        self.gameid_emb_w = tf.get_variable("gameid_emb_w", [self.args.gameid_count, self.args.embedding_size])
        self.model_emb_w = tf.get_variable("model_emb_w", [self.args.model_count, 15])
        self.version_emb_w = tf.get_variable("version_emb_w", [self.args.version_count, 10])
        self.connectiontype_emb_w = tf.get_variable("connectiontype_emb_w", [self.args.connectiontype_count, 5])

        gameid_feat = tf.layers.dense(tf.reshape(self.dense_features, [-1, 4]), 4, activation=None,
                                      name='missing_value_layer')

        # 用户行为序列
        hist_emb = tf.nn.embedding_lookup(self.gameid_emb_w,
                                          self.hist_click)  # (batch, sl, embedding_size) #sl=max(click_len)
        # 待排序游戏
        gameid_emb = tf.nn.embedding_lookup(self.gameid_emb_w, self.gameid)  # (batch, embedding_size)
        last_emb = tf.nn.embedding_lookup(self.gameid_emb_w, self.last_click)

        # attention polling  "a(Ej,Va)Ej = WjEj"
        hist_att_emb = AttentionSequencePoolingLayer(att_hidden_units=[80, 40], att_activation='dice').run(hist_emb,
                                                                                                       gameid_emb,
                                                                                                       self.click_len)

        # user profile
        model_emb = tf.nn.embedding_lookup(self.model_emb_w, self.modelid)
        version_emb = tf.nn.embedding_lookup(self.version_emb_w, self.version)
        connectiontype_emb = tf.nn.embedding_lookup(self.connectiontype_emb_w, self.connectiontype)

        inputs = tf.concat(
            [model_emb, version_emb, connectiontype_emb, hist_att_emb, gameid_emb, gameid_feat, last_emb],
            axis=-1)  # (batch, 226)

        outputs = DNN(hidden_units=[512, 256, 128], keep_prob=self.args.keep_prob, name='DNN').run(inputs)

        self.logits_click = tf.layers.dense(outputs, 2, activation=tf.nn.relu, name="logits_click")
        self.logits_download = tf.layers.dense(outputs, 2, activation=tf.nn.relu, name="logits_download")

    def train(self):
        self.click_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_click, labels=self.isclick))
        self.download_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_download, labels=self.isdownload))
        self.joint_loss = self.click_loss + self.download_loss
        tf.summary.scalar("loss", self.joint_loss)
        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.train_op = optimizer.minimize(self.joint_loss, global_step=self.global_step)

    def test(self):
        self.output = {
            'isclick': self.isclick,
            'isdownload': self.isdownload,
            'pred_click': tf.nn.softmax(self.logits_click)[:, 1],
            'pred_download': tf.nn.softmax(self.logits_download)[:, 1],
            'gameid': tf.gather(self.args.gameid_key, self.gameid),
            'model': tf.gather(self.args.model_key, self.modelid),
            'version': tf.gather(self.args.version_key, self.version),
            'connectiontype': tf.gather(self.args.connectiontype_key, self.connectiontype)
        }

    def pred(self):
        self.gameid = tf.gather(self.args.gameid_key, self.gameid)
        self.pred_click = tf.nn.softmax(self.logits_click, name="pred_click")[:, 1]
        self.pred_download = tf.nn.softmax(self.logits_download, name="pred_download")[:, 1]

