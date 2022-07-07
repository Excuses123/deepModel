import tensorflow.compat.v1 as tf


#输入
#embedding
#各种操作
#

class DeepFM(object):
    """
    支持id特征分开输入，以及序列id共享其他特征的embedding
    """
    def __init__(self, args, data_x, keep_prob):
        self.args = args
        self.keep_prob = keep_prob

        self.gameid = data_x['gameid']          # (batch, )
        self.brand = data_x['brand']            # (batch, )
        self.version = data_x['version']        # (batch, )
        self.click_hist = data_x['click_hist']  # (batch, max_len)
        self.click_len = data_x['click_len']
        self.dense_feats = data_x['dense_feats']  #(batch, 5)

        self.label = data_x['label']                   # shape (batch, )
        self.inference()
        self.train()


    def inference(self):
        ##FM部分
        # lr参数
        w1 = {
            'gameid_w1': tf.get_variable("gameid_w1", shape=[self.args['gameid_count'], 1]),
            'brand_w1': tf.get_variable("brand_w1", shape=[self.args['brand_count'], 1]),
            'version_w1': tf.get_variable("version_w1", shape=[self.args['version_count'], 1]),
            'dense_w1': tf.get_variable("dense_w1", shape=[self.args['dense_count'], 1])
        }
        b = tf.get_variable('bias', shape=[1], initializer=tf.constant_initializer(0.01))
        # 交叉项参数
        v = {
            'gameid_emb': tf.get_variable("gameid_emb_w", shape=[self.args['gameid_count'], self.args['K']]),
            'brand_emb': tf.get_variable("brand_emb_w", shape=[self.args['brand_count'], self.args['K']]),
            'version_emb': tf.get_variable("version_emb_w", shape=[self.args['version_count'], self.args['K']]),
            'dense_emb': tf.get_variable("dense_emb_w", shape=[self.args['dense_count'], self.args['K']])
        }

        # tf.sparse_tensor_dense_matmul(self.X, w1)
        self.linear_terms = tf.nn.embedding_lookup(w1['gameid_w1'], self.gameid) + \
                            tf.nn.embedding_lookup(w1['brand_w1'], self.brand) + \
                            tf.nn.embedding_lookup(w1['version_w1'], self.version) + \
                            SeqsPool(w1['gameid_w1'], self.click_hist, self.click_len) + \
                            tf.matmul(self.dense_feats, w1['dense_w1']) + \
                            b                                                         # (batch, 1)

        # tf.sparse_tensor_dense_matmul(self.X, v)
        x1 = tf.nn.embedding_lookup(v['gameid_emb'], self.gameid) + \
             tf.nn.embedding_lookup(v['brand_emb'], self.brand) + \
             tf.nn.embedding_lookup(v['version_emb'], self.version) + \
             SeqsPool(v['gameid_emb'], self.click_hist, self.click_len) + \
             tf.matmul(self.dense_feats, v['dense_emb'])             # (batch, k)


        # tf.sparse_tensor_dense_matmul(self.X, tf.pow(v, 2):   值都为1，tf.pow(self.X, 2) = self.X
        x2 = tf.nn.embedding_lookup(tf.pow(v['gameid_emb'], 2), self.gameid) + \
             tf.nn.embedding_lookup(tf.pow(v['brand_emb'], 2), self.brand) + \
             tf.nn.embedding_lookup(tf.pow(v['version_emb'], 2), self.version) + \
             SeqsPool(tf.pow(v['gameid_emb'], 2), self.click_hist, self.click_len) + \
             tf.matmul(tf.pow(self.dense_feats, 2), tf.pow(v['dense_emb'], 2))       # (batch, k)

        self.interaction_terms = tf.multiply(0.5,
                                             tf.reduce_sum(
                                                 tf.subtract(tf.pow(x1, 2), x2),  # (batch, K)
                                                 axis=1, keep_dims=True)
                                             )                                    # (batch, 1)

        self.y_fm = tf.add(self.linear_terms, self.interaction_terms)             # (batch, 1)

        # tf.nn.embedding_lookup(v, self.feature_inds)
        dnn_emb = tf.concat([
            tf.nn.embedding_lookup(v['gameid_emb'], self.gameid),        # (batch, k)
            tf.nn.embedding_lookup(v['brand_emb'], self.brand),          # (batch, k)
            tf.nn.embedding_lookup(v['version_emb'], self.version),      # (batch, k)
            SeqsPool(v['gameid_emb'], self.click_hist, self.click_len),  # (batch, k)
            tf.matmul(self.dense_feats, v['dense_emb'])                  # (batch, k)
            ], axis=1)                                                   # (batch, field_num * k)

        bn = tf.layers.batch_normalization(inputs=dnn_emb, name="b1")
        layer_1 = tf.layers.dense(bn, 512, activation=tf.nn.relu, name='f1')
        layer_1 = tf.nn.dropout(layer_1, keep_prob=self.keep_prob)
        layer_2 = tf.layers.dense(layer_1, 256, activation=tf.nn.relu, name='f2')
        layer_2 = tf.nn.dropout(layer_2, keep_prob=self.keep_prob)
        layer_3 = tf.layers.dense(layer_2, 128, activation=tf.nn.relu, name="f3")
        layer_3 = tf.nn.dropout(layer_3, keep_prob=self.keep_prob)
        self.y_dnn = tf.layers.dense(layer_3, 1, use_bias=False, activation=None, name="y_dnn")

        #输出
        self.logits = tf.add(self.y_fm, self.y_dnn)
        self.y_pred = tf.nn.sigmoid(self.logits)[:, 0]

    def train(self):
        self.log_loss = tf.losses.log_loss(self.label, self.y_pred)
        self.mse_loss = tf.reduce_mean(tf.pow(self.label - self.y_pred, 2))
        # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label))
        tf.summary.scalar("loss", self.log_loss)
        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args['learning_rate'])
        self.train_op = optimizer.minimize(self.log_loss, global_step=self.global_step)

    def save(self, sess, path, global_step):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path, global_step=global_step)

    def load(self, sess, path):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, save_path=ckpt.model_checkpoint_path)
            print("Load model of step %s success" % step)
        else:
            print("No checkpoint!")


def SeqsPool(embedding, item_seqs, click_len, keep_dims=False):
    seqs_emb = tf.nn.embedding_lookup(embedding, item_seqs)
    mask = tf.sequence_mask(click_len, tf.shape(seqs_emb)[1], dtype=tf.float32)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, tf.shape(seqs_emb)[2]])
    seqs_emb *= mask

    return tf.reduce_sum(seqs_emb, axis=1, keep_dims=keep_dims)
