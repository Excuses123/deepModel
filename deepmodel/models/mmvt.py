import numpy as np
import tensorflow.compat.v1 as tf
from keras.utils import to_categorical


class MMVT(object):
    """
    MutiModalVideoTag
    参考：https://aistudio.baidu.com/aistudio/projectdetail/3469740
    """
    def __init__(self, args):
        self.args = args
        self.params()
        self.build_model()


    def params(self):
        self.title = tf.placeholder(tf.int32, [None, self.args.max_seq], name="title")
        self.asr = tf.placeholder(tf.int32, [None, self.args.max_seq], name="asr")
        self.ocr = tf.placeholder(tf.int32, [None, self.args.max_seq], name="ocr")
        self.video = tf.placeholder(tf.float32, [None, self.args.T, 512], name="video")

        self.title_len = tf.placeholder(tf.int32, [None], name="title_len")
        self.asr_len = tf.placeholder(tf.int32, [None], name="asr_len")
        self.ocr_len = tf.placeholder(tf.int32, [None], name="ocr_len")
        self.video_len = tf.placeholder(tf.int32, [None], name="video_len")

        self.label = tf.placeholder(tf.float32, [None, self.args.num_class], name="label")
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.emb_word = tf.get_variable(name="emb_word", shape=[self.args.word_cnt, self.args.embedding_size])
        self.emb_len = tf.get_variable(name="emb_len", shape=[1000, 4])


    def build_model(self):

        title_len_emb = tf.nn.embedding_lookup(self.emb_len, self.title_len)  # (batch, 4)
        title_emb = tf.nn.embedding_lookup(self.emb_word, self.title)         # (batch, max_seq, emb)
        title_emb = tf.expand_dims(title_emb, -1)                             # (batch, max_seq, emb, 1)
        # todo: 多个特征共享参数，参数名相同
        title_emb = self.textcnn(title_emb, "title")                          # (batch, 3 * numFilters)

        asr_len_emb = tf.nn.embedding_lookup(self.emb_len, self.asr_len)  # (batch, 4)
        asr_emb = tf.nn.embedding_lookup(self.emb_word, self.asr)         # (batch, max_seq, emb)
        asr_emb = tf.expand_dims(asr_emb, -1)                             # (batch, max_seq, emb, 1)
        asr_emb = self.textcnn(asr_emb, "asr")                            # (batch, 3 * numFilters)

        ocr_len_emb = tf.nn.embedding_lookup(self.emb_len, self.ocr_len)  # (batch, 4)
        ocr_emb = tf.nn.embedding_lookup(self.emb_word, self.ocr)         # (batch, max_seq, emb)
        ocr_emb = tf.expand_dims(ocr_emb, -1)                             # (batch, max_seq, emb, 1)
        ocr_emb = self.textcnn(ocr_emb, "ocr")                            # (batch, 3 * numFilters)

        video_emb = self.bilstm(self.video, 256, self.video_len)          # (batch, max_T, 2 * 512)
        video_emb = self.attention(video_emb, title_emb, self.video_len)  # (batch, 2 * 512)

        inputs = tf.concat([title_emb, title_len_emb, asr_emb, asr_len_emb, ocr_emb, ocr_len_emb, video_emb], axis=1)
        fc1 = tf.layers.dense(inputs, 1024, activation=tf.nn.relu, name='fc1')
        fc1 = tf.nn.dropout(fc1, rate=1 - self.keep_prob)
        fc2 = tf.layers.dense(fc1, 512, activation=tf.nn.relu, name='fc2')
        fc2 = tf.nn.dropout(fc2, rate=1 - self.keep_prob)

        self.logits = tf.layers.dense(fc2, self.args.num_class, activation=None, name='logits')
        self.pctr = tf.nn.sigmoid(self.logits)

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.label))
        _, self.acc = tf.metrics.accuracy(labels=tf.argmax(self.label, axis=1), predictions=tf.argmax(self.pctr, axis=1), name="acc")
        self.acc_vars_initializer = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc"))

        tf.summary.scalar("loss", self.loss)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


    def run_train(self, sess, feed, summary_op, keep_prob):
        loss, acc, summary, global_step, _ = sess.run([self.loss, self.acc, summary_op, self.global_step, self.train_op], feed_dict={
            self.title: feed['title'],
            self.title_len: feed['title_len'],
            self.asr: feed['asr'],
            self.asr_len: feed['asr_len'],
            self.ocr: feed['ocr'],
            self.ocr_len: feed['ocr_len'],
            self.video: feed['video'],
            self.video_len: feed['video_len'],
            self.label: to_categorical(feed['label'], self.args.num_class),
            self.keep_prob: keep_prob
        })
        return loss, acc, summary, global_step


    def run_eval(self, sess, eval_data, keep_prob=1.0):
        total_loss = 0
        sess.run(self.acc_vars_initializer)
        for step, feed in eval_data:
            loss, acc = sess.run([self.loss, self.acc], feed_dict={
                self.title: feed['title'],
                self.title_len: feed['title_len'],
                self.asr: feed['asr'],
                self.asr_len: feed['asr_len'],
                self.ocr: feed['ocr'],
                self.ocr_len: feed['ocr_len'],
                self.video: feed['video'],
                self.video_len: feed['video_len'],
                self.label: to_categorical(feed['label'], self.args.num_class),
                self.keep_prob: keep_prob
            })
            total_loss += loss
            return total_loss / step, acc


    def run_pred(self, sess, feed, keep_prob=1.0):
        pctr = sess.run(self.pctr, feed_dict={
            self.title: feed['title'],
            self.title_len: feed['title_len'],
            self.asr: feed['asr'],
            self.asr_len: feed['asr_len'],
            self.ocr: feed['ocr'],
            self.ocr_len: feed['ocr_len'],
            self.video: feed['video'],
            self.video_len: feed['video_len'],
            self.label: to_categorical(feed['label'], self.args.num_class),
            self.keep_prob: keep_prob
        })
        return pctr


    def save(self, sess, path, global_step):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path, global_step=global_step)


    def load(self, sess, path):
        """ 加载模型 """
        saver = tf.train.Saver()
        ckpt_path = tf.train.latest_checkpoint(path)
        if ckpt_path:
            step = ckpt_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, save_path=ckpt_path)
            print("Load models of step %s success" % step)
        else:
            print("No checkpoint!")


    def bilstm(self, emb, hidden_dim, seq_len):
        """
        :param emb:   (batch, max_T, emb)
        :param hidden_dim:
        :return:
        """
        cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_dim)  # 正向
        cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_dim)  # 反向
        if self.args.lstm_layers > 1:
            cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * self.args.lstm_layers, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * self.args.lstm_layers, state_is_tuple=True)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=emb, sequence_length=seq_len, dtype=tf.float32)
        outputs = tf.concat(outputs, axis=2)  # (batch, seq_len, 2 * hidden_dim)

        return outputs


    def textcnn(self, text_emb, name, channel=1):
        """
        :param text_emb: 文本向量
        :param name:     特征名
        :param channel:  通道数
        :return:
        """
        pooled_outputs = []
        for i, kernelSize in enumerate(self.args.kernelSizes):
            with tf.name_scope("conv-maxpool-%s-%s" % (name, kernelSize)):
                filterShape = [kernelSize, self.args.embedding_size, channel, self.args.numFilters]  # (kernelSize, emb, channel, numFilters)
                W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W-%s-%s" % (name, kernelSize))
                b = tf.Variable(tf.constant(0.1, shape=[self.args.numFilters]), name="b-%s-%s" % (name, kernelSize))  # numFilters
                conv = tf.nn.conv2d(
                    text_emb,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")      # (batch, len - filterSize + 1, 1, numFilters)
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.args.max_seq - kernelSize + 1, 1, 1],  # ksize: (batch, height, width, numFilters)
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")                                          # (batch, 1, 1, numFilters)
                pooled_outputs.append(pooled)

        output = tf.reshape(tf.concat(pooled_outputs, axis=3), [-1, len(self.args.kernelSizes) * self.args.numFilters])
        output = tf.nn.dropout(output, rate=1 - self.keep_prob)                                 # (batch, 3 * 128)

        return output


    def attention(self, key, query, mask, weight_normalization=True):
        """
        :param key:    video_emb/audio_emb  (batch, max_T, emb1)
        :param query:  text_emb             (batch, emb2)
        :param mask:   mask
        :param weight_normalization: 权重归一化
        :return:
        """
        mask = tf.sequence_mask(mask, tf.shape(key)[1])  # (batch, max_T)
        mask = tf.expand_dims(mask, axis=1)                # (batch, 1, max_T)

        query_exp = tf.tile(tf.expand_dims(query, 1), [1, tf.shape(key)[1], 1])  # (batch, max_T, emb)

        value = tf.concat([key, query_exp], axis=2)                      # (batch, max_T, M+emb)
        value = tf.layers.dense(value, 1, activation=None)               # (batch, max_T, 1)
        value = tf.transpose(value, perm=[0, 2, 1])                      # (batch, 1, max_T)

        if weight_normalization:
            padding = tf.ones_like(value) * (-2 ** 32 + 1)               # (batch, 1, max_T)
        else:
            padding = tf.zeros_like(value)

        value = tf.where(mask, value, padding)

        if weight_normalization:
            value = tf.nn.softmax(value)

        # (batch, 1, max_T) # (batch, max_T, M)
        att = tf.squeeze(tf.matmul(value, key), axis=1)   # (batch, M)

        return att


    def __load_w2v(self, path, expectDim):
        print("loading embedding: %s !" % path)
        emb = np.load(path)['matrix'].astype("float32")
        assert emb.shape[1] == expectDim

        return emb

