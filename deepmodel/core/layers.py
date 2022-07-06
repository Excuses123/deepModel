
import tensorflow.compat.v1 as tf

class DNN(object):

    def __init__(self, hidden_units, activation='relu', l2_reg=0, keep_prob=1, use_bn=False, training=False, name="dnn", seed=1024):
        self.hidden_units = hidden_units
        self.activation = activation
        self.keep_prob = keep_prob
        self.training = training
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.name = name

    def run(self, inputs):

        outputs = inputs

        if self.use_bn:
            outputs = tf.layers.batch_normalization(inputs=outputs, training=self.training)

        for i, size in enumerate(self.hidden_units):
            fc = tf.layers.dense(outputs, size, name='%s_fc_%d' % (self.name, i))
            if self.activation == "relu":
                fc = tf.nn.relu(fc, name='%s_relu_%d' % (self.name, i))
            fc = tf.nn.dropout(fc, keep_prob=self.keep_prob)
            outputs = fc

        return outputs

    def get_config(self):
        config = {
            'activation': self.activation,
            'hidden_units': self.hidden_units,
            'l2_reg': self.l2_reg,
            'use_bn': self.use_bn,
            'dropout_rate': self.keep_prob,
            'seed': self.seed
            }
        return config