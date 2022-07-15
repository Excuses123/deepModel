import os
import tensorflow as tf
from deepmodel.data import TFRecordLoader, Feature
from deepmodel.models import YouTubeRank
from deepmodel.utils import save_ckpt, load_ckpt, ckpt2pb, load_pb


class Args(object):
    epoch = 10
    hidden_units = [512, 256]
    batch_size = 32
    learning_rate = 0.001
    save_path = './examples/checkpoint'
    model_name = 'test.pb'

features = [
    Feature(name='id', dtype='string', dim=1, for_train=False),
    Feature(name='a', dtype='int32', dim=1, emb_count=11, emb_size=5),
    Feature(name='b', dtype='int64', dim=1, emb_count=101, emb_size=10),
    Feature(name='c', dtype='int64', dim=2, dense=True, emb_share='b'),
    Feature(name='c_len', dtype='int64', dim=1, for_train=False),
    Feature(name='d', dtype='float32', dim=1),
    Feature(name='e', dtype='float32', dim=2, feat_size=10, dense=True),
    Feature(name='label', dtype='float32', dim=1, for_train=False)
]

args = Args()

inputs = TFRecordLoader(features, 'examples/test.tfrecord', repeats=args.epoch)
batch_x = inputs.load_batch(args.batch_size)

# train
sess = tf.Session()
model = YouTubeRank(args, features, batch_x, 0.8)
model.train()
sess.run(tf.global_variables_initializer())
l1_sum, step = 0, 0
while True:
    try:
        l1, _ = sess.run([model.loss, model.train_op])
        l1_sum += l1
        step += 1
        if step % 10 == 0:
            print(f'step: {step}   loss: {l1_sum}')
    except:
        print("End of dataset")
        break
save_ckpt(sess, args.save_path, global_step=step)
del model

# test
batch_x = TFRecordLoader(features, 'examples/test.tfrecord', repeats=args.epoch).load_batch(args.batch_size)
out_cols = ['id', 'label']
tf.reset_default_graph()
graph = tf.get_default_graph()
sess = tf.Session(graph=graph)
model = YouTubeRank(args, features, batch_x, 1.0)
model.test(out_cols)
sess.run(tf.global_variables_initializer())
sess = load_ckpt(sess, args.save_path)
output = sess.run(model.output)
print(output)

# ckpt转pb
ckpt2pb(args, features, YouTubeRank, out_name='pred')

# 加载pb并预测
path = os.path.join(args.save_path, args.model_name)
pd_loader = load_pb(path, features, out_name='pred')
pd_loader.predict(feed_dict={
    'a': [2, 0, 5],
    'b': [40, 12, 4],
    'c': [[3, 42, 80], [9, 25, 0], [81, 0, 0]],
    'c_len': [3, 2, 1],
    'd': [0.49, 0.123, 0.667],
    'e': [[0.48, 0.817, 0.5465, 0.913, 0.979, 0.931, 0.343, 0.364, 0.622, 0.318],
          [0.22, 0.532, 0.0838, 0.032, 0.423, 0.645, 0.865, 0.156, 0.363, 0.176],
          [0.60, 0.288, 0.3483, 0.152, 0.546, 0.935, 0.131, 0.825, 0.444, 0.674]],
})




