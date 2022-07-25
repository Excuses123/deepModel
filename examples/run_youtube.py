import os
import tensorflow as tf
from deepmodel.param import Param
from deepmodel.data import TFRecordLoader, Feature
from deepmodel.models import YouTubeRecall, YouTubeRank
from deepmodel.utils import save_ckpt, load_ckpt, ckpt2pb, load_pb


## 召回
args = Param(
    epoch=10,
    batch_size=32,
    learning_rate=0.001,
    hidden_units=[512, 256],
    bn_training=True,
    topK=5,
    model_path='./examples/checkpoint',
    model_name='test.pb',
    item_name='b'
)

features = [
    Feature(name='uid', name_from='id', dtype='string', dim=1, for_train=False),
    Feature(name='a', dtype='int32', dim=1, emb_count=11, emb_size=5),
    Feature(name='b', dtype='int64', dim=1, emb_count=101, emb_size=10),
    Feature(name='c', dtype='int64', dim=2, dense=True, emb_share='b'),
    Feature(name='c_len', dtype='int64', dim=1, for_train=False),
    Feature(name='d', dtype='float32', dim=1),
    Feature(name='e', dtype='float32', dim=2, feat_size=10, dense=True),
    Feature(name='f', dtype='float32', dtype_from='int64', dim=1),
    Feature(name='recall', dtype='int64', dim=2, dense=True, for_train=False, label=True)
]


# train
with tf.Session() as sess:
    inputs = TFRecordLoader(features, 'examples/test.tfrecord', repeats=args.epoch)
    batch_x = inputs.load_batch(args.batch_size)

    model = YouTubeRecall(args, features, batch_x, 0.8)
    model.train()
    sess.run(tf.global_variables_initializer())
    l1_sum, step = 0, 0
    while True:
        try:
            l1, _ = sess.run([model.loss, model.train_op])
            l1_sum += l1
            step += 1
            if step % 10 == 0:
                print(f'step: {step}   loss: {l1_sum/step:.4f}')
        except:
            print("End of dataset")
            break
    save_ckpt(sess, args.model_path, global_step=step)


# test
out_cols = ['uid', 'recall']
args.bn_training = False

tf.reset_default_graph()
with tf.Session() as sess:
    batch_x = TFRecordLoader(features, 'examples/test.tfrecord').load_batch(args.batch_size)

    model = YouTubeRecall(args, features, batch_x, 1.0)
    model.test(out_cols)
    sess.run(tf.global_variables_initializer())

    sess = load_ckpt(sess, args.model_path)
    for _ in range(3):
        output = sess.run(model.output)
        print(output)



## 排序
args = Param(
    epoch=10,
    batch_size=32,
    learning_rate=0.001,
    hidden_units=[512, 256],
    bn_training=True,
    model_path='./examples/checkpoint',
    model_name='test.pb',
    item_name='b'
)

features = [
    Feature(name='uid', name_from='id', dtype='string', dim=1, for_train=False),
    Feature(name='a', dtype='int32', dim=1, emb_count=11, emb_size=5),
    Feature(name='b', dtype='int64', dim=1, emb_count=101, emb_size=10),
    Feature(name='c', dtype='int64', dim=2, dense=True, emb_share='b'),
    Feature(name='c_len', dtype='int64', dim=1, for_train=False),
    Feature(name='d', dtype='float32', dim=1),
    Feature(name='e', dtype='float32', dim=2, feat_size=10, dense=True),
    Feature(name='f', dtype='float32', dtype_from='int64', dim=1),
    Feature(name='label', dtype='float32', dim=1, for_train=False, label=True),
    Feature(name='label2', dtype='float32', dim=1, for_train=False, label=True)
]

# train
with tf.Session() as sess:
    inputs = TFRecordLoader(features, 'examples/test.tfrecord', repeats=args.epoch)
    batch_x = inputs.load_batch(args.batch_size)

    model = YouTubeRank(args, features, batch_x, 0.8)
    model.train()
    sess.run(tf.global_variables_initializer())
    l_sum, l1_sum, l2_sum, step = 0, 0, 0, 0
    while True:
        try:
            l, (l1, l2), _ = sess.run([model.loss, model.loss_list, model.train_op])
            l_sum += l
            l1_sum += l1
            l2_sum += l2
            step += 1
            if step % 10 == 0:
                print(f'step: {step}   loss: {l_sum/step:.4f}   loss1: {l1_sum/step:.4f}   loss2: {l2_sum/step:.4f}')
        except:
            print("End of dataset")
            break
    save_ckpt(sess, args.model_path, global_step=step)


# test
out_cols = ['uid', 'label']
args.bn_training = False

tf.reset_default_graph()
with tf.Session() as sess:
    batch_x = TFRecordLoader(features, 'examples/test.tfrecord').load_batch(args.batch_size)

    model = YouTubeRank(args, features, batch_x, 1.0)
    model.test(out_cols)
    sess.run(tf.global_variables_initializer())

    sess = load_ckpt(sess, args.model_path)
    for _ in range(3):
        output = sess.run(model.output)
        print(output)


# ckpt转pb
ckpt2pb(args, features, YouTubeRank)


# 加载pb并预测
path = os.path.join(args.model_path, args.model_name)
pb_loader = load_pb(path, features, out_name='probability')
pb_loader.predict(feed_dict={
    'a': [2, 0, 5],
    'b': [40, 12, 4],
    'c': [[3, 42, 80], [9, 25, 0], [81, 0, 0]],
    'c_len': [3, 2, 1],
    'd': [0.49, 0.123, 0.667],
    'e': [[0.48, 0.817, 0.5465, 0.913, 0.979, 0.931, 0.343, 0.364, 0.622, 0.318],
          [0.22, 0.532, 0.0838, 0.032, 0.423, 0.645, 0.865, 0.156, 0.363, 0.176],
          [0.60, 0.288, 0.3483, 0.152, 0.546, 0.935, 0.131, 0.825, 0.444, 0.674]],
})




