## 召回
import os
import tensorflow as tf
from deepmodel.param import Param
from deepmodel.data import TFRecordLoader, Feature
from deepmodel.models import YouTubeRecall
from deepmodel.utils import save_ckpt, load_ckpt, ckpt2pb, load_pb


args = Param(
    epoch=10,
    batch_size=32,
    learning_rate=0.001,
    hidden_units=[512, 256],
    bn_training=True,
    use_weight=True,
    topK=5,
    model_path='./examples/checkpoint',
    model_name='test.pb',
    item_name='b'
)
import random
args.item_key = [0] + [random.randint(1000, 10000) for _ in range(100)]

features = [
    Feature(name='uid', name_from='id', dtype='string', dim=1, for_train=False),
    Feature(name='a', dtype='int32', dim=1, emb_count=11, emb_size=5),
    Feature(name='a_len', name_from='f', dtype='int64', dim=1, for_train=False),
    Feature(name='b', dtype='int64', dim=1, emb_count=101, emb_size=10),
    Feature(name='c', dtype='int64', dim=2, dense=True, emb_share='b'),
    Feature(name='c_weight', dtype='int64', dim=2, dense=True, for_train=False),
    Feature(name='c_len', dtype='int64', dim=1, for_train=False),
    Feature(name='d', dtype='float32', dim=1),
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

# ckpt转pb
pred_feature = Feature(name='recall_pool', dtype='int64', dim=2, dense=True)
ckpt2pb(args, features, YouTubeRecall, pred_feature=pred_feature)


# 加载pb并预测
features.append(pred_feature)
path = os.path.join(args.model_path, args.model_name)
pb_loader = load_pb(path, features, out_name='pred_topn_key')
print(pb_loader.inputMap)
pb_loader.predict(feed_dict={
    'a': [[2]],
    'a_len': [[1]],
    'c': [[3, 42, 80]],
    'c_weight': [[3, 1, 2]],
    'c_len': [[3]],

    'b': [[80]],
    'd': [[0.49]],

    'recall_pool': [[50, 51, 52, 53, 54, 55, 56, 57, 58, 59]]
})



###############################################################
## 排序
import os
import tensorflow as tf
from deepmodel.param import Param
from deepmodel.data import TFRecordLoader, Feature
from deepmodel.models import YouTubeRank
from deepmodel.utils import save_ckpt, load_ckpt, ckpt2pb, load_pb


args = Param(
    epoch=10,
    batch_size=32,
    learning_rate=0.001,
    hidden_units=[512, 256],
    bn_training=True,
    use_weight=True,
    model_path='./examples/checkpoint',
    model_name='test.pb',
    item_name='b',
    union_label=True
)
import random
args.item_key = [0] + [random.randint(1000, 10000) for _ in range(100)]

features = [
    Feature(name='uid', name_from='id', dtype='string', dim=1, for_train=False),
    Feature(name='a', dtype='int32', dim=1, emb_count=11, emb_size=5),
    Feature(name='a_len', name_from='f', dtype='int64', dim=1, for_train=False),
    Feature(name='b', dtype='int64', dim=1, emb_count=101, emb_size=10),
    Feature(name='c', dtype='int64', dim=2, dense=True, emb_share='b'),
    Feature(name='c_weight', dtype='int64', dim=2, dense=True, for_train=False),
    Feature(name='c_len', dtype='int64', dim=1, for_train=False),
    Feature(name='d', dtype='float32', dim=1),
    Feature(name='e', dtype='float32', dim=2, feat_size=10, dense=True),
    Feature(name='label', dtype='float32', dim=1, for_train=False, label=True, num_class=2, onehot=True),
    Feature(name='label2', dtype='float32', dim=1, for_train=False, label=True, num_class=2, onehot=True)
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
            l, l_list, _ = sess.run([model.loss, model.loss_list, model.train_op])
            l_sum += l
            l1_sum += l_list['label']
            l2_sum += l_list['label2']
            step += 1
            if step % 10 == 0:
                print(f'step: {step}   loss: {l_sum/step:.4f}   loss_label: {l1_sum/step:.4f}   loss_label2: {l2_sum/step:.4f}')
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
import numpy as np
args.dense_feats = {'e': np.random.randn(101*10).reshape(101, 10).astype('float32')}
ckpt2pb(args, features, YouTubeRank)


# 加载pb并预测
path = os.path.join(args.model_path, args.model_name)
pb_loader = load_pb(path, features, out_name='proba_union')
print(pb_loader.inputMap)
pb_loader.predict(feed_dict={
    'a': [[2, 2, 2]],
    'a_len': [[1, 1, 1]],
    'c': [[3, 42, 80], [3, 42, 80], [3, 42, 80]],
    'c_weight': [[3, 1, 2], [3, 1, 2], [3, 1, 2]],
    'c_len': [[3, 3, 3]],

    'b': [[40, 12, 4]],
    'd': [[0.49, 0.123, 0.667]]
})

