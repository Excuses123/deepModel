
import tensorflow as tf
from deepmodel.data import TFRecordLoader, Feature
from deepmodel.models import YouTubeRank


class Args(object):
    epoch = 10
    hidden_units = [512, 256]
    batch_size = 32
    learning_rate = 0.001

features = [
    Feature(name='id', dtype='string', dim=1),
    Feature(name='a', dtype='int32', dim=1, emb_count=11, emb_size=5, for_train=True),
    Feature(name='b', dtype='int64', dim=1, emb_count=101, emb_size=10, for_train=True),
    Feature(name='c', dtype='int64', dim=2, dense=True, emb_count=101, emb_size=10, for_train=True),
    Feature(name='c_len', dtype='int64', dim=1),
    Feature(name='d', dtype='float32', dim=1, for_train=True),
    Feature(name='e', dtype='float32', dim=2, feat_size=10, dense=True, for_train=True),
    Feature(name='label', dtype='float32', dim=1)
]

args = Args()

inputs = TFRecordLoader(features, 'examples/test.tfrecord', repeats=args.epoch)
batch_x = inputs.load_batch(args.batch_size)

model = YouTubeRank(args, features, batch_x, 0.8)
model.train()
sess = tf.Session()
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


out_cols = ['id', 'label']
model = YouTubeRank(args, features, batch_x, 1.0)
model.test(out_cols)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
output = sess.run(model.output)







