import tensorflow as tf
from deepmodel.data import TFRecordLoader, Feature
from deepmodel.models import DIN
from deepmodel.utils import save_ckpt, load_ckpt


class Args(object):
    epoch = 10
    hidden_units = [512, 256]
    batch_size = 32
    learning_rate = 0.001
    bn_training = True
    save_path = './examples/checkpoint'
    model_name = 'test.pb'


features = [
    Feature(name='id', dtype='string', dim=1, for_train=False),
    Feature(name='a', dtype='int32', dim=1, emb_count=11, emb_size=5),
    Feature(name='b', dtype='int64', dim=1, emb_count=101, emb_size=10, attention='query'),
    Feature(name='c', dtype='int64', dim=2, dense=True, emb_share='b', attention='key'),
    Feature(name='c_len', dtype='int64', dim=1, for_train=False),
    Feature(name='d', dtype='float32', dim=1),
    Feature(name='e', dtype='float32', dim=2, feat_size=10, dense=True),
    Feature(name='label', dtype='float32', dim=1, for_train=False)
]

args = Args()

# train
with tf.Session() as sess:
    inputs = TFRecordLoader(features, 'examples/test.tfrecord', repeats=args.epoch)
    batch_x = inputs.load_batch(args.batch_size)

    model = DIN(args, features, batch_x, 0.8)
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


# test
out_cols = ['id', 'label']
args.bn_training = False

tf.reset_default_graph()
with tf.Session() as sess:
    batch_x = TFRecordLoader(features, 'examples/test.tfrecord', repeats=args.epoch).load_batch(args.batch_size)

    model = DIN(args, features, batch_x, 1.0)
    model.test(out_cols)
    sess.run(tf.global_variables_initializer())

    sess = load_ckpt(sess, args.save_path)
    output = sess.run(model.output)
    print(output)
